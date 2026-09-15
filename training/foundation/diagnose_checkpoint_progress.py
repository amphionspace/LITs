"""Compare final-checkpoint predicted/MAS synthesis on prior Chinese probes."""
import argparse
import hashlib
import json
from pathlib import Path
import random


def main(root):
    import numpy as np
    import soundfile as sf
    import torch
    import torch.nn.functional as F
    from lits.models.lits import LITS
    from lits.utils.model import normalize, fix_len_compatibility, sequence_mask
    from vocos.vocoder import load_vocos_vocoder

    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.075)
    repo = Path(__file__).resolve().parents[2]
    old = root/'diagnostics/model_health_145000'
    checkpoint = root/'eval/step_00200000/checkpoint.ckpt'
    out = root/'diagnostics/checkpoint_progress_200000'
    out.mkdir(exist_ok=False)
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert ckpt['global_step'] == 200000
    def finite_tensors(value):
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all()
            return 1
        if isinstance(value, dict):
            return sum(finite_tensors(v) for v in value.values())
        if isinstance(value, (tuple, list)):
            return sum(finite_tensors(v) for v in value)
        return 0
    tensor_count = finite_tensors(ckpt['state_dict'])
    optimizer_count = finite_tensors(ckpt['optimizer_states'])
    checkpoint_hash = hashlib.file_digest(checkpoint.open('rb'), 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    expected = json.loads((checkpoint.parent/'checkpoint_metadata.json').read_text())['sha256']
    assert checkpoint_hash == expected
    final_hash = hashlib.sha256((root/'checkpoints/final.ckpt').read_bytes()).hexdigest()
    assert final_hash == checkpoint_hash
    del ckpt
    model = LITS.load_from_checkpoint(str(checkpoint), map_location='cpu', weights_only=False).to('cuda:0').eval().requires_grad_(False)
    vocoder, _ = load_vocos_vocoder(str(repo/'vocos/generator.ckpt'), torch.device('cuda:0'), repo)
    selected = json.loads((old/'selected_samples.json').read_text())
    previous = {r['id']:r for r in map(json.loads, (old/'synthesis.jsonl').read_text().splitlines())}
    meta = dict(checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash, final_checkpoint_matches=True,
        finite_state_tensors=tensor_count, finite_optimizer_tensors=optimizer_count,
        steps_compared=[145000,200000], chinese_samples=12, generated_audio_count=24, asr_audio_count=48,
        controls='Same selected Chinese texts, seed-0 noise, speaker 0, FP32, non-streaming, 10 Euler steps; MAS uses saved recording Mel and total frames',
        asr='All 48 old/new predicted/MAS clips re-transcribed in one batch-size-1 run')
    (out/'metadata.json').write_text(json.dumps(meta, indent=2)+'\n')
    with (out/'synthesis.jsonl').open('w', buffering=1) as stream, torch.inference_mode():
        for index, row in enumerate(selected):
            if not row['real_audio'] or row['source'] != 'Premium':
                continue
            name = row['diagnostic_id']
            dest = out/name
            dest.mkdir()
            x = torch.tensor([row['ids']], device='cuda:0')
            xt = torch.tensor([row['tones']], device='cuda:0')
            xl = torch.tensor([x.shape[-1]], device='cuda:0')
            sid = torch.tensor([0], device='cuda:0')
            speaker = model.spk_emb(sid)
            hidden = model.get_hidden_mel(x, xl, spks=sid, x_tones=xt)
            pred_n = int(hidden['y_max_length'])
            saved = np.load(old/name/'alignment.npz')
            real = torch.from_numpy(saved['real_mel']).to('cuda:0')
            n = real.shape[-1]
            yp = F.pad(normalize(real, model.mel_mean, model.mel_std), (0, fix_len_compatibility(n)-n))
            yl = torch.tensor([n], device='cuda:0')
            mask = sequence_mask(yl, yp.shape[-1]).unsqueeze(1).float()
            mu_x, _, _ = model.encoder(x, xl, speaker, x_tones=xt)
            random.seed(2)
            losses = model(x, xl, yp, yl, spks=sid, x_tones=xt)
            attn = losses[3]
            assert torch.equal(attn.sum(1), mask[:, 0])
            path = attn[0, :, :n].argmax(0)
            assert bool((path[1:] >= path[:-1]).all())
            mu = torch.bmm(attn.transpose(1, 2), mu_x.transpose(1, 2)).transpose(1, 2)
            torch.manual_seed(2026091202+index)
            z = torch.randn(1, 100, 3000, device='cuda:0')
            for mode, cond, ymask, frames in [('pred', hidden['mu_y'], hidden['y_mask'], pred_n), ('mas', mu, mask, n)]:
                mel = model.get_mel(cond, ymask, 10, 1., spks=speaker, streaming=False, z=z.clone())[:, :, :frames]
                wave = vocoder(mel).squeeze().cpu().numpy()[:frames*384]
                assert np.isfinite(wave).all()
                audio = dest/f'{mode}.wav'
                sf.write(audio, np.clip(wave, -1, 1), 24000, subtype='FLOAT')
                record = dict(id=f'{name}/{mode}_200000', sample=name, mode=f'{mode}_200000', source=row['source'],
                    split=row['split'], group=row['group'], ref_text=row['text'], audio=str(audio), duration=len(wave)/24000)
                stream.write(json.dumps(record, ensure_ascii=False)+'\n')
                prev = dict(previous[f'{name}/{mode}_10_s0'])
                prev.update(id=f'{name}/{mode}_145000', mode=f'{mode}_145000')
                stream.write(json.dumps(prev, ensure_ascii=False)+'\n')
            diag = dict(sample=name, text=row['text'], group=row['group'], real_frames=n, predicted_frames=pred_n,
                mas_durations=attn.sum(-1)[0].cpu().tolist(),
                changed_mas_frame_ratio=float((path.cpu() != torch.from_numpy(saved['alignment'])[0, :, :n].argmax(0)).float().mean()),
                prior_mse=float(((yp-mu).square()*mask).sum()/(mask.sum()*100)))
            (dest/'diagnostic.json').write_text(json.dumps(diag, ensure_ascii=False, indent=2)+'\n')
            np.savez_compressed(dest/'alignment.npz', alignment=attn.cpu().numpy(), mas_mu=mu.cpu().numpy())
            print(f'COMPLETE {name}', flush=True)
    print('CHECKPOINT_PROGRESS_SYNTHESIS_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    main(parser.parse_args().run)
