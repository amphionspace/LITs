"""Isolated, paired MAS bound ablations on the completed health sample set."""
import argparse
import json
import math
from pathlib import Path


def main(root):
    import numpy as np
    import soundfile as sf
    import torch
    import torch.nn.functional as F
    from lits.models.lits import LITS
    from lits.utils import monotonic_align
    from lits.utils.model import normalize, fix_len_compatibility, sequence_mask
    from vocos.vocoder import load_vocos_vocoder

    repo = Path(__file__).resolve().parents[2]
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.075)
    meta = json.loads((root/'metadata.json').read_text())
    model = LITS.load_from_checkpoint(meta['checkpoint'], map_location='cpu', weights_only=False).to('cuda:0').eval().requires_grad_(False)
    vocoder, _ = load_vocos_vocoder(str(repo/'vocos/generator.ckpt'), torch.device('cuda:0'), repo)
    out = root/'mas_constraints'
    out.mkdir(exist_ok=False)
    selected = json.loads((root/'selected_samples.json').read_text())
    (out/'metadata.json').write_text(json.dumps(dict(checkpoint=meta['checkpoint'], checkpoint_sha256=meta['checkpoint_sha256'],
        modes=['full', 'none', 'floor_only', 'ceiling_only', 'tone_only'], sample_count=12, expected_audio_count=60,
        precision='float32', steps=10, temperature=1, speaker_id=0, streaming=False,
        controls='Same recorded frames, saved real Mel, encoder, decoder, vocoder and seed-0 noise as model-health diagnostic',
        tone_only='Keep actual tone-token bounds; remove historical body-token bounds'), indent=2)+'\n')
    with (out/'synthesis.jsonl').open('w', buffering=1) as stream, torch.inference_mode():
        for index, row in enumerate(selected):
            if not row['real_audio'] or row['source'] != 'Premium':
                continue
            name = row['diagnostic_id']
            saved = np.load(root/name/'alignment.npz')
            real = torch.from_numpy(saved['real_mel']).to('cuda:0')
            n = real.shape[-1]
            y = normalize(real, model.mel_mean, model.mel_std)
            yp = F.pad(y, (0, fix_len_compatibility(n)-n))
            mask = sequence_mask(torch.tensor([n], device='cuda:0'), yp.shape[-1]).unsqueeze(1).float()
            x = torch.tensor([row['ids']], device='cuda:0')
            xt = torch.tensor([row['tones']], device='cuda:0')
            xl = torch.tensor([x.shape[-1]], device='cuda:0')
            sid = torch.tensor([0], device='cuda:0')
            speaker = model.spk_emb(sid)
            mu_x, _, xmask = model.encoder(x, xl, speaker, x_tones=xt)
            factor = -.5*torch.ones_like(mu_x)
            log_prior = (torch.matmul(factor.transpose(1, 2), yp**2)
                - torch.matmul(2*(factor*mu_x).transpose(1, 2), yp)
                + torch.sum(factor*mu_x**2, 1).unsqueeze(-1) - .5*math.log(2*math.pi)*model.n_feats)
            amask = (xmask.unsqueeze(-1)*mask.unsqueeze(2)).squeeze(1)
            floors = model._mas_floor_frames(x, sid, xt)
            ceilings = model._mas_ceiling_frames(x, sid, xt)
            zeros = torch.zeros_like(floors)
            tone = model._tone_mark_mask(x)
            variants = dict(full=(floors, ceilings), none=(None, None), floor_only=(floors, None),
                ceiling_only=(zeros, ceilings), tone_only=(torch.where(tone, floors, zeros), torch.where(tone, ceilings, zeros)))
            baseline = torch.from_numpy(saved['alignment']).to('cuda:0')
            torch.manual_seed(2026091202+index)
            z = torch.randn(1, 100, 3000, device='cuda:0')
            dest = out/name
            dest.mkdir()
            detail = dict(sample=name, text=row['text'], phonemes=row['phonemes'], ids=row['ids'], tones=row['tones'],
                tone_token_mask=tone[0].cpu().tolist(), floors=floors[0].cpu().tolist(), ceilings=ceilings[0].cpu().tolist(), modes={})
            paths = {}
            for mode, (lo, hi) in variants.items():
                if lo is None:
                    attn = monotonic_align.maximum_path(log_prior, amask)
                    stats = {}
                else:
                    attn, stats = monotonic_align.maximum_path_constrained(log_prior, amask, lo, hi)
                assert torch.equal(attn.sum(1), mask[:, 0])
                path = attn[0, :, :n].argmax(0)
                assert bool((path[1:] >= path[:-1]).all())
                durations = attn.sum(-1)[0]
                assert bool((durations >= 1).all())
                mu = torch.bmm(attn.transpose(1, 2), mu_x.transpose(1, 2)).transpose(1, 2)
                if mode == 'full':
                    assert torch.equal(attn, baseline), name
                    assert torch.equal(mu.cpu(), torch.from_numpy(saved['mas_mu'])), name
                mel = model.get_mel(mu, mask, 10, 1., spks=speaker, streaming=False, z=z.clone())[:, :, :n]
                wave = vocoder(mel).squeeze().cpu().numpy()[:n*384]
                assert np.isfinite(wave).all()
                wave = np.clip(wave, -1, 1)
                audio = dest/f'{mode}.wav'
                sf.write(audio, wave, 24000, subtype='FLOAT')
                baseline_audio_error = None
                if mode == 'full':
                    old, sr = sf.read(root/name/'mas_10_s0.wav', dtype='float32')
                    assert sr == 24000 and old.shape == wave.shape
                    baseline_audio_error = float(np.abs(old-wave).max())
                    assert baseline_audio_error < 1e-5, (name, baseline_audio_error)
                detail['modes'][mode] = dict(stats=stats, durations=durations.cpu().tolist(),
                    changed_frame_ratio=float((path != baseline[0, :, :n].argmax(0)).float().mean()),
                    prior_mse=float(((yp-mu).square()*mask).sum()/(mask.sum()*100)),
                    tone_frames=float(durations[tone[0]].sum()), max_token_frames=float(durations.max()),
                    baseline_audio_max_error=baseline_audio_error)
                paths[mode] = attn.cpu().numpy()
                record = dict(id=f'{name}/{mode}', sample=name, mode=mode, source=row['source'], split=row['split'],
                    group=row['group'], ref_text=row['text'], audio=str(audio), duration=len(wave)/24000)
                stream.write(json.dumps(record, ensure_ascii=False)+'\n')
            np.savez_compressed(dest/'alignments.npz', **paths)
            (dest/'diagnostic.json').write_text(json.dumps(detail, ensure_ascii=False, indent=2)+'\n')
            print(f'COMPLETE {name}', flush=True)
    print('MAS_ABLATION_COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
