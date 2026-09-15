"""Read-only checkpoint diagnostics for shared speaker conditioning.

All interventions affect a separate in-memory model. No optimizer step is taken.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

REPO = Path(__file__).resolve().parents[2]
ASSETS = Path('/119010446/tts-assets')


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def synthesize(args):
    import numpy as np
    import soundfile as sf
    import soxr
    import torch
    import torch.nn.functional as F
    from lits.models.lits import LITS
    from lits.utils.audio import mel_spectrogram
    from lits.utils.model import normalize, fix_len_compatibility
    from vocos.vocoder import load_vocos_vocoder

    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.075)
    device = 'cuda:0'
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'synthesis.jsonl').exists(), 'Use a new output directory'
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    step = checkpoint['global_step']
    current = checkpoint['state_dict']['spk_emb.weight'].clone()
    trajectory = []
    for n in [1000, 27000, 73000, step]:
        p = args.checkpoint.parents[1] / f'step_{n:08d}/checkpoint.ckpt'
        if not p.exists():
            continue
        d = torch.load(p, map_location='cpu', weights_only=False)
        emb = d['state_dict']['spk_emb.weight']
        moment_rows = []
        for optimizer in d['optimizer_states']:
            for group in optimizer['param_groups']:
                if group.get('name') == 'spk_emb':
                    assert len(group['params']) == 1
                    moments = optimizer['state'][group['params'][0]]
                    moment_rows = {k: moments[k].norm(dim=1).tolist()
                                   for k in ['exp_avg', 'exp_avg_sq']}
        trajectory.append(dict(step=n, row_norms=emb.norm(dim=1).tolist(),
                               distance_to_current=(emb-current).norm(dim=1).tolist(),
                               optimizer_moment_norms=moment_rows))
        del d
    del checkpoint
    model = LITS.load_from_checkpoint(str(args.checkpoint), map_location='cpu', weights_only=False).to(device).eval()
    model.requires_grad_(False)
    model.spk_emb.weight.requires_grad_(True)
    base_emb = model.spk_emb.weight[0].detach().clone()
    reserved_emb = model.spk_emb.weight[1].detach().clone()
    assert model.n_spks == 2 and model.encoder.use_spk_cond_in_encoder
    vocoder, cfg = load_vocos_vocoder(str(REPO/'vocos/generator.ckpt'), device, REPO)
    vocoder.requires_grad_(False)
    selected_path = ASSETS/'training_runs/hifitts_premium_stage1_20260911/diagnostics/generation_25000/selected_samples.json'
    selected = json.loads(selected_path.read_text())
    assert len(selected) == 8 and all(r['speaker'] == 0 for r in selected)
    write_json(out/'selected_samples.json', selected)
    write_json(out/'metadata.json', dict(checkpoint=str(args.checkpoint), checkpoint_step=step,
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        n_steps=10, temperature=1.0, streaming=False, precision='float32',
        selected_samples_source=str(selected_path), selection='Previously fixed 2 train/val samples per language; no outcome-based selection',
        gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), trajectory=trajectory,
        interventions='Same text and shared noise prefix across embedding interventions; separate noise-control seed.',
        limitations=['All training rows use speaker 0; row 1 is untrained.',
                     'Zero/reserved vectors are out of distribution and do not represent known voices.',
                     'Acoustic sensitivity and nonzero gradients do not establish speaker-identity control.',
                     'Raw Mel differences can include alignment/prosody changes; decoder_zero holds mu and durations fixed.'],
        started_at_unix=time.time()))

    with (out/'synthesis.jsonl').open('w', buffering=1) as stream:
        for index, row in enumerate(selected):
            name = row['diagnostic_id']
            dest = out/name
            dest.mkdir(exist_ok=False)
            x = torch.tensor([row['ids']], device=device)
            xt = torch.tensor([row['tones']], device=device)
            xl = torch.tensor([len(row['ids'])], device=device)
            sid = torch.tensor([0], device=device)
            audio, sr = sf.read(row['audio'], dtype='float32')
            if sr != 24000:
                audio = soxr.resample(audio, sr, 24000, quality='HQ')
            assert audio.ndim == 1 and np.isfinite(audio).all()
            with torch.no_grad():
                model.spk_emb.weight[0].copy_(base_emb)
                real_mel = mel_spectrogram(torch.from_numpy(audio)[None].to(device), 2048, 100, 24000, 384, 1536, 0, 12000)
                n = real_mel.shape[-1]
                y = F.pad(normalize(real_mel, model.mel_mean, model.mel_std), (0, fix_len_compatibility(n)-n))
            # Probe both training branches with fixed stochastic draws, without updating anything.
            gradients = []
            for streaming, seed in [(False, 2), (True, 1)]:
                random.seed(seed)  # random.Random(2).random() > .5; seed 1 < .5.
                torch.manual_seed(20260912 + index)
                losses = model(x, xl, y, torch.tensor([n], device=device), spks=sid, x_tones=xt)
                for k, loss in zip(['duration', 'prior', 'flow'], losses[:3]):
                    grad = torch.autograd.grad(loss, model.spk_emb.weight, retain_graph=True, allow_unused=True)[0] if loss.requires_grad else None
                    gradients.append(dict(streaming=streaming, loss=k, value=float(loss.detach()),
                        gradient_connected=grad is not None,
                        row_gradient_norms=grad.norm(dim=1).tolist() if grad is not None else [0.0, 0.0]))
                del losses
            with torch.inference_mode():
                baseline = model.get_hidden_mel(x, xl, spks=sid, x_tones=xt)
                baseline_mu, baseline_logw, _ = model.encoder(x, xl, base_emb[None], x_tones=xt)
                torch.manual_seed(20260912+index)
                z = torch.randn(1, 100, 2500, device=device)
                torch.manual_seed(20270912+index)
                noise_control = torch.randn_like(z)
                modes = {'baseline':base_emb, 'scale_0p9':base_emb*.9, 'scale_1p1':base_emb*1.1,
                         'zero':torch.zeros_like(base_emb), 'reserved_1':reserved_emb,
                         'decoder_zero':base_emb, 'different_noise':base_emb}
                results = []
                for mode, emb in modes.items():
                    model.spk_emb.weight[0].copy_(emb)
                    hidden = model.get_hidden_mel(x, xl, spks=sid, x_tones=xt)
                    mu, logw, _ = model.encoder(x, xl, emb[None], x_tones=xt)
                    speaker = hidden['spks']
                    if mode == 'decoder_zero':
                        hidden = baseline
                        speaker = torch.zeros_like(base_emb)[None]
                    frames = int(hidden['y_max_length'])
                    assert 0 < frames < z.shape[-1]
                    mel = model.get_mel(hidden['mu_y'], hidden['y_mask'], 10, 1., spks=speaker,
                                        streaming=False, z=(noise_control if mode=='different_noise' else z).clone())[:,:,:frames]
                    if mode == 'baseline':
                        baseline_mel = mel.clone()
                        repeat = model.get_mel(hidden['mu_y'], hidden['y_mask'], 10, 1., spks=speaker,
                                               streaming=False, z=z.clone())[:,:,:frames]
                        repeat_error = float((repeat-mel).abs().max())
                        assert repeat_error < 1e-5, repeat_error
                    norm_mel = normalize(mel, model.mel_mean, model.mel_std)
                    norm_baseline = normalize(baseline_mel, model.mel_mean, model.mel_std)
                    metrics = dict(mode=mode, frames=frames, duration_ratio=frames/int(baseline['y_max_length']),
                        encoder_mu_rmse=float((mu-baseline_mu).square().mean().sqrt()),
                        encoder_logw_rmse=float((logw-baseline_logw).square().mean().sqrt()))
                    if mel.shape == baseline_mel.shape:
                        metrics['normalized_mel_rmse'] = float((norm_mel-norm_baseline).square().mean().sqrt())
                        metrics['mel_relative_l2'] = float((norm_mel-norm_baseline).norm()/norm_baseline.norm().clamp_min(1e-8))
                    waveform = vocoder(mel).squeeze().cpu().numpy()[:frames*384]
                    assert np.isfinite(waveform).all()
                    sf.write(dest/f'{mode}.wav', np.clip(waveform,-1,1), 24000, subtype='FLOAT')
                    record = dict(id=f'{name}/{mode}', sample=name, mode=mode, source=row['source'],
                                  split=row['split'], ref_text=row['text'], audio=str(dest/f'{mode}.wav'),
                                  duration=len(waveform)/24000, diagnostics=metrics)
                    stream.write(json.dumps(record, ensure_ascii=False)+'\n')
                    results.append(metrics)
                sf.write(dest/'original.wav', audio, 24000, subtype='FLOAT')
                stream.write(json.dumps(dict(id=f'{name}/original', sample=name, mode='original', source=row['source'],
                    split=row['split'], ref_text=row['text'], audio=str(dest/'original.wav'), duration=len(audio)/24000), ensure_ascii=False)+'\n')
                write_json(dest/'diagnostic.json', dict(text=row['text'], gradients=gradients, interventions=results,
                    baseline_repeat_max_error=repeat_error))
                model.spk_emb.weight[0].copy_(base_emb)
            print(json.dumps(dict(completed=index+1, total=len(selected), sample=name,
                                 gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30)), flush=True)
    print('SYNTHESIS_COMPLETE', flush=True)


def speakers(args):
    import sys
    import numpy as np
    import torch
    sys.path.insert(0, str(REPO/'data_generation/majestic_voice'))
    from quality_metrics import SpeakerMetrics, audio16
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.075)
    rows = [json.loads(line) for line in (args.output/'synthesis.jsonl').read_text().splitlines()]
    scorer = SpeakerMetrics(rows[0]['audio'])
    embeddings = {}
    for i, row in enumerate(rows):
        w,c = scorer.embeddings(audio16(row['audio']))
        embeddings[row['id']] = (w.cpu().numpy().reshape(-1), c.cpu().numpy().reshape(-1))
        print(f'SPEAKER {i+1}/{len(rows)}', flush=True)
    with (args.output/'speaker_metrics.jsonl').open('w') as stream:
        for row in rows:
            v=embeddings[row['id']]
            baseline=embeddings[row['sample']+'/baseline']
            original=embeddings[row['sample']+'/original']
            result=dict(id=row['id'], sample=row['sample'], mode=row['mode'],
                        wavlm_cosine_to_baseline=float(np.dot(v[0],baseline[0])),
                        camp_cosine_to_baseline=float(np.dot(v[1],baseline[1])),
                        wavlm_cosine_to_original=float(np.dot(v[0],original[0])),
                        camp_cosine_to_original=float(np.dot(v[1],original[1])))
            stream.write(json.dumps(result)+'\n')
    np.savez_compressed(args.output/'speaker_vectors.npz', **{k.replace('/','__')+'__'+suffix:v[j]
        for k,v in embeddings.items() for j,suffix in enumerate(['wavlm','camp'])})
    print('SPEAKER_COMPLETE', flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['synthesize','speakers'], required=True)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    {'synthesize':synthesize,'speakers':speakers}[args.stage](args)
