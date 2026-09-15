"""Paired real-Mel, MAS-duration and predicted-duration synthesis diagnostics."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sqlite3
import time

REPO = Path(__file__).resolve().parents[2]
ASSETS = Path('/119010446/tts-assets')
DATA = ASSETS / 'data_24k/foundation'


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def synthesize(args):
    import numpy as np
    import soundfile as sf
    import soxr
    import torch
    import torch.nn.functional as F
    from lits.models.lits import LITS
    from lits.text import text_to_sequence_with_tones, phonemes_to_sequence_with_tones
    from lits.utils.audio import mel_spectrogram
    from lits.utils.model import normalize, denormalize, fix_len_compatibility, sequence_mask
    from vocos.vocoder import load_vocos_vocoder

    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.075)
    device = torch.device('cuda:0')
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'synthesis.jsonl').exists(), 'Use a new output directory'
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    step = int(ckpt['global_step'])
    del ckpt
    model = LITS.load_from_checkpoint(str(args.checkpoint), map_location='cpu', weights_only=False).to(device).eval()
    vocoder, cfg = load_vocos_vocoder(str(REPO / 'vocos/generator.ckpt'), device, REPO)
    assert (model.n_feats, model.n_vocab, cfg.sampling_rate, cfg.hop_size) == (100, 173, 24000, 384)
    stats = json.loads((DATA / 'mel_statistics.json').read_text())
    assert abs(float(model.mel_mean) - stats['mel_mean']) < 1e-5
    assert abs(float(model.mel_std) - stats['mel_std']) < 1e-5
    selected = []
    rng = random.Random(20260911)
    with sqlite3.connect(f'file:{DATA}/dataset.sqlite?mode=ro', uri=True) as db:
        for split in ['train', 'val']:
            for source in ['HiFiTTS', 'Premium']:
                candidates = db.execute('select id,payload from samples where split=? and source=? and duration between 3 and 6 order by id', (split, source)).fetchall()
                for i, (sample_id, payload) in enumerate(rng.sample(candidates, 2)):
                    selected.append(dict(json.loads(payload), sample_id=sample_id,
                                         diagnostic_id=f'{split}_{"en" if source == "HiFiTTS" else "zh"}_{i}'))
    write_json(out / 'selected_samples.json', selected)
    write_json(out / 'metadata.json', dict(checkpoint=str(args.checkpoint), checkpoint_step=step,
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        selection_seed=20260911, selection='2 samples per source/split, duration 3–6 seconds, uniformly sampled by row',
        n_steps=10, temperature=1.0, streaming=False, model_precision='float32',
        mel_statistics=stats | {'sample_ids': 'omitted'}, gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        started_at_unix=time.time(), mas_note='MAS is model-derived, not ground-truth phoneme alignment.'))

    with (out / 'synthesis.jsonl').open('w', buffering=1) as stream, torch.inference_mode():
        for idx, row in enumerate(selected):
            name = row['diagnostic_id']
            dest = out / name
            dest.mkdir()
            ids, tones, phonemes = text_to_sequence_with_tones(row['text'], ['en_zh_dict_mixed_rhyme_body_tone_cleaners'], prepend_sil=True)
            assert ids == row['ids'] and tones == row['tones'] and phonemes == row['phonemes'], name
            encoded, parallel, _ = phonemes_to_sequence_with_tones(row['phonemes'], ['en_zh_dict_mixed_rhyme_body_tone_cleaners'])
            assert [1] + encoded == ids and [0] + parallel == tones, name
            audio, sr = sf.read(row['audio'], dtype='float32')
            if sr != 24000:
                audio = soxr.resample(audio, sr, 24000, quality='HQ')
            assert audio.ndim == 1 and np.isfinite(audio).all()
            mel = mel_spectrogram(torch.from_numpy(audio)[None].to(device), 2048, 100, 24000, 384, 1536, 0, 12000)
            n = mel.shape[-1]
            y = normalize(mel, model.mel_mean, model.mel_std)
            reconstruction_mel = denormalize(y, model.mel_mean, model.mel_std)
            roundtrip = float((reconstruction_mel - mel).abs().max())
            assert roundtrip < 1e-5
            x = torch.tensor([ids], device=device)
            xt = torch.tensor([tones], device=device)
            xl = torch.tensor([len(ids)], device=device)
            yl = torch.tensor([n], device=device)
            yp = F.pad(y, (0, fix_len_compatibility(n) - n))
            ym = sequence_mask(yl, yp.shape[-1]).unsqueeze(1).float()
            spks = torch.tensor([0], device=device)
            spk_emb = model.spk_emb(spks)
            random.seed(20260911 + idx)
            torch.manual_seed(20260911 + idx)
            losses = model(x, xl, yp, yl, spks=spks, x_tones=xt)
            attn = losses[3]
            assert attn.shape == (1, len(ids), yp.shape[-1])
            assert torch.equal(attn.sum(1), ym[:, 0])
            mu_x, logw, xmask = model.encoder(x, xl, spk_emb, x_tones=xt)
            mas_mu = torch.bmm(attn.transpose(1, 2), mu_x.transpose(1, 2)).transpose(1, 2)
            normal = model.get_hidden_mel(x, xl, spks=spks, x_tones=xt)
            pred_n = int(normal['y_max_length'])
            assert 0 < pred_n < 2000
            torch.manual_seed(20260911 + idx)
            z = torch.randn(1, 100, max(yp.shape[-1], normal['mu_y'].shape[-1]), device=device)
            mas_mel = model.get_mel(mas_mu, ym, 10, 1.0, spks=spk_emb, streaming=False, z=z)[:, :, :n]
            pred_mel = model.get_mel(normal['mu_y'], normal['y_mask'], 10, 1.0,
                                     spks=spk_emb, streaming=False, z=z)[:, :, :pred_n]
            mas_durations = attn.sum(-1).squeeze(0)
            pred_durations = model._clamp_inference_tone_durations(torch.ceil(torch.exp(logw)) * xmask, x, xmask).squeeze()
            diagnostic = dict(sample_id=row['sample_id'], text=row['text'], phonemes=phonemes,
                token_ids=ids, frontend_reencode_matches=True, frontend_roundtrip_matches=True,
                normalization_roundtrip_max_error=roundtrip, real_frames=n, predicted_frames=pred_n,
                predicted_to_real_duration=pred_n/n, mas_duration_frames=mas_durations.cpu().tolist(),
                predicted_duration_frames=pred_durations.cpu().tolist(),
                duration_mae_frames=float((pred_durations-mas_durations).abs().mean()),
                sampled_losses={k:float(v) for k,v in zip(['duration','prior','flow'],losses[:3])})
            write_json(dest / 'diagnostic.json', diagnostic)
            np.savez_compressed(dest / 'tensors.npz', real_mel=mel.cpu().numpy(), mas_mel=mas_mel.cpu().numpy(),
                               predicted_mel=pred_mel.cpu().numpy(), alignment=attn.cpu().numpy())
            waves = {'original':audio}
            for mode, feat in [('reconstruction',reconstruction_mel),('mas_duration',mas_mel),('predicted_duration',pred_mel)]:
                waveform = vocoder(feat).squeeze().cpu().numpy()[:feat.shape[-1]*384]
                assert np.isfinite(waveform).all()
                waves[mode] = np.clip(waveform, -1, 1)
            for mode, waveform in waves.items():
                target = dest / f'{mode}.wav'
                sf.write(target, waveform, 24000, subtype='FLOAT')
                stream.write(json.dumps(dict(id=f'{name}/{mode}', sample=name, mode=mode,
                    source=row['source'], split=row['split'], ref_text=row['text'], audio=str(target),
                    duration=len(waveform)/24000, rms=float(np.sqrt(np.mean(waveform**2))),
                    peak=float(np.max(np.abs(waveform)))), ensure_ascii=False)+'\n')
            print(json.dumps(dict(completed=idx+1,total=len(selected),sample=name,
                                 duration_ratio=pred_n/n,gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30)),flush=True)
    print('SYNTHESIS_COMPLETE', flush=True)


def asr(args):
    import torch
    import sys
    from qwen_asr import Qwen3ASRModel
    sys.path.insert(0, str(REPO / 'data_generation/majestic_voice'))
    from quality_worker import cer, wer, normalize
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.075)
    batch_size = getattr(args, 'asr_batch_size', 2)
    model = Qwen3ASRModel.from_pretrained(str(ASSETS / 'Qwen3-ASR-1.7B'), dtype=torch.bfloat16,
        device_map='cuda:0', attn_implementation='sdpa', max_inference_batch_size=batch_size, max_new_tokens=256)
    rows = [json.loads(l) for l in (args.output/'synthesis.jsonl').read_text().splitlines()]
    with (args.output/'asr.jsonl').open('w',buffering=1) as stream:
        for offset in range(0,len(rows),batch_size):
            batch = rows[offset:offset+batch_size]
            predictions = model.transcribe(audio=[r['audio'] for r in batch], context='', language=None)
            for row,pred in zip(batch,predictions):
                result = dict(row, asr_text=pred.text, asr_language=pred.language,
                              cer=cer(row['ref_text'],pred.text), wer=wer(row['ref_text'],pred.text),
                              reference_characters=len(normalize(row['ref_text'])))
                stream.write(json.dumps(result,ensure_ascii=False)+'\n')
            print(f'ASR {min(offset+batch_size,len(rows))}/{len(rows)}',flush=True)
    print('ASR_COMPLETE',flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['synthesize','asr'], required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--asr-batch-size', type=int, choices=[1, 2], default=2)
    args = p.parse_args()
    {'synthesize':synthesize,'asr':asr}[args.stage](args)
