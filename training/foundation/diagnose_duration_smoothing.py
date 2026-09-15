"""Read-only duration-distribution diagnostic on the fixed foundation validation set."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def pair_metrics(target, prediction):
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if len(target) < 2:
        return None
    ts, ps = float(target.std()), float(prediction.std())
    tm, pm = float(target.mean()), float(prediction.mean())
    return dict(n=len(target), mas_mean=tm, predicted_mean=pm, mas_std=ts, predicted_std=ps,
                std_ratio=ps / ts if ts > 0 else None,
                mas_cv=ts / tm, predicted_cv=ps / pm,
                cv_ratio=(ps / pm) / (ts / tm) if ts > 0 else None,
                pearson=float(np.corrcoef(target, prediction)[0, 1]) if ts > 0 and ps > 0 else None,
                mae_frames=float(np.abs(prediction - target).mean()), mean_ratio=pm / tm,
                regression_slope=float(np.mean((target-tm)*(prediction-pm))/ts**2) if ts > 0 else None)


def summarize(records):
    result = {}
    for lang in ('en', 'zh', 'all'):
        rows = [r for r in records if lang == 'all' or r['language'] == lang]
        result[lang] = {}
        for subset in ('all_tokens', 'speech', 'supervised_speech'):
            result[lang][subset] = {}
            for variant in ('raw', 'ceil', 'inference'):
                pairs, sentence_metrics, ids = [], [], []
                for r in rows:
                    mask = np.asarray(r[subset], dtype=bool)
                    target, pred = np.asarray(r['mas'])[mask], np.asarray(r[variant])[mask]
                    if not len(target):
                        continue
                    pairs.append((target, pred))
                    ids.append(np.asarray(r['ids'])[mask])
                    met = pair_metrics(target, pred)
                    if met is not None:
                        sentence_metrics.append(met)
                if not pairs:
                    continue
                target = np.concatenate([p[0] for p in pairs])
                pred = np.concatenate([p[1] for p in pairs])
                stat = pair_metrics(target, pred)
                stat['sentences'] = len(pairs)
                stat['per_sentence'] = {}
                for metric in ('pearson', 'std_ratio', 'cv_ratio', 'mean_ratio'):
                    values = np.array([s[metric] for s in sentence_metrics if s[metric] is not None])
                    rng = np.random.default_rng(197000)
                    boot = np.median(rng.choice(values, (2000, len(values))), axis=1)
                    stat['per_sentence'][metric] = dict(median=float(np.median(values)),
                        p10=float(np.quantile(values, .1)), p90=float(np.quantile(values, .9)),
                        median_bootstrap_ci95=np.quantile(boot, [.025, .975]).tolist(),
                        fraction_below_one=float(np.mean(values < 1)))
                # Equalize each utterance's mean duration: tests spread independently of global pace.
                normalized_t = np.concatenate([t/t.mean() for t, p in pairs])
                normalized_p = np.concatenate([p/p.mean() for t, p in pairs])
                stat['sentence_mean_normalized'] = pair_metrics(normalized_t, normalized_p)
                # Remove token-identity means: distinguish within-phone variation from phone inventory.
                all_ids = np.concatenate(ids)
                residual_t, residual_p = [], []
                for token in np.unique(all_ids):
                    selected = all_ids == token
                    if selected.sum() >= 10:
                        residual_t.extend(target[selected] - target[selected].mean())
                        residual_p.extend(pred[selected] - pred[selected].mean())
                a, b = np.asarray(residual_t), np.asarray(residual_p)
                stat['within_token_centered'] = dict(n=len(a), std_ratio=float(b.std()/a.std()),
                    pearson=float(np.corrcoef(a, b)[0, 1]))
                result[lang][subset][variant] = stat
    return result


def main(args):
    import torch
    from torch.utils.data import DataLoader
    from lits.models.lits import LITS
    from lits.text.char_symbols.langs.zh_en_rhyme_body_tone_tokens import symbols, _punctuation
    from lits.text.bopomofo_utils import BOPOMOFO_TONES
    from lits.utils.infer_duration_floor import apply_infer_duration_patches
    from training.foundation.data import IndexedAudio, worker_init

    torch.set_num_threads(args.threads)
    torch.manual_seed(197000)
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    start = time.time()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    assert checkpoint['global_step'] == 197000
    model = LITS.load_from_checkpoint(str(args.checkpoint), map_location='cpu', weights_only=False)
    model.eval().requires_grad_(False)
    assert model.n_vocab == len(symbols) == 173
    assert model.duration_constrained_mas and not model.use_precomputed_durations
    sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    del checkpoint
    # This isolated, unsaved model retains its original MAS/duration/prior forward.
    # The unrelated flow computation is skipped; no model weights are changed.
    model.decoder.compute_loss = lambda x1, **kwargs: (x1.new_zeros(()), None)
    captured = {}
    handle = model.encoder.register_forward_hook(lambda module, inputs, outputs: captured.update(outputs=outputs))
    data = IndexedAudio(args.database, 'val',
                        dict(mel_mean=float(model.mel_mean), mel_std=float(model.mel_std)), args.per_source)
    loader = DataLoader(data, batch_size=None, num_workers=4, worker_init_fn=worker_init)
    excluded = {'<blank>', '<sil>', '<unk>', '_', *_punctuation, *BOPOMOFO_TONES}
    records = []
    metadata = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=sha, checkpoint_step=197000,
        database=str(args.database), selection='IndexedAudio val fixed seed 20260911, max 256/source (training validation selection)',
        per_source=args.per_source, device='cpu', precision='float32', sample_std_ddof=0,
        excluded_speech_symbols=sorted(excluded), length_scale=1.0,
        flow_skipped=True, original_mas_and_duration_forward=True,
        duration_patches=bool(getattr(model, 'apply_infer_duration_patches', False)),
        started_at_unix=start)
    (output/'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
    with (output/'samples.jsonl').open('w', buffering=1) as stream, torch.inference_mode():
        for i, batch in enumerate(loader):
            row = data.row(i)
            x = batch['x'].long()[None]
            xt = batch['x_tones'].long()[None] if batch['x_tones'] is not None else None
            y = batch['y'][None]
            xl, yl = torch.tensor([x.shape[-1]]), torch.tensor([y.shape[-1]])
            sid = torch.tensor([int(batch['spk'])])
            assert sid.item() == 0
            outputs = model(x, xl, y, yl, spks=sid, x_tones=xt)
            _, logw, xmask = captured['outputs']
            attn = outputs[3]
            mas = attn.sum(-1)
            assert torch.equal(attn.sum(1), torch.ones(1, y.shape[-1]))
            path = attn[0].argmax(0)
            assert bool((path[1:] >= path[:-1]).all()) and bool((mas >= 1).all())
            duration_mask, _ = model._mask_duration_outliers(x, mas[:, None], xmask, sid, xt)
            expected = (((logw-torch.log(mas[:, None]+1e-8))*xmask).square()*duration_mask).sum()/xl.sum()
            torch.testing.assert_close(expected, outputs[0])
            raw = torch.exp(logw)*xmask
            ceil = torch.ceil(raw)
            inference = model._clamp_inference_tone_durations(ceil, x, xmask)
            if getattr(model, 'apply_infer_duration_patches', False):
                inference = apply_infer_duration_patches(inference, x, xmask, n_vocab=model.n_vocab)
            speech = np.array([symbols[j] not in excluded for j in row['ids']])
            record = dict(sample_id=int(data.ids[i]), language='en' if row['source']=='HiFiTTS' else 'zh',
                source=row['source'], audio=row['audio'], text=row['text'], ids=row['ids'],
                symbols=[symbols[j] for j in row['ids']], mas=mas.flatten().tolist(),
                raw=raw.flatten().tolist(), ceil=ceil.flatten().tolist(), inference=inference.flatten().tolist(),
                all_tokens=[True]*len(row['ids']), speech=speech.tolist(),
                supervised_speech=(speech & duration_mask.flatten().bool().numpy()).tolist(),
                duration_supervision_mask=duration_mask.flatten().bool().tolist(),
                mas_frames=int(yl.item()), duration_loss=float(outputs[0]), mas_stats=outputs[4])
            # MAS stats may contain scalar tensors.
            record['mas_stats'] = {k:float(v) for k,v in record['mas_stats'].items()}
            for key in ('raw', 'ceil', 'inference'):
                assert np.isfinite(record[key]).all() and min(record[key]) > 0
            records.append(record)
            stream.write(json.dumps(record, ensure_ascii=False)+'\n')
            if (i+1)%32 == 0:
                print(json.dumps(dict(completed=i+1, total=len(data), seconds=time.time()-start)), flush=True)
    handle.remove()
    summary = summarize(records)
    (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    metadata.update(completed_at_unix=time.time(), elapsed_seconds=time.time()-start, samples=len(records),
                    all_alignment_and_duration_loss_checks_passed=True)
    (output/'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(status='complete',samples=len(records),seconds=time.time()-start)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--database', type=Path, default=Path('/119010446/tts-assets/data_24k/foundation/dataset.sqlite'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--per-source', type=int, default=256)
    parser.add_argument('--threads', type=int, default=4)
    main(parser.parse_args())
