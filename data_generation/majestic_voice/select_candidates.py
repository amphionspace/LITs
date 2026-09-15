"""Publish only fully scored candidates that pass every fixed quality gate."""
import json
import math
import os
from pathlib import Path

from quality_worker import ROOT, records, tonal_match, quality_config
from synthesize import save_json


def assess():
    thresholds = quality_config()
    asr = {r['id']: r for r in records(ROOT / 'quality/asr.jsonl')}
    metrics = {r['id']: r for r in records(ROOT / 'quality/metrics.jsonl')}
    generated = {r['id']: r for r in records(ROOT / 'logs/synthesis.jsonl') if r.get('status') == 'ok'}
    best, judged = {}, {}
    for key, row in generated.items():
        if key not in asr or key not in metrics:
            continue
        combined = {**row, **asr[key], **metrics[key]}
        combined['tonal_pinyin_match'] = tonal_match(row['text'], asr[key].get('asr_text', ''))
        pinyin_allowed = thresholds.get('allow_exact_tonal_pinyin', False) and combined['tonal_pinyin_match']
        reasons = []
        for setting, threshold in thresholds.items():
            if setting not in {'max_cer', 'max_wer', 'min_wavlm_similarity', 'min_camp_similarity',
                               'min_dnsmos_ovrl', 'min_dnsmos_sig', 'min_dnsmos_bak'}:
                continue
            direction, metric = setting.split('_', 1)
            value = combined.get(metric)
            if value is None or not math.isfinite(value):
                reasons.append(f'{metric}:missing_or_nonfinite')
            elif (direction == 'min' and value < threshold) or (direction == 'max' and value > threshold):
                if not (metric == 'cer' and pinyin_allowed):
                    reasons.append(f'{metric}:{value:.4f}')
        combined['rejection_reasons'] = reasons
        combined['passed'] = not reasons
        raw_cer = combined.get('cer')
        valid_cer = isinstance(raw_cer, (int, float)) and math.isfinite(raw_cer)
        combined['text_acceptance'] = ('cer' if valid_cer and raw_cer <= thresholds['max_cer']
                                       else 'exact_tonal_pinyin' if valid_cer and pinyin_allowed else 'rejected')
        effective_cer = 0.0 if pinyin_allowed else raw_cer if valid_cer else float('inf')
        combined['selection_score'] = None if reasons else (
            2 * combined['wavlm_similarity'] + combined['camp_similarity']
            + 0.25 * combined['dnsmos_ovrl'] - 2 * effective_cer)
        judged[key] = combined
        if not reasons:
            source = row['source_id']
            if source not in best or combined['selection_score'] > best[source]['selection_score']:
                best[source] = combined
    return generated, judged, best


def publish(best, complete=False):
    sources = records(ROOT / 'requests.jsonl')
    previous = {r['source_id'] for r in records(ROOT / 'quality/selected.jsonl')}
    stale = previous - best.keys()
    if complete:
        stale |= {r['id'] for r in sources} - best.keys()
    for identifier in stale:
        for rate in (24, 48):
            (ROOT / f'wavs_{rate}k' / (identifier + '.wav')).unlink(missing_ok=True)
    selected, manifests = [], {('train', rate): [] for rate in (24, 48)} | {('test', rate): [] for rate in (24, 48)}
    for source in sources:
        if source['id'] not in best:
            continue
        winner = dict(best[source['id']])
        for rate in (24, 48):
            target = ROOT / f'wavs_{rate}k' / (source['id'] + '.wav')
            target.parent.mkdir(parents=True, exist_ok=True)
            candidate = Path(winner[f'output_{rate}k'])
            if not target.exists() or not os.path.samefile(candidate, target):
                temp = target.with_suffix('.wav.tmp')
                temp.unlink(missing_ok=True)
                os.link(candidate, temp)
                temp.replace(target)
            winner[f'final_{rate}k'] = str(target)
            relative = target.relative_to(ROOT).with_suffix('')
            manifests[(source['split'], rate)].append(f"{relative}|{source['text']}\n")
        selected.append(winner)
    for (split, rate), rows in manifests.items():
        name = f'{split}.txt' if rate == 24 else f'{split}_48k.txt'
        path = ROOT / name
        temporary = path.with_suffix('.txt.tmp')
        temporary.write_text(''.join(rows))
        temporary.replace(path)
    (ROOT / 'quality/selected.jsonl.tmp').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in selected))
    (ROOT / 'quality/selected.jsonl.tmp').replace(ROOT / 'quality/selected.jsonl')
    remaining = [r['id'] for r in sources if r['id'] not in best]
    save_json(ROOT / 'quality/selection_summary.json', {
        'status': 'complete' if complete else 'running', 'target_texts': len(sources),
        'accepted_texts': len(selected), 'unaccepted_texts': len(remaining),
        'train': len(manifests[('train', 24)]), 'test': len(manifests[('test', 24)]),
        'unaccepted_ids': remaining})
    return remaining


if __name__ == '__main__':
    generated, judged, best = assess()
    remaining = publish(best)
    print(json.dumps({'generated': len(generated), 'scored': len(judged),
                      'accepted': len(best), 'remaining': len(remaining)}))
