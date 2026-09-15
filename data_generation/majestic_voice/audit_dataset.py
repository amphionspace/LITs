"""Validate the completed dataset against source texts, gates, and audio headers."""
import json
import math
import os
from collections import Counter

import soundfile as sf

from quality_worker import ROOT, records, tonal_match
from synthesize import save_json


def main():
    job = json.loads((ROOT / 'job.json').read_text())
    if job.get('status') != 'complete':
        raise RuntimeError('Wait for synthesis, quality checks, and retry rounds to finish')
    thresholds = json.loads((ROOT / 'quality/thresholds_used.json').read_text())
    sources = {r['id']: r for r in records(ROOT / 'requests.jsonl')}
    selected = records(ROOT / 'quality/selected.jsonl')
    selected_ids = [r['source_id'] for r in selected]
    assert len(selected_ids) == len(set(selected_ids)), 'Duplicate admitted source'
    assert set(selected_ids) <= sources.keys(), 'Unknown admitted source'
    expected_manifests = {(split, rate): [] for split in ('train', 'test') for rate in (24, 48)}
    total_seconds = 0.0
    for row in selected:
        source = sources[row['source_id']]
        assert row['text'] == source['text'], f"Changed target transcript: {row['source_id']}"
        assert row['dnsmos_protocol'] == 'dns-challenge-window-count-v1'
        assert math.isfinite(row['cer']) and (row['cer'] <= thresholds['max_cer'] or
            (thresholds.get('allow_exact_tonal_pinyin', False) and tonal_match(source['text'], row['asr_text'])))
        if 'max_wer' in thresholds:
            assert math.isfinite(row['wer']) and row['wer'] <= thresholds['max_wer']
        for key in ('wavlm_similarity', 'camp_similarity', 'dnsmos_ovrl', 'dnsmos_sig', 'dnsmos_bak'):
            assert math.isfinite(row[key]) and row[key] >= thresholds['min_' + key], (row['id'], key)
        duration = []
        for rate in (24, 48):
            path = ROOT / f'wavs_{rate}k' / (row['source_id'] + '.wav')
            info = sf.info(path)
            assert info.samplerate == rate * 1000 and info.channels == 1 and info.subtype == 'PCM_16'
            assert info.frames > 0 and path.stat().st_size >= info.frames * 2
            assert os.path.samefile(path, row[f'output_{rate}k']), 'Final link differs from scored candidate'
            duration.append(info.duration)
            relative = path.relative_to(ROOT).with_suffix('')
            expected_manifests[(source['split'], rate)].append(f"{relative}|{source['text']}")
        assert abs(duration[0] - duration[1]) <= 1 / 24000
        total_seconds += duration[0]
    for (split, rate), expected in expected_manifests.items():
        name = f'{split}.txt' if rate == 24 else f'{split}_48k.txt'
        assert (ROOT / name).read_text().splitlines() == expected, f'Manifest mismatch: {name}'
    accepted_set = set(selected_ids)
    for rate in (24, 48):
        directory = ROOT / f'wavs_{rate}k'
        actual = {str(p.relative_to(directory).with_suffix('')) for p in directory.rglob('*.wav')}
        assert actual == accepted_set, 'Final directory contains unadmitted or missing audio'
    scored = records(ROOT / 'quality/candidates_scored.jsonl')
    events = records(ROOT / 'logs/synthesis.jsonl')
    generated = {r['id']: r for r in events if r.get('status') == 'ok'}
    assert {r['id'] for r in scored} == generated.keys(), 'Unscored generated candidates'
    assert all(r.get('dnsmos_protocol') == 'dns-challenge-window-count-v1' for r in scored)
    failed = {r['id']: r for r in events if r.get('status') == 'failed' and r['id'] not in generated}
    attempted_sources = {r['source_id'] for r in [*generated.values(), *failed.values()]}
    assert attempted_sources == sources.keys(), 'Not all source texts were attempted'
    rejections = Counter(reason.split(':')[0] for r in scored for reason in r['rejection_reasons'])
    report = {'status': 'passed', 'source_texts': len(sources), 'attempted_texts': len(attempted_sources),
              'generated_candidates': len(generated), 'fully_scored_candidates': len(scored),
              'invalid_audio_candidates': len(failed),
              'accepted_texts': len(selected), 'excluded_texts': len(sources) - len(selected),
              'accepted_train': len(expected_manifests[('train', 24)]),
              'accepted_test': len(expected_manifests[('test', 24)]),
              'accepted_audio_hours': round(total_seconds / 3600, 4),
              'candidate_rejection_reasons': dict(rejections), 'thresholds': thresholds}
    save_json(ROOT / 'quality/final_audit.json', report)
    save_json(ROOT / 'pipeline_progress.json', {**report, 'status': 'complete'})
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
