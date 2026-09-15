"""Freeze the previous Seed-TTS and code-switch evaluation texts for this run."""
import hashlib
import json
from pathlib import Path

from prepare_training import ASSETS, CLEANERS, DATA, VOICE, text_to_sequence_with_tones


def main():
    output = ASSETS / 'chenmingjie/lits-tts/output'
    seed = output / 'teacher_student_step2_seedtts200_20260715/teacher/schedule_uniform'
    code_switch = output / 'eval_step230033_vs_meanflow_zh200_codeswitch50_20260717/input_archive/codeswitch_50/20260710_031154/manifest.jsonl'
    rows = []
    specs = [('ljs_en', 0, 'English', seed / 'en/seedtts_en_manifest.jsonl', 200),
             ('majestic_en', 1, 'English', seed / 'en/seedtts_en_manifest.jsonl', 200),
             ('majestic_zh', 1, 'Chinese', seed / 'zh/seedtts_zh_manifest.jsonl', 200),
             ('majestic_mixed', 1, 'Mixed', code_switch, 50)]
    for group, speaker, language, path, count in specs:
        sources = [json.loads(line) for line in path.read_text().splitlines()][:count]
        assert len(sources) == count
        for index, row in enumerate(sources):
            ids, tones, cleaned = text_to_sequence_with_tones(row['tts_text'], CLEANERS)
            assert ids and 2 not in ids and max(ids) < 173, row['tts_text']
            rows.append({'id': f'{group}/{index:04d}', 'group': group, 'speaker': speaker,
                         'language': language, 'text': row['tts_text'], 'ref_text': row['ref_text'],
                         'token_ids': ids, 'tone_ids': tones, 'phonemes': cleaned,
                         'source_manifest': str(path), 'source_index': index})
    path = DATA / 'eval_manifest.jsonl'
    content = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows)
    if path.exists() and path.read_text() != content:
        raise RuntimeError('Frozen evaluation set changed')
    path.write_text(content)
    references = {
        '0': '/ai_sds_wuzz/DATA_TTS/LJSpeech/LJSpeech_24k/wavs/LJ002-0321.wav',
        '1': str(VOICE / 'reference/ts004_cn_000001.wav'),
    }
    for reference in references.values():
        if not Path(reference).is_file():
            raise FileNotFoundError(reference)
    report = {'samples': len(rows), 'groups': {g: n for g, _, _, _, n in specs},
              'manifest_sha256': hashlib.sha256(content.encode()).hexdigest(),
              'speaker_references': references,
              'protocol': 'Fixed-text evaluation with learned speaker IDs; similarity uses fixed voice references, not Seed-TTS zero-shot prompt voices.'}
    (DATA / 'eval_protocol.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
