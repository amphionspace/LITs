"""Reproduce the text-only generation queue and verified reference selection."""
import json
import re
import shutil
from pathlib import Path

from quality_worker import ROOT, ASSETS
from synthesize import save_json

SOURCE = Path('/ai_sds_wuzz/DATA_TTS/vits_dataset_zh_biaobei/vits_dataset_zh_biaobei_wav')


def main():
    for directory in ('reference', 'logs', 'quality', 'candidates'):
        (ROOT / directory).mkdir(parents=True, exist_ok=True)
    folder = ASSETS / 'Archive/TS-004_大气女声/CN-000001-000253'
    line = next(line for line in (folder / 'text.txt').read_text().splitlines()
                if line.startswith('000001\t'))
    prompt = re.sub(r'#[0-9]+', '', line.split('\t', 1)[1]).strip()
    reference = ROOT / 'reference/ts004_cn_000001.wav'
    if not reference.exists():
        shutil.copy2(folder / 'wav/000001.wav', reference)
    (ROOT / 'reference/prompt_text.txt').write_text(prompt + '\n')
    requests, counts, seen = [], {}, set()
    for split in ('train', 'test'):
        original = (SOURCE / f'{split}.txt').read_text()
        (ROOT / f'{split}.source.txt').write_text(original)
        rows = []
        for line in original.splitlines():
            identifier, text = line.split('|', 1)
            if not identifier.startswith(split + '/') or '..' in Path(identifier).parts:
                raise ValueError(f'Invalid source ID: {identifier}')
            if identifier in seen or not text.strip():
                raise ValueError(f'Duplicate ID or empty text: {identifier}')
            seen.add(identifier)
            if not (SOURCE / (identifier + '.wav')).is_file():
                raise FileNotFoundError(identifier)
            rows.append({'id': identifier, 'split': split, 'text': text.strip(),
                         **{f'output_{rate}k': str(ROOT / f'candidates/wavs_{rate}k' / (identifier + '.wav'))
                            for rate in (24, 48)}})
        requests.extend(rows)
        counts[split] = len(rows)
    if counts != {'train': 9000, 'test': 1000}:
        raise ValueError(f'Unexpected source counts: {counts}')
    request_path = ROOT / 'requests.jsonl'
    if request_path.exists():
        old = [json.loads(line) for line in request_path.read_text().splitlines()]
        if old != requests:
            raise RuntimeError('Existing queue differs; use a new output directory for changed inputs')
    else:
        request_path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in requests))
    if not (ROOT / 'job.json').exists():
        save_json(ROOT / 'job.json', {'status': 'prepared', 'model': str(ASSETS / 'VoxCPM2'),
                                     'dataset_name': 'MajesticVoice', 'reference_audio': str(reference),
                                     'prompt_text': prompt, 'source': str(SOURCE), 'counts': counts})
    print(json.dumps({'counts': counts, 'reference': str(reference)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
