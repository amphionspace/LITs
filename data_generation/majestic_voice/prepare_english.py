"""English text queues for the MajesticVoice reference probe and augmentation."""
import argparse
import json
import random
import re
import shutil
from pathlib import Path

from quality_worker import ASSETS, normalize
from synthesize import save_json

ARCHIVE = ASSETS / 'chenmingjie/lits-tts/output'
HISTORICAL = ARCHIVE / 'ljspeech_biaobei_voxcpm2en2756_16k_balanced'
CHINESE = Path('/ai_sds_wuzz/DATA_TTS/MajesticVoice')


def text_pool():
    pool, seen = [], set()
    # Preserve the old English training/validation separation across voices.
    heldout = {normalize(line.split('|', 2)[2]) for line in (HISTORICAL / 'val.txt').read_text().splitlines()}
    def add(text, source):
        text = text.strip()
        key = normalize(text)
        if not key or key in seen or key in heldout or '|' in text:
            return
        if re.search(r'[\u4e00-\u9fff]', text):
            return
        seen.add(key)
        pool.append({'text': text, 'text_source': source})
    for line in (HISTORICAL / 'voxcpm2_biaobei_en_2756.txt').read_text().splitlines():
        wav, _, text = line.split('|', 2)
        add(text, 'historical_2756/' + Path(wav).stem)
    pack = ARCHIVE / 'voxcpm2_biaobei_trainpack_20260709'
    paths = sorted((pack / '02_text_selection').glob('tasks_b2_en_1995_gpu*.jsonl'))
    paths += sorted((pack / '04_synth_candidates').glob('*_en_ref*/tasks.jsonl'))
    for path in paths:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            add(row['text'], str(path.relative_to(ARCHIVE)) + ':' + row['utt_id'])
    train_lj = {Path(line.split('|', 1)[0]).stem for line in (HISTORICAL / 'train.txt').read_text().splitlines()
                if '/LJSpeech/' in line and line.split('|')[1] == '0'}
    metadata = Path('/ai_sds_wuzz/DATA_TTS/LJSpeech/LJSpeech-1.1/metadata.csv')
    fallback = [line.split('|', 2) for line in metadata.read_text().splitlines()]
    random.Random(20260910).shuffle(fallback)
    for identifier, _, text in fallback:
        if identifier in train_lj and 25 <= len(text) <= 140:
            add(text, 'LJSpeech_metadata_train/' + identifier)
    return pool


def prepare(root, reference_kind, count, probe):
    root = Path(root)
    for folder in ('reference', 'logs', 'quality', 'candidates'):
        (root / folder).mkdir(parents=True, exist_ok=True)
    anchor = root / 'reference/ts004_cn_000001.wav'
    shutil.copy2(CHINESE / 'reference/ts004_cn_000001.wav', anchor)
    if reference_kind == 'cn':
        reference = anchor
        prompt = (CHINESE / 'reference/prompt_text.txt').read_text().strip()
    else:
        folder = ASSETS / 'Archive/TS-004_大气女声/MIX-100001-100120'
        reference = root / 'reference/ts004_mix_100004.wav'
        shutil.copy2(folder / 'wav/100004.wav', reference)
        line = next(x for x in (folder / 'text.txt').read_text().splitlines() if x.startswith('100004\t'))
        prompt = re.sub(r'#[0-9]+', '', line.split('\t', 1)[1]).strip()
    (root / 'reference/prompt_text.txt').write_text(prompt + '\n')
    pool = text_pool()
    if probe:
        pool = random.Random(20260910).sample(pool[:2756], count)
    else:
        # Reserve new sentences absent from BOTH historic English training voices.
        primary = [r for r in pool if r['text_source'].startswith('historical_2756/')]
        reserve = [r for r in pool if not r['text_source'].startswith('historical_2756/')]
        test = [r for r in reserve if not r['text_source'].startswith('LJSpeech_metadata_train/')][:150]
        test_texts = {r['text'] for r in test}
        train = primary + [r for r in reserve if r['text'] not in test_texts]
        pool = [{**r, 'split': 'train'} for r in train[:count - len(test)]]
        pool += [{**r, 'split': 'test'} for r in test]
    requests = []
    counts = {'train': 0, 'test': 0}
    for row in pool[:count]:
        split = row.get('split', 'train')
        identifier = f'{split}/en_{counts[split]:05d}'
        counts[split] += 1
        requests.append({**row, 'id': identifier, 'split': split, 'language': 'English',
                         **{f'output_{rate}k': str(root / f'candidates/wavs_{rate}k' / (identifier + '.wav'))
                            for rate in (24, 48)}})
    assert len(requests) == count
    queue = root / 'requests.jsonl'
    content = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in requests)
    if queue.exists() and queue.read_text() != content:
        raise RuntimeError('Existing English queue differs; select a fresh output directory')
    queue.write_text(content)
    if not (root / 'job.json').exists():
        save_json(root / 'job.json', {'status': 'prepared', 'model': str(ASSETS / 'VoxCPM2'),
                                     'dataset_name': 'MajesticVoice-English', 'counts': counts,
                                     'reference_audio': str(reference), 'prompt_text': prompt,
                                     'similarity_reference_audio': str(anchor), 'reference_kind': reference_kind,
                                     'target_training_accepted': 2756, 'probe': probe})
    print(json.dumps({'root': str(root), 'counts': counts, 'reference': str(reference)}, ensure_ascii=False))


def extend_training(root, count, test_count=0):
    """Append fresh texts only after a fully scored batch; keep every existing ID."""
    root = Path(root)
    job = json.loads((root / 'job.json').read_text())
    audit = json.loads((root / 'quality/final_audit.json').read_text())
    assert job['status'] == 'complete' and audit['status'] == 'passed'
    queue = root / 'requests.jsonl'
    old = [json.loads(line) for line in queue.read_text().splitlines()]
    seen = {normalize(r['text']) for r in old}
    fresh = [r for r in text_pool() if normalize(r['text']) not in seen][:count]
    assert len(fresh) == count, 'Not enough fresh English texts'
    first = max(int(r['id'].rsplit('_', 1)[1]) for r in old if r['split'] == 'train') + 1
    rows = list(old)
    for index, source in enumerate(fresh, first):
        identifier = f'train/en_{index:05d}'
        rows.append({**source, 'id': identifier, 'split': 'train', 'language': 'English',
                     **{f'output_{rate}k': str(root / f'candidates/wavs_{rate}k' / (identifier + '.wav'))
                        for rate in (24, 48)}})
    if test_count:
        val_ids = {Path(line.split('|')[0]).stem for line in (HISTORICAL / 'val.txt').read_text().splitlines()
                   if line.split('|')[1] == '0'}
        used = {normalize(r['text']) for r in rows}
        metadata = Path('/ai_sds_wuzz/DATA_TTS/LJSpeech/LJSpeech-1.1/metadata.csv')
        testing = []
        for line in metadata.read_text().splitlines():
            identifier, _, text = line.split('|', 2)
            key = normalize(text)
            if identifier in val_ids and key not in used and 25 <= len(text) <= 160:
                used.add(key)
                testing.append({'text': text, 'text_source': 'historical_ljs_validation/' + identifier})
        random.Random(20260911).shuffle(testing)
        assert len(testing) >= test_count, 'Not enough unused held-out English texts'
        first_test = max(int(r['id'].rsplit('_', 1)[1]) for r in old if r['split'] == 'test') + 1
        for index, source in enumerate(testing[:test_count], first_test):
            identifier = f'test/en_{index:05d}'
            rows.append({**source, 'id': identifier, 'split': 'test', 'language': 'English',
                         **{f'output_{rate}k': str(root / f'candidates/wavs_{rate}k' / (identifier + '.wav'))
                            for rate in (24, 48)}})
    number = len(job.get('queue_extensions', [])) + 1
    shutil.copy2(queue, root / f'quality/requests_before_extension_{number}.jsonl')
    shutil.copy2(root / 'quality/final_audit.json', root / f'quality/audit_before_extension_{number}.json')
    temporary = queue.with_suffix('.jsonl.tmp')
    temporary.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    temporary.replace(queue)
    job.setdefault('queue_extensions', []).append({'number': number, 'added_training_texts': count,
                                                   'added_test_texts': test_count,
                                                   'previous_source_texts': len(old)})
    job['counts']['train'] += count
    job['counts']['test'] += test_count
    job['stop_when_accepted'] = {'train': 2800, 'test': 64}
    job['status'] = 'prepared'
    save_json(root / 'job.json', job)
    print(json.dumps({'root': str(root), 'counts': job['counts'], 'added': count}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--reference', choices=['cn', 'mixed'], default='cn')
    parser.add_argument('--count', type=int, default=3200)
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--extend-train', type=int, default=0)
    parser.add_argument('--extend-test', type=int, default=0)
    args = parser.parse_args()
    if args.extend_train or args.extend_test:
        extend_training(args.root, args.extend_train, args.extend_test)
    else:
        prepare(args.root, args.reference, args.count, args.probe)
