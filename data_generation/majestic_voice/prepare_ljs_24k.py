"""Relocate and resample the historical speaker-0 data without changing weights."""
import concurrent.futures
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ASSETS = Path('/119010446/tts-assets')
HISTORICAL = ASSETS / 'chenmingjie/lits-tts/output/ljspeech_biaobei_voxcpm2en2756_16k_balanced'
OUTPUT = ASSETS / 'data_24k/ljs_majestic'
TTS_DATA = Path('/ai_sds_wuzz/DATA_TTS')
AUDIO_OUTPUT = {
    'ljs_original': TTS_DATA / 'LJSpeech/LJSpeech_24k/wavs',
    'ljs_synthetic': TTS_DATA / 'LJSpeech_Synthetic/24k/wavs',
}


def source_path(path):
    if '/LJSpeech/LJSpeech_16k/' in path:
        return Path('/ai_sds_wuzz/DATA_TTS/LJSpeech/LJSpeech-1.1/wavs') / Path(path).name, 'ljs_original'
    prefix = '/chenmingjie/chenmingjie/'
    if not path.startswith(prefix) or '/LJS_synth/' not in path:
        raise ValueError(f'Unexpected historical speaker-0 source: {path}')
    return TTS_DATA / 'LJSpeech_Synthetic/16k/wavs' / Path(path).name, 'ljs_synthetic'


def convert(task):
    old, source, kind, target = task
    audio, rate = sf.read(source, dtype='float32', always_2d=True)
    if len(audio) == 0 or not np.isfinite(audio).all():
        raise ValueError(f'Empty/nonfinite source: {source}')
    audio = audio.mean(axis=1)
    if np.sqrt(np.mean(audio.astype('float64') ** 2)) < 1e-6:
        raise ValueError(f'Silent source: {source}')
    source_hash = hashlib.sha256(audio.tobytes() + str(rate).encode()).hexdigest()
    if rate != 24000:
        divisor = math.gcd(rate, 24000)
        audio = resample_poly(audio, 24000 // divisor, rate // divisor)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.wav.tmp')
    sf.write(temporary, audio, 24000, format='WAV', subtype='PCM_16')
    temporary.replace(target)
    return {'historical_path': old, 'source_path': str(source), 'source_rate': rate,
            'source_pcm_sha256': source_hash, 'output_path': str(target), 'kind': kind,
            'duration': len(audio) / 24000, 'frames': len(audio)}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lists, tasks = {}, {}
    for split in ('train', 'val'):
        lists[split] = [line.split('|', 2) for line in (HISTORICAL / f'{split}.txt').read_text().splitlines()
                        if line.split('|')[1] == '0']
        for old, _, _ in lists[split]:
            source, kind = source_path(old)
            if not source.is_file():
                raise FileNotFoundError(source)
            target = AUDIO_OUTPUT[kind] / source.name
            tasks[old] = old, source, kind, target
    assert len(lists['train']) == 27448 and len(lists['val']) == 232
    assert not ({r[0] for r in lists['train']} & {r[0] for r in lists['val']})
    destinations = [str(t[3]) for t in tasks.values()]
    assert len(destinations) == len(set(destinations)), 'Destination collision'
    metadata_path = OUTPUT / 'ljs_audio.jsonl'
    previous = {r['historical_path']: r for r in (json.loads(line) for line in metadata_path.read_text().splitlines())} if metadata_path.exists() else {}
    completed = {}
    with metadata_path.open('a', buffering=1) as stream:
        pending = []
        for old, task in tasks.items():
            saved = previous.get(old)
            if saved and Path(saved['output_path']).is_file():
                info = sf.info(saved['output_path'])
                if info.samplerate == 24000 and info.frames == saved['frames'] and info.channels == 1:
                    completed[old] = saved
                    continue
            pending.append(task)
        with concurrent.futures.ThreadPoolExecutor(12) as pool:
            for row in pool.map(convert, pending):
                completed[row['historical_path']] = row
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                if len(completed) % 1000 == 0:
                    print(json.dumps({'converted': len(completed), 'target': len(tasks)}), flush=True)
    hashes = {split: {completed[r[0]]['source_pcm_sha256'] for r in rows} for split, rows in lists.items()}
    assert not (hashes['train'] & hashes['val']), 'Identical source PCM in train and validation'
    for split, rows in lists.items():
        (OUTPUT / f'ljs_{split}.txt').write_text(''.join(
            f"{completed[old]['output_path']}|0|{text}\n" for old, _, text in rows))
    summary = {'status': 'complete', 'train_rows': len(lists['train']), 'val_rows': len(lists['val']),
               'unique_audio': len(tasks), 'unique_hours': sum(r['duration'] for r in completed.values()) / 3600,
               'source_rates': dict(Counter(r['source_rate'] for r in completed.values())),
               'train_val_source_pcm_overlap': 0, 'sample_rate': 24000,
               'note': 'Original LJSpeech uses 22050-Hz originals; archived synthetic audio uses its available source rate.'}
    (OUTPUT / 'ljs_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
