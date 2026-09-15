"""Exact weighted Mel statistics, with the dataset's segment and padding rules."""
import argparse
import concurrent.futures
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from lits.utils.audio import mel_spectrogram

DATA = Path('/119010446/tts-assets/data_24k/ljs_majestic')


def unit_key(path, start=None, end=None):
    return json.dumps([path, start, end], separators=(',', ':'))


def analyze(key):
    path, start, end = json.loads(key)
    info = sf.info(path)
    assert info.samplerate == 24000 and info.channels == 1 and info.subtype == 'PCM_16'
    first = max(0, int(round(start * 24000))) if start is not None else 0
    last = min(info.frames, int(round(end * 24000))) if end is not None else info.frames
    assert last > first, key
    audio, rate = sf.read(path, start=first, stop=last, dtype='float32')
    assert np.isfinite(audio).all(), path
    mel = mel_spectrogram(torch.from_numpy(audio)[None], 2048, 100, rate, 384, 1536, 0, 12000)
    assert torch.isfinite(mel).all()
    double = mel.double()
    return {'key': key, 'sum': double.sum().item(), 'sum_squares': double.square().sum().item(),
            'mel_frames': mel.shape[-1], 'audio_frames': len(audio),
            'pcm_sha256': hashlib.sha256(audio.tobytes()).hexdigest()}


def main(args):
    torch.set_num_threads(1)
    # Initialize the one shared Mel basis before concurrent calls.
    mel_spectrogram(torch.zeros(1, 4096), 2048, 100, 24000, 384, 1536, 0, 12000)
    manifest = DATA / ('ljs_train.txt' if args.cache_ljs else 'train.txt')
    units = []
    for line in manifest.read_text().splitlines():
        fields = line.split('|')
        units.append(unit_key(fields[0], float(fields[2]) if len(fields) == 5 else None,
                              float(fields[3]) if len(fields) == 5 else None))
    counts = Counter(units)
    cache_file = DATA / 'mel_statistics_units.jsonl'
    cache = {r['key']: r for r in (json.loads(line) for line in cache_file.read_text().splitlines())} if cache_file.exists() else {}
    pending = [key for key in counts if key not in cache]
    with cache_file.open('a', buffering=1) as stream, concurrent.futures.ThreadPoolExecutor(8) as pool:
        for row in pool.map(analyze, pending):
            cache[row['key']] = row
            stream.write(json.dumps(row) + '\n')
            if len(cache) % 1000 == 0:
                print(json.dumps({'cached_units': len(cache), 'requested_units': len(counts)}), flush=True)
    n = sum(cache[key]['mel_frames'] * 100 * count for key, count in counts.items())
    mean = math.fsum(cache[key]['sum'] * count for key, count in counts.items()) / n
    variance = math.fsum(cache[key]['sum_squares'] * count for key, count in counts.items()) / n - mean ** 2
    result = {'mel_mean': mean, 'mel_std': math.sqrt(variance), 'weighted_mel_elements': n,
              'manifest_rows': len(units), 'distinct_segments': len(counts),
              'weighted_audio_hours': sum(cache[k]['audio_frames'] * v for k, v in counts.items()) / 24000 / 3600,
              'distinct_audio_hours': sum(cache[k]['audio_frames'] for k in counts) / 24000 / 3600,
              'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
              'sample_rate': 24000, 'n_fft': 2048, 'n_feats': 100, 'hop_length': 384,
              'win_length': 1536, 'f_min': 0, 'f_max': 12000}
    if not args.cache_ljs:
        training = [json.loads(line) for line in (DATA / 'train.jsonl').read_text().splitlines()]
        text_cache = {r['text']: r for r in (json.loads(line) for line in (DATA / 'text_preflight.jsonl').read_text().splitlines())}
        for row in training:
            key = unit_key(row['audio'], row.get('start'), row.get('end'))
            assert len(text_cache[row['text']]['tokens']) <= cache[key]['mel_frames'], f'Infeasible MAS text/audio pair: {row}'
        result['mas_text_length_check'] = 'passed'
    out = DATA / ('ljs_mel_statistics.json' if args.cache_ljs else 'mel_statistics.json')
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-ljs', action='store_true')
    main(parser.parse_args())
