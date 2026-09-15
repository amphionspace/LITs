"""Consume completed candidates without blocking VoxCPM2 production."""
import argparse
import json
import os
import time
import unicodedata
from functools import lru_cache
from pathlib import Path

ROOT = Path(os.environ.get('MAJESTIC_VOICE_DATA_ROOT', '/ai_sds_wuzz/DATA_TTS/MajesticVoice'))
ASSETS = Path(os.environ.get('MAJESTIC_VOICE_ASSETS_ROOT', '/119010446/tts-assets'))


def quality_config():
    path = Path(os.environ.get('MAJESTIC_QUALITY_CONFIG', Path(__file__).with_name('quality_thresholds.json')))
    return json.loads(path.read_text())


def normalize(text):
    return ''.join(c for c in unicodedata.normalize('NFKC', text).lower()
                   if unicodedata.category(c)[0] in ('L', 'N'))


@lru_cache(maxsize=100_000)
def tonal_sequence(text):
    """Tone-number pinyin for fully Chinese text; unknown readings fail closed."""
    import re
    from pypinyin import pinyin, Style
    text = normalize(text)
    if not text or not all('\u4e00' <= c <= '\u9fff' for c in text):
        return None
    sequence = tuple(item[0] for item in pinyin(text, style=Style.TONE3,
                                               heteronym=False, neutral_tone_with_five=True))
    if not all(re.fullmatch(r'[a-zvü]+[1-5]', token) for token in sequence):
        return None
    return sequence


def tonal_match(reference, hypothesis):
    expected = tonal_sequence(reference)
    return expected is not None and expected == tonal_sequence(hypothesis)


def cer(reference, hypothesis):
    reference, hypothesis = normalize(reference), normalize(hypothesis)
    return error_rate(reference, hypothesis)


def wer(reference, hypothesis):
    import re
    def words(text):
        text = unicodedata.normalize('NFKC', text).lower().replace('’', "'")
        return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)*", text)
    return error_rate(words(reference), words(hypothesis))


def error_rate(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, a in enumerate(reference, 1):
        current = [i]
        for j, b in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (a != b)))
        previous = current
    return previous[-1] / max(1, len(reference))


def records(path):
    if not path.exists():
        return []
    result = []
    for line in path.read_text().splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # A concurrently appended final line can be incomplete.
    return result


def main(args):
    import torch
    torch.set_num_threads(4)
    (ROOT / 'quality').mkdir(parents=True, exist_ok=True)
    job = json.loads((ROOT / 'job.json').read_text())
    reference = Path(job.get('similarity_reference_audio', ROOT / 'reference/ts004_cn_000001.wav'))
    if args.kind == 'asr':
        from qwen_asr import Qwen3ASRModel
        model = Qwen3ASRModel.from_pretrained(
            str(ASSETS / 'Qwen3-ASR-1.7B'), dtype=torch.bfloat16,
            device_map='cuda:0', attn_implementation='sdpa',
            max_inference_batch_size=args.batch_size, max_new_tokens=256)
    else:
        from quality_metrics import SpeakerMetrics, DNSMOS, audio16
        from concurrent.futures import ThreadPoolExecutor
        config = quality_config()
        speakers = SpeakerMetrics(reference, fast_matmul=not args.calibrate,
                                  admission_thresholds=(config['min_wavlm_similarity'], config['min_camp_similarity']))
        dnsmos = DNSMOS()
        cpu = ThreadPoolExecutor(max_workers=8)
    print(f'{args.kind} models ready', flush=True)
    if args.calibrate:
        if args.kind != 'metrics':
            raise ValueError('Calibration requires metrics worker')
        result = []
        for folder, group in [('TS-004_大气女声', 'same_voice'), ('TS-050_自然女声', 'different_voice')]:
            for path in sorted((ASSETS / 'Archive' / folder).rglob('*.wav')):
                audio = audio16(path)
                row = {'path': str(path), 'group': group,
                       **speakers(audio), **dnsmos(audio)}
                result.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
        (ROOT / 'quality/calibration.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
        return
    output = ROOT / f'quality/{args.kind}.jsonl'
    output.parent.mkdir(parents=True, exist_ok=True)
    done = {r['id'] for r in records(output)}
    with output.open('a', buffering=1) as stream:
        while True:
            candidates = {r['id']: r for r in records(ROOT / 'logs/synthesis.jsonl')
                          if r.get('status') == 'ok'}
            pending = [r for key, r in candidates.items() if key not in done]
            if args.kind == 'metrics':
                pending.sort(key=lambda r: r['duration'])
            if args.limit:
                pending = pending[:max(0, args.limit - len(done))]
            if not pending:
                if not args.watch or (args.limit and len(done) >= args.limit):
                    break
                time.sleep(5)
                continue
            for offset in range(0, len(pending), args.batch_size):
                batch = pending[offset:offset + args.batch_size]
                started = time.monotonic()
                if args.kind == 'asr':
                    # Deliberately no target transcript in context: prevent answer leakage.
                    predictions = model.transcribe(audio=[r['output_24k'] for r in batch],
                                                   context='', language=None)
                    if len(predictions) != len(batch):
                        raise RuntimeError('ASR output count mismatch')
                    results = [{'asr_text': p.text, 'asr_language': p.language,
                                'cer': cer(r['text'], p.text),
                                'wer': wer(r['text'], p.text)} for r, p in zip(batch, predictions)]
                else:
                    audio = [audio16(r['output_24k']) for r in batch]
                    mos_futures = [cpu.submit(dnsmos, a) for a in audio]
                    similarity = speakers.batch(audio)
                    results = [{**sim, **future.result()} for sim, future in zip(similarity, mos_futures)]
                for row, result in zip(batch, results):
                    stream.write(json.dumps({'id': row['id'], 'source_id': row['source_id'],
                                             'status': 'ok', **result}, ensure_ascii=False) + '\n')
                    done.add(row['id'])
                print(json.dumps({'kind': args.kind, 'completed': len(done),
                                  'batch_seconds': round(time.monotonic() - started, 2)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=['asr', 'metrics'], required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--calibrate', action='store_true')
    main(parser.parse_args())
