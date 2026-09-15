"""Resumeable dual-server VoxCPM2 synthesis with atomic 48/24 kHz outputs."""
import argparse
import asyncio
import base64
import io
import json
import os
import time
import zlib
from pathlib import Path

import aiohttp
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(os.environ.get("MAJESTIC_VOICE_DATA_ROOT", "/ai_sds_wuzz/DATA_TTS/MajesticVoice"))


def save_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def validate_wave(path, expected_rate):
    try:
        info = sf.info(path)
        return info.samplerate == expected_rate and info.channels == 1 and info.duration >= 0.25
    except (OSError, RuntimeError):
        return False


def save_audio(payload, row):
    wav, rate = sf.read(io.BytesIO(payload), dtype='float32')
    if rate != 48000 or wav.ndim != 1:
        raise ValueError(f'Unexpected audio format: rate={rate}, shape={wav.shape}')
    duration = len(wav) / rate
    if not 0.25 <= duration <= 120 or not np.isfinite(wav).all():
        raise ValueError(f'Invalid duration or samples: {duration:.3f}s')
    rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2)))
    if rms < 0.0001:
        raise ValueError(f'Nearly silent output: RMS={rms}')
    if duration > max(15, len(row['text']) * 1.0):
        raise ValueError(f'Unusually long output for text: {duration:.3f}s')
    min_seconds_per_character = 0.02 if row.get('language') == 'English' else 0.045
    if duration < max(0.25, len(row['text']) * min_seconds_per_character):
        raise ValueError(f'Unusually short output for text: {duration:.3f}s')
    for field, audio, sample_rate in (
        ('output_48k', wav, 48000),
        ('output_24k', resample_poly(wav, 1, 2), 24000),
    ):
        path = Path(row[field])
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.wav.tmp')
        sf.write(temporary, audio, sample_rate, format='WAV', subtype='PCM_16')
        temporary.replace(path)
    return {'duration': duration, 'rms': rms, 'peak': float(np.abs(wav).max())}


async def run(args):
    all_rows = [json.loads(line) for line in (ROOT / 'requests.jsonl').read_text().splitlines()]
    selected_sources = all_rows[:args.limit] if args.limit else all_rows
    if args.ids_file:
        wanted = set(Path(args.ids_file).read_text().splitlines())
        selected_sources = [r for r in selected_sources if r['id'] in wanted]
    selected = []
    for source in selected_sources:
        for variant in range(args.candidates):
            candidate_index = args.round * args.candidates + variant
            row = dict(source)
            row['source_id'] = source['id']
            row['candidate_index'] = candidate_index
            row['id'] = f"{source['id']}__c{candidate_index:03d}"
            for field in ('output_48k', 'output_24k'):
                path = Path(source[field])
                row[field] = str(path.with_stem(path.stem + f'__c{candidate_index:03d}'))
            selected.append(row)
    pending = [r for r in selected if not (
        validate_wave(r['output_48k'], 48000) and validate_wave(r['output_24k'], 24000)
    )]
    previous = len(selected) - len(pending)
    job = json.loads((ROOT / 'job.json').read_text())
    # VoxCPM2's reference encoder operates at 16 kHz. Keep the original reference,
    # and use a compact PCM16 data URI for local HTTP requests.
    ref, rate = sf.read(job['reference_audio'], dtype='float32')
    if rate != 48000 or ref.ndim != 1:
        raise ValueError('Expected the verified original 48 kHz mono reference')
    ref16 = resample_poly(ref, 1, 3)
    buffer = io.BytesIO()
    sf.write(buffer, ref16, 16000, format='WAV', subtype='PCM_16')
    reference_uri = 'data:audio/wav;base64,' + base64.b64encode(buffer.getvalue()).decode()
    queue = asyncio.Queue()
    for row in pending:
        queue.put_nowait(row)
    started = time.monotonic()
    completed, failed, audio_seconds = previous, [], 0.0
    last_report = 0.0
    per_endpoint = {url: 0 for url in args.endpoints}
    events = (ROOT / 'logs' / 'synthesis.jsonl').open('a', buffering=1)

    def report(force=False):
        nonlocal last_report
        now = time.monotonic()
        if not force and now - last_report < 30:
            return
        last_report = now
        elapsed = now - started
        status = {
            'status': 'running', 'target': len(selected), 'target_texts': len(selected_sources),
            'candidates_per_text': args.candidates, 'round': args.round, 'completed': completed,
            'previously_completed': previous, 'failed': len(failed),
            'queued': queue.qsize(), 'elapsed_seconds': round(elapsed, 2),
            'new_audio_seconds': round(audio_seconds, 2),
            'samples_per_second': round((completed - previous) / max(elapsed, 1), 4),
            'completed_by_endpoint': per_endpoint, 'updated_at_unix': time.time(),
        }
        save_json(ROOT / 'progress.json', status)
        print(json.dumps(status, ensure_ascii=False), flush=True)

    async def worker(endpoint, session):
        nonlocal completed, audio_seconds
        while True:
            if len(failed) >= 16:
                raise RuntimeError("Stopping after 16 failed requests; inspect failures.json before resuming")
            try:
                row = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            error = None
            for attempt in range(args.retries):
                try:
                    body = {
                        'model': job['model'], 'input': row['text'],
                        'ref_audio': reference_uri, 'ref_text': job['prompt_text'],
                        'response_format': 'wav', 'stream': False,
                        'seed': args.seed + zlib.crc32(row['id'].encode()) + attempt * 100000,
                        'max_new_tokens': 2000,
                    }
                    before = time.monotonic()
                    async with session.post(endpoint + '/v1/audio/speech', json=body) as response:
                        payload = await response.read()
                        if response.status != 200:
                            raise RuntimeError(f'HTTP {response.status}: {payload[:500].decode(errors="replace")}')
                    metrics = await asyncio.to_thread(save_audio, payload, row)
                    completed += 1
                    per_endpoint[endpoint] += 1
                    audio_seconds += metrics['duration']
                    events.write(json.dumps({
                        **row, 'status': 'ok', 'endpoint': endpoint,
                        'reference_audio': job['reference_audio'],
                        'attempt': attempt + 1, 'seconds': time.monotonic() - before,
                        **metrics,
                    }, ensure_ascii=False) + '\n')
                    error = None
                    break
                except Exception as exc:
                    error = f'{type(exc).__name__}: {exc}'
                    events.write(json.dumps({'id': row['id'], 'status': 'retry',
                                             'attempt': attempt + 1, 'error': error},
                                            ensure_ascii=False) + '\n')
                    await asyncio.sleep(min(2 ** attempt, 10))
            if error:
                failed.append({'id': row['id'], 'error': error})
                events.write(json.dumps({**row, 'status': 'failed', 'error': error},
                                        ensure_ascii=False) + '\n')
                save_json(ROOT / 'failures.json', failed)
            queue.task_done()
            report()

    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(limit=len(args.endpoints) * args.concurrency)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        for endpoint in args.endpoints:
            async with session.get(endpoint + '/health') as response:
                if response.status != 200:
                    raise RuntimeError(f'Server is not healthy: {endpoint}')
        report(True)
        await asyncio.gather(*[
            worker(endpoint, session)
            for endpoint in args.endpoints for _ in range(args.concurrency)
        ])
    events.close()
    report(True)
    save_json(ROOT / 'failures.json', failed)
    if not args.limit and completed == len(selected) and not failed:
        job['status'] = 'awaiting_quality_checks'
        job['synthesis_status'] = 'complete'
        job['completed_at_unix'] = time.time()
        job['generated_count'] = completed
        save_json(ROOT / 'job.json', job)
    summary = json.loads((ROOT / 'progress.json').read_text())
    summary['status'] = 'failed' if failed else 'preview_complete' if args.limit else 'awaiting_quality_checks'
    save_json(ROOT / 'progress.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--endpoints', nargs='+', default=['http://127.0.0.1:8100', 'http://127.0.0.1:8101'])
    parser.add_argument('--concurrency', type=int, default=8, help='Concurrent requests per GPU server')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--candidates', type=int, default=2)
    parser.add_argument('--round', type=int, default=0)
    parser.add_argument('--ids-file', help='Optional newline-separated source IDs for regeneration')
    parser.add_argument('--retries', type=int, default=3)
    parser.add_argument('--timeout', type=int, default=1800)
    parser.add_argument('--seed', type=int, default=20260910)
    asyncio.run(run(parser.parse_args()))
