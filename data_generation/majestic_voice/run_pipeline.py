"""Two candidates per round; concurrent quality checks; retry only rejected texts."""
import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path

from quality_worker import ROOT, ASSETS, quality_config
from select_candidates import assess, publish
from synthesize import save_json

CODE = Path(__file__).resolve().parent


def main():
    config = quality_config()
    save_json(ROOT / 'quality/thresholds_used.json', config)
    job = json.loads((ROOT / 'job.json').read_text())
    job.update(status='synthesizing_and_filtering', quality_thresholds=config)
    save_json(ROOT / 'job.json', job)
    workers, handles = [], []

    def start(command, log, gpu=None):
        env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
                   NUMEXPR_NUM_THREADS='1',
                   HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
        if gpu is not None:
            env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        handle = (ROOT / 'logs' / log).open('a', buffering=1)
        handles.append(handle)
        return subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env)

    try:
        workers.append(start([str(ASSETS / '.venv-voxcpm2/bin/python'), str(CODE / 'quality_worker.py'),
                              '--kind', 'asr', '--watch', '--batch-size', '16'], 'quality-asr.log', 1))
        metrics_python = os.environ.get('MAJESTIC_METRICS_PYTHON',
                                        '/119010446/UltraEval-Audio/envs/metrics/bin/python')
        workers.append(start([metrics_python, str(CODE / 'quality_worker.py'), '--kind', 'metrics',
                              '--watch', '--batch-size', '64'], 'quality-metrics.log', 0))
        for round_index in range(config['max_rounds']):
            generated, judged, best = assess()
            remaining = publish(best)
            if not remaining:
                break
            ids = ROOT / f'quality/round_{round_index}_ids.txt'
            ids.write_text('\n'.join(remaining) + '\n')
            command = [str(ASSETS / '.venv-voxcpm2/bin/python'), str(CODE / 'synthesize.py'),
                       '--round', str(round_index), '--candidates', str(config['candidates_per_round']),
                       '--ids-file', str(ids)]
            producer = start(command, f'synthesis-round{round_index}.log')
            workers.append(producer)
            while True:
                for worker in workers[:2]:
                    if worker.poll() is not None:
                        raise RuntimeError(f'Quality worker exited with code {worker.returncode}; inspect logs')
                generated, judged, best = assess()
                remaining = publish(best)
                status = {'round': round_index, 'generated_candidates': len(generated),
                          'scored_candidates': len(judged), 'accepted_texts': len(best),
                          'remaining_texts': len(remaining), 'updated_at_unix': time.time()}
                save_json(ROOT / 'pipeline_progress.json', status)
                print(json.dumps(status), flush=True)
                if producer.poll() is not None:
                    if producer.returncode:
                        progress = json.loads((ROOT / 'progress.json').read_text())
                        failures = json.loads((ROOT / 'failures.json').read_text())
                        # A completed queue can contain explicitly rejected audio.
                        # Infrastructure failures or interrupted queues still stop the run.
                        if (producer.returncode != 1 or progress.get('status') != 'failed'
                                or progress.get('queued') != 0 or not failures
                                or progress['completed'] + len(failures) != progress['target']
                                or any(not r['error'].startswith('ValueError:') for r in failures)):
                            raise RuntimeError(f'Synthesis failed with code {producer.returncode}; inspect logs')
                        print(json.dumps({'event': 'invalid_audio_excluded', 'candidates': len(failures)}), flush=True)
                    if generated.keys() <= judged.keys():
                        break
                time.sleep(30)
            workers.remove(producer)
            # Optional dataset quota is checked only after the whole round was
            # generated and scored, so no new source can be left unattempted.
            quota = job.get('stop_when_accepted', {})
            accepted_by_split = Counter(row['split'] for row in best.values())
            if quota and all(accepted_by_split[split] >= count for split, count in quota.items()):
                break
        generated, judged, best = assess()
        remaining = publish(best, complete=True)
        with (ROOT / 'quality/candidates_scored.jsonl').open('w') as stream:
            for row in judged.values():
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        job = json.loads((ROOT / 'job.json').read_text())
        job.update(status='complete', accepted_texts=len(best), excluded_texts=len(remaining),
                   generated_candidates=len(generated), quality_thresholds=config,
                   completed_at_unix=time.time())
        save_json(ROOT / 'job.json', job)
        print(json.dumps({'status': 'complete', 'accepted': len(best), 'excluded': len(remaining)}), flush=True)
    finally:
        for process in workers:
            if process.poll() is None:
                process.terminate()
        for process in workers:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for handle in handles:
            handle.close()


if __name__ == '__main__':
    main()
