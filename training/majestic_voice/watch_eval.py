"""Evaluate immutable copies of new checkpoints, sequentially on one GPU."""
import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

CODE = Path(__file__).resolve().parent
ASSETS = Path('/119010446/tts-assets')
PYTHONS = {'synthesize': ASSETS / 'chenmingjie/lits-tts/envs/lits/bin/python',
           'asr': ASSETS / '.venv-voxcpm2/bin/python',
           'metrics': Path('/119010446/UltraEval-Audio/envs/metrics/bin/python'),
           'summarize': ASSETS / 'chenmingjie/lits-tts/envs/lits/bin/python'}


def main(args):
    import torch
    output = args.run_dir / 'eval'
    output.mkdir(exist_ok=True)
    done_file = output / 'evaluated.jsonl'
    done = [json.loads(line) for line in done_file.read_text().splitlines()] if done_file.exists() else []
    last_step = max((r['global_step'] for r in done), default=-1)
    last_seen = None
    while True:
        alive = Path(f'/proc/{args.training_pid}').exists()
        candidates = list((args.run_dir / 'checkpoints').glob('*.ckpt'))
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        checkpoint = candidates[0] if candidates else None
        if checkpoint is not None:
            identity = (str(checkpoint), checkpoint.stat().st_mtime_ns, checkpoint.stat().st_size)
            if identity != last_seen:
                try:
                    checkpoint_data = torch.load(checkpoint, map_location='cpu', weights_only=False)
                    step = int(checkpoint_data['global_step'])
                    del checkpoint_data
                    gc.collect()
                    if step > last_step and (last_step < 0 or step - last_step >= args.interval_steps or not alive):
                        destination = output / f'step_{step:08d}'
                        destination.mkdir(exist_ok=True)
                        snapshot = destination / 'checkpoint.ckpt'
                        temporary = destination / 'checkpoint.ckpt.tmp'
                        shutil.copyfile(checkpoint, temporary)
                        if identity[1:] != (checkpoint.stat().st_mtime_ns, checkpoint.stat().st_size):
                            temporary.unlink(missing_ok=True)
                            continue
                        temporary.replace(snapshot)
                        digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
                        startup_smoke = last_step < 0
                        manifest = args.run_dir / 'data' / 'eval_manifest.jsonl'
                        sample_count = len(manifest.read_text().splitlines()) if manifest.exists() else 650
                        metadata = {'source_checkpoint': str(checkpoint), 'global_step': step, 'sha256': digest,
                                    'scope': 'startup_smoke_8_per_group' if startup_smoke else f'full_{sample_count}',
                                    'started_at_unix': time.time()}
                        (destination / 'checkpoint_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
                        (output / 'latest_eval.json').write_text(json.dumps({**metadata, 'status': 'running',
                                                                          'output_dir': str(destination)}, indent=2) + '\n')
                        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS='4',
                                   MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
                                   HF_HUB_OFFLINE='1')
                        print(json.dumps({'event': 'eval_start', 'global_step': step}), flush=True)
                        for stage, python in PYTHONS.items():
                            extra = ['--per-group-limit', '8'] if stage == 'synthesize' and startup_smoke else []
                            with (destination / f'{stage}.log').open('a') as log:
                                subprocess.run([str(python), str(CODE / 'evaluate_checkpoint.py'), '--stage', stage,
                                                '--checkpoint', str(snapshot), '--output', str(destination), *extra],
                                               env=env, stdout=log, stderr=subprocess.STDOUT, check=True,
                                               timeout=7200)
                        with done_file.open('a') as stream:
                            stream.write(json.dumps({**metadata, 'output_dir': str(destination),
                                                     'completed_at_unix': time.time()}) + '\n')
                        last_step = step
                        (output / 'latest_eval.json').write_text(json.dumps({**metadata, 'status': 'complete',
                                                                          'summary': str(destination / 'summary.json')}, indent=2) + '\n')
                        print(json.dumps({'event': 'eval_complete', 'global_step': step,
                                          'summary': str(destination / 'summary.json')}), flush=True)
                    last_seen = identity
                except Exception as exc:
                    print(json.dumps({'event': 'eval_error', 'checkpoint': str(checkpoint),
                                      'error': f'{type(exc).__name__}: {exc}'}), flush=True)
        if not alive:
            return
        time.sleep(30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--training-pid', type=int, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--interval-steps', type=int, default=2000)
    main(parser.parse_args())
