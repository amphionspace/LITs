"""Evaluate the copied speaker embeddings before any stage-two update."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from training.majestic_voice.watch_eval import PYTHONS


def main(args):
    output=args.run_dir/'baseline'
    output.mkdir(exist_ok=False)
    samples=len((args.run_dir/'data/eval_manifest.jsonl').read_text().splitlines())
    meta=dict(status='running',global_step=0,checkpoint=str(args.run_dir/'initialization.ckpt'),samples=samples,started_at_unix=time.time())
    (output/'status.json').write_text(json.dumps(meta,indent=2)+'\n')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),STAGE2_DATA=str(args.run_dir/'data'))
    try:
        for stage,python in PYTHONS.items():
            with (output/f'{stage}.log').open('w') as log:
                subprocess.run([str(python),str(Path(__file__).with_name('evaluate_checkpoint.py')),'--stage',stage,
                    '--checkpoint',meta['checkpoint'],'--output',str(output)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=7200)
        meta.update(status='complete',finished_at_unix=time.time())
    except Exception as exc:
        meta.update(status='failed',error=f'{type(exc).__name__}: {exc}',finished_at_unix=time.time())
        raise
    finally:
        (output/'status.json').write_text(json.dumps(meta,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--gpu',type=int,default=3)
    main(parser.parse_args())
