"""Drain every exact-step checkpoint, including after training exits."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from training.majestic_scratch.config import REPO
from training.majestic_voice.watch_eval import PYTHONS
from training.stage2.prepare import digest


def alive(pid):
    try:return (Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
    except FileNotFoundError:return False


def main(args):
    root=args.run_dir/'eval';root.mkdir(exist_ok=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),STAGE2_DATA=str(args.run_dir/'data'),
             OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    done_file=root/'evaluated.jsonl'
    done={r['global_step'] for r in map(json.loads,done_file.read_text().splitlines())} if done_file.exists() else set()
    while True:
        pending=[p for p in sorted((args.run_dir/'evaluation_checkpoints').glob('step_*.ckpt')) if int(p.stem.split('_')[-1]) not in done]
        for checkpoint in pending:
            step=int(checkpoint.stem.split('_')[-1]);out=root/f'step_{step:08d}';out.mkdir(exist_ok=True)
            try:
                import torch
                # save_checkpoint can be observed before write completion; retry until readable.
                ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False)
                assert ckpt['global_step']==step;del ckpt
                snapshot=out/'checkpoint.ckpt'
                if not snapshot.exists():
                    tmp=out/'checkpoint.ckpt.tmp';shutil.copyfile(checkpoint,tmp);tmp.replace(snapshot)
                metadata=dict(global_step=step,source_checkpoint=str(checkpoint),sha256=digest(snapshot),started_at_unix=time.time(),
                              scope='startup_smoke_8_per_group' if step==1000 else f'full_{len((args.run_dir/"data/eval_manifest.jsonl").read_text().splitlines())}')
                (out/'checkpoint_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
                (root/'latest_eval.json').write_text(json.dumps(dict(metadata,status='running',output_dir=str(out)),indent=2)+'\n')
                commands=[]
                if step!=1000:
                    commands.append(('duration',[sys.executable,str(REPO/'training/majestic_scratch/diagnose_duration.py'),
                        '--checkpoint',str(snapshot),'--data-dir',str(args.run_dir/'data'),'--output',str(out)]))
                for stage,python in PYTHONS.items():
                    extra=['--per-group-limit','8'] if stage=='synthesize' and step==1000 else []
                    commands.append((stage,[str(python),str(REPO/'training/stage2/evaluate_checkpoint.py'),
                        '--stage',stage,'--checkpoint',str(snapshot),'--output',str(out),*extra]))
                for stage,command in commands:
                    with (out/f'{stage}.log').open('a') as log:
                        subprocess.run(command,env=env,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=7200)
                with done_file.open('a') as f:f.write(json.dumps(dict(metadata,completed_at_unix=time.time(),output_dir=str(out)))+'\n')
                done.add(step)
                (root/'latest_eval.json').write_text(json.dumps(dict(metadata,status='complete',summary=str(out/'summary.json')),indent=2)+'\n')
                print(json.dumps(dict(event='evaluation_complete',global_step=step)),flush=True)
            except Exception as exc:
                (root/'latest_eval.json').write_text(json.dumps(dict(status='failed',global_step=step,error=repr(exc)),indent=2)+'\n')
                print(json.dumps(dict(event='evaluation_error',step=step,error=repr(exc))),flush=True)
                if not alive(args.training_pid):raise
                break
        remaining=[p for p in (args.run_dir/'evaluation_checkpoints').glob('step_*.ckpt') if int(p.stem.split('_')[-1]) not in done]
        if not alive(args.training_pid) and not remaining:return
        time.sleep(30)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--training-pid',type=int,required=True)
    p.add_argument('--gpu',type=int,default=0);main(p.parse_args())
