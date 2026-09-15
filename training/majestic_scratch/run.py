"""Launch random-initialized joint training after dataset and GPU preflight gates."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from training.majestic_scratch.config import REPO, environment, model_overrides
from training.stage2.prepare import digest


def command_for(run,plan):
    command=[sys.executable,str(REPO/'lits/train.py'),*model_overrides(plan['peak_lr']),
        f'run_name={run.name}',f'seed={plan["seed"]}',f'data.seed={plan["seed"]}',
        'data._target_=training.majestic_scratch.data.ScratchDataModule',f'+data.data_dir={run}/data','data.batch_size=48','data.num_workers=8',
        'trainer.devices=[0,1,2,3]','trainer.precision=bf16-mixed','+trainer.use_distributed_sampler=false',
        '+trainer.accumulate_grad_batches=1',
        f'+trainer.max_steps={plan["max_steps"]}','trainer.check_val_every_n_epoch=null','+trainer.val_check_interval=1000',
        f'trainer.max_epochs={plan.get("max_epochs",-1)}',
        '+trainer.num_sanity_val_steps=2','trainer.gradient_clip_val=5.0',
        'callbacks.model_checkpoint.every_n_epochs=null','callbacks.model_checkpoint.every_n_train_steps=1000',
        'callbacks.model_checkpoint.monitor=step',"callbacks.model_checkpoint.filename='step_{step:08.0f}'",'callbacks.model_checkpoint.save_top_k=3',
        'callbacks.freeze_encoder.enabled=false','callbacks.lr_override.enabled=false',
        'callbacks.encoder_lr_downshift.enabled=false','callbacks.duration_lr_downshift.enabled=false',
        '+callbacks.scratch_schedule._target_=training.majestic_scratch.schedule.WarmupStableDecay',
        f'+callbacks.scratch_schedule.decay_fraction={plan["decay_fraction"]}',
        f'+callbacks.scratch_schedule.warmup_steps={plan["warmup_steps"]}',f'+callbacks.scratch_schedule.total_steps={plan["max_steps"]}',
        f'+callbacks.scratch_schedule.peak_lr={plan["peak_lr"]}',f'+callbacks.scratch_schedule.final_lr={plan["final_lr"]}',
        '+callbacks.scratch_audit._target_=training.majestic_scratch.callbacks.ScratchAudit',f'+callbacks.scratch_audit.run_dir={run}',
        '+callbacks.training_state._target_=training.majestic_voice.callbacks.TrainingState',
        'callbacks.rich_progress_bar=null','trainer.enable_progress_bar=false','test=false',f'hydra.run.dir={run}']
    if plan['mode']=='backbone_init':
        command=[f'init_ckpt_path={run}/initialization.ckpt' if arg=='init_ckpt_path=null' else arg for arg in command]
    return command


def main(args):
    run=args.run_dir;plan=json.loads((run/'plan.json').read_text())
    assert plan['mode'] in ('scratch','backbone_init')
    assert bool(plan['source_checkpoint'])==(plan['mode']=='backbone_init')
    preflight=json.loads((run/'preflight.json').read_text())
    assert preflight['status']=='passed' and preflight['mode']==plan['mode']
    if plan['mode']=='backbone_init':assert digest(run/'initialization.ckpt')==plan['initialization_sha256']
    assert not (run/'launch.json').exists(), 'Run already launched; explicit resume required'
    for name,expected in plan['manifest_hashes'].items(): assert digest(run/'data'/name)==expected
    command=command_for(run,plan)
    env=dict(os.environ,**environment(run,plan['data_statistics']))
    env.update(PATH=str(Path(sys.executable).parent)+':'+os.environ['PATH'],PYTHONPATH=str(REPO),CUDA_VISIBLE_DEVICES='0,1,2,3',
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
        PYTHONUNBUFFERED='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    for name in ['lits','configs','training']:
        shutil.copytree(REPO/name,run/'source'/name,ignore=shutil.ignore_patterns('__pycache__','.ipynb_checkpoints'))
    with (run/'training.log').open('w') as log:
        child=subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
    with (run/'eval_watcher.log').open('w') as log:
        watcher=subprocess.Popen([sys.executable,str(REPO/'training/majestic_scratch/watch_eval.py'),
            '--run-dir',str(run),'--training-pid',str(child.pid),'--gpu','0'],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    launch=dict(status='running',mode=plan['mode'],training_pid=child.pid,eval_watcher_pid=watcher.pid,
                started_at_unix=time.time(),command=command,source_checkpoint=plan['source_checkpoint'],ckpt_path=None,init_ckpt_path=plan['init_ckpt_path'],
                active_speaker_ids=plan['active_speaker_ids'],max_steps=plan['max_steps'],batch_size_per_gpu=48,effective_batch=192)
    (run/'launch.json').write_text(json.dumps(launch,indent=2)+'\n');print(json.dumps(launch),flush=True)
    code=child.wait()
    launch.update(status='complete' if code==0 else 'failed',returncode=code,finished_at_unix=time.time())
    (run/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
    raise SystemExit(code)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);main(p.parse_args())
