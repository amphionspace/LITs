"""Launch authorized stage-two adaptation with frozen provenance and evaluation."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from training.stage2.prepare import digest


def main(args):
    run=args.run_dir
    plan=json.loads((run/'plan.json').read_text())
    if plan.get('mode') in ('scratch','backbone_init'):
        from training.majestic_scratch.run import main as scratch_run
        return scratch_run(args)
    max_steps=int(plan['max_steps'])
    rates=plan['learning_rates']
    schedule=plan['schedule']
    schedule_target=plan.get('schedule_callback','training.stage2.callbacks.AdaptationSchedule')
    assert max_steps > 1000
    assert json.loads((run/'preflight.json').read_text())['status']=='passed'
    if (run/'launch.json').exists():
        raise RuntimeError('This run was already launched; use an explicit resume, not a fresh start')
    assert digest(run/'initialization.ckpt')==plan['initialization_sha256']
    for name,expected in plan['manifest_hashes'].items():
        assert digest(run/'data'/name)==expected
    repo=Path(__file__).resolve().parents[2]
    stats=plan['data_statistics']
    env=dict(os.environ,PATH=str(Path(sys.executable).parent)+':'+os.environ['PATH'],PYTHONPATH=str(repo),PROJECT_ROOT=str(repo),
        CUDA_VISIBLE_DEVICES='0,1,2,3',TRAIN_FILELIST=str(run/'data/train.txt'),VALID_FILELIST=str(run/'data/val.txt'),
        STAGE2_DATA=str(run/'data'),N_SPKS='2',MEL_MEAN=str(stats['mel_mean']),MEL_STD=str(stats['mel_std']),
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
        PYTHONUNBUFFERED='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    command=['bash',str(repo/'training.sh'),f'run_name={run.name}',
        'ckpt_path=null',f'init_ckpt_path={run}/initialization.ckpt','seed=20260913','data.seed=20260913',
        'data._target_=training.stage2.data.Stage2DataModule',f'+data.data_dir={run}/data','data.batch_size=48','data.num_workers=8',
        f'model.optimizer.lr={rates["decoder"]}',f'model.prior_encoder_lr={rates["prior_encoder"]}',f'model.duration_predictor_lr={rates["duration_predictor"]}',
        'trainer.devices=[0,1,2,3]','trainer.precision=bf16-mixed','+trainer.use_distributed_sampler=false',
        f'+trainer.max_steps={max_steps}','trainer.check_val_every_n_epoch=null','+trainer.val_check_interval=250',
        '+trainer.num_sanity_val_steps=2','trainer.gradient_clip_val=5.0',
        'callbacks.model_checkpoint.every_n_epochs=null','callbacks.model_checkpoint.every_n_train_steps=250',
        'callbacks.model_checkpoint.monitor=step',"callbacks.model_checkpoint.filename='step_{step:08.0f}'",'callbacks.model_checkpoint.save_top_k=5',
        'callbacks.freeze_encoder.enabled=false','callbacks.lr_override.enabled=false',
        'callbacks.encoder_lr_downshift.enabled=false','callbacks.duration_lr_downshift.enabled=false',
        f'+callbacks.stage2_schedule._target_={schedule_target}',f'+callbacks.stage2_schedule.run_dir={run}',
        f'+callbacks.stage2_schedule.adaptation_steps={schedule["encoder_update_start_step"]}',
        f'+callbacks.stage2_schedule.warmup_steps={schedule["warmup_steps"]}',
        f'+callbacks.stage2_schedule.encoder_warmup_steps={schedule["encoder_warmup_steps"]}',
        f'+callbacks.stage2_schedule.total_steps={max_steps}',
        '+callbacks.training_state._target_=training.majestic_voice.callbacks.TrainingState',
        'callbacks.rich_progress_bar=null','trainer.enable_progress_bar=false','test=false',f'hydra.run.dir={run}']
    for name in ['lits','configs','training']:
        shutil.copytree(repo/name,run/'source'/name,ignore=shutil.ignore_patterns('__pycache__','.ipynb_checkpoints'))
    with (run/'training.log').open('w') as log:
        child=subprocess.Popen(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
        with (run/'eval_watcher.log').open('w') as log:
            watcher=subprocess.Popen([sys.executable,str(repo/'training/stage2/watch_eval.py'),'--run-dir',str(run),
                '--training-pid',str(child.pid),'--gpu','0','--interval-steps','1000'],cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        with (run/'baseline.log').open('w') as log:
            baseline=subprocess.Popen([sys.executable,str(repo/'training/stage2/baseline.py'),'--run-dir',str(run),
                '--gpu','3'],cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        launch=dict(status='running',stage=2,training_pid=child.pid,eval_watcher_pid=watcher.pid,baseline_pid=baseline.pid,
            started_at_unix=time.time(),command=command,run_dir=str(run),plan=str(run/'plan.json'),
            source_checkpoint=plan['source_checkpoint'],source_global_step=plan['source_global_step'],
            speaker_ids=plan['speakers'],batch_size_per_gpu=48,effective_batch=192,devices=4,max_steps=max_steps)
        (run/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
        print(json.dumps(launch),flush=True)
        code=child.wait()
        launch.update(status='complete' if code==0 else 'failed',returncode=code,finished_at_unix=time.time())
        (run/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
        raise SystemExit(code)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,required=True)
    main(parser.parse_args())
