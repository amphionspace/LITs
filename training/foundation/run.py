"""Launch stage one from random weights using four GPUs."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
DATA = Path('/119010446/tts-assets/data_24k/foundation')

def main(args):
    if not (math.isfinite(args.peak_lr) and math.isfinite(args.final_lr)
            and 0 < args.final_lr <= args.peak_lr):
        raise ValueError('Require finite 0 < final_lr <= peak_lr')
    run = args.run_dir
    run.mkdir(parents=True,exist_ok=True)
    if (run/'launch.json').exists():
        raise RuntimeError('Run already launched; never silently restart training')
    stats=json.loads((DATA/'mel_statistics.json').read_text())
    capacity=json.loads((DATA/'capacity.json').read_text())
    batch=capacity['chosen_batch_size_per_gpu']
    env=dict(os.environ,PATH=str(Path(sys.executable).parent)+':'+os.environ['PATH'],
             PYTHONPATH=str(REPO),PROJECT_ROOT=str(REPO),CUDA_VISIBLE_DEVICES='0,1,2,3',
             TRAIN_FILELIST=str(DATA/'dataset.sqlite'),VALID_FILELIST=str(DATA/'dataset.sqlite'),N_SPKS='2',
             MEL_MEAN=str(stats['mel_mean']),MEL_STD=str(stats['mel_std']),
             OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',
             HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',
             PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    command=['bash',str(REPO/'training.sh'),f'run_name={args.run_name}',
             'ckpt_path=null','init_ckpt_path=null','seed=20260911','data.seed=20260911',
             'data._target_=training.foundation.data.FoundationDataModule',f'+data.db_path={DATA}/dataset.sqlite',
             f'data.batch_size={batch}','data.num_workers=8',f'model.optimizer.lr={args.peak_lr}',
             'trainer.devices=[0,1,2,3]','trainer.precision=bf16-mixed','+trainer.use_distributed_sampler=false',
             '+trainer.max_steps=200000','trainer.check_val_every_n_epoch=null','+trainer.val_check_interval=1000',
             '+trainer.num_sanity_val_steps=2','callbacks.model_checkpoint.every_n_epochs=null',
             'callbacks.model_checkpoint.every_n_train_steps=1000','callbacks.model_checkpoint.monitor=step',
             "callbacks.model_checkpoint.filename='step_{step:08.0f}'",'callbacks.model_checkpoint.save_top_k=3',
             'callbacks.encoder_lr_downshift.enabled=false','callbacks.duration_lr_downshift.enabled=false',
             '+callbacks.warmup._target_=training.foundation.callbacks.WarmupCosine',
             f'+callbacks.warmup.peak_lr={args.peak_lr}',
             f'+callbacks.warmup.final_lr={args.final_lr}',
             '+callbacks.warmup.warmup_steps=1000','+callbacks.warmup.total_steps=200000',
             '+callbacks.training_state._target_=training.common.callbacks.TrainingState',
             'callbacks.rich_progress_bar=null','trainer.enable_progress_bar=false',
             'test=false',f'hydra.run.dir={run}']
    # Preserve code/config to make the run reproducible after later edits.
    import shutil
    for name in ['lits','configs','training']:
        shutil.copytree(REPO/name,run/'source'/name,ignore=shutil.ignore_patterns('__pycache__','.ipynb_checkpoints'),dirs_exist_ok=True)
    with (run/'training.log').open('a') as log:
        child=subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
        launch={'status':'running','training_pid':child.pid,'command':command,'started_at_unix':time.time(),
                'run_dir':str(run),'batch_size_per_gpu':batch,'effective_batch':batch*4,'devices':4,
                'learning_rate':{'peak':args.peak_lr,'final':args.final_lr,'warmup_steps':1000,'total_steps':200000},
                'initialization':'random','active_speaker_ids':[0],'reserved_stage2_speaker_id':1,
                'data_summary':json.loads((DATA/'summary.json').read_text()),
                'dataset_sha256':json.loads((DATA/'integrity.json').read_text())['dataset_sha256']}
        (run/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
        print(json.dumps(launch),flush=True)
        with (run/'eval_watcher.log').open('a') as eval_log:
            watcher=subprocess.Popen([sys.executable,str(REPO/'training/foundation/watch_eval.py'),'--run-dir',str(run),'--training-pid',str(child.pid),'--gpu','0'],cwd=REPO,env=env,stdout=eval_log,stderr=subprocess.STDOUT,start_new_session=True)
        launch['eval_watcher_pid']=watcher.pid
        (run/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
        code=child.wait()
        launch.update(status='complete' if code==0 else 'failed',returncode=code,finished_at_unix=time.time())
        (run/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
        raise SystemExit(code)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--run-name',default='hifitts_premium_stage1_fromscratch')
    parser.add_argument('--peak-lr',type=float,default=1e-4)
    parser.add_argument('--final-lr',type=float,default=2e-5)
    main(parser.parse_args())
