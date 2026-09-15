"""Launch the existing training entry point and its checkpoint evaluator."""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
ASSETS = Path('/119010446/tts-assets')
DATA = ASSETS / 'data_24k/ljs_majestic'
PYTHON = ASSETS / 'chenmingjie/lits-tts/envs/lits/bin/python'


def main(args):
    stats = json.loads((DATA / 'mel_statistics.json').read_text())
    assert stats['mas_text_length_check'] == 'passed'
    assert stats['manifest_sha256'] == hashlib.sha256((DATA / 'train.txt').read_bytes()).hexdigest()
    summary = json.loads((DATA / 'training_summary.json').read_text())
    assert summary['majestic_unique_english_train'] == 2756
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if (args.run_dir / 'launch.json').exists():
        raise RuntimeError('Run directory already launched; use an explicit checkpoint resume command')
    env = dict(os.environ, PATH=str(PYTHON.parent) + ':' + os.environ['PATH'],
               PYTHONPATH=str(REPO), PROJECT_ROOT=str(REPO), CUDA_VISIBLE_DEVICES='0,1',
               TRAIN_FILELIST=str(DATA / 'train.txt'), VALID_FILELIST=str(DATA / 'val.txt'), N_SPKS='2',
               MEL_MEAN=str(stats['mel_mean']), MEL_STD=str(stats['mel_std']),
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
               TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', PYTHONUNBUFFERED='1',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    command = ['bash', str(REPO / 'training.sh'), 'run_name=ljs_majestic_24k_fromscratch',
               'ckpt_path=null', 'init_ckpt_path=null', 'seed=20260910', 'data.seed=20260910',
               'data.batch_size=32', 'data.num_workers=6', 'model.optimizer.lr=0.0001',
               'trainer.devices=[0,1]', 'trainer.precision=bf16-mixed',
               '+trainer.accumulate_grad_batches=3', '+trainer.max_steps=200000',
               'trainer.check_val_every_n_epoch=1', 'trainer.gradient_clip_val=5.0',
               'callbacks.model_checkpoint.every_n_epochs=5', 'callbacks.model_checkpoint.save_top_k=3',
               '+callbacks.training_state._target_=training.majestic_voice.callbacks.TrainingState',
               'callbacks.rich_progress_bar=null', 'trainer.enable_progress_bar=false',
               'test=false', f'hydra.run.dir={args.run_dir}']
    with (args.run_dir / 'training.log').open('a') as log, (args.run_dir / 'eval_watcher.log').open('a') as eval_log:
        training = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        watcher = subprocess.Popen([str(PYTHON), str(CODE / 'watch_eval.py'), '--run-dir', str(args.run_dir),
                                    '--training-pid', str(training.pid), '--gpu', '0'],
                                   cwd=REPO, env=env, stdout=eval_log, stderr=subprocess.STDOUT, start_new_session=True)
        launch = {'status': 'running', 'training_pid': training.pid, 'eval_watcher_pid': watcher.pid,
                  'command': command, 'train_manifest_sha256': stats['manifest_sha256'],
                  'run_dir': str(args.run_dir), 'started_at_unix': time.time()}
        (args.run_dir / 'launch.json').write_text(json.dumps(launch, indent=2) + '\n')
        workflow_path = Path(summary['source_chinese_audit']).parents[1] / 'workflow_status.json'
        workflow = json.loads(workflow_path.read_text()) if workflow_path.exists() else {}
        workflow.update(phase='training_and_evaluating', training_status='running',
                        english_status='audited_complete', english_training_texts=2756,
                        run_dir=str(args.run_dir), training_pid=training.pid, eval_watcher_pid=watcher.pid)
        workflow_path.write_text(json.dumps(workflow, indent=2) + '\n')
        print(json.dumps(launch), flush=True)
        code = training.wait()
        launch.update(status='complete' if code == 0 else 'failed', returncode=code, finished_at_unix=time.time())
        (args.run_dir / 'launch.json').write_text(json.dumps(launch, indent=2) + '\n')
        workflow.update(phase='training_finished' if code == 0 else 'training_failed',
                        training_status=launch['status'], returncode=code)
        workflow_path.write_text(json.dumps(workflow, indent=2) + '\n')
        print(json.dumps(launch), flush=True)
        if code:
            if watcher.poll() is None:
                os.killpg(watcher.pid, signal.SIGTERM)
            watcher.wait()
        else:
            watcher.wait()
        raise SystemExit(code)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    main(parser.parse_args())
