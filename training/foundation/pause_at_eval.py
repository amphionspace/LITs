"""Pause only the current training process group after an immutable eval copy exists."""
import argparse,hashlib,json,os,signal,time
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--run-dir',type=Path,required=True)
p.add_argument('--step',type=int,required=True)
a=p.parse_args()
launch=json.loads((a.run_dir/'launch.json').read_text())
training_pid=launch['training_pid']
pgid=os.getpgid(training_pid)
assert pgid!=os.getpgrp() and pgid!=launch['eval_watcher_pid']
output=a.run_dir/'transition_state.json'
state={'status':'waiting_for_checkpoint','target_step':a.step,'training_pid':training_pid,'training_pgid':pgid,'eval_watcher_pid':launch['eval_watcher_pid']}
output.write_text(json.dumps(state,indent=2)+'\n')
while True:
    destination=a.run_dir/'eval'/f'step_{a.step:08d}'
    meta=destination/'checkpoint_metadata.json'
    checkpoint=destination/'checkpoint.ckpt'
    if meta.exists() and checkpoint.exists():
        metadata=json.loads(meta.read_text())
        if metadata['global_step']==a.step:
            h=hashlib.sha256()
            with checkpoint.open('rb') as f:
                for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
            assert h.hexdigest()==metadata['sha256']
            assert os.getpgid(training_pid)==pgid
            os.killpg(pgid,signal.SIGSTOP)
            progress=json.loads((a.run_dir/'training_state.json').read_text())
            (a.run_dir/'training_state_before_transition.json').write_text(json.dumps(progress,indent=2)+'\n')
            progress.update(status='paused_for_checkpoint_eval',resume_step=a.step,updated_at_unix=time.time())
            (a.run_dir/'training_state.json').write_text(json.dumps(progress,indent=2)+'\n')
            state.update(status='paused_for_checkpoint_eval',checkpoint=str(checkpoint),sha256=metadata['sha256'],last_logged_step=progress['global_step'],paused_at_unix=time.time())
            output.write_text(json.dumps(state,indent=2)+'\n')
            print(json.dumps(state),flush=True)
            break
    if not Path(f'/proc/{training_pid}').exists():raise RuntimeError('Training exited before target checkpoint')
    time.sleep(2)
