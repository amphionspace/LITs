"""Resume a paused, fully evaluated run with the valid-frame flow loss fix."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--run-dir', type=Path, required=True)
a = p.parse_args()
run = a.run_dir
repo = Path(__file__).resolve().parents[2]
launch = json.loads((run / 'launch.json').read_text())
transition = json.loads((run / 'transition_state.json').read_text())
checkpoint = Path(transition['checkpoint'])
summary = json.loads((checkpoint.parent / 'summary.json').read_text())
latest = json.loads((run / 'eval/latest_eval.json').read_text())
assert transition['status'] == 'paused_for_checkpoint_eval'
assert latest['status'] == 'complete' and latest['global_step'] == transition['target_step']
assert summary['status'] == 'complete' and summary['samples'] == 450
assert all(g['evaluation_failures'] == 0 for g in summary['groups'].values())
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == transition['sha256']
old_pid = launch['training_pid']
pgid = transition['training_pgid']
assert os.getpgid(old_pid) == pgid != os.getpgrp()
assert b'training.sh' in Path(f'/proc/{old_pid}/cmdline').read_bytes()
assert '\nState:\tT' in Path(f'/proc/{old_pid}/status').read_text()
watcher = launch['eval_watcher_pid']
assert b'watch_eval.py' in Path(f'/proc/{watcher}/cmdline').read_bytes()
assert os.getpgid(watcher) != pgid
# Preserve the launch environment in memory, without logging credentials.
env = dict(item.split('=', 1) for item in Path(f'/proc/{old_pid}/environ').read_bytes().decode().split('\0') if item)
command = [f'ckpt_path={checkpoint}' if item.startswith('ckpt_path=') else item for item in launch['command']]
assert 'init_ckpt_path=null' in command
archive = run / 'transitions/padding_fix_step_00009000'
archive.mkdir(parents=True, exist_ok=False)
for name in ['launch.json', 'training_state.json', 'transition_state.json', '.hydra']:
    source = run / name
    if source.is_dir():
        shutil.copytree(source, archive / name)
    elif source.exists():
        shutil.copy2(source, archive / name)
shutil.copy2(repo / 'lits/models/components/flow_matching.py', archive / 'flow_matching.py')
fix_hash = hashlib.sha256((archive / 'flow_matching.py').read_bytes()).hexdigest()
os.kill(watcher, signal.SIGTERM)
# The process group is already stopped and its checkpoint is immutable.
# Kill it while stopped, so no additional old-code updates can be written.
os.killpg(pgid, signal.SIGKILL)
for _ in range(60):
    live = []
    for proc in Path('/proc').iterdir():
        if proc.name.isdigit():
            try:
                if os.getpgid(int(proc.name)) == pgid and proc.joinpath('stat').read_text().split(') ')[1][0] != 'Z':
                    live.append(proc.name)
            except (FileNotFoundError, ProcessLookupError):
                pass
    if not live:
        break
    time.sleep(1)
assert not live, live
with (run / 'training_padding_resume.log').open('a') as log:
    child = subprocess.Popen(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
launch.update(status='running', training_pid=child.pid, command=command,
              resume_checkpoint=str(checkpoint), resume_step=transition['target_step'],
              resumed_at_unix=time.time(), resume_supervisor_pid=os.getpid(),
              flow_loss='valid_frames_only', flow_source_sha256=fix_hash)
with (run / 'eval_watcher.log').open('a') as log:
    new_watcher = subprocess.Popen([sys.executable, str(repo / 'training/foundation/watch_eval.py'),
                                   '--run-dir', str(run), '--training-pid', str(child.pid), '--gpu', '0'],
                                  cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
launch['eval_watcher_pid'] = new_watcher.pid
(run / 'launch.json').write_text(json.dumps(launch, indent=2) + '\n')
transition.update(status='resuming_with_padding_fix', resumed_at_unix=time.time(),
                  new_training_pid=child.pid, new_eval_watcher_pid=new_watcher.pid,
                  flow_source_sha256=fix_hash)
(run / 'transition_state.json').write_text(json.dumps(transition, indent=2) + '\n')
print(json.dumps(transition), flush=True)
code = child.wait()
launch.update(status='complete' if code == 0 else 'failed', returncode=code, finished_at_unix=time.time())
(run / 'launch.json').write_text(json.dumps(launch, indent=2) + '\n')
sys.exit(code)
