"""Foreground supervisor intended for a detached session; owns only its workers."""
import fcntl
import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path
from common import ROOT, ASSETS, CODE, REPO, atomic_json, config, connection


def main():
    lock=(ROOT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=config();workers={};handles=[]
    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    start=time.time()
    while True:
        try:
            for url in cfg['tts_endpoints']:
                with urllib.request.urlopen(url+'/health',timeout=3) as response:
                    assert response.status==200
            break
        except Exception:
            if time.time()-start>600:raise RuntimeError('TTS health check timeout')
            time.sleep(3)
    db=connection()
    # Exclusive lock means old supervisor is gone. Never reset live independent workers.
    previous=ROOT/'workers.json'
    if previous.exists():
        for record in json.loads(previous.read_text()).values():
            proc=Path(f'/proc/{record["pid"]}')
            try:
                state=(proc/'stat').read_text().rsplit(')',1)[1].split()[0]
                command=(proc/'cmdline').read_bytes()
            except FileNotFoundError:continue
            if state!='Z' and str(CODE/'run_workers.py').encode() in command:
                raise RuntimeError('A previous worker is still alive; inspect workers.json')
    for busy,ready in [('synthesizing','queued'),('asr_running','synthesized'),('metrics_running','asr_done')]:
        db.execute('update candidates set state=?,lease_until=0 where state=?',(ready,busy))
    complete=False
    try:
        for kind,gpu in [('feeder',None),('synth',None),('asr',2),('metrics',3)]:
            python=ASSETS/'.venv-voxcpm2/bin/python' if kind!='metrics' else Path('/119010446/UltraEval-Audio/envs/metrics/bin/python')
            childenv=dict(env)
            if gpu is not None:childenv['CUDA_VISIBLE_DEVICES']=str(gpu)
            if kind=='metrics' and cfg.get('dnsmos_provider')=='cuda':
                childenv['CUDA_VISIBLE_DEVICES']=','.join(str(g) for g in cfg['quality_metrics_gpus'])
                childenv['PYTHONPATH']=cfg['dnsmos_runtime_path']+':'+env.get('PYTHONPATH','')
            log=(ROOT/'logs'/f'{kind}.log').open('a',buffering=1);handles.append(log)
            workers[kind]=subprocess.Popen([str(python),'-u',str(CODE/'run_workers.py'),kind],env=childenv,stdout=log,stderr=subprocess.STDOUT)
        atomic_json(ROOT/'workers.json',{k:dict(pid=p.pid) for k,p in workers.items()})
        job=json.loads((ROOT/'job.json').read_text())
        for key in ('error','stopped_at_unix','failed_stage','returncode'):job.pop(key,None)
        job.update(status='synthesizing_and_filtering',supervisor_pid=os.getpid(),active_languages=cfg['active_languages'],started_at_unix=time.time());atomic_json(ROOT/'job.json',job)
        def stop(signum,frame):raise KeyboardInterrupt
        signal.signal(signal.SIGTERM,stop)
        while True:
            for kind,process in workers.items():
                if process.poll() is not None:raise RuntimeError(f'{kind} exited {process.returncode}; inspect logs/{kind}.log')
            atomic_json(ROOT/'supervisor_heartbeat.json',dict(pid=os.getpid(),updated_at_unix=time.time(),workers={k:p.pid for k,p in workers.items()}))
            if (ROOT/'collection_ready.json').exists():
                complete=True;break
            time.sleep(5)
    except BaseException as exc:
        job=json.loads((ROOT/'job.json').read_text());job.update(status='stopped',error=str(exc),stopped_at_unix=time.time());atomic_json(ROOT/'job.json',job)
        raise
    finally:
        for p in workers.values():
            if p.poll() is None:p.terminate()
        for p in workers.values():
            try:p.wait(timeout=20)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        for h in handles:h.close()
    if complete:
        # Release only the VoxCPM2 services started for this collection.
        for record in json.loads((ROOT/'servers.json').read_text()).values():
            pid=record['pid'];cmd=Path(f'/proc/{pid}/cmdline')
            if cmd.exists() and str(ASSETS/'VoxCPM2').encode() in cmd.read_bytes() and os.getpgid(pid)==pid:
                os.killpg(pid,signal.SIGTERM)
        time.sleep(15)
        python=str(ASSETS/'chenmingjie/lits-tts/envs/lits/bin/python');run=Path(config()['training']['run_dir'])
        childenv=dict(env,PYTHONPATH=str(CODE)+':'+str(REPO))
        # Audit rechecks every quota, winner and audio header before any model work.
        commands=[[python,str(CODE/'prepare_training.py')],
                  [python,str(REPO/'training/stage2/preflight.py'),'--run-dir',str(run)],
                  [python,str(REPO/'training/stage2/run.py'),'--run-dir',str(run)]]
        for label,command in zip(['prepare','preflight','training'],commands):
            job=json.loads((ROOT/'job.json').read_text());job.update(status=label,training_run_dir=str(run));atomic_json(ROOT/'job.json',job)
            with (ROOT/'logs'/f'{label}_handoff.log').open('a') as log:
                result=subprocess.run(command,env=childenv,cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                job.update(status='handoff_failed',failed_stage=label,returncode=result.returncode);atomic_json(ROOT/'job.json',job)
                raise RuntimeError(f'{label} failed; inspect handoff log')
        job.update(status='training_complete');atomic_json(ROOT/'job.json',job)


if __name__=='__main__':main()
