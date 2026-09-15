"""Finish preparation and capacity checks, then launch a fresh four-GPU run."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[2]
DATA=Path('/119010446/tts-assets/data_24k/foundation')
RUN=Path('/119010446/tts-assets/training_runs/hifitts_premium_stage1_20260911')

def main():
    deadline=time.time()+3600
    while not (DATA/'summary.json').exists():
        if time.time()>deadline:
            raise TimeoutError('Data preparation did not finish within one hour')
        time.sleep(5)
    for script in ['compute_statistics.py','benchmark.py']:
        with (DATA/(script.removesuffix('.py')+'.log')).open('a') as log:
            subprocess.run([sys.executable,str(REPO/'training/foundation'/script)],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
    RUN.mkdir(parents=True,exist_ok=True)
    with (RUN/'launcher.log').open('a') as log:
        subprocess.run([sys.executable,str(REPO/'training/foundation/run.py'),'--run-dir',str(RUN)],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)

if __name__=='__main__':main()
