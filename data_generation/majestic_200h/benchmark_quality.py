"""Measure quality-stage overlap and CPU/GPU DNSMOS score parity on identical audio."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import sys
import threading
import time

from common import ROOT, REPO, atomic_json, config, connection
sys.path.append(str(REPO/'data_generation/majestic_voice'))


class TimedSession:
    def __init__(self,session):self.session=session;self.seconds=0.;self.calls=0;self.lock=threading.Lock()
    def run(self,*args,**kwargs):
        start=time.perf_counter();result=self.session.run(*args,**kwargs)
        with self.lock:self.seconds+=time.perf_counter()-start;self.calls+=1
        return result


def main(provider):
    import torch
    from quality_metrics import DNSMOS,SpeakerMetrics,audio16
    torch.set_num_threads(4);cfg=config();directory=ROOT/'reports/dnsmos_benchmark';directory.mkdir(exist_ok=True)
    manifest=directory/'samples.json'
    if manifest.exists():rows=json.loads(manifest.read_text())
    else:
        db=connection();rows=[]
        for lang in ['en','zh','mixed']:
            rows += [dict(r) for r in db.execute('''select c.id,c.audio_24k,c.duration,t.language from candidates c join texts t on t.id=c.text_id
                where t.language=? and c.metrics is not null order by c.id desc limit 8''',(lang,))]
        db.close();atomic_json(manifest,rows)
    start=time.perf_counter();audios=[audio16(r['audio_24k']) for r in rows];read_seconds=time.perf_counter()-start
    mos=DNSMOS(provider);main_provider=mos.main.get_providers();p808_provider=mos.p808.get_providers()
    speakers=SpeakerMetrics(cfg['reference_audio'],fast_matmul=True,admission_thresholds=(.7,.68))
    # Exclude imports, model loading and first-request kernel setup from timed repetitions.
    mos(audios[0]);speakers.batch(audios[:2]);torch.cuda.synchronize()
    mos.main=TimedSession(mos.main);mos.p808=TimedSession(mos.p808)
    repetitions=[];last=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for repeat in range(3):
            start=time.perf_counter();futures=[pool.submit(mos,a) for a in audios]
            speaker_start=time.perf_counter();speakers.batch(audios);torch.cuda.synchronize();speaker_seconds=time.perf_counter()-speaker_start
            wait_start=time.perf_counter();last=[f.result() for f in futures];remaining_dns_wait=time.perf_counter()-wait_start
            repetitions.append(dict(wall_seconds=time.perf_counter()-start,speaker_seconds=speaker_seconds,remaining_dns_wait_seconds=remaining_dns_wait))
    results=[dict(id=r['id'],**m) for r,m in zip(rows,last)]
    report=dict(provider=provider,providers=dict(main=main_provider,p808=p808_provider),samples=len(rows),read_seconds=read_seconds,
        protocol='Same 24 recordings, CPU pool 8 plus concurrent speaker metrics, 3 warm repetitions; shares GPU with live synthesis',
        repetitions=repetitions,main_onnx_thread_seconds=mos.main.seconds,p808_onnx_thread_seconds=mos.p808.seconds,
        onnx_windows=mos.main.calls,results=results,completed_at_unix=time.time())
    atomic_json(directory/f'{provider}.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--provider',choices=['cpu','cuda'],required=True);a=p.parse_args();main(a.provider)
