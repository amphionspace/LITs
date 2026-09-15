"""Paired English synthesis probe using the user's selected fourth listening sample."""
import argparse
import asyncio
import base64
from collections import Counter
import io
import json
import math
import sys
import time
import zlib
from pathlib import Path

from common import ROOT, ASSETS, REPO, atomic_json, config, connection, digest

PROBE=ROOT/'reports/english_reference4_probe'
LEGACY=REPO/'data_generation/majestic_voice'
sys.path.append(str(LEGACY))


def prepare():
    cfg=config();db=connection()
    anchor=dict(db.execute('''select t.id text_id,t.text,t.split,c.* from candidates c join texts t on t.id=c.text_id
        where c.id=72003 and t.accepted_candidate=c.id and t.language='en' ''').fetchone())
    assert anchor['split']=='train' and anchor['state']=='passed'
    assert anchor['text']=='Archie sends his love, and will, with me, be very glad to welcome you to your old home, should you care to visit it.'
    targets=[]
    for length in ['short','medium','long']:
        rows=db.execute("select id,payload,length_bin from pool where language='en' and split='train' and state='available' and length_bin=? order by rank_key limit 6",(length,)).fetchall()
        for row in rows:
            p=json.loads(row['payload']);assert p['text']!=anchor['text']
            targets.append(dict(id=str(row['id']),text=p['text'],ids=p['ids'],length_bin=row['length_bin']))
    assert len(targets)==18
    PROBE.mkdir(exist_ok=True);(PROBE/'logs').mkdir(exist_ok=True)
    refs={'original_cn':dict(audio=cfg['reference_audio'],text=cfg['prompt_text']),
          'selected_en4':dict(audio=anchor['audio_24k'],text=anchor['text'])}
    for value in refs.values():value['sha256']=digest(value['audio'])
    atomic_json(PROBE/'plan.json',dict(targets=targets,references=refs,candidates_per_text_per_arm=2,
        seeds='20260914 + crc32(text_id:variant); identical across arms',thresholds=cfg['thresholds'],
        similarity_reference=cfg['reference_audio'],anchor_candidate_id=72003,anchor_text_id=anchor['text_id'],
        policy='diagnostic only; no admission or quota credit; original speaker reference retained for QC',created_at_unix=time.time()))
    atomic_json(PROBE/'job.json',dict(model=cfg['tts_model'],similarity_reference_audio=cfg['reference_audio']))


async def synthesize():
    import aiohttp
    import soundfile as sf
    from scipy.signal import resample_poly
    from synthesize import save_audio
    cfg=config();plan=json.loads((PROBE/'plan.json').read_text());uris={}
    for name,ref in plan['references'].items():
        audio,sr=sf.read(ref['audio'],dtype='float32');div=math.gcd(sr,16000)
        audio=resample_poly(audio,16000//div,sr//div);buff=io.BytesIO();sf.write(buff,audio,16000,format='WAV',subtype='PCM_16')
        uris[name]='data:audio/wav;base64,'+base64.b64encode(buff.getvalue()).decode()
    done=set()
    log=PROBE/'logs/synthesis.jsonl'
    if log.exists():done={json.loads(x)['id'] for x in log.read_text().splitlines()}
    queue=asyncio.Queue()
    for target in plan['targets']:
        for variant in range(2):
            for arm in plan['references']:
                key=f'{arm}_{target["id"]}_{variant}'
                if key not in done:queue.put_nowait((key,target,variant,arm))
    async def worker(endpoint,session,stream):
        while not queue.empty():
            try:key,target,variant,arm=queue.get_nowait()
            except asyncio.QueueEmpty:return
            seed=20260914+zlib.crc32(f'{target["id"]}:{variant}'.encode())
            row=dict(id=key,source_id=target['id'],text=target['text'],language='English',arm=arm,seed=seed,
                output_24k=str(PROBE/'wavs_24k'/f'{key}.wav'),output_48k=str(PROBE/'wavs_48k'/f'{key}.wav'))
            body=dict(model=cfg['tts_model'],input=target['text'],ref_audio=uris[arm],ref_text=plan['references'][arm]['text'],
                response_format='wav',stream=False,seed=seed,max_new_tokens=1000)
            async with session.post(endpoint+'/v1/audio/speech',json=body) as response:
                data=await response.read()
                if response.status!=200:raise RuntimeError(f'HTTP {response.status}: {data[:200]!r}')
            stats=await asyncio.to_thread(save_audio,data,row)
            stream.write(json.dumps(dict(row,**stats,status='ok',endpoint=endpoint))+'\n');stream.flush()
            done.add(key);atomic_json(PROBE/'progress.json',dict(completed=len(done),target=72,updated_at_unix=time.time()))
            print(f'synthesized {len(done)}/72',flush=True)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800)) as session:
        with log.open('a') as stream:
            await asyncio.gather(*(worker(ep,session,stream) for ep in cfg['tts_endpoints'] for _ in range(2)))


def summarize():
    from run_workers import judge,wave_checks
    def records(path):return [json.loads(x) for x in path.read_text().splitlines()]
    plan=json.loads((PROBE/'plan.json').read_text());cfg=config()
    waves=records(PROBE/'logs/synthesis.jsonl');asrs={r['id']:r for r in records(PROBE/'quality/asr.jsonl')}
    metrics={r['id']:r for r in records(PROBE/'quality/metrics.jsonl')};targets={r['id']:r for r in plan['targets']}
    assert len(waves)==len(asrs)==len(metrics)==72
    arms={}
    for name in plan['references']:
        results=[]
        for row in waves:
            if row['arm']!=name:continue
            met=metrics[row['id']];a=asrs[row['id']];wave=wave_checks(row['output_24k'])
            reasons=judge(dict(language='en',asr=json.dumps(a),synthesis=json.dumps(wave)),met,cfg['thresholds'])
            if not cfg['min_audio_seconds']<=row['duration']<=cfg['max_audio_seconds']:reasons.append('duration')
            if len(targets[row['source_id']]['ids'])>round(row['duration']*24000)//384:reasons.append('mas_length')
            results.append(dict(id=row['id'],source_id=row['source_id'],text=row['text'],audio=row['output_24k'],duration=row['duration'],
                passed=not reasons,reasons=reasons,metrics=met,asr=a,score=2*met['wavlm_similarity']+met['camp_similarity']+.25*met['dnsmos_ovrl']))
        arms[name]=dict(candidates=len(results),passed_candidates=sum(x['passed'] for x in results),
            texts_with_passing_candidate=len({x['source_id'] for x in results if x['passed']}),
            mean_metrics={k:sum(x['metrics'][k] for x in results)/len(results) for k in ['wavlm_similarity','camp_similarity','dnsmos_ovrl','dnsmos_sig','dnsmos_bak']},
            mean_wer=sum(x['asr']['wer'] for x in results)/len(results),reasons=dict(Counter(r for x in results for r in x['reasons'])),
            examples=sorted(results,key=lambda x:(x['passed'],x['score']),reverse=True))
    result=dict(arms=arms,targets=18,candidates_per_arm=36,similarity_reference=plan['similarity_reference'],
        paired_seeds=True,created_at_unix=time.time(),caveat='Small stratified pilot; not a full-production acceptance estimate or human accent assessment')
    atomic_json(PROBE/'comparison.json',result)
    print(json.dumps({name:{k:v for k,v in arm.items() if k!='examples'} for name,arm in arms.items()},indent=2))


def anchor_metrics():
    import torch
    from quality_metrics import SpeakerMetrics,audio16
    from run_workers import judge,wave_checks
    torch.set_num_threads(4)
    plan=json.loads((PROBE/'plan.json').read_text());report=json.loads((PROBE/'comparison.json').read_text());cfg=config()
    model=SpeakerMetrics(plan['references']['selected_en4']['audio'],fast_matmul=True,
        admission_thresholds=(cfg['thresholds']['min_wavlm_similarity'],cfg['thresholds']['min_camp_similarity']))
    output={}
    for arm,summary in report['arms'].items():
        examples=summary['examples'];scores=model.batch([audio16(x['audio']) for x in examples]);rows=[]
        for row,similarity in zip(examples,scores):
            metrics={**row['metrics'],**similarity}
            reasons=judge(dict(language='en',asr=json.dumps(row['asr']),synthesis=json.dumps(wave_checks(row['audio']))),metrics,cfg['thresholds'])
            reasons += [r for r in row['reasons'] if r in ('duration','mas_length')]
            rows.append(dict(id=row['id'],text=row['text'],audio=row['audio'],duration=row['duration'],**similarity,
                passed_against_selected_reference=not reasons,reasons_against_selected_reference=reasons,
                dnsmos_ovrl=row['metrics']['dnsmos_ovrl'],asr=row['asr']))
        output[arm]=dict(candidates=len(rows),passed_against_selected_reference=sum(r['passed_against_selected_reference'] for r in rows),
            wavlm_mean=sum(r['wavlm_similarity'] for r in rows)/len(rows),camp_mean=sum(r['camp_similarity'] for r in rows)/len(rows),
            examples=sorted(rows,key=lambda r:(r['passed_against_selected_reference'],2*r['wavlm_similarity']+r['camp_similarity']+.25*r['dnsmos_ovrl']),reverse=True))
    atomic_json(PROBE/'selected_anchor_comparison.json',dict(reference=plan['references']['selected_en4'],arms=output,
        production_quality_reference_changed=False,caveat='Same numeric thresholds applied to a different reference; not equivalent to passing original Chinese identity checks'))
    print(json.dumps({k:{a:b for a,b in v.items() if a!='examples'} for k,v in output.items()},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','synthesize','summarize','anchor_metrics']);a=p.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='synthesize':asyncio.run(synthesize())
    elif a.stage=='summarize':summarize()
    else:anchor_metrics()
