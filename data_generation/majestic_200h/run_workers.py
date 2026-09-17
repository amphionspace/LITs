"""Persistent, quota-driven text admission, VoxCPM2 synthesis and independent QC."""
import argparse
import asyncio
import json
import math
import os
import re
import sqlite3
import sys
import time
from functools import lru_cache
from pathlib import Path

from common import ROOT, ASSETS, REPO, atomic_json, config, connection, initialize, claim, reject, near_match, totals, goals, digest

LEGACY=REPO/'data_generation/majestic_voice'
sys.path.append(str(LEGACY))


def enqueue(db, text_id, start, count):
    now=time.time()
    for attempt in range(start,start+count):
        db.execute('insert or ignore into candidates(text_id,attempt,created_at,updated_at) values(?,?,?,?)',
                   (text_id,attempt,now,now))
    db.execute('update texts set attempts=? where id=?',(start+count,text_id))


def synthesis_reference(cfg, language):
    reference=cfg.get('synthesis_references',{}).get(language)
    if reference is None:return dict(reference_audio=cfg['reference_audio'],prompt_text=cfg['prompt_text'])
    if not reference.get('reference_audio') or not reference.get('prompt_text'):
        raise ValueError('Language reference requires both audio and exact transcript')
    return reference


def quality_reference(cfg, language):
    return cfg.get('quality_reference_audio_by_language',{}).get(language,cfg['reference_audio'])


def candidate_limit(cfg, language, split):
    return cfg.get('max_candidates_per_text_by_split',{}).get(split,{}).get(language,
        cfg.get('max_candidates_per_text_by_language',{}).get(language,cfg['max_candidates_per_text']))


@lru_cache(maxsize=8)
def reference_encoding(path):
    import base64,io
    import soundfile as sf
    from scipy.signal import resample_poly
    audio,sr=sf.read(path,dtype='float32')
    if audio.ndim!=1:raise ValueError('Reference audio must be mono')
    div=math.gcd(sr,16000);audio=resample_poly(audio,16000//div,sr//div)
    buffer=io.BytesIO();sf.write(buffer,audio,16000,format='WAV',subtype='PCM_16')
    return 'data:audio/wav;base64,'+base64.b64encode(buffer.getvalue()).decode(),digest(path)


def synthesis_languages(cfg, index):
    active=cfg.get('active_languages',['zh','en','mixed'])
    slots=[lang for lang in cfg.get('synthesis_language_slots',['zh','zh','en','mixed']) if lang in active]
    if not active:return []
    preferred=(slots or active)[index%len(slots or active)]
    return [preferred]+[lang for lang in active if lang!=preferred]


def active_candidate_count(db, active):
    if not active:return 0
    placeholders=','.join('?' for _ in active)
    return db.execute(f'''select count(*) from candidates c join texts t on t.id=c.text_id
        where c.state not in ('passed','rejected') and t.language in ({placeholders})''',active).fetchone()[0]


def feeder():
    db=connection();old=sqlite3.connect('file:'+str(ROOT/'text/exclusions.sqlite')+'?mode=ro',uri=True);old.row_factory=sqlite3.Row
    last_manifest_export=time.monotonic()
    while True:
        cfg=config();active=cfg.get('active_languages',['zh','en']);current=totals(db);targets=goals(cfg)
        caps={lang:max(candidate_limit(cfg,lang,s) for s in ('train','val','test')) for lang in ('zh','en','mixed')}
        retry_budget=max(0,2048-active_candidate_count(db,active))
        # Decide between the scored variants only after the current round has finished.
        for text in db.execute('''select t.id,t.attempts,t.language,t.split from texts t where accepted_candidate is null
            and (t.attempts<case t.language when 'en' then ? when 'mixed' then ? else ? end
                 or exists(select 1 from candidates c where c.text_id=t.id and c.state='passed'))
            and exists(select 1 from candidates c where c.text_id=t.id)
            and not exists(select 1 from candidates c where c.text_id=t.id and c.state not in ('passed','rejected'))''',
            (caps['en'],caps['mixed'],caps['zh'])).fetchall():
            winner=db.execute("select * from candidates where text_id=? and state='passed' order by score desc,id limit 1",(text['id'],)).fetchone()
            if winner:
                for rate in (24,48):
                    source=Path(winner[f'audio_{rate}k']);target=ROOT/f'accepted/wavs_{rate}k'/f'{text["id"]}.wav';target.parent.mkdir(parents=True,exist_ok=True)
                    if not target.exists():os.link(source,target)
                db.execute('update texts set accepted_candidate=? where id=?',(winner['id'],text['id']))
            elif (text['language'] in active and text['attempts']<candidate_limit(cfg,text['language'],text['split'])
                  and current.get((text['split'],text['language']),{}).get('seconds',0)<targets[(text['split'],text['language'])] and retry_budget>0):
                count=min(cfg['candidates_per_round'],candidate_limit(cfg,text['language'],text['split'])-text['attempts'],retry_budget)
                enqueue(db,text['id'],text['attempts'],count);retry_budget-=count
        current=totals(db)
        # Round robin across language/split queues; bounds disk and QC backlog.
        if active_candidate_count(db,active)<2048:
            for split in ('val','test','train'):
                for lang in active:
                    if current.get((split,lang),{}).get('seconds',0)>=targets[(split,lang)]:continue
                    outstanding=db.execute('''select count(*) from texts where split=? and language=? and accepted_candidate is null
                        and (attempts<? or exists(select 1 from candidates c where c.text_id=texts.id and c.state not in ('passed','rejected')))''',
                        (split,lang,candidate_limit(cfg,lang,split))).fetchone()[0]
                    if outstanding>=192:continue
                    rows=db.execute("select * from pool where state='available' and split=? and language=? order by rank_key limit 64",(split,lang)).fetchall()
                    for row in rows:
                        payload=json.loads(row['payload']);text=payload['text']
                        match=near_match(old,'old_fts',text) or near_match(db,'new_fts',text)
                        if match:
                            db.execute("update pool set state='near_duplicate' where id=?",(row['id'],));continue
                        db.execute('begin immediate')
                        try:
                            cur=db.execute('''insert or ignore into texts(id,language,split,domain,length_bin,text,norm,phone_key,payload,estimated_seconds,created_at)
                                values(?,?,?,?,?,?,?,?,?,?,?)''',(row['id'],lang,split,payload.get('source','external'),row['length_bin'],text,row['norm'],row['phone_key'],row['payload'],row['estimated_seconds'],time.time()))
                            if cur.rowcount:
                                db.execute('insert into new_fts(rowid,norm) values(?,?)',(row['id'],row['norm']))
                                enqueue(db,row['id'],0,cfg['candidates_per_round'])
                            db.execute("update pool set state='admitted' where id=?",(row['id'],));db.execute('commit')
                        except BaseException:db.execute('rollback');raise
        export_manifests=time.monotonic()-last_manifest_export>=60
        export_progress(db,export_manifests=export_manifests)
        if export_manifests:last_manifest_export=time.monotonic()
        time.sleep(.5)


def export_progress(db,export_manifests=True):
    cfg=config();counts=dict(db.execute('select state,count(*) from candidates group by state').fetchall())
    accepted={s:{l:dict(hours=0.0,texts=0) for l in ('zh','en','mixed')} for s in ('train','val','test')}
    for (s,l),r in totals(db).items():accepted[s][l]=dict(hours=r['seconds']/3600,texts=r['count'])
    atomic_json(ROOT/'pipeline_progress.json',dict(status='running',active_languages=cfg.get('active_languages',['zh','en']),
        admitted_texts=db.execute('select count(*) from texts').fetchone()[0],candidate_states=counts,accepted=accepted,updated_at_unix=time.time()))
    busy=sum(n for state,n in counts.items() if state not in ('passed','rejected'))
    complete=all(accepted[s][l]['hours']*3600>=target for (s,l),target in goals(cfg).items())
    # Never signal completion with stale manifests, even during a throttled cycle.
    if not export_manifests and not complete:return
    # The DB is authoritative; refresh portable manifests without duplicating audio.
    for split in ('train','val','test'):
        p=ROOT/'accepted'/f'{split}.jsonl';temp=p.with_suffix('.jsonl.tmp')
        with temp.open('w') as out:
            for row in db.execute('''select t.*,c.duration,c.asr,c.metrics,c.score from texts t join candidates c on c.id=t.accepted_candidate where t.split=? order by t.id''',(split,)):
                payload=json.loads(row['payload']);payload['source_audio']=payload.get('audio');payload['source_duration']=payload.get('duration')
                payload.update(id=str(row['id']),text=row['text'],language=row['language'],split=split,speaker=1,
                    audio=str(ROOT/'accepted/wavs_24k'/f'{row["id"]}.wav'),duration=row['duration'],sample_rate=24000,
                    synthetic=True,synthesis_model='VoxCPM2',quality=dict(asr=json.loads(row['asr']),metrics=json.loads(row['metrics']),score=row['score']))
                out.write(json.dumps(payload,ensure_ascii=False)+'\n')
        temp.replace(p)
    if complete and not busy:
        atomic_json(ROOT/'collection_ready.json',dict(ready=True,accepted=accepted,updated_at_unix=time.time()))


def wave_checks(path):
    import numpy as np
    import soundfile as sf
    audio,sr=sf.read(path,dtype='float32');n=max(1,int(sr*.02));pad=(-len(audio))%n
    frames=np.pad(audio,(0,pad)).reshape(-1,n);rms=np.sqrt(np.mean(frames**2,axis=1))
    threshold=max(.002,float(rms.max())*.02);active=rms>threshold;idx=np.flatnonzero(active)
    longest=0;run=0
    if len(idx):
        for v in active[idx[0]:idx[-1]+1]:
            run=0 if v else run+1;longest=max(longest,run)
    return dict(clip_fraction=float(np.mean(np.abs(audio)>=.999)),active_fraction=float(active.mean()),
        leading_silence=float(idx[0]*.02) if len(idx) else len(audio)/sr,
        trailing_silence=float((len(active)-1-idx[-1])*.02) if len(idx) else len(audio)/sr,
        internal_silence=longest*.02)


async def synthesizer():
    import aiohttp,zlib
    from synthesize import save_audio
    cfg=config()
    async def worker(endpoint,index,session):
        db=connection()
        while True:
            current_cfg=config()
            rows=[]
            for language in synthesis_languages(current_cfg,int(index.split('-')[-1])):
                rows=claim(db,'queued','synthesizing',f'synth-{index}',lease_seconds=2100,language=language)
                if rows:break
            if not rows:await asyncio.sleep(1);continue
            r=rows[0];paths={f'output_{rate}k':str(ROOT/f'candidates/wavs_{rate}k'/f'{r["id"]}.wav') for rate in (24,48)}
            reference=synthesis_reference(current_cfg,r['language'])
            uri,reference_sha256=reference_encoding(reference['reference_audio'])
            body=dict(model=cfg['tts_model'],input=r['text'],ref_audio=uri,ref_text=reference['prompt_text'],response_format='wav',stream=False,
                      seed=cfg['seed']+zlib.crc32(f'{r["text_id"]}:{r["attempt"]}'.encode()),max_new_tokens=1000)
            last=None
            for attempt in range(3):
                try:
                    async with session.post(endpoint+'/v1/audio/speech',json=body) as response:
                        data=await response.read()
                        if response.status!=200:raise RuntimeError(f'HTTP {response.status}: {data[:300]!r}')
                    info=await asyncio.to_thread(save_audio,data,dict(text=r['text'],language='English' if r['language']=='en' else 'Chinese',**paths))
                    if not cfg['min_audio_seconds']<=info['duration']<=cfg['max_audio_seconds']:
                        reject(db,r,['duration'],duration=info['duration']);break
                    if len(json.loads(r['payload'])['ids'])>int(round(info['duration']*24000))//384:
                        reject(db,r,['mas_length'],duration=info['duration']);break
                    checks=await asyncio.to_thread(wave_checks,paths['output_24k']);info.update(checks,endpoint=endpoint,seed=body['seed'],
                        reference_audio=reference['reference_audio'],reference_sha256=reference_sha256,reference_text=reference['prompt_text'])
                    db.execute("update candidates set state='synthesized',audio_24k=?,audio_48k=?,duration=?,synthesis=?,lease_until=0,updated_at=? where id=?",
                        (paths['output_24k'],paths['output_48k'],info['duration'],json.dumps(info),time.time(),r['id']));break
                except ValueError as exc:reject(db,r,[str(exc)]);break
                except Exception as exc:
                    last=exc;await asyncio.sleep(2**attempt)
            else:
                db.execute("update candidates set state='queued',lease_until=0 where id=?",(r['id'],))
                raise RuntimeError(f'Synthesis infrastructure failed: {last}')
    timeout=aiohttp.ClientTimeout(total=1800)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        await asyncio.gather(*(worker(ep,f'{i}-{j}',session) for i,ep in enumerate(cfg['tts_endpoints']) for j in range(8)))


def quality(kind):
    import torch
    from quality_worker import cer,wer,tonal_match
    torch.set_num_threads(4);cfg=config();thresholds=cfg['thresholds'];db=connection()
    if kind=='asr':
        from qwen_asr import Qwen3ASRModel
        model=Qwen3ASRModel.from_pretrained(str(ASSETS/'Qwen3-ASR-1.7B'),dtype=torch.bfloat16,device_map='cuda:0',
            attn_implementation='sdpa',max_inference_batch_size=16,max_new_tokens=256)
    else:
        from quality_metrics import SpeakerMetrics,DNSMOS,audio16
        from concurrent.futures import ThreadPoolExecutor
        speakers=SpeakerMetrics(cfg['reference_audio'],fast_matmul=True,
            admission_thresholds=(thresholds['min_wavlm_similarity'],thresholds['min_camp_similarity']))
        speaker_references={cfg['reference_audio']:speakers.reference}
        reference_hashes={cfg['reference_audio']:digest(cfg['reference_audio'])}
        mos=DNSMOS(provider=cfg.get('dnsmos_provider','cpu'),device_id=cfg.get('dnsmos_device_id',0));cpu=ThreadPoolExecutor(max_workers=8)
        print(json.dumps(dict(dnsmos_provider=mos.provider,dnsmos_main_providers=mos.main.get_providers(),dnsmos_p808_providers=mos.p808.get_providers(),
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),dnsmos_device_id=cfg.get('dnsmos_device_id',0))),flush=True)
    print(kind+' models ready',flush=True)
    while True:
        rows=claim(db,'synthesized' if kind=='asr' else 'asr_done',kind+'_running',kind,16,lease_seconds=900)
        if not rows:time.sleep(1);continue
        if kind=='asr':
            predictions=model.transcribe(audio=[r['audio_24k'] for r in rows],context='',language=None)
            if len(predictions)!=len(rows):raise RuntimeError('ASR count mismatch')
            for r,p in zip(rows,predictions):
                result=dict(text=p.text,language=p.language,cer=cer(r['text'],p.text),wer=wer(r['text'],p.text))
                result['tonal_match']=tonal_match(r['text'],p.text) if r['language']=='zh' else False
                if r['language']=='mixed':
                    result['chinese_cer']=cer(''.join(re.findall('[\u4e00-\u9fff]',r['text'])),''.join(re.findall('[\u4e00-\u9fff]',p.text)))
                db.execute("update candidates set asr=?,state='asr_done',lease_until=0,updated_at=? where id=?",(json.dumps(result,ensure_ascii=False),time.time(),r['id']))
        else:
            quality_started=time.perf_counter()
            audios=[audio16(r['audio_24k']) for r in rows];futures=[cpu.submit(mos,a) for a in audios]
            read_seconds=time.perf_counter()-quality_started;speaker_started=time.perf_counter()
            current_cfg=config();reference_groups={};similarity=[None]*len(rows)
            for i,row in enumerate(rows):reference_groups.setdefault(quality_reference(current_cfg,row['language']),[]).append(i)
            for path,indices in reference_groups.items():
                if path not in speaker_references:
                    speaker_references[path]=speakers.embeddings(audio16(path));reference_hashes[path]=digest(path)
                speakers.reference=speaker_references[path]
                for i,sim in zip(indices,speakers.batch([audios[i] for i in indices])):
                    similarity[i]=dict(sim,speaker_reference_audio=path,speaker_reference_sha256=reference_hashes[path],quality_policy='language_reference_v1')
            speaker_seconds=time.perf_counter()-speaker_started;wait_started=time.perf_counter()
            dns_metrics=[future.result() for future in futures];dns_wait_seconds=time.perf_counter()-wait_started
            for r,sim,dns_result in zip(rows,similarity,dns_metrics):
                metrics={**sim,**dns_result,'dnsmos_provider':mos.provider};reasons=judge(r,metrics,thresholds)
                score=2*metrics['wavlm_similarity']+metrics['camp_similarity']+.25*metrics['dnsmos_ovrl']
                db.execute("update candidates set metrics=?,state=?,reasons=?,score=?,lease_until=0,updated_at=? where id=?",
                    (json.dumps(metrics),'rejected' if reasons else 'passed',json.dumps(reasons),score,time.time(),r['id']))
        timing=dict(total_seconds=time.perf_counter()-quality_started,read_and_dispatch_seconds=read_seconds,speaker_seconds=speaker_seconds,dns_wait_seconds=dns_wait_seconds) if kind=='metrics' else {}
        print(json.dumps(dict(kind=kind,batch=len(rows),last_id=rows[-1]['id'],time=time.time(),**timing)),flush=True)


def judge(row,metrics,thresholds):
    asr=json.loads(row['asr']);wave=json.loads(row['synthesis']);reasons=[]
    for key,value in thresholds.items():
        if key in ('min_wavlm_similarity','min_camp_similarity','min_dnsmos_ovrl','min_dnsmos_sig','min_dnsmos_bak'):
            actual=metrics.get(key[4:],float('nan'))
            if not math.isfinite(actual) or actual<value:reasons.append(key)
    for key in ('clip_fraction','leading_silence','trailing_silence','internal_silence'):
        if not math.isfinite(wave[key]) or wave[key]>thresholds['max_'+key]:reasons.append(key)
    if wave['active_fraction']<thresholds['min_active_fraction']:reasons.append('active_fraction')
    if not all(math.isfinite(asr.get(k,float('nan'))) for k in ('cer','wer')):reasons.append('nonfinite_asr')
    if row['language']=='en':
        if asr['wer']>thresholds['max_english_wer'] or asr['cer']>thresholds['max_cer']:reasons.append('asr')
    elif row['language']=='zh':
        allowed=thresholds['allow_exact_tonal_pinyin'] and asr['tonal_match'] and asr['cer']<=thresholds['max_raw_cer_for_tonal_exception']
        if asr['cer']>thresholds['max_cer'] and not allowed:reasons.append('asr')
    elif asr['wer']>thresholds['max_mixed_english_wer'] or asr['chinese_cer']>thresholds['max_mixed_chinese_cer']:reasons.append('asr')
    return reasons


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('kind',choices=['feeder','synth','asr','metrics']);args=parser.parse_args()
    initialize()
    if args.kind=='feeder':feeder()
    elif args.kind=='synth':asyncio.run(synthesizer())
    else:quality(args.kind)
