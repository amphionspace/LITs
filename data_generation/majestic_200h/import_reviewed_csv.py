"""Import the user-approved, externally generated and reviewed DOTA text."""
import csv
import hashlib
import json
import multiprocessing as mp
import time
from collections import Counter
from common import ROOT,atomic_json,connection,norm
from prepare_corpus import classify,eligibility
from import_emilia_text import init_worker,process


def main():
    folder=ROOT/'text/external/asr-code-switch';source=folder/'short_audio.csv';report=folder/'import_report.json'
    if report.exists():print(report.read_text());return
    manifest=json.loads((folder/'download_manifest.json').read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest()==manifest['sha256']
    db=connection();db.execute('attach database ? as excluded',(str(ROOT/'text/exclusions.sqlite'),))
    old_text={r[0] for r in db.execute('select norm from excluded.old')};old_phones={r[0] for r in db.execute('select key from excluded.phones')}
    seen={r[0] for r in db.execute('select norm from pool')};phones={r[0] for r in db.execute('select phone_key from pool')}
    counts=Counter();selected=[];seconds=0;started=time.time()
    for index,row in enumerate(csv.DictReader(source.open(encoding='utf-8-sig'),delimiter='|'),2):
        text=row['Text'].strip();lang,han,words,letters=classify(text)
        if lang!='mixed':counts['not_mixed']+=1;continue
        reason=eligibility(dict(text=text,ids=[3]*20))
        if reason:counts[reason]+=1;continue
        key=norm(text)
        if key in seen or key in old_text:counts['exact_text_duplicate_or_exclusion']+=1;continue
        selected.append(dict(text=text,source='DOTA-ME-CS_reviewed_GPT4o',source_id=row['id'],source_line=index,
            source_group='DOTA-ME-CS/reviewed-scripts',source_file=str(source),source_revision=manifest['revision'],
            scene=row['Scene'],source_tag=row['Tag'],text_origin='GPT-4o generated; manually checked; user approved',
            estimated_seconds=han/4.5+words/2.5,source_seconds=None,language='mixed'))
    with mp.get_context('spawn').Pool(16,initializer=init_worker) as workers:
        for row,reason in workers.imap(process,selected,chunksize=16):
            if reason:counts[reason]+=1;continue
            key=norm(row['text']);phone=hashlib.sha256(bytes(row['ids'])).hexdigest()
            if key in old_text or phone in old_phones:counts['historical_exclusion']+=1;continue
            if key in seen or phone in phones:counts['pool_duplicate']+=1;continue
            rid=-int(hashlib.sha256(('DOTA-ME-CS:'+row['source_id']).encode()).hexdigest()[:15],16)-1
            estimate=row['estimated_seconds'];row.update(speaker=1,split='train',duration=None,sample_rate=24000,text_key=phone)
            cur=db.execute('''insert or ignore into pool(id,language,split,source_group,length_bin,norm,phone_key,payload,estimated_seconds,source_seconds,rank_key)
                values(?,?,?,?,?,?,?,?,?,?,?)''',(rid,'mixed','train',row['source_group'],
                'short' if estimate<7 else 'medium' if estimate<12 else 'long',key,phone,json.dumps(row,ensure_ascii=False),estimate,None,
                hashlib.sha256(('20260913:'+str(rid)).encode()).hexdigest()))
            if cur.rowcount:
                seen.add(key);phones.add(phone);counts['imported_train']+=1;seconds+=estimate
            if counts['imported_train']%500==0:print(json.dumps(dict(counts=dict(counts),estimated_hours=seconds/3600)),flush=True)
    result=dict(status='imported',counts=dict(counts),estimated_text_hours=seconds/3600,source_audio_hours=None,
        policy='User explicitly accepts externally GPT-4o generated, manually reviewed corpus; no new LLM generation.',
        split_policy='train only; existing validation/test text kept independent',audio_downloaded=False,
        source_sha256=manifest['sha256'],near_duplicate_screening='before synthesis admission',elapsed_seconds=time.time()-started)
    atomic_json(report,result);manifest['pool_status']='imported';manifest['user_approval']='GPT-4o origin accepted on 2026-09-13'
    atomic_json(folder/'download_manifest.json',manifest);print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
