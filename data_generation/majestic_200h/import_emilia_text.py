"""Pretokenize and deduplicate real Emilia2 transcripts into the expandable pool."""
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from common import ROOT,REPO,atomic_json,connection,norm
from prepare_corpus import eligibility


def init_worker():
    sys.path.insert(0,str(REPO))
    import torch
    torch.set_num_threads(1)
    global encode
    from lits.text import text_to_sequence_with_tones
    encode=text_to_sequence_with_tones
    encode('你好，Hello world.',['en_zh_dict_mixed_rhyme_body_tone_cleaners'],prepend_sil=True)


def process(row):
    try:
        ids,tones,phones=encode(row['text'],['en_zh_dict_mixed_rhyme_body_tone_cleaners'],prepend_sil=True)
        reason=eligibility(dict(text=row['text'],ids=ids))
        if reason:return None,reason
        if len(ids)!=len(tones) or not all(0<=x<173 for x in ids):return None,'frontend_shape'
        row.update(ids=ids,tones=tones,phonemes=phones)
        return row,None
    except Exception as exc:return None,type(exc).__name__+':'+str(exc)[:100]


def main():
    folder=ROOT/'text/external/Emilia2';source=folder/'mixed_candidates.jsonl';report_path=folder/'import_report.json'
    if report_path.exists():print(report_path.read_text());return
    db=connection();db.execute('attach database ? as excluded',(str(ROOT/'text/exclusions.sqlite'),))
    old_text={r[0] for r in db.execute('select norm from excluded.old')};old_phones={r[0] for r in db.execute('select key from excluded.phones')}
    pool_text={r[0] for r in db.execute('select norm from pool')};pool_phones={r[0] for r in db.execute('select phone_key from pool')}
    counts=Counter();estimates=Counter();original=Counter();started=time.time();batch=[]
    def insert_batch():
        if not batch:return
        db.execute('begin immediate')
        try:
            db.executemany('''insert or ignore into pool(id,language,split,source_group,length_bin,norm,phone_key,payload,estimated_seconds,source_seconds,rank_key)
                values(?,?,?,?,?,?,?,?,?,?,?)''',batch);db.execute('commit');batch.clear()
        except BaseException:db.execute('rollback');raise
    def rows():
        for line in source.open():
            row=json.loads(line)
            if norm(row['text']) not in pool_text:yield row
    with mp.get_context('spawn').Pool(int(os.environ.get('TEXT_IMPORT_WORKERS','32')),initializer=init_worker) as workers:
        for index,(row,reason) in enumerate(workers.imap(process,rows(),chunksize=32),1):
            if reason:counts[reason]+=1;continue
            key=norm(row['text']);phones=hashlib.sha256(bytes(row['ids'])).hexdigest()
            if key in old_text or phones in old_phones:counts['historical_exclusion']+=1;continue
            if key in pool_text or phones in pool_phones:counts['pool_duplicate']+=1;continue
            # Keep external IDs negative; existing foundation IDs are positive.
            rid=-int(hashlib.sha256(('Emilia2:'+row['source_id']).encode()).hexdigest()[:15],16)-1
            group=row['source_group'];bucket=int(hashlib.sha256(group.encode()).hexdigest()[:8],16)%1000
            split='val' if bucket<20 else 'test' if bucket<40 else 'train';estimate=row['estimated_seconds']
            row.update(speaker=1,split=split,duration=row['source_seconds'],sample_rate=24000,text_key=phones)
            batch.append((rid,'mixed',split,group,'short' if estimate<7 else 'medium' if estimate<12 else 'long',key,phones,
                json.dumps(row,ensure_ascii=False),estimate,row['source_seconds'],hashlib.sha256(('20260913:'+str(rid)).encode()).hexdigest()))
            pool_text.add(key);pool_phones.add(phones);counts['imported_'+split]+=1;estimates[split]+=estimate;original[split]+=row['source_seconds']
            if len(batch)>=128:insert_batch()
            if index%2000==0:print(json.dumps(dict(processed=index,counts=dict(counts),estimated_train_hours=estimates['train']/3600)),flush=True)
    insert_batch()
    cumulative={r[0]:dict(texts=r[1],estimated_hours=r[2]/3600,source_audio_hours=r[3]/3600)
        for r in db.execute("select split,count(*),sum(estimated_seconds),sum(source_seconds) from pool where source_group like 'Emilia2/%' group by split")}
    report=dict(status='imported',counts=dict(counts),cumulative=cumulative,estimated_hours={k:v/3600 for k,v in estimates.items()},
        source_audio_hours={k:v/3600 for k,v in original.items()},audio_downloaded=False,source=str(source),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),near_duplicate_screening='at synthesis admission',
        completed_at_unix=time.time(),elapsed_seconds=time.time()-started)
    atomic_json(report_path,report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
