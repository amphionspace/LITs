"""Reproducible proportional selection of accepted audio into per-language quotas."""
from collections import defaultdict, Counter
import hashlib
import re


def stratum(row):
    words=len(re.findall(r"[A-Za-z]+(?:['’-][A-Za-z]+)*",row['text']))
    english_bin=('1-2' if words<=2 else '3-5' if words<=5 else '6-10' if words<=10 else '11+') if row['language']=='Mixed' else 'na'
    return (row.get('source','unknown'),row.get('source_length_bin','unknown'),english_bin)


def select(rows, target_seconds, seed):
    if sum(r['duration'] for r in rows)+1e-6<target_seconds:raise ValueError('Insufficient accepted audio')
    buckets=defaultdict(list)
    for row in rows:buckets[stratum(row)].append(row)
    ordered=[]
    for key,bucket in buckets.items():
        total=sum(r['duration'] for r in bucket);elapsed=0.
        bucket.sort(key=lambda r:hashlib.sha256(f'{seed}:{r["audio"]}'.encode()).digest())
        for row in bucket:
            ordered.append(((elapsed+row['duration']/2)/total,str(row['audio']),row))
            elapsed+=row['duration']
    chosen=[];seconds=0.
    for _,_,row in sorted(ordered,key=lambda x:(x[0],x[1])):
        if seconds>=target_seconds:break
        chosen.append(row);seconds+=row['duration']
    before=Counter();after=Counter()
    for row in rows:before[stratum(row)]+=row['duration']
    for row in chosen:after[stratum(row)]+=row['duration']
    report=dict(available_texts=len(rows),available_hours=sum(before.values())/3600,selected_texts=len(chosen),selected_hours=seconds/3600,
        target_hours=target_seconds/3600,seed=seed,policy='Duration-proportional strata by source/text-length/English-word count; seeded within-stratum ordering; at most one-record global overshoot',
        strata=[dict(source=k[0],text_length_bin=k[1],english_word_bin=k[2],available_seconds=before[k],selected_seconds=after[k]) for k in sorted(before)])
    return chosen,report
