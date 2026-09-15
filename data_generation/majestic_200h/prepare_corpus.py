"""Inventory intact foundation TRAIN utterances; no text generation or concatenation."""
import hashlib
import json
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path
from common import ROOT,ASSETS,atomic_json,config,connection,initialize,norm


def classify(text):
    han=re.findall('[\u4e00-\u9fff]',text);words=re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?",text)
    lang='mixed' if han and words else 'zh' if han else 'en'
    return lang,len(han),len(words),sum(map(len,words))


def eligibility(r):
    t=r['text'].strip();lang,han,words,letters=classify(t)
    if not t or '|' in t or '\n' in t:return 'format'
    if re.search(r'https?://|www\.|@|[<>\[\]{}]',t):return 'nonspoken_markup'
    if not re.search(r'[。！？.!?][”"\u2019\']?$',t):return 'unfinished_punctuation'
    if not 18<=len(r['ids'])<=340 or 2 in r['ids']:return 'token_length_or_unknown'
    if re.search(r'(.{4,12})\1\1',norm(t)):return 'repetition'
    if len(re.findall('[呃嗯]',t))>2:return 'disfluency'
    if lang=='zh' and not 12<=han<=95:return 'zh_length'
    if lang=='en' and not 8<=words<=55:return 'en_length'
    if lang=='mixed' and not (han>=5 and words>=1 and letters>=3 and han<=90 and words<=45):return 'mixed_content'
    estimate=han/4.5+words/2.5
    if not 3<=estimate<=22:return 'estimated_length'
    return None


def source_group(r):
    p=Path(r['audio'])
    return 'HiFiTTS/'+p.parent.name if r['source']=='HiFiTTS' else 'Premium/'+p.stem.split('_S')[0]


def ensure_english_heldouts(db):
    """Small source-group inventories can leave a hash-defined split empty."""
    moves=[]
    for split in ('val','test'):
        if db.execute("select count(*) from pool where language='en' and split=?",(split,)).fetchone()[0]:
            continue
        groups=db.execute("""select source_group,count(*) n,sum(estimated_seconds) seconds
            from pool where split='train' group by source_group
            having min(language)='en' and max(language)='en' and count(*)>=300
            and sum(estimated_seconds)>=3600""").fetchall()
        if not groups:
            raise RuntimeError('No intact English source group available for '+split)
        group=min(groups,key=lambda r:hashlib.sha256(('20260913:'+split+':'+r[0]).encode()).hexdigest())
        db.execute('update pool set split=? where source_group=?',(split,group[0]))
        moves.append(dict(source_group=group[0],split=split,texts=group[1]))
    return moves


def main():
    initialize();db=connection()
    if (ROOT/'text/corpus_inventory.json').exists():return
    excluded=sqlite3.connect('file:'+str(ROOT/'text/exclusions.sqlite')+'?mode=ro',uri=True)
    old_text={r[0] for r in excluded.execute('select norm from old')}
    old_phones={r[0] for r in excluded.execute('select key from phones')};excluded.close()
    assert json.loads((ROOT/'text/exclusions_complete.json').read_text())['policy_version']==2
    db.executescript('''CREATE TABLE IF NOT EXISTS pool(
      id INTEGER PRIMARY KEY,language TEXT,split TEXT,source_group TEXT,length_bin TEXT,
      norm TEXT UNIQUE,phone_key TEXT UNIQUE,payload TEXT,estimated_seconds REAL,
      source_seconds REAL,rank_key TEXT,state TEXT DEFAULT 'available');
      CREATE INDEX IF NOT EXISTS pool_select ON pool(state,split,language,length_bin,rank_key);''')
    db.execute('delete from pool')
    source=sqlite3.connect('file:'+str(ASSETS/'data_24k/foundation/dataset.sqlite')+'?mode=ro',uri=True)
    counts=Counter();seconds=Counter();rejected=Counter();groups=Counter()
    db.execute('begin')
    for index,(rid,key,payload) in enumerate(source.execute("select id,text_key,payload from samples where split='train'")):
        r=json.loads(payload);language,han,words,letters=classify(r['text']);reason=eligibility(r)
        if not reason and (key in old_phones or norm(r['text']) in old_text):reason='old_text_or_heldout_overlap'
        if reason:rejected[(language,reason)]+=1;continue
        group=source_group(r);bucket=int(hashlib.sha256(group.encode()).hexdigest()[:8],16)%1000
        split='val' if bucket<20 else 'test' if bucket<40 else 'train'
        estimate=han/4.5+words/2.5;length='short' if estimate<7 else 'medium' if estimate<12 else 'long'
        rank=hashlib.sha256(('20260913:'+str(rid)).encode()).hexdigest()
        cur=db.execute('insert or ignore into pool(id,language,split,source_group,length_bin,norm,phone_key,payload,estimated_seconds,source_seconds,rank_key) values(?,?,?,?,?,?,?,?,?,?,?)',
                       (rid,language,split,group,length,norm(r['text']),key,payload,estimate,r['duration'],rank))
        if cur.rowcount:counts[(split,language)]+=1;seconds[(split,language)]+=r['duration'];groups[group]+=1
        else:rejected[(language,'duplicate_normalized_text_or_phones')]+=1
        if index%10000==0:db.execute('commit');print('scanned',index,flush=True);db.execute('begin')
    db.execute('commit');source.close()
    split_repairs=ensure_english_heldouts(db)
    counts=Counter();seconds=Counter()
    for split,language,n,duration in db.execute('select split,language,count(*),sum(source_seconds) from pool group by split,language'):
        counts[(split,language)]=n;seconds[(split,language)]=duration
    inventory={'status':'prepared','source':'foundation TRAIN only','text_policy':'intact real text; no rewriting, concatenation, or LLM text',
       'eligible':{s:{l:{'texts':counts[(s,l)],'original_audio_hours':seconds[(s,l)]/3600,
                         'absolute_upper_hours_at_20s':counts[(s,l)]*20/3600} for l in ('zh','en','mixed')} for s in ('train','val','test')},
       'rejections':{str(k):v for k,v in rejected.items()},'source_groups':len(groups),'split_repairs':split_repairs,'dedup_status':'exact exclusion complete; near-match checked before synthesis admission',
       'created_at_unix':time.time()}
    atomic_json(ROOT/'text/corpus_inventory.json',inventory);print(json.dumps(inventory,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
