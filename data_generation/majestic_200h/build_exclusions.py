"""Index historical text and phonemes; never use these sentences as new requests."""
import json
from pathlib import Path
import sqlite3
import time
from collections import Counter
from common import ROOT,ASSETS,norm,phone_key,atomic_json,initialize


def main():
    initialize();dest=ROOT/'text/exclusions.sqlite'
    if (ROOT/'text/exclusions_complete.json').exists():
        old=json.loads((ROOT/'text/exclusions_complete.json').read_text())
        if old.get('policy_version')==2:return
        dest.rename(ROOT/'text/exclusions_all_foundation_abandoned.sqlite')
        (ROOT/'text/exclusions_complete.json').rename(ROOT/'text/exclusions_all_foundation_abandoned.json')
    if dest.exists():dest.unlink()
    db=sqlite3.connect(dest);db.row_factory=sqlite3.Row
    db.execute('pragma journal_mode=WAL');db.execute('pragma synchronous=NORMAL')
    db.executescript('''CREATE TABLE old(id INTEGER PRIMARY KEY,norm TEXT UNIQUE,text TEXT,source TEXT);
                       CREATE TABLE phones(key TEXT PRIMARY KEY);
                       CREATE VIRTUAL TABLE old_fts USING fts5(norm,tokenize='trigram');''')
    counts=Counter();language=Counter();lengths=Counter();inventory=[]
    def add(text,source,ids=None,key=None):
        n=norm(text)
        if not n:return
        cur=db.execute('insert or ignore into old(norm,text,source) values(?,?,?)',(n,text,source))
        if cur.rowcount:counts[source]+=1
        if ids is not None:key=phone_key(ids)
        if key:db.execute('insert or ignore into phones values(?)',(key,))
    foundation=ASSETS/'data_24k/foundation/dataset.sqlite'
    source=sqlite3.connect('file:'+str(foundation)+'?mode=ro',uri=True)
    for index,(split,kind,key,payload) in enumerate(source.execute("select split,source,text_key,payload from samples where split!='train'")):
        r=json.loads(payload);text=r['text'];add(text,'foundation/'+kind,key=key)
        if index%10000==0:db.commit();print('foundation',index,flush=True)
    source.close();inventory.append(str(foundation))
    p=ASSETS/'data_24k/ljs_majestic'
    paths=[p/f'{split}.jsonl' for split in ('train','val','test')]+[p/'eval_manifest.jsonl']
    old=Path('/ai_sds_wuzz/DATA_TTS/MajesticVoice')
    paths+=sorted(old.glob('**/requests.jsonl'))
    for path in paths:
        if not path.is_file():continue
        inventory.append(str(path))
        for line in path.read_text().splitlines():
            r=json.loads(line);text=r.get('text',r.get('ref_text',''))
            add(text,str(path),r.get('text_key',r.get('token_ids')))
        db.commit()
    metadata=Path('/ai_sds_wuzz/DATA_TTS/LJSpeech/LJSpeech-1.1/metadata.csv')
    if metadata.is_file():
        inventory.append(str(metadata))
        for line in metadata.read_text().splitlines():add(line.split('|',2)[2],'LJSpeech/all_metadata')
    # Historical train packs are additional exclusions, not generation material.
    base=ASSETS/'chenmingjie/lits-tts/output'
    for path in [base/'ljspeech_biaobei_voxcpm2en2756_16k_balanced'/n for n in ('train.txt','val.txt','voxcpm2_biaobei_en_2756.txt')]:
        if path.exists():
            inventory.append(str(path))
            for line in path.read_text().splitlines():add(line.split('|')[-1],str(path))
    db.commit();print('building FTS',flush=True)
    db.execute('insert into old_fts(rowid,norm) select id,norm from old');db.commit()
    result=dict(status='complete',policy_version=2,texts=db.execute('select count(*) from old').fetchone()[0],
                phoneme_keys=db.execute('select count(*) from phones').fetchone()[0],source_counts=dict(counts),
                source_files=inventory,completed_at_unix=time.time(),policy='Foundation training text allowed; old Majestic, historical adaptation text, ALL heldouts/evaluations excluded by exact text/phonemes and near duplicates')
    db.execute('pragma wal_checkpoint(TRUNCATE)');db.close()
    atomic_json(ROOT/'text/exclusions_complete.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
