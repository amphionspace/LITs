"""Shared persistent state for new-text, quota-driven MajesticVoice synthesis."""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
import unicodedata

ROOT=Path(os.environ.get('MAJESTIC_200H_ROOT','/ai_sds_wuzz/DATA_TTS/MajesticVoice_200h_20260913'))
ASSETS=Path('/119010446/tts-assets')
REPO=Path(__file__).resolve().parents[2]
CODE=Path(__file__).resolve().parent
LANGS=('zh','en','mixed')
DOMAINS=('日常生活','家庭相处','朋友交流','工作协作','学习方法','校园生活','阅读写作','科技产品',
         '软件与互联网','科学常识','自然观察','动物植物','城市交通','旅行体验','餐饮烹饪','购物服务',
         '运动休闲','音乐艺术','影视文化','历史故事','空间与建筑','手工修理','时间安排','情绪与感受')
ACTS=('陈述说明','叙述经历','提问求助','建议请求','解释比较','表达观点','回应对话','描写场景')
KINDS=('轻松口语','清楚的讲解','生活叙事','礼貌沟通','细致描述','简短讨论')
LENGTHS=('short','medium','long')
SWITCHES=('chinese_matrix_phrase','chinese_matrix_clause','english_matrix','alternating')


def atomic_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def config():return json.loads((ROOT/'config.json').read_text())


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def norm(text):
    return ''.join(c for c in unicodedata.normalize('NFKC',text).lower() if unicodedata.category(c)[0] in ('L','N'))


def phone_key(ids):return hashlib.sha256(bytes(ids)).hexdigest()


def connection():
    db=sqlite3.connect(ROOT/'pipeline.sqlite',timeout=120,isolation_level=None)
    db.row_factory=sqlite3.Row;db.execute('pragma busy_timeout=120000');db.execute('pragma foreign_keys=ON')
    return db


def initialize():
    db=connection();db.execute('pragma journal_mode=WAL');db.execute('pragma synchronous=NORMAL')
    db.executescript('''
    CREATE TABLE IF NOT EXISTS texts(
      id INTEGER PRIMARY KEY,language TEXT NOT NULL,split TEXT NOT NULL,domain TEXT NOT NULL,
      act TEXT,style TEXT,length_bin TEXT,switch_style TEXT,scene TEXT,batch_key TEXT,
      text TEXT NOT NULL,norm TEXT NOT NULL UNIQUE,phone_key TEXT NOT NULL UNIQUE,payload TEXT NOT NULL,
      estimated_seconds REAL NOT NULL,attempts INTEGER NOT NULL DEFAULT 0,
      accepted_candidate INTEGER,created_at REAL NOT NULL);
    CREATE TABLE IF NOT EXISTS candidates(
      id INTEGER PRIMARY KEY,text_id INTEGER NOT NULL REFERENCES texts(id),attempt INTEGER NOT NULL,
      state TEXT NOT NULL DEFAULT 'queued',owner TEXT,lease_until REAL DEFAULT 0,
      audio_24k TEXT,audio_48k TEXT,duration REAL,synthesis TEXT,asr TEXT,metrics TEXT,
      reasons TEXT,score REAL,created_at REAL NOT NULL,updated_at REAL NOT NULL,
      UNIQUE(text_id,attempt));
    CREATE INDEX IF NOT EXISTS candidate_state ON candidates(state,id);
    CREATE INDEX IF NOT EXISTS candidate_text ON candidates(text_id);
    CREATE INDEX IF NOT EXISTS text_quota ON texts(split,language,domain,accepted_candidate);
    CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY,time REAL,kind TEXT,payload TEXT);
    CREATE TABLE IF NOT EXISTS text_rejections(id INTEGER PRIMARY KEY,time REAL,language TEXT,reason TEXT,payload TEXT);
    CREATE TABLE IF NOT EXISTS batches(batch_key TEXT PRIMARY KEY,language TEXT,split TEXT,domain TEXT,metadata TEXT,created_at REAL);
    CREATE VIRTUAL TABLE IF NOT EXISTS new_fts USING fts5(norm,tokenize='trigram');
    ''');db.close()


def event(kind,payload):
    db=connection();db.execute('insert into events(time,kind,payload) values(?,?,?)',(time.time(),kind,json.dumps(payload,ensure_ascii=False)));db.close()


def goals(cfg=None):
    cfg=cfg or config()
    return {(s,l):3600*(cfg['targets_train_hours'][l] if s=='train' else cfg['targets_heldout_hours_per_language'][s])
            for s in ('train','val','test') for l in LANGS}


def totals(db):
    return {(r['split'],r['language']):dict(seconds=r['seconds'],count=r['count']) for r in db.execute('''
      SELECT t.split,t.language,sum(c.duration) seconds,count(*) count FROM texts t
      JOIN candidates c ON c.id=t.accepted_candidate GROUP BY t.split,t.language''')}


def claim(db,from_state,to_state,owner,limit=1,lease_seconds=900,language=None):
    db.execute('begin immediate')
    try:
        rows=db.execute('''SELECT c.*,t.language,t.split,t.text,t.payload,t.domain,t.length_bin
             FROM candidates c JOIN texts t ON t.id=c.text_id
             WHERE c.state=? AND (? IS NULL OR t.language=?) ORDER BY c.id LIMIT ?''',(from_state,language,language,limit)).fetchall()
        now=time.time()
        db.executemany('update candidates set state=?,owner=?,lease_until=?,updated_at=? where id=?',
                       [(to_state,owner,now+lease_seconds,now,r['id']) for r in rows])
        db.execute('commit');return [dict(r) for r in rows]
    except BaseException:db.execute('rollback');raise


def reject(db,row,reasons,**fields):
    fields.update(state='rejected',reasons=json.dumps(reasons,ensure_ascii=False),lease_until=0,updated_at=time.time())
    db.execute('update candidates set '+','.join(k+'=?' for k in fields)+' where id=?',list(fields.values())+[row['id']])


def log_rejection(db,language,reason,payload):
    db.execute('insert into text_rejections(time,language,reason,payload) values(?,?,?,?)',
               (time.time(),language,reason,json.dumps(payload,ensure_ascii=False)))


def near_match(db,table,text):
    """FTS substring retrieval plus normalized edit/containment checks; not semantic identity."""
    from rapidfuzz.fuzz import ratio
    key=norm(text)
    if len(key)<12:return None
    width=max(4,min(10,len(key)//10))
    positions=sorted(set([0,max(0,len(key)-width)]+list(range(0,len(key)-width+1,max(1,len(key)//12)))))
    chunks=list(dict.fromkeys(key[i:i+width] for i in positions))
    query=' OR '.join('"'+p.replace('"','""')+'"' for p in chunks)
    found=db.execute(f'SELECT rowid,norm FROM {table} WHERE {table} MATCH ? ORDER BY rank LIMIT 96',(query,)).fetchall()
    for row in found:
        other=row['norm']
        similarity=ratio(key,other)
        contained=min(len(key),len(other))>=20 and min(len(key),len(other))/max(len(key),len(other))>=.65 and (key in other or other in key)
        if similarity>=88 or contained:return dict(rowid=row['rowid'],similarity=similarity,contained=contained)
    return None
