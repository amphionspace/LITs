"""Trace the existing English frontend's actual dictionary fallback decisions."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sqlite3
import sys

REPO=Path(__file__).resolve().parents[2]
DATA=Path('/119010446/tts-assets/data_24k/foundation')
sys.path[:0]=[str(REPO/'temp_cmu_g2p'),str(REPO)]
from english_frontend import get_default_g2p,preprocess_english_input
from g2p_engine import CMUDictG2P


def main(out):
    rng=random.Random(2026091203)
    samples=[]
    with sqlite3.connect(f'file:{DATA}/dataset.sqlite?mode=ro',uri=True) as db:
        for source in ['HiFiTTS','Premium']:
            ids=[r[0] for r in db.execute("select id from samples where split='train' and source=? order by id",(source,))]
            chosen=rng.sample(ids,1000)
            for start in range(0,len(chosen),400):
                chunk=chosen[start:start+400]
                for sid,payload in db.execute('select id,payload from samples where id in ('+','.join('?'*len(chunk))+')',chunk):
                    r=json.loads(payload);samples.append(dict(id=f'train_{sid}',scope='train_'+source,text=r['text']))
    for line in (DATA/'eval_manifest.jsonl').read_text().splitlines():
        r=json.loads(line);samples.append(dict(id=r['id'],scope=r['group'],text=r['text']))
    engine=get_default_g2p()
    original=CMUDictG2P.lookup_phrase_phonemes
    traces=[]
    def traced(self,word,**kwargs):
        phones,source=original(self,word,**kwargs)
        traces.append(dict(word=word,source=source,phones=phones))
        return phones,source
    CMUDictG2P.lookup_phrase_phonemes=traced
    summary={};details=[]
    try:
        for row in samples:
            traces.clear()
            processed=preprocess_english_input(row['text'],engine)
            stats=summary.setdefault(row['scope'],dict(texts=0,texts_with_spelling=0,texts_with_lowercase_spelling=0,
                phrase_lookups=0,spelling_lookups=0,lowercase_spelling_lookups=0,lookup_sources=Counter(),spelled_words=Counter()))
            stats['texts']+=1;stats['phrase_lookups']+=len(traces)
            stats['lookup_sources'].update(r['source'] for r in traces)
            spelled=[r for r in traces if r['source']=='spell']
            ordinary=[r for r in spelled if len(r['word'])>1 and any(c.islower() for c in r['word'])]
            stats['texts_with_spelling']+=bool(spelled);stats['texts_with_lowercase_spelling']+=bool(ordinary)
            stats['spelling_lookups']+=len(spelled);stats['lowercase_spelling_lookups']+=len(ordinary)
            stats['spelled_words'].update(r['word'] for r in spelled)
            if spelled:details.append(dict(row,spelled=spelled,processed=processed))
    finally:
        CMUDictG2P.lookup_phrase_phonemes=original
    report=dict(sample_seed=2026091203,train_sample_count=2000,full_eval_count=450,summary=summary,
        interpretation='Spelling is intentional for some acronyms. Lowercase-containing multi-letter words are review candidates, not automatically errors. Lookup traces omit isolated letters and tokens passed through as existing phonemes.',
        example_paler=engine.lookup_phrase_phonemes('paler'))
    (out/'frontend_fallback_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (out/'frontend_fallback_details.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in details))
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    main(p.parse_args().output)
