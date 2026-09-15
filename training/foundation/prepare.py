"""Prepare a read-only, pretokenized index of HiFiTTS and Premium."""
import concurrent.futures as cf
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sqlite3
import time

ROOT = Path('/ai_sds_wuzz/DATA_TTS')
OUT = Path('/119010446/tts-assets/data_24k/foundation')
CLEANERS = ['en_zh_dict_mixed_rhyme_body_tone_cleaners']

def init_worker():
    import torch
    torch.set_num_threads(1)
    global sf, encode
    import soundfile as sf
    from lits.text import text_to_sequence_with_tones
    encode = text_to_sequence_with_tones
    encode('你好，Hello world.', CLEANERS, prepend_sil=True)

def analyze(r):
    try:
        if r.get('textfile'):
            r['text'] = Path(r.pop('textfile')).read_text().splitlines()[0].split('\t', 1)[1].strip()
        text = r['text'].strip().replace('|', ' ')
        info = sf.info(r['audio'])
        dur = info.frames / info.samplerate
        if not 0.5 <= dur <= 20.0:
            return {'excluded': 'duration_outside_0.5_20s', 'audio': r['audio']}
        if info.channels != 1 or info.samplerate not in (16000, 22050, 24000, 44100, 48000):
            raise ValueError(f'audio format {info}')
        ids, tones, phonemes = encode(text, CLEANERS, prepend_sil=True)
        if not ids or 2 in ids or not all(0 <= x < 173 for x in ids):
            return {'excluded': 'empty_unknown_token', 'audio': r['audio'], 'text': text}
        frames = int(round(dur * 24000)) // 384
        if len(ids) > frames:
            return {'excluded': 'text_longer_than_mel', 'audio': r['audio']}
        key = hashlib.sha256(bytes(ids)).hexdigest()
        return {'audio': r['audio'], 'text': text, 'ids': ids, 'tones': tones,
                'phonemes': phonemes, 'duration': dur, 'sample_rate': info.samplerate,
                'split': r['split'], 'source': r['source'], 'key': key, 'speaker': 0}
    except Exception as e:
        return {'excluded': type(e).__name__, 'reason': str(e), 'audio': r['audio']}

def source_rows():
    hifi = ROOT / 'HiFiTTS/hi_fi_tts_v0'
    for manifest in sorted(hifi.glob('*manifest*.json')):
        split = manifest.stem.rsplit('_', 1)[1]
        split = {'dev': 'val'}.get(split, split)
        with manifest.open() as stream:
            for line in stream:
                row = json.loads(line)
                yield {'audio': str(hifi / row['audio_filepath']), 'text': row.get('text_normalized', row['text']),
                       'source': 'HiFiTTS', 'split': split}
    for shard in sorted((ROOT / 'WenetSpeech4TTS/Premium').glob('WenetSpeech4TTS_Premium_*')):
        if not shard.is_dir():
            continue
        for txt in sorted((shard / 'txts').glob('*.txt')):
            group = txt.stem.rsplit('_S', 1)[0]
            bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 1000
            split = 'val' if bucket < 5 else 'test' if bucket < 10 else 'train'
            yield {'audio': str(shard / 'wavs' / (txt.stem + '.wav')), 'textfile': str(txt),
                   'source': 'Premium', 'split': split}

def main():
    from collections import Counter
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / 'dataset.sqlite'
    if target.exists():
        print('Already prepared:', target, flush=True)
        return
    path = OUT / 'dataset.building.sqlite'
    if path.exists():
        raise RuntimeError('Incomplete build exists; inspect before rerunning')
    db = sqlite3.connect(path)
    db.execute('pragma journal_mode=OFF')
    db.execute('pragma synchronous=OFF')
    db.execute('create table samples (id integer primary key, split text, source text, duration real, text_key text, payload text)')
    started = time.time()
    counter = Counter()
    ctx = mp.get_context('spawn')
    with ctx.Pool(32, initializer=init_worker) as pool, (OUT / 'exclusions.jsonl').open('w') as errors:
        for n, r in enumerate(pool.imap_unordered(analyze, source_rows(), chunksize=64), 1):
            if 'excluded' in r:
                counter[r['excluded']] += 1
                errors.write(json.dumps(r, ensure_ascii=False) + '\n')
            else:
                db.execute('insert into samples(split,source,duration,text_key,payload) values(?,?,?,?,?)',
                           (r['split'], r['source'], r['duration'], r['key'], json.dumps(r, ensure_ascii=False)))
            if n % 10000 == 0:
                db.commit()
                progress = {'processed': n, 'excluded': dict(counter), 'elapsed_s': time.time()-started}
                (OUT / 'progress.json').write_text(json.dumps(progress))
                print(json.dumps(progress), flush=True)
    db.execute('create index split_idx on samples(split, source)')
    db.execute('create index key_idx on samples(text_key)')
    # Keep the established target-voice validation/test and fixed external evaluation unseen.
    heldout = set()
    prior = Path('/119010446/tts-assets/data_24k/ljs_majestic')
    from lits.text import text_to_sequence_with_tones
    for name in ['val.txt', 'test.txt']:
        for line in (prior / name).read_text().splitlines():
            ids, _, _ = text_to_sequence_with_tones(line.split('|')[-1], CLEANERS, prepend_sil=True)
            heldout.add(hashlib.sha256(bytes(ids)).hexdigest())
    for line in (prior / 'eval_manifest.jsonl').read_text().splitlines():
        r = json.loads(line)
        ids, _, _ = text_to_sequence_with_tones(r['text'], CLEANERS, prepend_sil=True)
        heldout.add(hashlib.sha256(bytes(ids)).hexdigest())
    db.execute('create temp table heldout(key text primary key)')
    db.executemany('insert or ignore into heldout values(?)', [(x,) for x in heldout])
    db.execute("insert or ignore into heldout select text_key from samples where split!='train'")
    deleted = db.execute("delete from samples where split='train' and text_key in (select key from heldout)").rowcount
    # Remove validation text that is also in the test set.
    val_deleted = db.execute("delete from samples where split='val' and text_key in (select text_key from samples where split='test')").rowcount
    db.commit()
    summary = {'status': 'prepared', 'processed': n, 'excluded': dict(counter), 'train_heldout_overlap_removed': deleted,
               'val_test_overlap_removed': val_deleted, 'train_heldout_phoneme_overlap': 0,
               'groups': [dict(zip(['split','source','rows','hours'], r)) for r in db.execute('select split,source,count(*),sum(duration)/3600 from samples group by split,source')],
               'active_speaker_id': 0, 'reserved_stage2_speaker_id': 1, 'sample_rate': 24000,
               'audio_conversion': 'on read: mono float32, soxr HQ resample to 24000 Hz',
               'cleaners': CLEANERS, 'seed': 20260911, 'elapsed_s': time.time()-started}
    db.close()
    path.replace(target)
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
