"""Fail closed on all quotas/QC, then freeze the new single-speaker recipe."""
import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from common import ROOT,ASSETS,REPO,atomic_json,config,connection,goals,totals
from run_workers import judge,quality_reference
from select_training_quota import select as select_quota


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def audit():
    import soundfile as sf
    cfg=config();db=connection();accepted=totals(db);missing=[]
    for (split,lang),target in goals(cfg).items():
        actual=accepted.get((split,lang),{}).get('seconds',0)
        if actual<target:missing.append(dict(split=split,language=lang,seconds=actual,target=target))
    if missing:
        result=dict(status='not_ready',missing_quotas=missing,training_started=False)
        atomic_json(ROOT/'reports/training_readiness.json',result);return None,result
    rows=db.execute('''select t.id,t.language,t.split,t.payload,t.norm,t.phone_key,t.length_bin,c.*
        from texts t join candidates c on c.id=t.accepted_candidate order by t.id''').fetchall()
    seen_text=set();seen_phone=set();groups={};paths=set();splits={s:[] for s in ('train','val','test')}
    reference_hashes={p:digest(p) for p in {cfg['reference_audio'],*[quality_reference(cfg,l) for l in ('zh','en','mixed')]}}
    for r in rows:
        r=dict(r);payload=json.loads(r['payload']);text_id=r['text_id']
        metrics=json.loads(r['metrics'])
        assert r['state']=='passed' and not judge(r,metrics,cfg['thresholds'])
        reference=metrics.get('speaker_reference_audio',cfg['reference_audio'])
        assert reference in {cfg['reference_audio'],quality_reference(cfg,r['language'])}
        if 'speaker_reference_sha256' in metrics:assert metrics['speaker_reference_sha256']==reference_hashes[reference]
        assert r['norm'] not in seen_text and r['phone_key'] not in seen_phone
        seen_text.add(r['norm']);seen_phone.add(r['phone_key'])
        group=db.execute('select source_group from pool where id=?',(text_id,)).fetchone()[0]
        assert groups.setdefault(group,r['split'])==r['split']
        audio=ROOT/'accepted/wavs_24k'/f'{text_id}.wav';assert str(audio) not in paths;paths.add(str(audio))
        assert cfg['min_audio_seconds']<=r['duration']<=cfg['max_audio_seconds']
        assert 2 not in payload['ids'] and hashlib.sha256(bytes(payload['ids'])).hexdigest()==r['phone_key']
        record=dict(audio=str(audio),speaker=cfg['training'].get('speaker_id',1),text=payload['text'],language={'zh':'Chinese','en':'English','mixed':'Mixed'}[r['language']],
            text_key=payload['ids'],ids=payload['ids'],tones=payload['tones'],phonemes=payload['phonemes'],duration=r['duration'],
            split=r['split'],source_group=group,source=payload.get('source'),source_length_bin=r['length_bin'],synthetic=True)
        splits[r['split']].append(record)
    audio_validation_rows=[r for rs in splits.values() for r in rs]
    quota_selection={};selected_train=[]
    for lang,label in [('zh','Chinese'),('en','English'),('mixed','Mixed')]:
        selected,selection=select_quota([r for r in splits['train'] if r['language']==label],cfg['targets_train_hours'][lang]*3600,cfg['seed'])
        selected_train.extend(selected);quota_selection[lang]=selection
    splits['train']=selected_train
    atomic_json(ROOT/'reports/training_quota_selection.json',quota_selection)
    def check_audio(r):
        info=sf.info(r['audio']);assert info.samplerate==24000 and info.channels==1 and info.subtype=='PCM_16'
        assert abs(info.duration-r['duration'])<1/24000
        assert len(r['text_key'])<=info.frames//384
        return dict(key=json.dumps([r['audio'],None,None],separators=(',',':')),audio_frames=info.frames)
    with ThreadPoolExecutor(max_workers=32) as workers:
        all_units=list(workers.map(check_audio,audio_validation_rows))
    selected_audio={r['audio'] for rs in splits.values() for r in rs}
    units=[u for r,u in zip(audio_validation_rows,all_units) if r['audio'] in selected_audio]
    db.execute('attach database ? as excluded',(str(ROOT/'text/exclusions.sqlite'),))
    assert db.execute('''select count(*) from texts t join excluded.old o on t.norm=o.norm where t.accepted_candidate is not null''').fetchone()[0]==0
    assert db.execute('''select count(*) from texts t join excluded.phones p on t.phone_key=p.key where t.accepted_candidate is not null''').fetchone()[0]==0
    result=dict(status='passed',accepted_hours={s:{l:accepted[(s,l)]['seconds']/3600 for l in ('zh','en','mixed')} for s in splits},
        selected_train_hours={lang:selection['selected_hours'] for lang,selection in quota_selection.items()},training_quota_selection=quota_selection,
        unique_texts=len(seen_text),unique_audio=len(paths),cross_split_group_overlap=0,historical_text_phone_overlap=0,
        quality_thresholds=cfg['thresholds'],audio_format_and_mas_length='passed',completed_at_unix=time.time())
    atomic_json(ROOT/'reports/final_dataset_audit.json',result)
    return (splits,units),result


def prepare():
    mode = config()['training'].get('mode')
    if mode not in ('scratch', 'backbone_init'):
        raise ValueError(f'Unsupported retired training mode: {mode!r}')
    from training.majestic_scratch.prepare import prepare as prepare_joint
    return prepare_joint()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--audit-only',action='store_true');args=parser.parse_args()
    if args.audit_only:
        _,report=audit();print(json.dumps(report));raise SystemExit(0 if report['status']=='passed' else 2)
    raise SystemExit(prepare())
