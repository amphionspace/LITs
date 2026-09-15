"""Fail closed on all quotas/QC, then freeze the new single-speaker recipe."""
import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from collections import Counter
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
    if config()['training'].get('mode') in ('scratch','backbone_init'):
        from training.majestic_scratch.prepare import prepare as prepare_scratch
        return prepare_scratch()
    content,result=audit()
    if content is None:print(json.dumps(result));return 2
    import torch
    cfg=config();training=cfg['training'];run=Path(training['run_dir']);source=Path(training['source_checkpoint'])
    if (run/'plan.json').exists():return 0
    assert not run.exists(),'Partial training preparation exists; inspect before retry'
    ckpt=torch.load(source,map_location='cpu',weights_only=False)
    expected='f3cfa4a9eaa7c0036ce78bb3b5857a55ebcccf4b40943062c1191e7d28b31a35'
    assert digest(source)==expected and ckpt['global_step']==197000
    h=ckpt['hyper_parameters'];assert h['n_vocab']==173 and h['n_feats']==100 and h['n_spks']==2
    state={k:v.clone() for k,v in ckpt['state_dict'].items()};assert all(torch.isfinite(v).all() for v in state.values())
    assert state['spk_emb.weight'].shape==(2,64);state['spk_emb.weight'][1].copy_(state['spk_emb.weight'][0])
    stage=run.with_name('.'+run.name+'.preparing');stage.mkdir(exist_ok=False);data=stage/'data';data.mkdir()
    splits,units=content
    def write_jsonl(path,rows):path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    for split,rows in splits.items():
        write_jsonl(data/f'{split}.jsonl',rows)
        (data/f'{split}.txt').write_text(''.join(f'{r["audio"]}|1|{r["text"]}\n' for r in rows))
    write_jsonl(data/'text_preflight.jsonl',[dict(text=r['text'],tokens=r['ids'],tones=r['tones'],phonemes=r['phonemes']) for rs in splits.values() for r in rs])
    write_jsonl(data/'mel_statistics_units.jsonl',units)
    original=ASSETS/'data_24k/ljs_majestic'
    evaluation=[r for r in map(json.loads,(original/'eval_manifest.jsonl').read_text().splitlines()) if r['speaker']==1]
    assert not {tuple(r['text_key']) for r in splits['train']} & {tuple(r['token_ids']) for r in evaluation}
    write_jsonl(data/'eval_manifest.jsonl',evaluation)
    protocol=json.loads((original/'eval_protocol.json').read_text());protocol.update(samples=len(evaluation),groups=dict(Counter(r['group'] for r in evaluation)),manifest_sha256=digest(data/'eval_manifest.jsonl'))
    atomic_json(data/'eval_protocol.json',protocol)
    stats=dict(mel_mean=float(state['mel_mean']),mel_std=float(state['mel_std']))
    atomic_json(data/'mel_statistics.json',dict(**stats,manifest_sha256=digest(data/'train.txt'),sample_rate=24000,n_feats=100,
        normalization_policy='preserve stage-one scale',normalization_source_checkpoint=str(source),mas_text_length_check='passed'))
    atomic_json(data/'training_summary.json',dict(train_rows=len(splits['train']),validation_rows=len(splits['val']),test_rows=len(splits['test']),
        language_counts=dict(Counter(r['language'] for r in splits['train'])),source_dataset_audit=str(ROOT/'reports/final_dataset_audit.json')))
    init={k:ckpt[k] for k in ['pytorch-lightning_version','hyper_parameters']};init.update(state_dict=state,global_step=0,epoch=0)
    torch.save(init,stage/'initialization.ckpt')
    plan=dict(stage=2,recipe=f'majestic_{sum(cfg["targets_train_hours"].values()):g}h_embedding_warmup_then_joint',active_speaker_ids=[1],
        source_checkpoint=str(source),source_global_step=197000,source_sha256=expected,
        initialization_sha256=digest(stage/'initialization.ckpt'),initialization='weights only; fresh Adam',
        speakers={'0':'reserved, unused','1':'MajesticVoice'},speaker_embedding_initialization='row 1 copied from source row 0',
        data_dir=str(run/'data'),data_source=str(ROOT),manifest_hashes={p.name:digest(p) for p in data.iterdir()},
        train_rows=len(splits['train']),validation_rows=len(splits['val']),test_rows=len(splits['test']),
        train_rows_per_speaker={'1':len(splits['train'])},data_statistics=stats,unique_audio_hours=result['selected_train_hours'],
        batch_size_per_gpu=48,devices=[0,1,2,3],effective_batch=192,precision='bf16-mixed',max_steps=training['max_steps'],
        learning_rates=dict(decoder=training['other_joint_lr'],spk_emb=training['embedding_joint_lr'],prior_encoder=training['other_joint_lr'],duration_predictor=training['other_joint_lr']),
        schedule_callback='training.majestic_finetune.callbacks.MajesticSchedule',
        schedule=dict(warmup_steps=50,encoder_update_start_step=training['embedding_only_steps'],encoder_warmup_steps=200,final_lr_ratio=.2,embedding_adaptation_lr=training['embedding_adaptation_lr']),
        validation_every_steps=250,checkpoint_every_steps=250,evaluation_interval_steps=1000,evaluation_groups=protocol['groups'],seed=cfg['seed'])
    atomic_json(stage/'plan.json',plan);shutil.copyfile(ROOT/'reports/final_dataset_audit.json',stage/'dataset_audit.json');stage.rename(run)
    print(json.dumps(dict(status='prepared',run_dir=str(run),train_rows=plan['train_rows'])));return 0


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--audit-only',action='store_true');args=parser.parse_args()
    if args.audit_only:
        _,report=audit();print(json.dumps(report));raise SystemExit(0 if report['status']=='passed' else 2)
    raise SystemExit(prepare())
