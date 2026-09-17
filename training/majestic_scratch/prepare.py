"""Freeze audited synthetic/original data and estimate combined train-only Mel statistics."""
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

from training.majestic_scratch.config import REPO, make_model, parameter_digest, backbone_metadata, load_backbone, resolve_budget


def mel_statistics(rows, seed):
    import numpy as np
    import soundfile as sf
    import torch
    from lits.utils.audio import mel_spectrogram
    torch.set_num_threads(4)
    rng = np.random.default_rng(seed)
    # Same population as natural utterance sampling; all speakers/languages share
    # one acoustic scale. The complete selected list is recorded for reproducibility.
    selected = [rows[int(i)] for i in rng.permutation(len(rows))[:4096]]
    total = square = 0.; elements = 0
    with torch.inference_mode():
        for i, row in enumerate(selected):
            wave, rate = sf.read(row['audio'], dtype='float32')
            assert rate==24000 and wave.ndim==1 and np.isfinite(wave).all()
            mel = mel_spectrogram(torch.from_numpy(wave)[None],2048,100,24000,384,1536,0,12000).double()
            assert torch.isfinite(mel).all()
            total += float(mel.sum()); square += float(mel.square().sum()); elements += mel.numel()
            if (i+1)%256==0: print(json.dumps(dict(mel_statistics_completed=i+1,total=len(selected))),flush=True)
    mean = total/elements; std = (square/elements-mean*mean)**.5
    assert 0 < std < 100
    # Model buffers are float32: freeze the actual values used by every component.
    return dict(mel_mean=float(np.float32(mean)),mel_std=float(np.float32(std)),
                normalization_policy='estimate from combined training split only; uniform unique utterance sample',
                sample_counts=dict(Counter(f'{r["speaker"]}/{r["language"]}' for r in selected)),
                sample_count=len(selected),mel_elements=elements,seed=seed,
                selected_audio=[r['audio'] for r in selected], sample_rate=24000,n_feats=100)


def load_ljspeech(data_dir):
    """Keep only audited original LJS recordings, never historical synthetic rows."""
    import soundfile as sf
    from concurrent.futures import ThreadPoolExecutor
    from training.stage2.prepare import digest
    data_dir=Path(data_dir)
    cache={r['text']:r for r in map(json.loads,(data_dir/'text_preflight.jsonl').read_text().splitlines())}
    splits={};units=[]
    for split in ('train','val','test'):
        rows=[]
        for r in map(json.loads,(data_dir/f'{split}.jsonl').read_text().splitlines()):
            if r['speaker']!=0 or '/LJSpeech/LJSpeech_24k/wavs/' not in r['audio']:continue
            assert 'start' not in r and 'end' not in r
            text=cache[r['text']];assert text['tokens']==r['text_key']
            rows.append(dict(r,ids=text['tokens'],tones=text['tones'],phonemes=text['phonemes'],
                             synthetic=False,source='LJSpeech',split=split))
        assert len(rows)==len({r['audio'] for r in rows})
        def check(row):
            info=sf.info(row['audio']);assert info.samplerate==24000 and info.channels==1
            assert info.frames>0 and len(row['ids'])<=info.frames//384
            row['duration']=info.duration
            return dict(key=json.dumps([row['audio'],None,None],separators=(',',':')),audio_frames=info.frames)
        with ThreadPoolExecutor(max_workers=16) as pool:units.extend(pool.map(check,rows))
        splits[split]=rows
    assert len(splits['train'])==11550 and len(splits['val'])==232
    assert not splits['test']  # Existing protocol has LJS val + external English eval, no internal LJS test.
    report=dict(source=str(data_dir),original_recordings_only=True,historical_synthetic_rows=0,
        rows={s:len(rs) for s,rs in splits.items()},hours={s:sum(r['duration'] for r in rs)/3600 for s,rs in splits.items()},
        source_hashes={n:digest(data_dir/n) for n in ['train.jsonl','val.jsonl','test.jsonl','text_preflight.jsonl']},
        internal_test='none in inherited LJS split; use 232 val records and fixed 200-text external LJS evaluation')
    return splits,units,report


def check_joint_splits(splits,evaluation):
    phones={s:{tuple(r['text_key']) for r in rs} for s,rs in splits.items()}
    audio={s:{r['audio'] for r in rs} for s,rs in splits.items()}
    for i,a in enumerate(('train','val','test')):
        for b in ('train','val','test')[i+1:]:
            assert not phones[a]&phones[b], f'Phoneme overlap: {a}/{b}'
            assert not audio[a]&audio[b], f'Audio overlap: {a}/{b}'
    assert not phones['train']&{tuple(r['token_ids']) for r in evaluation}
    return dict(cross_split_audio_overlap=0,cross_split_phoneme_overlap=0,train_external_eval_phoneme_overlap=0)


def prepare():
    sys.path.insert(0,str(REPO/'data_generation/majestic_200h'))
    from common import ROOT, ASSETS, atomic_json, config
    from prepare_training import audit, digest
    cfg = config(); training = cfg['training']; run = Path(training['run_dir'])
    joint=training.get('include_ljspeech',False)
    warm=training['mode']=='backbone_init'
    active=[0,1] if joint else [0]
    assert training['mode'] in ('scratch','backbone_init') and training['speaker_id']==int(joint)
    assert bool(training['source_checkpoint'])==warm
    if (run/'plan.json').exists():
        plan=json.loads((run/'plan.json').read_text())
        assert plan['mode']==training['mode'] and plan['training_config']==training
        return 0
    content, result = audit()
    if content is None: print(json.dumps(result)); return 2
    assert not run.exists(), 'Inspect partial run before retry'
    stage=run.with_name('.'+run.name+'.preparing'); stage.mkdir(exist_ok=False)
    data=stage/'data'; data.mkdir(); splits,units=content
    ljs_report=None
    if joint:
        ljs,ljs_units,ljs_report=load_ljspeech(training['ljs_data_dir'])
        for split in splits:
            assert all(r['speaker']==1 for r in splits[split])
            splits[split]=ljs[split]+splits[split]
        units=units+ljs_units
        atomic_json(data/'ljs_source_audit.json',ljs_report)
    def jsonl(path,rows): path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    for split, rows in splits.items():
        assert all(r['speaker'] in active for r in rows)
        jsonl(data/f'{split}.jsonl',rows)
        (data/f'{split}.txt').write_text(''.join(f'{r["audio"]}|{r["speaker"]}|{r["text"]}\n' for r in rows))
    jsonl(data/'text_preflight.jsonl',[dict(text=r['text'],tokens=r['ids'],tones=r['tones'],phonemes=r['phonemes']) for rows in splits.values() for r in rows])
    jsonl(data/'mel_statistics_units.jsonl',units)
    original=ASSETS/'data_24k/ljs_majestic'
    all_eval=list(map(json.loads,(original/'eval_manifest.jsonl').read_text().splitlines()))
    evaluation=all_eval if joint else [dict(r,speaker=0) for r in all_eval if r['speaker']==1]
    assert len(evaluation)==(650 if joint else 450)
    overlap=check_joint_splits(splits,evaluation)
    atomic_json(data/'joint_split_audit.json',overlap)
    jsonl(data/'eval_manifest.jsonl',evaluation)
    protocol=json.loads((original/'eval_protocol.json').read_text())
    reference=cfg['reference_audio']; english=cfg['quality_reference_audio_by_language']['en']
    ljs_reference=protocol['speaker_references']['0']
    group_refs={r['group']:(ljs_reference if joint and r['speaker']==0 else english if r['language']=='English' else reference) for r in evaluation}
    protocol.update(samples=len(evaluation),groups=dict(Counter(r['group'] for r in evaluation)),
                    manifest_sha256=digest(data/'eval_manifest.jsonl'),speaker_references={'0':ljs_reference if joint else reference,'1':reference},
                    group_speaker_references=group_refs,
                    group_reference_sha256={g:digest(p) for g,p in group_refs.items()},active_speaker_ids=active)
    atomic_json(data/'eval_protocol.json',protocol)
    source_info=backbone_metadata(training['source_checkpoint'],training['source_sha256']) if warm else None
    if warm:
        assert source_info['source_global_step']==training['source_global_step']
        stats_report=dict(**source_info['data_statistics'],normalization_policy='preserve selected backbone checkpoint acoustic scale',
                          normalization_source_checkpoint=source_info['source_checkpoint'],normalization_source_sha256=source_info['source_sha256'],
                          sample_rate=24000,n_feats=100)
    else:stats_report=mel_statistics(splits['train'],cfg['seed'])
    stats_report['manifest_sha256']=digest(data/'train.txt')
    atomic_json(data/'mel_statistics.json',stats_report)
    stats={k:stats_report[k] for k in ('mel_mean','mel_std')}
    model,model_cfg=make_model(run,stats,training['peak_lr'],cfg['seed'])
    initialization_sha256=None
    if warm:
        import torch
        import lightning as L
        transfer=load_backbone(model,training['source_checkpoint'],training['source_sha256'])
        atomic_json(data/'backbone_transfer_audit.json',dict(**source_info,**transfer))
        torch.save(dict(state_dict=model.state_dict(),hyper_parameters=dict(model.hparams),global_step=0,epoch=0,
                        **{'pytorch-lightning_version':L.__version__}),stage/'initialization.ckpt')
        initialization_sha256=digest(stage/'initialization.ckpt')
    atomic_json(data/'model_config.json',model_cfg)
    initial_digest=parameter_digest(model)
    budget=resolve_budget(training,len(splits['train']))
    sampling_counts=Counter(r['speaker'] for r in splits['train'])
    sampling_report=dict(policy='natural_unique_utterances',train_rows=len(splits['train']),
        speaker_rows=dict(sampling_counts),speaker_row_shares={str(k):v/len(splits['train']) for k,v in sampling_counts.items()},
        speaker_audio_hours={str(k):sum(r['duration'] for r in splits['train'] if r['speaker']==k)/3600 for k in sampling_counts},
        optimizer_steps_per_epoch=len(splits['train'])//192,
        approximate_epochs_at_step_cap=budget['approximate_epochs'],
        note='Row shares apply before the incomplete global tail batch is dropped; batches are length-bucketed, not fixed speaker quotas.')
    atomic_json(data/'sampling_plan.json',sampling_report)
    atomic_json(data/'training_summary.json',dict(train_rows=len(splits['train']),validation_rows=len(splits['val']),test_rows=len(splits['test']),
                language_counts=dict(Counter(r['language'] for r in splits['train'])),speaker_counts=dict(Counter(r['speaker'] for r in splits['train'])),
                ljs_source_audit=ljs_report,source_dataset_audit=str(ROOT/'reports/final_dataset_audit.json')))
    plan=dict(mode=training['mode'],recipe=('ljs_majestic_100h_joint_online_mas_backbone_init' if warm else 'ljs_majestic_100h_joint_online_mas_from_scratch' if joint else 'majestic_100h_joint_online_mas_from_scratch'),training_config=training,
              source_checkpoint=training['source_checkpoint'],source_global_step=source_info['source_global_step'] if warm else None,
              source_sha256=source_info['source_sha256'] if warm else None,initialization_sha256=initialization_sha256,
              ckpt_path=None,init_ckpt_path=str(run/'initialization.ckpt') if warm else None,
              initialization='backbone weights only; seeded random speaker rows; fresh Adam and scheduler' if warm else 'seeded random model; fresh Adam; no checkpoint loaded',initial_parameter_sha256=initial_digest,
              active_speaker_ids=active,speakers={'0':'LJSpeech','1':'MajesticVoice'} if joint else {'0':'MajesticVoice','1':'unused reserved slot'},n_spks=2,
              data_dir=str(run/'data'),data_source=str(ROOT),data_statistics=stats,
              manifest_hashes={p.name:digest(p) for p in data.iterdir()},unique_audio_hours=dict(majestic=result['selected_train_hours'],ljs=ljs_report['hours'] if joint else {}),
              sampling='natural_unique_utterances',train_rows_per_speaker=dict(Counter(r['speaker'] for r in splits['train'])),
              sampling_plan=sampling_report,
              evaluation_samples=len(evaluation),
              train_rows=len(splits['train']),validation_rows=len(splits['val']),test_rows=len(splits['test']),
              batch_size_per_gpu=48,devices=[0,1,2,3],effective_batch=192,precision='bf16-mixed',
              max_steps=budget['max_steps'],max_epochs=budget['max_epochs'],budget_unit=budget['budget_unit'],peak_lr=training['peak_lr'],final_lr=training['final_lr'],warmup_steps=training['warmup_steps'],
              lr_schedule=training['lr_schedule'],decay_fraction=training['decay_fraction'],
              loss_weights=dict(duration=1,prior=1,flow=1),seed=cfg['seed'],
              validation_every_steps=1000,checkpoint_every_steps=1000,evaluation_interval_steps=5000,
              diagnostic_steps=sorted(set(s for s in training['diagnostic_steps']+[budget['max_steps']] if s<=budget['max_steps'])),
              stopping_policy='resolved epoch/step cap; review generated CER/WER and listening for earlier stopping; no loss-only automatic stop')
    atomic_json(stage/'plan.json',plan)
    shutil.copyfile(ROOT/'reports/final_dataset_audit.json',stage/'dataset_audit.json')
    stage.rename(run)
    print(json.dumps(dict(status='prepared',run_dir=str(run),mode=plan['mode'],train_rows=plan['train_rows'])),flush=True)
    return 0
