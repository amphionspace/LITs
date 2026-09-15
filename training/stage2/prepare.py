"""Validate the existing two-speaker recipe and make an explicit weights-only init."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import shutil


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def main(args):
    import torch
    if args.max_steps <= 1000:
        raise ValueError('max_steps must exceed the initial adaptation and warmup phases')
    data = Path('/119010446/tts-assets/data_24k/ljs_majestic')
    run = args.run_dir
    run.mkdir(parents=True,exist_ok=False)
    checkpoint = args.checkpoint
    ckpt = torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert ckpt['global_step'] == 197000
    h = ckpt['hyper_parameters']
    assert h['n_vocab']==173 and h['n_feats']==100 and h['n_spks']==2
    assert all(torch.isfinite(v).all() for v in ckpt['state_dict'].values())
    source_hash = digest(checkpoint)
    assert source_hash == json.loads((checkpoint.parent/'checkpoint_metadata.json').read_text())['sha256']
    summary = json.loads((data/'training_summary.json').read_text())
    for field in ['source_chinese_audit','source_english_audit']:
        assert json.loads(Path(summary[field]).read_text())['status']=='passed'
    stats = json.loads((data/'mel_statistics.json').read_text())
    assert stats['manifest_sha256']==digest(data/'train.txt') and stats['mas_text_length_check']=='passed'
    splits = {split:[json.loads(line) for line in (data/f'{split}.jsonl').read_text().splitlines()] for split in ['train','val','test']}
    source_train = splits['train']
    assert len(source_train)==54788 and Counter(r['speaker'] for r in source_train)=={0:27394,1:27394}
    ljs=[r for r in source_train if r['speaker']==0 and '/LJSpeech/LJSpeech_24k/' in r['audio']]
    majestic=list({(r['audio'],r.get('start'),r.get('end'),tuple(r['text_key'])):r for r in source_train if r['speaker']==1}.values())
    assert len(ljs)==11550 and len(majestic)==11034
    rng=random.Random(20260913)
    train=ljs+majestic+rng.choices(majestic,k=len(ljs)-len(majestic))
    rng.shuffle(train)
    splits['train']=train
    assert len(train)==23100 and Counter(r['speaker'] for r in train)=={0:11550,1:11550}
    assert all((r['speaker']==0 and '/LJSpeech' in r['audio']) or (r['speaker']==1 and '/MajesticVoice/' in r['audio']) for rs in splits.values() for r in rs)
    train_keys = {tuple(r['text_key']) for r in train}
    heldout = {tuple(r['text_key']) for split in ['val','test'] for r in splits[split]}
    eval_rows = [json.loads(line) for line in (data/'eval_manifest.jsonl').read_text().splitlines()]
    heldout |= {tuple(r['token_ids']) for r in eval_rows}
    assert not train_keys & heldout
    assert not {r['audio'] for r in train} & {r['audio'] for split in ['val','test'] for r in splits[split]}
    paths = {r['audio'] for rs in splits.values() for r in rs}
    assert all(Path(p).is_file() for p in paths)
    snapshot = run/'data'
    snapshot.mkdir()
    files = ['train.txt','train.jsonl','val.txt','val.jsonl','test.txt','test.jsonl','text_preflight.jsonl',
        'mel_statistics_units.jsonl','training_summary.json','mel_statistics.json','eval_manifest.jsonl','eval_protocol.json']
    hashes = {}
    for name in files:
        shutil.copyfile(data/name,snapshot/name)
        hashes[name]=digest(snapshot/name)
    def line(row):
        fields=[row['audio'],str(row['speaker'])]
        if 'start' in row:fields += [str(row['start']),str(row['end'])]
        return '|'.join(fields+[row['text']])+'\n'
    (snapshot/'train.txt').write_text(''.join(line(r) for r in train))
    (snapshot/'train.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in train))
    stage_summary=dict(summary,stage=2,train_rows=23100,train_rows_per_speaker={'0':11550,'1':11550},
        train_unique_audio=len({r['audio'] for r in train}),train_segment_rows=sum('start' in r for r in train),
        ljs_source='original recordings only',ljs_historical_synthetic_rows=0,seed=20260913)
    (snapshot/'training_summary.json').write_text(json.dumps(stage_summary,ensure_ascii=False,indent=2)+'\n')
    state = {k:v.clone() for k,v in ckpt['state_dict'].items()}
    assert state['spk_emb.weight'].shape==(2,64)
    state['spk_emb.weight'][1].copy_(state['spk_emb.weight'][0])
    assert all(torch.equal(v,ckpt['state_dict'][k]) for k,v in state.items() if k!='spk_emb.weight')
    assert torch.equal(state['spk_emb.weight'][0],ckpt['state_dict']['spk_emb.weight'][0])
    init = {key:ckpt[key] for key in ['pytorch-lightning_version','hyper_parameters']}
    init.update(state_dict=state,global_step=0,epoch=0)
    torch.save(init,run/'initialization.ckpt')
    inherited_statistics = dict(mel_mean=float(state['mel_mean']),mel_std=float(state['mel_std']))
    shutil.copyfile(snapshot/'mel_statistics.json',snapshot/'source_recipe_mel_statistics.json')
    (snapshot/'mel_statistics.json').write_text(json.dumps(dict(**inherited_statistics,
        normalization_source_checkpoint=str(checkpoint),normalization_policy='preserve stage-one scale',
        manifest_sha256=digest(snapshot/'train.txt'),sample_rate=24000,n_feats=100,mas_text_length_check='passed'),indent=2)+'\n')
    hashes={name:digest(snapshot/name) for name in files}
    plan = dict(stage=2,source_checkpoint=str(checkpoint),source_global_step=197000,source_sha256=source_hash,
        initialization_sha256=digest(run/'initialization.ckpt'),initialization='weights only; fresh Adam and stage-two step counter',
        speakers={'0':'LJSpeech','1':'MajesticVoice'},speaker_embedding_initialization='both rows start from trained stage-one row 0',
        data_dir=str(snapshot),data_source=str(data),manifest_hashes=hashes,train_rows=23100,validation_rows=464,test_rows=1004,
        train_rows_per_speaker={'0':11550,'1':11550},majestic_unique_chinese_train=8278,majestic_unique_english_train=2756,
        ljs_original_train_rows=11550,ljs_historical_synthetic_train_rows=0,
        unique_audio_hours={'ljs_original':21.04746983796303,'majestic_chinese':8.670755555555672,'majestic_english':2.9952888888888705},
        train_heldout_audio_overlap=0,train_heldout_phoneme_overlap=0,audio_paths_checked=len(paths),
        data_statistics=inherited_statistics,normalization='stage-one model statistics retained for both data and model',
        batch_size_per_gpu=48,devices=[0,1,2,3],effective_batch=192,precision='bf16-mixed',max_steps=args.max_steps,
        learning_rates={'decoder':2e-5,'spk_emb':1e-4,'prior_encoder':2e-6,'duration_predictor':5e-6},
        schedule={'warmup_steps':200,'encoder_update_start_step':500,'encoder_warmup_steps':500,'final_lr_ratio':.2},
        validation_every_steps=250,checkpoint_every_steps=250,evaluation_interval_steps=1000,
        evaluation_groups={'ljs_en':200,'majestic_en':200,'majestic_zh':200,'majestic_mixed':50},
        seed=20260913)
    (run/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(plan,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--max-steps',type=int,default=6000)
    main(parser.parse_args())
