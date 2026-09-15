"""Same-text audio comparisons for reported Chinese accent and missing mixed data."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re


def main(run):
    import numpy as np
    import soundfile as sf
    import soxr
    import torch
    from lits.models.lits import LITS
    from lits.text import text_to_sequence_with_tones
    from vocos.vocoder import load_vocos_vocoder

    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.075)
    repo=Path(__file__).resolve().parents[2]
    out=run/'diagnostics/accent_2250'
    out.mkdir(parents=True,exist_ok=False)
    archive=Path('/119010446/tts-assets/Archive/TS-004_大气女声')
    original=[]
    for folder in sorted(archive.iterdir()):
        for line in (folder/'text.txt').read_text().splitlines():
            if not re.match(r'^\d+\t',line):continue
            identifier,text=line.split('\t',1)
            path=folder/'wav'/f'{identifier}.wav'
            assert path.is_file()
            original.append(dict(sample=f'native_{identifier}',text=re.sub(r'#\d+','',text).strip(),audio=str(path),
                group='native_mixed' if folder.name.startswith('MIX') else 'native_zh',split='reference'))
    chosen={'native_000001','native_000003','native_100001','native_100004'}
    selected=[r for r in original if r['sample'] in chosen]
    val=[r for r in map(json.loads,(run/'data/val.jsonl').read_text().splitlines()) if r['speaker']==1 and r['language']=='Chinese']
    rng=np.random.default_rng(2026091305)
    for i in rng.choice(len(val),size=4,replace=False):
        row=val[int(i)]
        selected.append(dict(sample='teacher_'+Path(row['audio']).stem,text=row['text'],audio=row['audio'],group='teacher_zh',split='val'))
    train=[json.loads(s) for s in (run/'data/train.jsonl').read_text().splitlines()]
    metadata=dict(checkpoint=str(run/'eval/step_00002250/checkpoint.ckpt'),baseline=str(run/'initialization.ckpt'),
        original_inventory=original,original_cn_audio_count=sum(r['group']=='native_zh' for r in original),
        original_mixed_audio_count=sum(r['group']=='native_mixed' for r in original),
        mixed_training_rows=sum(bool(re.search('[\u4e00-\u9fff]',r['text']) and re.search('[a-zA-Z]',r['text'])) for r in train if r['speaker']==1),
        train_language_counts=dict(Counter(r['language'] for r in train)),speaker_id=1,steps=10,streaming=False,
        precision='float32',noise='fixed per-text 100x3000 Gaussian; shared across all variants',
        selected=selected,limitations=['Module replacement is an inference intervention, not an experiment proving which training data caused an accent.',
            'ASR CER does not measure foreign accent or naturalness. User listening is required.',
            'Native 000001 is the teacher voice reference, not an independent reference-selection test.'])
    current=torch.load(metadata['checkpoint'],map_location='cpu',weights_only=False)['state_dict']
    initial=torch.load(metadata['baseline'],map_location='cpu',weights_only=False)['state_dict']
    model=LITS.load_from_checkpoint(metadata['checkpoint'],map_location='cpu',weights_only=False).cuda().eval().requires_grad_(False)
    vocoder,_=load_vocos_vocoder(str(repo/'vocos/generator.ckpt'),torch.device('cuda:0'),repo)
    variants=dict(current=lambda k:False,initial=lambda k:True,
        restore_prior=lambda k:k.startswith('encoder.') and not k.startswith('encoder.proj_w.'),
        restore_duration=lambda k:k.startswith('encoder.proj_w.'),
        restore_speaker=lambda k:k.startswith('spk_emb.'),
        restore_decoder=lambda k:k.startswith('decoder.'))
    metadata['modes']=list(variants)
    metadata['audio_count']=len(selected)*(len(variants)+1)
    (out/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    details={}
    with (out/'synthesis.jsonl').open('w',buffering=1) as stream,torch.inference_mode():
        def save(row,mode,wave,extra=None):
            dest=out/row['sample'];dest.mkdir(exist_ok=True)
            wave=np.asarray(wave,dtype=np.float32).reshape(-1)
            assert np.isfinite(wave).all() and len(wave)>0
            path=dest/f'{mode}.wav'
            sf.write(path,np.clip(wave,-1,1),24000,subtype='FLOAT')
            record=dict(id=f'{row["sample"]}/{mode}',sample=row['sample'],group=row['group'],mode=mode,split=row['split'],
                source='TS004_native' if row['sample'].startswith('native') else 'MajesticVoice_teacher',
                ref_text=row['text'],audio=str(path),duration=len(wave)/24000,**(extra or {}))
            stream.write(json.dumps(record,ensure_ascii=False)+'\n')
        for row in selected:
            wave,sr=sf.read(row['audio'],dtype='float32')
            assert wave.ndim==1
            if sr!=24000:wave=soxr.resample(wave,sr,24000,quality='HQ')
            save(row,'reference',wave)
            ids,tones,phonemes=text_to_sequence_with_tones(row['text'],['en_zh_dict_mixed_rhyme_body_tone_cleaners'],prepend_sil=True)
            assert ids and 2 not in ids
            details[row['sample']]=dict(ids=ids,tones=tones,phonemes=phonemes,modes={})
        for mode,predicate in variants.items():
            model.load_state_dict({k:initial[k] if predicate(k) else value for k,value in current.items()},strict=True)
            for index,row in enumerate(selected):
                d=details[row['sample']]
                x=torch.tensor([d['ids']],device='cuda:0');xt=torch.tensor([d['tones']],device='cuda:0')
                xl=torch.tensor([len(d['ids'])],device='cuda:0');sid=torch.tensor([1],device='cuda:0')
                model.decoder.reset_encoder_cache()
                hidden=model.get_hidden_mel(x,xl,spks=sid,x_tones=xt)
                n=int(hidden['y_max_length']);assert 0<n<3000
                torch.manual_seed(2026091305+index)
                z=torch.randn(1,100,3000,device='cuda:0')
                mel=model.get_mel(hidden['mu_y'],hidden['y_mask'],10,1.,spks=hidden['spks'],streaming=False,z=z)[:,:,:n]
                wave=vocoder(mel).squeeze().cpu().numpy()[:n*384]
                _,logw,xmask=model.encoder(x,xl,hidden['spks'],x_tones=xt)
                durations=model._clamp_inference_tone_durations(torch.ceil(torch.exp(logw))*xmask,x,xmask)[0,0]
                d['modes'][mode]=dict(frames=n,durations=durations.cpu().tolist())
                save(row,mode,wave,dict(frames=n))
            print(f'COMPLETE {mode}',flush=True)
    (out/'details.json').write_text(json.dumps(details,ensure_ascii=False,indent=2)+'\n')
    print('ACCENT_COMPARISON_COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,required=True)
    main(parser.parse_args().run_dir)
