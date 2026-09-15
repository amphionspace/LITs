"""Checkpoint, frontend, alignment, text-conditioning and sampling diagnostics.

Uses an isolated model without optimization or changes to the active experiment.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sqlite3
import time

REPO = Path(__file__).resolve().parents[2]
ASSETS = Path('/119010446/tts-assets')
DATA = ASSETS/'data_24k/foundation'


def write_json(path, obj):
    def encode(value):
        if hasattr(value, 'detach'):
            return value.detach().cpu().tolist()
        raise TypeError(f'Unsupported diagnostic value: {type(value).__name__}')
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=encode)+'\n')


def main(args):
    import numpy as np
    import soundfile as sf
    import soxr
    import torch
    import torch.nn.functional as F
    from lits.models.lits import LITS
    from lits.text import text_to_sequence_with_tones
    from lits.utils.audio import mel_spectrogram
    from lits.utils.model import normalize, denormalize, fix_len_compatibility, sequence_mask
    from vocos.vocoder import load_vocos_vocoder

    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.075)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    assert not (out/'synthesis.jsonl').exists(), 'Choose a new output directory'
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    step = checkpoint['global_step']
    previous_path = args.checkpoint.parents[1]/f'step_{step-2000:08d}/checkpoint.ckpt'
    previous = torch.load(previous_path, map_location='cpu', weights_only=False)
    model = LITS.load_from_checkpoint(str(args.checkpoint), map_location='cpu', weights_only=False).eval()
    parameters = dict(model.named_parameters())
    names_by_id = {id(p):name for name,p in parameters.items()}
    group_specs = model._build_optimizer_param_groups()
    saved_groups = checkpoint['optimizer_states'][0]['param_groups']
    optimizer_states = checkpoint['optimizer_states'][0]['state']
    audited = []
    covered = []
    for spec, saved in zip(group_specs,saved_groups):
        assert spec['name']==saved['name'] and len(spec['params'])==len(saved['params'])
        group_names=[names_by_id[id(p)] for p in spec['params']]
        covered.extend(group_names)
        states=[]
        for name,pid in zip(group_names,saved['params']):
            state=optimizer_states.get(pid,{})
            a=checkpoint['state_dict'][name]
            b=previous['state_dict'][name]
            states.append(dict(name=name,numel=a.numel(),finite=bool(torch.isfinite(a).all()),
                optimizer_state_present=bool(state), optimizer_step=float(state['step']) if state else None,
                moments_finite=all(bool(torch.isfinite(v).all()) for v in state.values() if torch.is_tensor(v)),
                weight_norm=float(a.norm()),update_norm=float((a-b).norm()),
                exp_avg_norm=float(state['exp_avg'].norm()) if state else None))
        audited.append(dict(group=spec['name'],lr=saved['lr'],parameters=states))
    assert len(covered)==len(set(covered)) and set(covered)==set(parameters)
    assert all(p['finite'] and p['moments_finite'] for g in audited for p in g['parameters'])
    write_json(out/'optimizer_audit.json',dict(checkpoint_step=step,previous_step=previous['global_step'],
        parameter_tensors=len(parameters),parameter_count=sum(p.numel() for p in parameters.values()),
        optimizer_exact_coverage=True,groups=audited,
        finite_checkpoint_tensors=all(bool(torch.isfinite(v).all()) for v in checkpoint['state_dict'].values() if torch.is_tensor(v))))
    del checkpoint,previous
    model=model.to('cuda:0').eval().requires_grad_(False)
    vocoder,cfg=load_vocos_vocoder(str(REPO/'vocos/generator.ckpt'),torch.device('cuda:0'),REPO)
    vocoder.requires_grad_(False)
    assert (cfg.sampling_rate,cfg.num_mels,cfg.hop_size)==(24000,100,384)
    selected=[]
    rng=random.Random(2026091202)
    with sqlite3.connect(f'file:{DATA}/dataset.sqlite?mode=ro',uri=True) as db:
        for split in ['train','val']:
            for source in ['HiFiTTS','Premium']:
                i=0
                for lo,hi in [(2,4),(4,7),(7,10)]:
                    candidates=db.execute('select id,payload from samples where split=? and source=? and duration>=? and duration<? order by id',(split,source,lo,hi)).fetchall()
                    for sid,payload in rng.sample(candidates,2):
                        row=json.loads(payload)
                        selected.append(dict(row,sample_id=sid,diagnostic_id=f'{split}_{"en" if source=="HiFiTTS" else "zh"}_{i}',group=f'{split}_{"en" if source=="HiFiTTS" else "zh"}',real_audio=True))
                        i+=1
    mixed=[json.loads(l) for l in (DATA/'eval_manifest.jsonl').read_text().splitlines() if json.loads(l)['group']=='foundation_mixed']
    for i,row in enumerate(rng.sample(mixed,12)):
        selected.append(dict(row,ids=row['token_ids'],tones=row['tone_ids'],source='mixed',split='external',
            diagnostic_id=f'mixed_{i}',group='mixed',real_audio=False))
    write_json(out/'selected_samples.json',selected)
    write_json(out/'metadata.json',dict(checkpoint=str(args.checkpoint),checkpoint_step=step,
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),selection_seed=2026091202,
        selection='24 real samples: 2 in each 2–4/4–7/7–10-second bin, each source and train/val split; 12 uniformly sampled fixed mixed evaluation texts',
        precision='float32',streaming=False,speaker_id=0,temperature=1.,
        inference_conditions=['10 and 30 Euler steps crossed with two fixed noise seeds','MAS durations at 10 steps, seed 0, for real samples','real Mel reconstruction and original audio'],
        expected_audio_count=216,gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),started_at_unix=time.time(),
        limitations=['MAS durations are derived from this model, not ground truth.',
                     'ASR error on real recordings does not independently prove transcription errors.',
                     'Reversing/zeroing mu is an intervention outside the normal conditioning distribution.',
                     '24 real examples and 12 mixed texts cannot exclude rare defects or estimate full-set quality.']))
    with (out/'synthesis.jsonl').open('w',buffering=1) as stream, torch.inference_mode():
        for index,row in enumerate(selected):
            name=row['diagnostic_id'];dest=out/name;dest.mkdir(exist_ok=False)
            x=torch.tensor([row['ids']],device='cuda:0');xt=torch.tensor([row['tones']],device='cuda:0');xl=torch.tensor([x.shape[-1]],device='cuda:0')
            sid=torch.tensor([0],device='cuda:0');speaker=model.spk_emb(sid)
            assert x.min()>=0 and x.max()<model.n_vocab and len(row['ids'])==len(row['tones'])
            ids,tones,phonemes=text_to_sequence_with_tones(row['text'],['en_zh_dict_mixed_rhyme_body_tone_cleaners'],prepend_sil=True)
            frontend=dict(ids_match=ids==row['ids'],tones_match=tones==row['tones'],phonemes_match=phonemes==row['phonemes'])
            hidden=model.get_hidden_mel(x,xl,spks=sid,x_tones=xt)
            pred_n=int(hidden['y_max_length']);assert 0<pred_n<3000
            mu_x,logw,xmask=model.encoder(x,xl,speaker,x_tones=xt)
            diagnostic=dict(text=row['text'],group=row['group'],frontend=frontend,predicted_frames=pred_n)
            torch.manual_seed(2026091202+index)
            z0=torch.randn(1,100,3000,device='cuda:0')
            torch.manual_seed(2027091202+index)
            z1=torch.randn_like(z0)
            def save(mode,wav,extra=None):
                waveform=np.asarray(wav,dtype=np.float32).reshape(-1)
                assert len(waveform)>0 and np.isfinite(waveform).all()
                target=dest/f'{mode}.wav'
                clipped_fraction=float(np.mean(np.abs(waveform)>1))
                sf.write(target,np.clip(waveform,-1,1),24000,subtype='FLOAT')
                stream.write(json.dumps(dict(id=f'{name}/{mode}',sample=name,mode=mode,source=row['source'],
                    split=row['split'],group=row['group'],ref_text=row['text'],audio=str(target),
                    duration=len(waveform)/24000,unclipped_peak=float(np.abs(waveform).max()),clipped_fraction=clipped_fraction,
                    **(extra or {})),ensure_ascii=False)+'\n')
            def decode(mu,mask,frames,steps,z):
                mel=model.get_mel(mu,mask,steps,1.,spks=speaker,streaming=False,z=z.clone())[:,:,:frames]
                assert torch.isfinite(mel).all()
                wave=vocoder(mel).squeeze().cpu().numpy()[:frames*384]
                return mel,wave
            if row['real_audio']:
                audio,sr=sf.read(row['audio'],dtype='float32');assert audio.ndim==1
                if sr!=24000:audio=soxr.resample(audio,sr,24000,quality='HQ')
                real=mel_spectrogram(torch.from_numpy(audio)[None].to('cuda:0'),2048,100,24000,384,1536,0,12000)
                n=real.shape[-1];yl=torch.tensor([n],device='cuda:0')
                y=normalize(real,model.mel_mean,model.mel_std);yp=F.pad(y,(0,fix_len_compatibility(n)-n));mask=sequence_mask(yl,yp.shape[-1]).unsqueeze(1).float()
                torch.manual_seed(2026091202+index);random.seed(2)
                losses=model(x,xl,yp,yl,spks=sid,x_tones=xt)
                attn=losses[3]
                assert torch.equal(attn.sum(1),mask[:,0])
                path=attn[0,:,:n].argmax(0);assert bool((path[1:]>=path[:-1]).all())
                durations=attn.sum(-1).squeeze(0)
                predicted=model._clamp_inference_tone_durations(torch.ceil(torch.exp(logw))*xmask,x,xmask).squeeze()
                mas_mu=torch.bmm(attn.transpose(1,2),mu_x.transpose(1,2)).transpose(1,2)
                floors=model._mas_floor_frames(x,sid,xt);ceilings=model._mas_ceiling_frames(x,sid,xt)
                diagnostic.update(real_frames=n,predicted_to_real_duration=pred_n/n,
                    normalized_real_mel_mean=float(y.mean()),normalized_real_mel_std=float(y.std()),
                    duration_mae_frames=float((predicted-durations).abs().mean()),
                    mas_durations=durations.cpu().tolist(),predicted_durations=predicted.cpu().tolist(),
                    mas_stats=losses[4],mas_monotonic=True,mas_valid_frame_coverage=True,
                    prior_mse=float((((yp-mas_mu)**2)*mask).sum()/(mask.sum()*100)),
                    prior_zero_mean_mse=float(((yp**2)*mask).sum()/(mask.sum()*100)),
                    padding_roundtrip_error=float((denormalize(y,model.mel_mean,model.mel_std)-real).abs().max()))
                conditioning=[]
                reversed_mu=mas_mu.clone();reversed_mu[:,:,:n]=mas_mu[:,:,:n].flip(-1)
                conditions={'correct':mas_mu,'reversed':reversed_mu,'zero':torch.zeros_like(mas_mu)}
                eps=z0[:,:,:yp.shape[-1]]
                for condition,mu in conditions.items():
                    h=model.decoder.encoder(mu,mask,streaming=False)
                    for t in [.05,.25,.5,.75,.95]:
                        noisy=(1-(1-model.decoder.sigma_min)*t)*eps+t*yp
                        target=yp-(1-model.decoder.sigma_min)*eps
                        v=model.decoder.estimator(noisy,mask,h,torch.tensor([t],device='cuda:0'),speaker,None,streaming=False)
                        error=float(((v-target).square()*mask).sum()/(mask.sum()*100))
                        conditioning.append(dict(condition=condition,t=t,flow_mse=error))
                diagnostic['text_conditioning_loss']=conditioning
                save('original',audio)
                reconstructed=vocoder(real).squeeze().cpu().numpy()[:n*384];save('reconstruction',reconstructed)
                mas_mel,mas_audio=decode(mas_mu,mask,n,10,z0);save('mas_10_s0',mas_audio)
                np.savez_compressed(dest/'alignment.npz',alignment=attn.cpu().numpy(),real_mel=real.cpu().numpy(),
                    mas_mu=mas_mu.cpu().numpy(),predicted_mu=hidden['mu_y'].cpu().numpy())
                if index==0:
                    torch.manual_seed(777)
                    direct=model.synthesise(x,xl,10,spks=sid,x_tones=xt)['mel'][:,:,:pred_n]
                    torch.manual_seed(777)
                    separate=model.get_mel(hidden['mu_y'],hidden['y_mask'],10,1.,spks=speaker,streaming=False)[:,:,:pred_n]
                    diagnostic['inference_entrypoint_max_error']=float((direct-separate).abs().max())
                    assert diagnostic['inference_entrypoint_max_error']<1e-5
            for seed,z in [(0,z0),(1,z1)]:
                for steps in [10,30]:
                    mel,audio=decode(hidden['mu_y'],hidden['y_mask'],pred_n,steps,z)
                    save(f'pred_{steps}_s{seed}',audio,dict(ode_steps=steps,noise_seed_index=seed))
            write_json(dest/'diagnostic.json',diagnostic)
            print(json.dumps(dict(completed=index+1,total=len(selected),sample=name,frontend=frontend,
                                 gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30)),flush=True)
    print('HEALTH_SYNTHESIS_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
