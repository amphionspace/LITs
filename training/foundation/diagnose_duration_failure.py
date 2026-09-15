"""Audit Chinese MAS A/B failures with controlled alignment/solver ablations."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time

import numpy as np

from training.foundation.diagnose_duration_ab import allocate_frames


def main(args):
    import soundfile as sf
    import soxr
    import torch
    from torch.nn import functional as F
    from lits.models.lits import LITS
    from lits.utils import monotonic_align
    from lits.utils.audio import mel_spectrogram
    from lits.utils.model import normalize,fix_len_compatibility,sequence_mask,generate_path
    from lits.text.bopomofo_utils import BOPOMOFO_TONES
    from vocos.vocoder import load_vocos_vocoder

    torch.set_num_threads(4)
    root=args.output;root.mkdir(parents=True,exist_ok=False)
    repo=Path(__file__).resolve().parents[2];start=time.time()
    protocol=json.loads((args.ab/'protocol.json').read_text())
    cases=[c for c in json.loads((args.ab/'cases.json').read_text()) if c['language']=='zh']
    selected={r['sample_id']:r for r in json.loads((args.ab/'selected_samples.json').read_text())}
    checkpoint=Path(protocol['checkpoint'])
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==protocol['checkpoint_sha256']
    model=LITS.load_from_checkpoint(str(checkpoint),map_location='cpu',weights_only=False).eval().requires_grad_(False)
    vocoder,_=load_vocos_vocoder(str(repo/'vocos/generator.ckpt'),torch.device('cpu'),repo)
    vocoder.eval().requires_grad_(False)
    metadata=dict(checkpoint=str(checkpoint),checkpoint_sha256=protocol['checkpoint_sha256'],device='cpu',precision='float32',
        source_ab=str(args.ab),samples=3,started_at_unix=start,training_modified=False)
    (root/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    all_details=[]
    with (root/'synthesis.jsonl').open('w',buffering=1) as stream,torch.inference_mode():
        for case in cases:
            row=selected[case['sample_id']];name=str(case['sample_id']);dest=root/name;dest.mkdir()
            with sqlite3.connect('file:/119010446/tts-assets/data_24k/foundation/dataset.sqlite?mode=ro',uri=True) as db:
                source=json.loads(db.execute('select payload from samples where id=?',(case['sample_id'],)).fetchone()[0])
            x=torch.tensor([row['ids']]);xt=torch.tensor([source['tones']]);xl=torch.tensor([x.shape[-1]]);sid=torch.tensor([0]);spk=model.spk_emb(sid)
            audio,sr=sf.read(source['audio'],dtype='float32')
            if sr!=24000:audio=soxr.resample(audio,sr,24000,quality='HQ')
            real=mel_spectrogram(torch.from_numpy(audio)[None],2048,100,24000,384,1536,0,12000)
            n=real.shape[-1];padded=int(fix_len_compatibility(n));assert n==case['frames']
            y=F.pad(normalize(real,model.mel_mean,model.mel_std),(0,padded-n))
            mask=sequence_mask(torch.tensor([n]),padded)[:,None].float()
            mu_x,logw,xmask=model.encoder(x,xl,spk,x_tones=xt)
            factor=-.5*torch.ones_like(mu_x)
            scores=(torch.matmul(factor.transpose(1,2),y**2)-torch.matmul(2*factor.transpose(1,2)*mu_x.transpose(1,2),y)
                    +torch.sum(factor*mu_x**2,1).unsqueeze(-1)-.5*math.log(2*math.pi)*100)
            amask=(xmask.unsqueeze(-1)*mask.unsqueeze(2)).squeeze(1)
            floors=model._mas_floor_frames(x,sid,xt);ceilings=model._mas_ceiling_frames(x,sid,xt)
            full,stats=monotonic_align.maximum_path_constrained(scores,amask,floors,ceilings)
            mas=full.sum(-1)[0].numpy().astype(np.int64)
            np.testing.assert_array_equal(mas,case['mas_durations'])
            # Verify the original model.forward uses exactly this alignment and target duration.
            compute=model.decoder.compute_loss
            model.decoder.compute_loss=lambda x1,**kwargs:(x1.new_zeros(()),None)
            original_forward=model(x,xl,y,torch.tensor([n]),spks=sid,x_tones=xt)
            model.decoder.compute_loss=compute
            torch.testing.assert_close(original_forward[3],full,rtol=0,atol=0)
            baseline=case['matched_predicted_durations'];pred=np.array(baseline,dtype=np.int64)
            tone=model._tone_mark_mask(x);tone_np=tone[0].numpy();speech=np.array(row['speech'])
            variants={'predicted_matched':pred,'mas':mas}
            tone_path,tone_stats=monotonic_align.maximum_path_constrained(scores,amask,
                torch.where(tone,floors,torch.zeros_like(floors)),torch.where(tone,ceilings,torch.zeros_like(ceilings)))
            variants['tone_only_mas']=tone_path.sum(-1)[0].numpy().astype(np.int64)
            variants['free_mas']=monotonic_align.maximum_path(scores,amask).sum(-1)[0].numpy().astype(np.int64)
            variants['blend50'],_=allocate_frames(.5*pred+.5*mas,n,tone_np)
            # Preserve each MAS syllable's total, changing only initial/rhyme allocation within it.
            internal=mas.copy();pending=[];syllables=[]
            for i,symbol in enumerate(row['symbols']):
                if speech[i]:pending.append(i)
                elif symbol in BOPOMOFO_TONES:
                    if pending:
                        total=int(mas[pending].sum())
                        internal[pending],_=allocate_frames(pred[pending],total,np.zeros(len(pending),dtype=bool))
                        syllables.append(dict(indices=pending+[i],symbols=[row['symbols'][j] for j in pending+[i]],
                            mas_total=int(mas[pending+[i]].sum()),pred_total=int(pred[pending+[i]].sum())))
                    pending=[]
                else:pending=[]
            variants['mas_syllable_pred_internal']=internal
            # Keep prediction's pauses/tone symbols, allocating the remainder using MAS speech proportions.
            body=pred.copy();body[speech],_=allocate_frames(mas[speech],n-int(pred[~speech].sum()),np.zeros(speech.sum(),dtype=bool))
            variants['mas_body_pred_special']=body
            torch.manual_seed(case['seed']);z=torch.randn(1,100,padded)
            assert hashlib.sha256(z.numpy().tobytes()).hexdigest()==case['noise_sha256']
            detail=dict(sample_id=case['sample_id'],text=row['text'],symbols=row['symbols'],ids=row['ids'],frames=n,
                floors=floors[0].tolist(),ceilings=ceilings[0].tolist(),syllables=syllables,
                full_stats={k:float(v) for k,v in stats.items()},tone_only_stats={k:float(v) for k,v in tone_stats.items()},
                checks=dict(recomputed_mas_equals_saved=True,original_forward_alignment_equal=True),variants={})
            original_npz=np.load(args.ab/Path(case['audio']['A']).parent/'mel_outputs.npz')
            for mode,durations in variants.items():
                assert int(durations.sum())==n and (durations>=1).all()
                path=generate_path(torch.tensor(durations)[None],amask)
                expanded=torch.repeat_interleave(mu_x,torch.tensor(durations),dim=2)
                mu=F.pad(expanded,(0,padded-n))
                bmm=torch.bmm(path.transpose(1,2),mu_x.transpose(1,2)).transpose(1,2)
                torch.testing.assert_close(mu,bmm,rtol=0,atol=0)
                stats_mode=dict(durations=durations.tolist(),
                    prior_mse=float(((y-mu)**2*mask).sum()/(mask.sum()*100)),
                    speech_one_or_two_frame_fraction=float(np.mean(durations[speech]<=2)),
                    max_speech_frames=int(durations[speech].max()),special_frames=int(durations[~speech].sum()),
                    changed_frame_fraction=float((path[0,:,:n].argmax(0)!=full[0,:,:n].argmax(0)).float().mean()))
                detail['variants'][mode]=stats_mode
                for steps in ([10,30] if mode=='mas' else [10]):
                    model.decoder.reset_encoder_cache()
                    mel=model.get_mel(mu,mask,steps,1.,spks=spk,streaming=False,z=z.clone())[:,:,:n]
                    wave=vocoder(mel).flatten().numpy()[:n*384]
                    assert np.isfinite(wave).all() and len(wave)==n*384
                    if mode in ('mas','predicted_matched') and steps==10:
                        error=float(np.max(np.abs(mel.numpy()-original_npz[mode])))
                        assert error<1e-5,(mode,error)
                        stats_mode['baseline_mel_max_error']=error
                        arm=next(k for k,v in case['mapping'].items() if v==mode)
                        existing,sr=sf.read(args.ab/case['audio'][arm],dtype='float32')
                        wav_error=float(np.max(np.abs(existing-wave*case['pair_gain'])))
                        assert wav_error<=1/32768+1e-6,(mode,wav_error)
                        stats_mode['baseline_pcm_max_error']=wav_error
                        path_audio=args.ab/case['audio'][arm]
                    else:
                        path_audio=dest/f'{mode}_{steps}.wav'
                        # Use original pair gain across interventions; fail rather than silently clip.
                        gain=case['pair_gain']
                        peak=float(np.abs(wave*gain).max())
                        if peak>=1:gain=min(gain,.98/float(np.abs(wave).max()))
                        sf.write(path_audio,wave*gain,24000,subtype='PCM_16')
                        stats_mode[f'gain_{steps}']=gain
                    entry=dict(id=f'{name}/{mode}_{steps}',sample=name,mode=f'{mode}_{steps}',source='Premium',split='val',
                        group='val_zh',ref_text=row['text'],audio=str(path_audio),duration=n*.016)
                    stream.write(json.dumps(entry,ensure_ascii=False)+'\n')
            for mode in ('original','reconstruction'):
                path_audio=args.ab/case['audio'][mode]
                stream.write(json.dumps(dict(id=f'{name}/{mode}',sample=name,mode=mode,source='Premium',split='val',group='val_zh',
                    ref_text=row['text'],audio=str(path_audio),duration=sf.info(path_audio).duration),ensure_ascii=False)+'\n')
            (dest/'diagnostic.json').write_text(json.dumps(detail,ensure_ascii=False,indent=2)+'\n');all_details.append(detail)
            print(json.dumps(dict(completed=name,seconds=time.time()-start)),flush=True)
    metadata.update(completed_at_unix=time.time(),elapsed_seconds=time.time()-start,baseline_reproduced=True,alignment_expansion_exact=True)
    (root/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (root/'details.json').write_text(json.dumps(all_details,ensure_ascii=False,indent=2)+'\n')
    print('DURATION_FAILURE_AUDIT_SYNTHESIS_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ab',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
