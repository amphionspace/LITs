"""Duration agreement with MAS on the frozen validation split; MAS is not ground truth."""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def pair_metrics(target,prediction):
    a=np.asarray(target,dtype=np.float64);b=np.asarray(prediction,dtype=np.float64)
    assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    if len(a)<2: return None
    am,bm=float(a.mean()),float(b.mean());ast,bst=float(a.std()),float(b.std())
    return dict(tokens=len(a),mas_mean=am,predicted_mean=bm,mas_std=ast,predicted_std=bst,
                pearson=float(np.corrcoef(a,b)[0,1]) if ast>0 and bst>0 else None,
                std_ratio=bst/ast if ast>0 else None,
                cv_ratio=(bst/bm)/(ast/am) if ast>0 and bm>0 and am>0 else None,
                mean_ratio=bm/am if am>0 else None,mae_frames=float(np.abs(a-b).mean()))


def summarize(rows):
    report={}
    speaker_groups=sorted({f'speaker_{r.get("speaker",0)}/{r["language"]}' for r in rows})
    for lang in ['English','Chinese','Mixed','all',*speaker_groups]:
        chosen=[r for r in rows if lang=='all' or r['language']==lang or f'speaker_{r.get("speaker",0)}/{r["language"]}'==lang]
        result={}
        for subset in ('all_tokens','speech','supervised_speech','tone'):
            result[subset]={}
            for variant in ('raw','ceil','inference'):
                pairs=[];sentence=[]
                for r in chosen:
                    mask=np.asarray(r[subset],dtype=bool)
                    a=np.asarray(r['mas'])[mask]; b=np.asarray(r[variant])[mask]
                    if len(a):
                        pairs.append((a,b));m=pair_metrics(a,b)
                        if m is not None: sentence.append(m)
                if not pairs: continue
                pooled=pair_metrics(np.concatenate([a for a,b in pairs]),np.concatenate([b for a,b in pairs]))
                if pooled is None: continue
                pooled['sentences']=len(pairs)
                pooled['per_sentence']={}
                for key in ('pearson','std_ratio','cv_ratio','mean_ratio'):
                    values=[s[key] for s in sentence if s[key] is not None]
                    pooled['per_sentence'][key]=dict(median=float(np.median(values)),p10=float(np.quantile(values,.1)),p90=float(np.quantile(values,.9))) if values else None
                positive=[(a,b) for a,b in pairs if a.mean()>0 and b.mean()>0]
                pooled['sentence_mean_normalized']=pair_metrics(np.concatenate([a/a.mean() for a,b in positive]),np.concatenate([b/b.mean() for a,b in positive])) if positive else None
                result[subset][variant]=pooled
        tones=[d for r in chosen for d,m in zip(r['mas'],r['tone']) if m]
        result['tone_allocation']=dict(tokens=len(tones),histogram=dict(sorted(Counter(tones).items())),
            mean_frames=float(np.mean(tones)) if tones else None,max_frames=max(tones) if tones else None,
            fraction_gt3=float(np.mean(np.asarray(tones)>3)) if tones else None)
        report[lang]=result
    return report


def main(args):
    import torch
    from lits.models.lits import LITS
    from lits.data.text_mel_datamodule import TextMelBatchCollate
    from lits.text.char_symbols.langs.zh_en_rhyme_body_tone_tokens import symbols,_punctuation
    from lits.text.bopomofo_utils import BOPOMOFO_TONES
    from training.stage2.data import Stage2Audio
    from training.stage2.prepare import digest
    torch.set_num_threads(4);torch.manual_seed(20260914)
    model=LITS.load_from_checkpoint(args.checkpoint,map_location='cpu',weights_only=False).to(args.device).eval().requires_grad_(False)
    # Isolated diagnostic model: preserve the exact production MAS/outlier code;
    # skip unrelated flow work. No optimizer, saving or mutation of training weights.
    model.decoder.compute_loss=lambda x1,**kwargs:(x1.new_zeros(()),None)
    captured={}
    hook=model.encoder.register_forward_hook(lambda m,i,o:captured.update(output=o))
    ds=Stage2Audio(args.data_dir,'val',dict(mel_mean=float(model.mel_mean),mel_std=float(model.mel_std)))
    excluded={'<blank>','<sil>','<unk>','_',*_punctuation,*BOPOMOFO_TONES}
    rows=[];collate=TextMelBatchCollate(2)
    with torch.inference_mode():
        # Individual records preserve the ordering of the frozen validation manifest;
        # collate sorts by length for multi-record batches.
        for i in range(len(ds)):
            batch=collate([ds[i]])
            batch={k:v.to(args.device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
            losses=model(batch['x'],batch['x_lengths'],batch['y'],batch['y_lengths'],spks=batch['spks'],x_tones=batch['x_tones'])
            _,logw,mask=captured['output'];mas=losses[3].sum(-1).unsqueeze(1)
            raw=logw.exp()*mask;ceil=raw.ceil();inf=model._clamp_inference_tone_durations(ceil.clone(),batch['x'],mask)
            supervised,_=model._mask_duration_outliers(batch['x'],mas,mask,batch['spks'],batch['x_tones'])
            source=ds.rows[i];n=int(batch['x_lengths'][0]);ids=batch['x'][0,:n].tolist()
            speech=np.asarray([symbols[j] not in excluded for j in ids]);tone=[symbols[j] in BOPOMOFO_TONES for j in ids]
            targets=mas[0,0,:n].tolist();assert sum(targets)==int(batch['y_lengths'][0])
            rows.append(dict(index=i,audio=source['audio'],text=source['text'],language=source['language'],speaker=source['speaker'],ids=ids,
                symbols=[symbols[j] for j in ids],mas=targets,raw=raw[0,0,:n].tolist(),ceil=ceil[0,0,:n].tolist(),inference=inf[0,0,:n].tolist(),
                all_tokens=[True]*n,speech=speech.tolist(),supervised_speech=(speech & supervised[0,0,:n].bool().cpu().numpy()).tolist(),tone=tone,
                mas_stats={k:float(v) for k,v in losses[4].items()}))
    hook.remove();args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'duration_records.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    report=dict(checkpoint=str(args.checkpoint),checkpoint_sha256=digest(args.checkpoint),validation_manifest_sha256=digest(args.data_dir/'val.jsonl'),
                samples=len(rows),device=args.device,precision='float32',std_ddof=0,
                interpretation='Agreement with model-derived MAS, not independent phonetic alignment accuracy. No fixed Corr/std threshold proves causality.',
                groups=summarize(rows))
    (args.output/'duration_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='complete',samples=len(rows),output=str(args.output))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda');main(p.parse_args())
