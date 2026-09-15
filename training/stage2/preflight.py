"""Exercise text/audio equivalence, speaker gradients, update schedule and capacity."""
import argparse
import json
from pathlib import Path
import time


def main(run):
    import numpy as np
    import torch
    from lits.models.lits import LITS
    from lits.data.text_mel_datamodule import TextMelBatchCollate
    from lits.text import text_to_sequence_with_tones
    from training.foundation.data import BilingualBuckets
    from training.stage2.data import Stage2Audio
    from training.stage2.callbacks import AdaptationSchedule

    torch.set_num_threads(4)
    torch.manual_seed(20260913)
    plan=json.loads((run/'plan.json').read_text())
    ds=Stage2Audio(plan['data_dir'],'train',plan['data_statistics'])
    rng=np.random.default_rng(20260913)
    selected=rng.choice(len(ds),size=96,replace=False).tolist()
    selected += [int(np.flatnonzero(ds.sources==s)[np.argmax(ds.lengths[ds.sources==s])]) for s in [0,1]]
    for index in selected:
        row=ds.rows[index]
        ids,tones,phonemes=text_to_sequence_with_tones(row['text'],['en_zh_dict_mixed_rhyme_body_tone_cleaners'],prepend_sil=True)
        cached=ds.text_cache[row['text']]
        assert ids==cached['tokens'] and tones==cached['tones'] and phonemes==cached['phonemes']
        direct=ds.mel_loader.get_datapoint([row['audio'],str(row['speaker']),*([str(row['start']),str(row['end'])] if 'start' in row else []),row['text']])
        actual=ds[index]
        assert torch.equal(direct['x'],actual['x']) and torch.equal(direct['y'],actual['y'])
    rank_batches=[list(BilingualBuckets(ds,48,rank,4,20260913)) for rank in range(4)]
    assert len({len(b) for b in rank_batches})==1
    all_indices=[i for b in rank_batches for batch in b for i in batch]
    assert len(all_indices)==len(set(all_indices)) and len(all_indices)==len(rank_batches[0])*192
    assert abs(sum(ds.sources[i]==0 for i in all_indices)-sum(ds.sources[i]==1 for i in all_indices)) <= 192
    schedule=AdaptationSchedule(str(run),total_steps=plan['max_steps'])
    assert schedule.rates(0)['prior_encoder']==schedule.rates(schedule.adaptation_steps-1)['duration_predictor']==0
    assert schedule.rates(schedule.adaptation_steps)['prior_encoder']>0 and schedule.rates(schedule.adaptation_steps)['duration_predictor']>0
    assert abs(schedule.rates(schedule.total_steps)['decoder']-4e-6)<1e-12
    model=LITS.load_from_checkpoint(run/'initialization.ckpt',map_location='cpu',weights_only=False).cuda().train()
    optimizer=model.configure_optimizers()
    if isinstance(optimizer,dict):optimizer=optimizer['optimizer']
    before={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
    # A 48-example batch of the longest utterances is conservative for both phases.
    longest=np.argsort(ds.lengths)[-48:]
    examples=[ds[int(i)] for i in longest]
    batch=TextMelBatchCollate(2)(examples)
    batch={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
    peak=[]
    for step in [0,schedule.adaptation_steps]:
        torch.cuda.reset_peak_memory_stats()
        started=time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        for g in optimizer.param_groups:g['lr']=schedule.rates(step)[g['name']]
        with torch.autocast('cuda',dtype=torch.bfloat16):
            outputs=model(batch['x'],batch['x_lengths'],batch['y'],batch['y_lengths'],spks=batch['spks'],x_tones=batch['x_tones'])
            loss=sum(outputs[:3])
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        if step==0:
            for g in optimizer.param_groups:
                if g['name'] in ('prior_encoder','duration_predictor'):
                    for p in g['params']:p.grad=None
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        optimizer.step()
        if step==0:
            assert all(torch.equal(p.detach().cpu(),before[n]) for n,p in model.named_parameters() if n.startswith('encoder.'))
        else:
            assert any(not torch.equal(p.detach().cpu(),before[n]) for n,p in model.named_parameters() if n.startswith('encoder.') and not n.startswith('encoder.proj_w.'))
        torch.cuda.synchronize()
        peak.append(dict(phase_step=step,loss=float(loss.detach()),peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            seconds=time.monotonic()-started,mel_frames=batch['y'].shape[-1],max_tokens=batch['x'].shape[-1]))
        assert peak[-1]['peak_allocated_gib']<60
        print(json.dumps(peak[-1]),flush=True)
    # Explicitly include both speakers, since the longest-duration bucket can be one-sided.
    del outputs,loss,batch
    optimizer.zero_grad(set_to_none=True)
    examples=[ds[int(i)] for s in [0,1] for i in np.flatnonzero(ds.sources==s)[:4]]
    batch=TextMelBatchCollate(2)(examples)
    batch={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
    with torch.autocast('cuda',dtype=torch.bfloat16):
        loss=sum(model(batch['x'],batch['x_lengths'],batch['y'],batch['y_lengths'],spks=batch['spks'],x_tones=batch['x_tones'])[:3])
    loss.backward()
    grads=model.spk_emb.weight.grad.norm(dim=1).detach().cpu().tolist()
    assert all(v>0 and np.isfinite(v) for v in grads)
    result=dict(status='passed',text_and_loader_equivalence_samples=len(selected),rank_batches=len(rank_batches[0]),
        distributed_bucket_partition_passed=True,batch_size_per_gpu=48,capacity=peak,speaker_row_gradient_norms=grads,
        frozen_update_and_unfreeze_checks_passed=True,benchmark_weights_discarded=True,
        schedule_total_steps=schedule.total_steps,schedule_final_rates=schedule.rates(schedule.total_steps))
    (run/'preflight.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,required=True)
    main(parser.parse_args().run_dir)
