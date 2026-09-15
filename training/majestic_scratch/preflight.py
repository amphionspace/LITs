"""Check the real scratch model, data partition and first joint optimization step."""
import json
import time
from pathlib import Path


def main(run):
    import numpy as np
    import torch
    from lits.data.text_mel_datamodule import TextMelBatchCollate
    from lits.text import text_to_sequence_with_tones
    from training.stage2.data import Stage2Audio
    from training.majestic_scratch.data import NaturalBuckets
    from training.majestic_scratch.config import make_initialized_model,parameter_digest
    torch.set_num_threads(4)
    plan=json.loads((run/'plan.json').read_text()); assert plan['mode'] in ('scratch','backbone_init')
    ds=Stage2Audio(plan['data_dir'],'train',plan['data_statistics'])
    active=set(plan['active_speaker_ids'])
    assert len(ds)==plan['train_rows'] and set(ds.sources)==active
    indices=np.random.default_rng(plan['seed']).choice(len(ds),16,replace=False).tolist()+[int(np.argmax(ds.lengths))]
    indices+= [int(np.flatnonzero(ds.sources==speaker)[0]) for speaker in sorted(active)]
    for i in indices:
        r=ds.rows[i]; ids,tones,phones=text_to_sequence_with_tones(r['text'],['en_zh_dict_mixed_rhyme_body_tone_cleaners'],prepend_sil=True)
        cache=ds.text_cache[r['text']]
        assert ids==cache['tokens'] and tones==cache['tones'] and phones==cache['phonemes']
        actual=ds[i];direct=ds.mel_loader.get_datapoint([r['audio'],str(r['speaker']),r['text']])
        assert torch.equal(actual['x'],direct['x']) and torch.equal(actual['y'],direct['y'])
    partitions=[list(NaturalBuckets(ds,48,rank,4,plan['seed'])) for rank in range(4)]
    batches=len(ds)//192; assert all(len(p)==batches for p in partitions)
    flat=[i for p in partitions for b in p for i in b]
    assert len(flat)==len(set(flat))==batches*192
    other=NaturalBuckets(ds,48,0,4,plan['seed']);other.sampler.set_epoch(1)
    assert list(other)!=partitions[0]
    model,_=make_initialized_model(run,plan)
    assert parameter_digest(model)==plan['initial_parameter_sha256']
    model=model.cuda().train(); optimizer=model.configure_optimizers()
    if isinstance(optimizer,dict): optimizer=optimizer['optimizer']
    assert not optimizer.state and not model.use_precomputed_durations
    assert model._dur_loss_weight()==model._prior_loss_weight()==1
    initial={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
    longest=np.argsort(ds.lengths)[-48:].tolist()
    for speaker in active:
        if not any(ds.sources[i]==speaker for i in longest):
            pool=np.flatnonzero(ds.sources==speaker)
            longest[0]=int(pool[np.argmax(ds.lengths[pool])])
    batch=TextMelBatchCollate(2)([ds[int(i)] for i in longest])
    batch={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
    lr=plan['peak_lr']*(.1+.9/plan['warmup_steps'])
    for group in optimizer.param_groups: group['lr']=lr
    torch.cuda.reset_peak_memory_stats(); started=time.monotonic()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        losses=model(batch['x'],batch['x_lengths'],batch['y'],batch['y_lengths'],spks=batch['spks'],x_tones=batch['x_tones'])
        loss=sum(losses[:3])
    assert torch.isfinite(loss);loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(model.spk_emb.weight.grad[row].norm()>0 for row in active)
    assert all(torch.count_nonzero(model.spk_emb.weight.grad[row])==0 for row in set(range(2))-active)
    torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();torch.cuda.synchronize()
    groups={}
    for name,p in model.named_parameters():
        group=('speaker' if name.startswith('spk_emb.') else 'duration' if name.startswith('encoder.proj_w.')
               else 'prior_encoder' if name.startswith('encoder.') else 'frame_encoder' if name.startswith('decoder.encoder.') else 'flow')
        groups[group]=groups.get(group,0)+int(not torch.equal(p.detach().cpu(),initial[name]))
    assert all(v>0 for v in groups.values()), groups
    row_updates=(model.spk_emb.weight.detach().cpu()-initial['spk_emb.weight']).norm(dim=1).tolist()
    assert all(row_updates[row]>0 for row in active)
    assert all(row_updates[row]==0 for row in set(range(2))-active)
    memory=torch.cuda.max_memory_allocated()/2**30; assert memory<60
    report=dict(status='passed',mode=plan['mode'],active_speakers=sorted(active),initialization_verified=True,
                frontend_loader_equivalence_samples=len(indices),rank_batches=batches,distributed_partition_unique=True,
                all_acoustic_groups_update_at_first_step=True,updated_tensors=groups,speaker_row_update_norms=row_updates,
                loss=float(loss.detach()),peak_allocated_gib=memory,seconds=time.monotonic()-started,
                benchmark_weights_discarded=True,source_checkpoint=plan['source_checkpoint'])
    (run/'preflight.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);main(p.parse_args().run_dir)
