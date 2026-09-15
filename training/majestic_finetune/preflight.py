"""Check the single-speaker partition and both real optimization phases."""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace


def main(run):
    if json.loads((run/'plan.json').read_text()).get('mode') in ('scratch','backbone_init'):
        from training.majestic_scratch.preflight import main as scratch_preflight
        return scratch_preflight(run)
    import numpy as np
    import torch
    from lits.models.lits import LITS
    from lits.data.text_mel_datamodule import TextMelBatchCollate
    from lits.text import text_to_sequence_with_tones
    from training.stage2.data import Stage2Audio
    from training.foundation.data import BilingualBuckets
    from training.majestic_finetune.callbacks import MajesticSchedule
    torch.set_num_threads(4);torch.manual_seed(20260913)
    plan=json.loads((run/'plan.json').read_text())
    ds=Stage2Audio(plan['data_dir'],'train',plan['data_statistics'])
    assert len(ds)==plan['train_rows'] and set(ds.sources)=={1}
    indices=np.random.default_rng(20260913).choice(len(ds),16,replace=False).tolist()+[int(np.argmax(ds.lengths))]
    for i in indices:
        row=ds.rows[i];ids,tones,phones=text_to_sequence_with_tones(row['text'],['en_zh_dict_mixed_rhyme_body_tone_cleaners'],prepend_sil=True)
        cache=ds.text_cache[row['text']]
        assert ids==cache['tokens'] and tones==cache['tones'] and phones==cache['phonemes']
        actual=ds[i];direct=ds.mel_loader.get_datapoint([row['audio'],'1',row['text']])
        assert torch.equal(actual['x'],direct['x']) and torch.equal(actual['y'],direct['y'])
    partitions=[list(BilingualBuckets(ds,48,rank,4,20260913)) for rank in range(4)]
    expected_batches=len(ds)//192
    assert all(len(p)==expected_batches for p in partitions)
    flat=[i for p in partitions for b in p for i in b]
    assert len(flat)==len(set(flat))==expected_batches*192 and all(ds.sources[i]==1 for i in flat)
    changed_sampler=BilingualBuckets(ds,48,0,4,20260913);changed_sampler.sampler.set_epoch(1)
    assert list(changed_sampler)!=partitions[0]
    schedule=MajesticSchedule(run,total_steps=plan['max_steps'])
    for step in [0,49,100,499]:
        rates=schedule.rates(step)
        assert rates['spk_emb']>0 and all(v==0 for k,v in rates.items() if k!='spk_emb')
    assert schedule.rates(49)['spk_emb']==5e-4
    assert all(v>0 for v in schedule.rates(500).values())
    assert schedule.rates(500)['spk_emb']==1e-4
    assert all(abs(schedule.rates(plan['max_steps'])[k]-.2*v)<1e-12 for k,v in schedule.peaks.items())
    model=LITS.load_from_checkpoint(run/'initialization.ckpt',map_location='cpu',weights_only=False).cuda().train()
    optimizer=model.configure_optimizers()
    if isinstance(optimizer,dict):optimizer=optimizer['optimizer']
    initial={k:p.detach().cpu().clone() for k,p in model.named_parameters()}
    batch=TextMelBatchCollate(2)([ds[int(i)] for i in np.argsort(ds.lengths)[-48:]])
    batch={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
    records=[]
    for step in [0,500]:
        optimizer.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();started=time.monotonic()
        for group in optimizer.param_groups:group['lr']=schedule.rates(step)[group['name']]
        with torch.autocast('cuda',dtype=torch.bfloat16):
            loss=sum(model(batch['x'],batch['x_lengths'],batch['y'],batch['y_lengths'],spks=batch['spks'],x_tones=batch['x_tones'])[:3])
        assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        schedule.on_before_optimizer_step(SimpleNamespace(global_step=step,is_global_zero=True),model,optimizer)
        assert model.spk_emb.weight.grad[1].norm()>0
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
        groups={}
        for name,p in model.named_parameters():
            group='speaker' if name.startswith('spk_emb.') else 'duration' if name.startswith('encoder.proj_w.') else 'text_encoder' if name.startswith('encoder.') else 'frame_encoder' if name.startswith('decoder.encoder.') else 'flow'
            groups[group]=groups.get(group,0)+int(not torch.equal(p.detach().cpu(),initial[name]))
        assert torch.equal(model.spk_emb.weight[0].detach().cpu(),initial['spk_emb.weight'][0])
        assert groups['speaker']==1
        if step==0:
            assert all(n==0 for k,n in groups.items() if k!='speaker')
            assert all(p not in optimizer.state for group in optimizer.param_groups if group['name']!='spk_emb' for p in group['params'])
        else:assert all(n>0 for n in groups.values())
        torch.cuda.synchronize()
        record=dict(step=step,loss=float(loss.detach()),updated_tensors=groups,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,seconds=time.monotonic()-started)
        assert record['peak_allocated_gib']<60;records.append(record);print(json.dumps(record),flush=True)
    result=dict(status='passed',active_speakers=[1],frontend_loader_equivalence_samples=len(indices),rank_batches=expected_batches,
        distributed_partition_unique=True,epoch_shuffle_verified=True,phase_checks=records,
        row0_unchanged=True,only_embedding_updates_before_500=True,all_acoustic_groups_update_after_500=True,
        frozen_optimizer_state_empty=True,benchmark_weights_discarded=True,schedule_total_steps=plan['max_steps'])
    (run/'preflight.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);main(p.parse_args().run_dir)
