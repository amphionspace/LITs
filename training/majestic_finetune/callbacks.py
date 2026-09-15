"""Embedding-only warmup, followed by low-rate joint acoustic adaptation."""
import json
import math
import torch
from training.stage2.callbacks import AdaptationSchedule


class MajesticSchedule(AdaptationSchedule):
    def __init__(self,run_dir,adaptation_steps=500,warmup_steps=50,total_steps=12000,encoder_warmup_steps=200):
        super().__init__(run_dir,adaptation_steps,warmup_steps,total_steps,encoder_warmup_steps)
        plan=json.loads((self.run_dir/'plan.json').read_text())
        self.peaks=plan['learning_rates']
        self.adaptation_lr=plan['schedule']['embedding_adaptation_lr']
        assert adaptation_steps==500 and plan['active_speaker_ids']==[1]

    def rates(self,step):
        if step < self.adaptation_steps:
            scale=min(1.,.1+.9*(step+1)/self.warmup_steps)
            return {k:self.adaptation_lr*scale if k=='spk_emb' else 0. for k in self.peaks}
        progress=min(1.,(step-self.adaptation_steps)/(self.total_steps-self.adaptation_steps))
        scale=.2+.8*.5*(1+math.cos(math.pi*progress))
        ramp=min(1.,.1+.9*(step-self.adaptation_steps+1)/self.encoder_warmup_steps)
        return {k:v*scale*(1. if k=='spk_emb' else ramp) for k,v in self.peaks.items()}

    def on_before_optimizer_step(self,trainer,pl_module,optimizer):
        grad=pl_module.spk_emb.weight.grad
        if grad is not None:
            assert torch.count_nonzero(grad[0])==0, 'Unexpected supervision of reserved row 0'
            if trainer.is_global_zero and torch.isfinite(grad[1]).all() and grad[1].norm()>0:
                self.speaker_gradient_seen[1]=True
        if trainer.global_step < self.adaptation_steps:
            for group in optimizer.param_groups:
                if group['name']!='spk_emb':
                    for parameter in group['params']:parameter.grad=None

    def on_train_batch_end(self,trainer,pl_module,outputs,batch,batch_idx):
        if not trainer.is_global_zero or trainer.global_step not in (1,100,500,510) or not hasattr(self,'initial'):
            return
        changes={}
        for name,p in pl_module.named_parameters():
            diff=p.detach().cpu()-self.initial[name]
            assert torch.isfinite(diff).all()
            group=('spk_emb' if name.startswith('spk_emb.') else
                   'duration_predictor' if name.startswith('encoder.proj_w.') else
                   'prior_encoder' if name.startswith('encoder.') else
                   'frame_encoder' if name.startswith('decoder.encoder.') else 'flow_estimator')
            entry=changes.setdefault(group,dict(tensors=0,updated_tensors=0,squared_update_norm=0.))
            entry['tensors']+=1
            entry['updated_tensors']+=int(torch.count_nonzero(diff)>0)
            entry['squared_update_norm']+=float(diff.square().sum())
        rows=(pl_module.spk_emb.weight.detach().cpu()-self.initial['spk_emb.weight']).norm(dim=1).tolist()
        assert rows[0]==0 and rows[1]>0 and self.speaker_gradient_seen[1]
        frozen=trainer.global_step<=self.adaptation_steps
        for group in ['prior_encoder','duration_predictor','frame_encoder','flow_estimator']:
            assert (changes[group]['updated_tensors']==0) if frozen else (changes[group]['updated_tensors']>0)
        if frozen:
            for group in trainer.optimizers[0].param_groups:
                if group['name']!='spk_emb':
                    assert all(p not in trainer.optimizers[0].state for p in group['params'])
        record=dict(passed=True,global_step=trainer.global_step,phase='embedding_only' if frozen else 'joint_finetuning',
                    parameter_updates=changes,speaker_row_update_norms=rows,reserved_row0_unchanged=True,
                    target_row1_received_gradient=True,
                    group_learning_rates={g['name']:g['lr'] for g in trainer.optimizers[0].param_groups})
        name={1:'first_step_verification.json',100:'startup_verification.json',500:'embedding_phase_verification.json',510:'unfreeze_verification.json'}[trainer.global_step]
        (self.run_dir/name).write_text(json.dumps(record,indent=2)+'\n')
        print('[MAJESTIC_VERIFICATION] '+json.dumps(record),flush=True)
        if trainer.global_step==510:del self.initial
