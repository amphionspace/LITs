"""Stage-two learning rates and auditable parameter-update checks."""
import json
import math
from pathlib import Path

import torch
from lightning import Callback


class AdaptationSchedule(Callback):
    def __init__(self,run_dir,adaptation_steps=500,warmup_steps=200,total_steps=6000,encoder_warmup_steps=500):
        self.run_dir=Path(run_dir)
        self.adaptation_steps=adaptation_steps
        self.warmup_steps=warmup_steps
        self.total_steps=total_steps
        self.encoder_warmup_steps=encoder_warmup_steps
        self.peaks=dict(prior_encoder=2e-6,duration_predictor=5e-6,spk_emb=1e-4,decoder=2e-5)
        self.speaker_gradient_seen=[False,False]

    def rates(self,step):
        if step < self.warmup_steps:
            scale=.1+.9*(step+1)/self.warmup_steps
        else:
            progress=min(1.,(step-self.warmup_steps)/(self.total_steps-self.warmup_steps))
            scale=.2+.8*.5*(1+math.cos(math.pi*progress))
        rates={key:value*scale for key,value in self.peaks.items()}
        for key in ['prior_encoder','duration_predictor']:
            rates[key] *= max(0.,min(1.,(step-self.adaptation_steps+1)/self.encoder_warmup_steps)) if step >= self.adaptation_steps else 0.
        return rates

    def on_fit_start(self,trainer,pl_module):
        optimizer=trainer.optimizers[0]
        assert {g['name'] for g in optimizer.param_groups}==set(self.peaks)
        expected={id(p) for p in pl_module.parameters()}
        actual=[id(p) for g in optimizer.param_groups for p in g['params']]
        assert len(actual)==len(set(actual)) and set(actual)==expected
        plan=json.loads((self.run_dir/'plan.json').read_text())
        assert self.total_steps == trainer.max_steps == plan['max_steps']
        assert self.peaks == plan['learning_rates']
        assert self.adaptation_steps == plan['schedule']['encoder_update_start_step']
        assert self.warmup_steps == plan['schedule']['warmup_steps']
        assert self.encoder_warmup_steps == plan['schedule']['encoder_warmup_steps']
        assert float(pl_module.mel_mean)==plan['data_statistics']['mel_mean']
        assert float(pl_module.mel_std)==plan['data_statistics']['mel_std']
        if trainer.global_step==0 and trainer.is_global_zero:
            initial=torch.load(self.run_dir/'initialization.ckpt',map_location='cpu',weights_only=False)['state_dict']
            state=pl_module.state_dict()
            assert state.keys()==initial.keys()
            assert all(torch.equal(value.detach().cpu(),initial[key]) for key,value in state.items())
            assert not optimizer.state
            self.initial={name:p.detach().cpu().clone() for name,p in pl_module.named_parameters()}
            record=dict(weights_exactly_match_initialization=True,optimizer_state_empty=True,
                optimizer_exact_parameter_coverage=True,parameter_tensors=len(actual),
                speaker_rows_equal_at_start=bool(torch.equal(state['spk_emb.weight'][0],state['spk_emb.weight'][1])),
                normalization_matches_source=True,source_step=plan['source_global_step'],
                trainer_max_steps=trainer.max_steps,schedule_total_steps=self.total_steps,
                peak_learning_rates=self.peaks,final_learning_rates=self.rates(self.total_steps))
            (self.run_dir/'initialization_verification.json').write_text(json.dumps(record,indent=2)+'\n')

    def on_train_batch_start(self,trainer,pl_module,batch,batch_idx):
        rates=self.rates(trainer.global_step)
        for group in trainer.optimizers[0].param_groups:
            group['lr']=rates[group['name']]

    def on_before_optimizer_step(self,trainer,pl_module,optimizer):
        # Keep the DDP graph stable. Suppress these updates after gradient reduction,
        # so frozen parameters accumulate neither Adam moments nor weight decay.
        if trainer.is_global_zero and trainer.global_step < 100:
            grad=pl_module.spk_emb.weight.grad
            if grad is not None:
                for i in [0,1]:
                    self.speaker_gradient_seen[i] |= bool(torch.isfinite(grad[i]).all() and grad[i].norm()>0)
        if trainer.global_step < self.adaptation_steps:
            for group in optimizer.param_groups:
                if group['name'] in ('prior_encoder','duration_predictor'):
                    for parameter in group['params']:
                        parameter.grad=None

    def on_train_batch_end(self,trainer,pl_module,outputs,batch,batch_idx):
        if not trainer.is_global_zero or trainer.global_step not in (100,self.adaptation_steps+10) or not hasattr(self,'initial'):
            return
        changes={}
        for name,p in pl_module.named_parameters():
            difference=p.detach().cpu()-self.initial[name]
            assert torch.isfinite(difference).all()
            group='spk_emb' if name.startswith('spk_emb.') else 'duration_predictor' if name.startswith('encoder.proj_w.') else 'prior_encoder' if name.startswith('encoder.') else 'decoder'
            entry=changes.setdefault(group,dict(tensors=0,updated_tensors=0,squared_update_norm=0.))
            entry['tensors']+=1
            entry['updated_tensors']+=int(torch.count_nonzero(difference)>0)
            entry['squared_update_norm']+=float(difference.square().sum())
        embedding=pl_module.spk_emb.weight.detach().cpu()
        row_changes=(embedding-self.initial['spk_emb.weight']).norm(dim=1).tolist()
        assert all(v>0 for v in row_changes) and all(self.speaker_gradient_seen)
        assert changes['decoder']['updated_tensors']>0
        if trainer.global_step==100:
            assert changes['prior_encoder']['updated_tensors']==changes['duration_predictor']['updated_tensors']==0
        else:
            assert changes['prior_encoder']['updated_tensors']>0 and changes['duration_predictor']['updated_tensors']>0
        record=dict(global_step=trainer.global_step,parameter_updates=changes,speaker_row_update_norms=row_changes,
            both_speaker_rows_received_gradient=all(self.speaker_gradient_seen),speaker_row_distance=float((embedding[0]-embedding[1]).norm()),
            group_learning_rates={g['name']:g['lr'] for g in trainer.optimizers[0].param_groups},
            phase='voice_adaptation' if trainer.global_step < self.adaptation_steps else 'joint_finetuning',passed=True)
        name='startup_verification.json' if trainer.global_step==100 else 'unfreeze_verification.json'
        (self.run_dir/name).write_text(json.dumps(record,indent=2)+'\n')
        print('[STAGE2_VERIFICATION] '+json.dumps(record),flush=True)
        if trainer.global_step==self.adaptation_steps+10:
            del self.initial
