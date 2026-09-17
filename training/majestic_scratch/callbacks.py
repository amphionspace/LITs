"""Verify initialization/joint updates and preserve exact evaluation steps."""
import json
from pathlib import Path

import torch
from lightning import Callback
from training.majestic_scratch.config import parameter_digest


class ScratchAudit(Callback):
    def __init__(self, run_dir):
        self.run=Path(run_dir)
        self.plan=json.loads((self.run/'plan.json').read_text())

    def on_fit_start(self, trainer, pl_module):
        p=self.plan; opt=trainer.optimizers[0]
        assert p['mode'] in ('scratch','backbone_init')
        assert bool(p['source_checkpoint'])==(p['mode']=='backbone_init')
        assert trainer.global_step==0 and trainer.ckpt_path is None and not opt.state
        assert trainer.max_steps==p['max_steps']
        if p.get('max_epochs',-1)>0:assert trainer.max_epochs==p['max_epochs']
        assert not pl_module.use_precomputed_durations and pl_module.duration_constrained_mas
        assert pl_module._dur_loss_weight()==pl_module._prior_loss_weight()==1
        assert parameter_digest(pl_module)==p['initial_parameter_sha256']
        assert float(pl_module.mel_mean)==p['data_statistics']['mel_mean']
        assert float(pl_module.mel_std)==p['data_statistics']['mel_std']
        covered=[id(q) for g in opt.param_groups for q in g['params']]
        assert len(covered)==len(set(covered)) and set(covered)=={id(q) for q in pl_module.parameters()}
        if trainer.is_global_zero:
            self.initial={n:q.detach().cpu().clone() for n,q in pl_module.named_parameters()}
            report=dict(status='passed',initialization_verified=True,mode=p['mode'],initial_parameter_sha256=p['initial_parameter_sha256'],
                        checkpoint_weights_loaded=p['mode']=='backbone_init',optimizer_state_empty=True,global_step=0,all_groups_train_from_step0=True,
                        source_checkpoint=p['source_checkpoint'],speaker_rows_seeded_random=True,
                        loss_weights=p['loss_weights'],normalization='source backbone scale' if p['mode']=='backbone_init' else 'combined training data',active_speaker_ids=p['active_speaker_ids'])
            (self.run/'initialization_verification.json').write_text(json.dumps(report,indent=2)+'\n')
        (self.run/'evaluation_checkpoints').mkdir(exist_ok=True)

    def on_train_batch_start(self,trainer,pl_module,batch,batch_idx):
        assert set(batch['spks'].tolist())<=set(self.plan['active_speaker_ids'])
        assert pl_module._dur_loss_weight()==pl_module._prior_loss_weight()==1

    def on_before_optimizer_step(self,trainer,pl_module,optimizer):
        if trainer.global_step<100:
            grad=pl_module.spk_emb.weight.grad
            assert grad is not None and torch.isfinite(grad).all()
            for row in set(range(pl_module.n_spks))-set(self.plan['active_speaker_ids']):
                assert torch.count_nonzero(grad[row])==0

    def on_train_batch_end(self,trainer,pl_module,outputs,batch,batch_idx):
        step=trainer.global_step
        if trainer.is_global_zero and step in (1,100):
            groups={}
            for name,q in pl_module.named_parameters():
                group=('speaker' if name.startswith('spk_emb.') else 'duration' if name.startswith('encoder.proj_w.')
                       else 'prior_encoder' if name.startswith('encoder.') else 'frame_encoder' if name.startswith('decoder.encoder.') else 'flow')
                groups[group]=groups.get(group,0)+int(not torch.equal(q.detach().cpu(),self.initial[name]))
            assert all(v>0 for v in groups.values()), groups
            row_updates=(pl_module.spk_emb.weight.detach().cpu()-self.initial['spk_emb.weight']).norm(dim=1).tolist()
            for row in set(range(pl_module.n_spks))-set(self.plan['active_speaker_ids']): assert row_updates[row]==0
            # A natural length bucket can contain only one speaker. Require both
            # rows to have learned by the startup audit, not in every local batch.
            if step==100: assert all(row_updates[row]>0 for row in self.plan['active_speaker_ids'])
            record=dict(status='passed',global_step=step,updated_tensors=groups,speaker_row_update_norms=row_updates,
                        learning_rates={g['name']:g['lr'] for g in trainer.optimizers[0].param_groups})
            (self.run/('first_step_verification.json' if step==1 else 'startup_verification.json')).write_text(json.dumps(record,indent=2)+'\n')
            if step==100: del self.initial
        # All ranks participate in save_checkpoint barriers. These files are never pruned
        # by the routine ModelCheckpoint callback, so a slower evaluator cannot skip 5k/10k/etc.
        if step==1000 or step%5000==0 or step in self.plan['diagnostic_steps'] or step==trainer.max_steps:
            trainer.save_checkpoint(str(self.run/'evaluation_checkpoints'/f'step_{step:08d}.ckpt'))
