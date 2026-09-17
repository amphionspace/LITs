"""Initialization and exact-step evaluation checks for accumulated iMF updates."""

import torch

from training.majestic_scratch.callbacks import ScratchAudit


class IMFAudit(ScratchAudit):
    def on_fit_start(self, trainer, pl_module):
        super().on_fit_start(trainer, pl_module)
        assert pl_module.decoder.objective == 'imf'
        assert list(pl_module.decoder.sampling_time_grid) == [0.0, 0.5, 1.0]
        assert pl_module.decoder.estimator.interval_time_scale == self.plan['interval_time_scale']
        assert trainer.accumulate_grad_batches == self.plan['accumulate_grad_batches']
        self.last_completed_step = 0

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        super().on_before_optimizer_step(trainer, pl_module, optimizer)
        if trainer.global_step in (0, 9, 99):
            for name, parameter in pl_module.named_parameters():
                if parameter.grad is not None:
                    assert torch.isfinite(parameter.grad).all(), name

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # global_step stays unchanged between accumulated microbatches. Do not
        # repeat startup audits or overwrite the same evaluation checkpoint.
        if trainer.global_step <= self.last_completed_step:
            return
        self.last_completed_step = trainer.global_step
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
