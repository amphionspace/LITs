"""Warmup, stable peak, then linear decay; steps are optimizer updates."""
from lightning import Callback


class WarmupStableDecay(Callback):
    def __init__(self, warmup_steps, total_steps, peak_lr, final_lr, decay_fraction=0.2):
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        self.peak_lr = float(peak_lr)
        self.final_lr = float(final_lr)
        self.decay_steps = max(1, int(self.total_steps * decay_fraction))
        self.decay_start = self.total_steps - self.decay_steps
        if not (0 < decay_fraction < 1 and 0 < self.warmup_steps < self.decay_start):
            raise ValueError('WSD requires nonempty warmup, stable and decay stages')
        if not 0 <= self.final_lr <= self.peak_lr:
            raise ValueError('Expected 0 <= final_lr <= peak_lr')

    def lr_at(self, step):
        if step < self.warmup_steps:
            return self.peak_lr * (0.1 + 0.9 * (step + 1) / self.warmup_steps)
        if step < self.decay_start:
            return self.peak_lr
        progress = min(1.0, (step - self.decay_start + 1) / self.decay_steps)
        return self.peak_lr + (self.final_lr - self.peak_lr) * progress

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        lr = self.lr_at(trainer.global_step)
        for optimizer in trainer.optimizers:
            for group in optimizer.param_groups:
                group['lr'] = lr
