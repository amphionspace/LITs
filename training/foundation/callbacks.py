import math
from lightning import Callback

class WarmupCosine(Callback):
    def __init__(self, warmup_steps=1000, total_steps=200000, peak_lr=1e-4, final_lr=2e-5):
        self.warmup_steps=warmup_steps
        self.total_steps=total_steps
        self.peak_lr=peak_lr
        self.final_lr=final_lr

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        step=trainer.global_step
        if step < self.warmup_steps:
            lr=self.peak_lr*(0.1+0.9*(step+1)/self.warmup_steps)
        else:
            progress=min(1.,(step-self.warmup_steps)/(self.total_steps-self.warmup_steps))
            lr=self.final_lr+(self.peak_lr-self.final_lr)*0.5*(1+math.cos(math.pi*progress))
        for optimizer in trainer.optimizers:
            for group in optimizer.param_groups:
                group['lr']=lr
