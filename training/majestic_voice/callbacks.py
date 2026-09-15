"""Persist compact training/validation progress alongside Lightning checkpoints."""
import json
import math
import time
from pathlib import Path

from lightning import Callback


class TrainingState(Callback):
    def _write(self, trainer, status):
        if not trainer.is_global_zero:
            return
        metrics = {}
        for key, value in trainer.callback_metrics.items():
            if 'loss' in key and hasattr(value, 'numel') and value.numel() == 1:
                number = float(value.detach().cpu())
                if not math.isfinite(number):
                    raise RuntimeError(f'Nonfinite training metric: {key}={number}')
                metrics[key] = number
        row = {'status': status, 'global_step': trainer.global_step, 'epoch': trainer.current_epoch,
               'updated_at_unix': time.time(), 'metrics': metrics}
        output = Path(trainer.default_root_dir)
        temporary = output / 'training_state.json.tmp'
        temporary.write_text(json.dumps(row, indent=2) + '\n')
        temporary.replace(output / 'training_state.json')
        return row

    def on_train_start(self, trainer, pl_module):
        self.last_step = -1
        self._write(trainer, 'training')

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step != self.last_step and trainer.global_step % 10 == 0:
            self.last_step = trainer.global_step
            row = self._write(trainer, 'training')
            if row is not None:
                print('[TRAIN_STATE] ' + json.dumps(row), flush=True)

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        row = self._write(trainer, 'training')
        if row is not None:
            with (Path(trainer.default_root_dir) / 'validation.jsonl').open('a') as stream:
                stream.write(json.dumps(row) + '\n')

    def on_fit_end(self, trainer, pl_module):
        # All DDP ranks participate in Lightning's save barrier.
        trainer.save_checkpoint(str(Path(trainer.default_root_dir) / 'checkpoints/final.ckpt'))
        self._write(trainer, 'complete')
