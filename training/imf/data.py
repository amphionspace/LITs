"""Natural Stage 2 sampling with a capacity-bounded validation batch."""

from torch.utils.data import DataLoader

from lits.data.text_mel_datamodule import TextMelBatchCollate
from training.foundation.data import worker_init
from training.majestic_scratch.data import ScratchDataModule


class IMFDataModule(ScratchDataModule):
    def val_dataloader(self):
        indices = range(self.trainer.global_rank, len(self.validset), self.trainer.world_size)
        workers = min(2, self.hparams.num_workers)
        return DataLoader(
            self.validset, batch_size=min(16, self.hparams.batch_size), sampler=indices,
            num_workers=workers, persistent_workers=workers > 0, pin_memory=True,
            worker_init_fn=worker_init, collate_fn=TextMelBatchCollate(2),
        )
