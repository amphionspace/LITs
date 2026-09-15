"""Natural unique-utterance sampling with global duration buckets."""
import numpy as np
from torch.utils.data import DataLoader
from lits.data.text_mel_datamodule import TextMelBatchCollate
from training.foundation.data import BilingualBuckets,worker_init
from training.stage2.data import Stage2DataModule


class NaturalBuckets(BilingualBuckets):
    def __init__(self,dataset,batch_size,rank,world_size,seed=20260913):
        super().__init__(dataset,batch_size,rank,world_size,seed)
        # One sampling pool, while every row retains its real speaker condition.
        self.pools=[np.arange(len(dataset),dtype=np.int64)]
        self.per_source=len(dataset)


class ScratchDataModule(Stage2DataModule):
    def train_dataloader(self):
        sampler=NaturalBuckets(self.trainset,self.hparams.batch_size,self.trainer.global_rank,self.trainer.world_size,self.hparams.seed)
        return DataLoader(self.trainset,batch_sampler=sampler,num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.num_workers>0,pin_memory=True,worker_init_fn=worker_init,collate_fn=TextMelBatchCollate(2))
