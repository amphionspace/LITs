"""Reuse audited speaker-balanced manifests with cached text and length buckets."""
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from lits.data.text_mel_datamodule import TextMelBatchCollate, TextMelDataset
from training.foundation.data import BilingualBuckets, worker_init


class Stage2Audio(Dataset):
    def __init__(self, data_dir, split, statistics):
        data = Path(data_dir)
        self.rows = [json.loads(line) for line in (data/f'{split}.jsonl').read_text().splitlines()]
        self.text_cache = {r['text']:r for r in map(json.loads,(data/'text_preflight.jsonl').read_text().splitlines())}
        units = {r['key']:r for r in map(json.loads,(data/'mel_statistics_units.jsonl').read_text().splitlines())}
        lengths = []
        for row in self.rows:
            key = json.dumps([row['audio'],row.get('start'),row.get('end')],separators=(',',':'))
            if key in units:
                lengths.append(units[key]['audio_frames']/24000)
            else:
                info = sf.info(row['audio'])
                assert info.samplerate == 24000 and info.channels == 1
                lengths.append(row['end']-row['start'] if 'start' in row else info.duration)
            text = self.text_cache[row['text']]
            assert text['tokens'] == row['text_key']
            assert text['tones'] is None or len(text['tones']) == len(text['tokens'])
        self.lengths = np.asarray(lengths,dtype=np.float32)
        self.sources = np.asarray([r['speaker'] for r in self.rows],dtype=np.int8)
        # Delegate audio slicing and Mel extraction to the already validated loader.
        self.mel_loader = TextMelDataset(str(data/f'{split}.txt'),2,
            ['en_zh_dict_mixed_rhyme_body_tone_cleaners'],False,True,2048,100,24000,384,1536,0,12000,statistics,20260913,False)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self,index):
        row = self.rows[index]
        cached = self.text_cache[row['text']]
        mel = self.mel_loader.get_mel(row['audio'],start=row.get('start'),end=row.get('end'))
        if not torch.isfinite(mel).all() or len(cached['tokens']) > mel.shape[-1]:
            raise ValueError(f'Invalid Mel or MAS length: {row["audio"]}')
        return dict(x=torch.IntTensor(cached['tokens']),
            x_tones=torch.IntTensor(cached['tones']) if cached['tones'] is not None else None,
            y=mel,spk=row['speaker'],filepath=row['audio'],x_text=cached['phonemes'],durations=None)


class Stage2DataModule(LightningDataModule):
    def __init__(self,data_dir,**kwargs):
        super().__init__()
        self.save_hyperparameters(dict(data_dir=data_dir,**kwargs),logger=False)

    def setup(self,stage=None):
        self.trainset = Stage2Audio(self.hparams.data_dir,'train',self.hparams.data_statistics)
        self.validset = Stage2Audio(self.hparams.data_dir,'val',self.hparams.data_statistics)

    def train_dataloader(self):
        sampler = BilingualBuckets(self.trainset,self.hparams.batch_size,self.trainer.global_rank,self.trainer.world_size,self.hparams.seed)
        return DataLoader(self.trainset,batch_sampler=sampler,num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.num_workers>0,pin_memory=True,worker_init_fn=worker_init,
            collate_fn=TextMelBatchCollate(2))

    def val_dataloader(self):
        indices = list(range(self.trainer.global_rank,len(self.validset),self.trainer.world_size))
        return DataLoader(self.validset,batch_size=16,sampler=indices,num_workers=2,
            persistent_workers=True,pin_memory=True,worker_init_fn=worker_init,collate_fn=TextMelBatchCollate(2))
