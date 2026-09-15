"""Pretokenized audio data and equal-rank bilingual length buckets."""
import json
import math
import os
import sqlite3
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Sampler

from lits.data.text_mel_datamodule import TextMelBatchCollate
from lits.utils.audio import mel_spectrogram
from lits.utils.model import normalize


class IndexedAudio(Dataset):
    def __init__(self, db_path, split, statistics, max_per_source=None):
        self.db_path = str(db_path)
        self.statistics = statistics
        self.connection = None
        with sqlite3.connect(f'file:{self.db_path}?mode=ro', uri=True) as db:
            rows = db.execute('select id,duration,source from samples where split=? order by id', (split,)).fetchall()
        if max_per_source:
            rng = np.random.default_rng(20260911)
            selected = []
            for source in sorted({r[2] for r in rows}):
                pool = [r for r in rows if r[2] == source]
                indices = rng.permutation(len(pool))[:max_per_source]
                selected.extend(pool[int(i)] for i in indices)
            rows = selected
        self.ids = np.array([r[0] for r in rows], dtype=np.int64)
        self.lengths = np.array([r[1] for r in rows], dtype=np.float32)
        self.sources = np.array([r[2] == 'Premium' for r in rows], dtype=np.int8)
        self.owner_pid = None

    def row(self, index):
        if self.connection is None or self.owner_pid != os.getpid():
            self.connection = sqlite3.connect(f'file:{self.db_path}?mode=ro&immutable=1', uri=True)
            self.owner_pid = os.getpid()
        return json.loads(self.connection.execute('select payload from samples where id=?', (int(self.ids[index]),)).fetchone()[0])

    def __getitem__(self, index):
        r = self.row(index)
        audio, rate = sf.read(r['audio'], dtype='float32')
        if rate != 24000:
            audio = soxr.resample(audio, rate, 24000, quality='HQ')
        if not np.isfinite(audio).all():
            raise ValueError(f"Nonfinite audio: {r['audio']}")
        mel = mel_spectrogram(torch.from_numpy(audio)[None], 2048, 100, 24000, 384, 1536, 0, 12000).squeeze(0)
        if len(r['ids']) > mel.shape[-1]:
            raise ValueError(f"Infeasible MAS lengths: {r['audio']}")
        return {'x': torch.IntTensor(r['ids']), 'x_tones': torch.IntTensor(r['tones']) if r['tones'] is not None else None,
                'y': normalize(mel, self.statistics['mel_mean'], self.statistics['mel_std']),
                'spk': r['speaker'], 'filepath': r['audio'], 'x_text': r['phonemes'], 'durations': None}

    def __len__(self):
        return len(self.ids)


class EpochSeed:
    def __init__(self):
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch


class BilingualBuckets(Sampler):
    """Balance source counts, bucket globally, then split each batch over ranks."""
    def __init__(self, dataset, batch_size, rank, world_size, seed=20260911):
        self.dataset = dataset
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.sampler = EpochSeed()  # Lightning propagates its epoch here.
        self.pools = [np.flatnonzero(dataset.sources == s) for s in np.unique(dataset.sources)]
        self.per_source = max(map(len, self.pools))

    def __len__(self):
        return self.per_source * len(self.pools) // (self.batch_size * self.world_size)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.sampler.epoch)
        balanced = []
        for pool in self.pools:
            pieces = [rng.permutation(pool) for _ in range(math.ceil(self.per_source / len(pool)))]
            balanced.append(np.concatenate(pieces)[:self.per_source])
        indices = rng.permutation(np.concatenate(balanced))
        global_batch = self.batch_size * self.world_size
        batches = []
        window = global_batch * 50
        for start in range(0, len(indices), window):
            bucket = indices[start:start+window]
            bucket = bucket[np.argsort(self.dataset.lengths[bucket], kind='stable')]
            for offset in range(0, len(bucket)-global_batch+1, global_batch):
                batches.append(bucket[offset:offset+global_batch])
        for i in rng.permutation(len(batches)):
            batch = rng.permutation(batches[int(i)])
            start = self.rank * self.batch_size
            yield batch[start:start+self.batch_size].tolist()


def worker_init(_):
    torch.set_num_threads(1)


class FoundationDataModule(LightningDataModule):
    def __init__(self, db_path, validation_per_source=256, **kwargs):
        super().__init__()
        self.save_hyperparameters({'db_path': db_path, 'validation_per_source': validation_per_source, **kwargs}, logger=False)

    def setup(self, stage=None):
        self.trainset = IndexedAudio(self.hparams.db_path, 'train', self.hparams.data_statistics)
        self.validset = IndexedAudio(self.hparams.db_path, 'val', self.hparams.data_statistics, self.hparams.validation_per_source)

    def train_dataloader(self):
        sampler = BilingualBuckets(self.trainset, self.hparams.batch_size, self.trainer.global_rank, self.trainer.world_size, self.hparams.seed)
        return DataLoader(self.trainset, batch_sampler=sampler, num_workers=self.hparams.num_workers,
                          persistent_workers=self.hparams.num_workers > 0, pin_memory=True,
                          worker_init_fn=worker_init, collate_fn=TextMelBatchCollate(2))

    def val_dataloader(self):
        # Explicit, nonoverlapping rank partition; 512 samples divides evenly over 4 ranks.
        indices = list(range(self.trainer.global_rank, len(self.validset), self.trainer.world_size))
        return DataLoader(self.validset, batch_size=16, sampler=indices, num_workers=2,
                          persistent_workers=True, pin_memory=True, worker_init_fn=worker_init,
                          collate_fn=TextMelBatchCollate(2))
