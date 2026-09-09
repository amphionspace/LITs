import random
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import soundfile as sf
from lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader

from lits.text import text_to_sequence, text_to_sequence_with_tones
from lits.utils.audio import mel_spectrogram
from lits.utils.model import fix_len_compatibility, normalize
from lits.utils.utils import intersperse


def parse_filelist(filelist_path, split_char="|"):
    with open(filelist_path, encoding="utf-8") as f:
        filepaths_and_text = [line.strip().split(split_char) for line in f]
    return filepaths_and_text


class TextMelDataModule(LightningDataModule):
    def __init__(  # pylint: disable=unused-argument
        self,
        name,
        train_filelist_path,
        valid_filelist_path,
        batch_size,
        num_workers,
        pin_memory,
        cleaners,
        add_blank,
        prepend_sil,
        n_spks,
        n_fft,
        n_feats,
        sample_rate,
        hop_length,
        win_length,
        f_min,
        f_max,
        data_statistics,
        seed,
        load_durations,
        sanity_check_samples=5,
        text_normalization=True,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

    def setup(self, stage: Optional[str] = None):  # pylint: disable=unused-argument
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # TN module is not used in the pinyin pipeline.

        # load and split datasets only if not loaded already
        self.trainset = TextMelDataset(  # pylint: disable=attribute-defined-outside-init
            self.hparams.train_filelist_path,
            self.hparams.n_spks,
            self.hparams.cleaners,
            self.hparams.add_blank,
            self.hparams.prepend_sil,
            self.hparams.n_fft,
            self.hparams.n_feats,
            self.hparams.sample_rate,
            self.hparams.hop_length,
            self.hparams.win_length,
            self.hparams.f_min,
            self.hparams.f_max,
            self.hparams.data_statistics,
            self.hparams.seed,
            self.hparams.load_durations,
        )
        print("[DM_SETUP] building validset", flush=True)
        self.validset = TextMelDataset(  # pylint: disable=attribute-defined-outside-init
            self.hparams.valid_filelist_path,
            self.hparams.n_spks,
            self.hparams.cleaners,
            self.hparams.add_blank,
            self.hparams.prepend_sil,
            self.hparams.n_fft,
            self.hparams.n_feats,
            self.hparams.sample_rate,
            self.hparams.hop_length,
            self.hparams.win_length,
            self.hparams.f_min,
            self.hparams.f_max,
            self.hparams.data_statistics,
            self.hparams.seed,
            self.hparams.load_durations,
        )
        self._run_data_sanity_check(self.trainset, "train")
        self._run_data_sanity_check(self.validset, "valid")

    def _run_data_sanity_check(self, dataset, split_name: str):
        # Avoid duplicated logs in DDP workers.
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        if rank != 0:
            return

        print(f"[DATA_SANITY] enter split={split_name}", flush=True)

        num_samples = max(1, int(self.hparams.sanity_check_samples))
        rows = dataset.filepaths_and_text[:num_samples]
        if not rows:
            raise ValueError(f"[DATA_SANITY][{split_name}] Empty filelist: {dataset.filelist_path}")

        numeric_text_count = 0
        print(
            f"[DATA_SANITY][{split_name}] Checking {len(rows)} samples, "
            f"n_spks={dataset.n_spks}, cleaners={dataset.cleaners}, add_blank={dataset.add_blank}"
        )
        for i, row in enumerate(rows):
            filepath, _, text, start, end = dataset._parse_entry(row)
            print(f"[DATA_SANITY] before text_to_sequence split={split_name} row={i}", flush=True)
            token_ids, _ = text_to_sequence(text, dataset.cleaners)
            print(f"[DATA_SANITY] after text_to_sequence split={split_name} row={i}", flush=True)
            if dataset.add_blank:
                token_ids = intersperse(token_ids, 0)
            if len(token_ids) == 0:
                raise ValueError(
                    f"[DATA_SANITY][{split_name}] Empty token sequence at row {i}: row={row}"
                )
            if text.strip().isdigit():
                numeric_text_count += 1

            col_1 = row[1] if len(row) > 1 else ""
            preview = text[:80].replace("\n", " ")
            segment = ""
            if start is not None and end is not None:
                segment = f" segment=({start:.3f},{end:.3f})"
            print(
                f"[DATA_SANITY][{split_name}] row={i} cols={len(row)} "
                f"col1='{col_1}' parsed_text='{preview}' token_len={len(token_ids)} "
                f"path={filepath}{segment}"
            )
            if i < min(3, len(rows)):
                self._print_token_debug_example(dataset, split_name, i, text)

        if dataset.n_spks == 1 and numeric_text_count == len(rows):
            raise ValueError(
                f"[DATA_SANITY][{split_name}] All sampled texts are numeric-only. "
                "This usually means the filelist text column is parsed incorrectly "
                "(e.g. using spk_id as text)."
            )

    def _print_token_debug_example(self, dataset, split_name: str, row_idx: int, text: str):
        token_ids, cleaned = text_to_sequence(text, dataset.cleaners)
        if dataset.add_blank:
            token_ids = intersperse(token_ids, 0)
        print(f"[TOKEN_DEBUG][{split_name}] row={row_idx} text={text}")
        print(f"[TOKEN_DEBUG][{split_name}] row={row_idx} cleaned={cleaned}")
        print(f"[TOKEN_DEBUG][{split_name}] row={row_idx} token_ids={token_ids}")

    def train_dataloader(self):
        return DataLoader(
            dataset=self.trainset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            multiprocessing_context="spawn" if self.hparams.num_workers > 0 else None,
            persistent_workers=self.hparams.num_workers > 0,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=TextMelBatchCollate(self.hparams.n_spks),
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.validset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            multiprocessing_context="spawn" if self.hparams.num_workers > 0 else None,
            persistent_workers=self.hparams.num_workers > 0,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=TextMelBatchCollate(self.hparams.n_spks),
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass  # pylint: disable=unnecessary-pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass  # pylint: disable=unnecessary-pass


class TextMelDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        filelist_path,
        n_spks,
        cleaners,
        add_blank=True,
        prepend_sil=True,
        n_fft=1024,
        n_mels=80,
        sample_rate=22050,
        hop_length=256,
        win_length=1024,
        f_min=0.0,
        f_max=8000,
        data_parameters=None,
        seed=None,
        load_durations=False,
    ):
        self.filepaths_and_text = parse_filelist(filelist_path)
        self.n_spks = n_spks
        self.cleaners = cleaners
        self.add_blank = add_blank
        self.prepend_sil = prepend_sil
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        self.load_durations = load_durations
        self.filelist_path = filelist_path

        if data_parameters is not None:
            self.data_parameters = data_parameters
        else:
            self.data_parameters = {"mel_mean": 0, "mel_std": 1}
        random.seed(seed)
        random.shuffle(self.filepaths_and_text)

    def _parse_entry(self, filepath_and_text):
        filepath = filepath_and_text[0]
        start = None
        end = None

        if self.n_spks > 1:
            if len(filepath_and_text) == 3:
                _, spk_raw, text = filepath_and_text
            elif len(filepath_and_text) == 5:
                _, spk_raw, start_raw, end_raw, text = filepath_and_text
                start = float(start_raw)
                end = float(end_raw)
            else:
                raise ValueError(
                    "Invalid multi-speaker filelist entry. Expected "
                    "`wav|spk|text` or `wav|spk|start|end|text`, got "
                    f"{filepath_and_text}"
                )
            spk = int(spk_raw)
        else:
            # Be tolerant to multiple single-speaker formats:
            # 1) wav|text
            # 2) wav|spk|text
            # 3) wav|start|end|text
            # 4) wav|spk|start|end|text
            if len(filepath_and_text) == 2:
                _, text = filepath_and_text
            elif len(filepath_and_text) == 3:
                _, _, text = filepath_and_text
            elif len(filepath_and_text) == 4:
                _, start_raw, end_raw, text = filepath_and_text
                start = float(start_raw)
                end = float(end_raw)
            elif len(filepath_and_text) == 5:
                _, _, start_raw, end_raw, text = filepath_and_text
                start = float(start_raw)
                end = float(end_raw)
            else:
                raise ValueError(
                    "Invalid single-speaker filelist entry. Expected "
                    "`wav|text`, `wav|spk|text`, `wav|start|end|text`, or "
                    f"`wav|spk|start|end|text`, got {filepath_and_text}"
                )
            spk = None

        if (start is None) != (end is None):
            raise ValueError(f"Segment start/end must appear together: {filepath_and_text}")
        if start is not None:
            if start < 0 or end <= start:
                raise ValueError(f"Invalid segment bounds in filelist entry: {filepath_and_text}")

        return filepath, spk, text, start, end

    def get_datapoint(self, filepath_and_text):
        filepath, spk, text, start, end = self._parse_entry(filepath_and_text)
        if self.load_durations and start is not None:
            raise NotImplementedError(
                "Segmented filelist entries are not yet compatible with load_durations=True. "
                "Please use full-utterance entries or regenerate duration annotations for segments."
            )

        text, tones, cleaned_text = self.get_text(text, add_blank=self.add_blank)
        mel = self.get_mel(filepath, start=start, end=end)

        durations = self.get_durations(filepath, text) if self.load_durations else None

        return {
            "x": text,
            "x_tones": tones,
            "y": mel,
            "spk": spk,
            "filepath": filepath,
            "x_text": cleaned_text,
            "durations": durations,
        }

    def get_text_datapoint(self, filepath_and_text):
        """Text-only datapoint for duration-prediction diagnostics (no audio I/O)."""
        filepath, spk, text, start, end = self._parse_entry(filepath_and_text)
        if start is not None:
            raise ValueError(
                "Text-only duration diagnostics do not support segmented filelist entries. "
                "Use full-utterance `wav|spk|text` or `wav|text` rows."
            )
        text, tones, cleaned_text = self.get_text(text, add_blank=self.add_blank)
        return {"x": text, "x_tones": tones, "spk": spk, "filepath": filepath, "x_text": cleaned_text}

    def get_durations(self, filepath, text):
        filepath = Path(filepath)
        data_dir, name = filepath.parent.parent, filepath.stem

        try:
            dur_loc = data_dir / "durations" / f"{name}.npy"
            durs = torch.from_numpy(np.load(dur_loc).astype(int))

        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Tried loading the durations but durations didn't exist at {dur_loc}, make sure you've generate the durations first using: python lits/utils/get_durations_from_trained_model.py \n"
            ) from e

        assert len(durs) == len(text), f"Length of durations {len(durs)} and text {len(text)} do not match"

        return durs

    def get_mel(self, filepath, start=None, end=None):
        # Torchaudio>=2.9 routes through torchcodec by default, which can fail
        # when codec runtime libs are mismatched. Use soundfile for robust wav loading.
        if start is None and end is None:
            audio_np, sr = sf.read(filepath, always_2d=True, dtype="float32")
        else:
            file_info = sf.info(filepath)
            sr = file_info.samplerate
            start_frame = max(0, int(round(start * sr)))
            end_frame = min(file_info.frames, int(round(end * sr)))
            if end_frame <= start_frame:
                raise ValueError(
                    f"Empty audio segment after frame conversion: path={filepath}, "
                    f"start={start}, end={end}, sr={sr}"
                )
            audio_np, sr = sf.read(
                filepath,
                start=start_frame,
                stop=end_frame,
                always_2d=True,
                dtype="float32",
            )
        audio = torch.from_numpy(audio_np.T)
        assert sr == self.sample_rate, f"expected {self.sample_rate} but got {sr}"
        mel = mel_spectrogram(
            audio,
            self.n_fft,
            self.n_mels,
            self.sample_rate,
            self.hop_length,
            self.win_length,
            self.f_min,
            self.f_max,
            center=False,
        ).squeeze()
        mel = normalize(mel, self.data_parameters["mel_mean"], self.data_parameters["mel_std"])
        return mel

    def get_text(self, text, add_blank=True):
        text_norm, text_tones, cleaned_text = text_to_sequence_with_tones(
            text, self.cleaners, prepend_sil=self.prepend_sil
        )
        if self.add_blank:
            text_norm = intersperse(text_norm, 0)
            if text_tones is not None:
                text_tones = intersperse(text_tones, 0)
        text_norm = torch.IntTensor(text_norm)
        text_tones = torch.IntTensor(text_tones) if text_tones is not None else None
        return text_norm, text_tones, cleaned_text

    def __getitem__(self, index):
        datapoint = self.get_datapoint(self.filepaths_and_text[index])
        return datapoint

    def __len__(self):
        return len(self.filepaths_and_text)


class TextOnlyBatchCollate:
    def __init__(self, n_spks):
        self.n_spks = n_spks

    def __call__(self, batch):
        batch_size = len(batch)
        x_max_length = max(item["x"].shape[-1] for item in batch)
        has_tones = batch[0].get("x_tones") is not None
        x = torch.zeros((batch_size, x_max_length), dtype=torch.long)
        x_tones = torch.zeros((batch_size, x_max_length), dtype=torch.long) if has_tones else None
        x_lengths = []
        spks = []
        filepaths = []
        x_texts = []
        for i, item in enumerate(batch):
            x_ = item["x"]
            x_lengths.append(x_.shape[-1])
            x[i, : x_.shape[-1]] = x_
            if has_tones:
                x_tones[i, : item["x_tones"].shape[-1]] = item["x_tones"]
            spks.append(item["spk"])
            filepaths.append(item["filepath"])
            x_texts.append(item["x_text"])
        return {
            "x": x,
            "x_tones": x_tones,
            "x_lengths": torch.tensor(x_lengths, dtype=torch.long),
            "spks": torch.tensor(spks, dtype=torch.long) if self.n_spks > 1 else None,
            "filepaths": filepaths,
            "x_texts": x_texts,
        }


class TextMelBatchCollate:
    def __init__(self, n_spks):
        self.n_spks = n_spks

    def __call__(self, batch):
        B = len(batch)
        y_max_length = max([item["y"].shape[-1] for item in batch])  # pylint: disable=consider-using-generator
        y_max_length = fix_len_compatibility(y_max_length)
        x_max_length = max([item["x"].shape[-1] for item in batch])  # pylint: disable=consider-using-generator
        n_feats = batch[0]["y"].shape[-2]

        y = torch.zeros((B, n_feats, y_max_length), dtype=torch.float32)
        x = torch.zeros((B, x_max_length), dtype=torch.long)
        has_tones = batch[0].get("x_tones") is not None
        x_tones = torch.zeros((B, x_max_length), dtype=torch.long) if has_tones else None
        durations = torch.zeros((B, x_max_length), dtype=torch.long)

        y_lengths, x_lengths = [], []
        spks = []
        filepaths, x_texts = [], []
        for i, item in enumerate(batch):
            y_, x_ = item["y"], item["x"]
            y_lengths.append(y_.shape[-1])
            x_lengths.append(x_.shape[-1])
            y[i, :, : y_.shape[-1]] = y_
            x[i, : x_.shape[-1]] = x_
            if has_tones:
                x_tones[i, : item["x_tones"].shape[-1]] = item["x_tones"]
            spks.append(item["spk"])
            filepaths.append(item["filepath"])
            x_texts.append(item["x_text"])
            if item["durations"] is not None:
                durations[i, : item["durations"].shape[-1]] = item["durations"]

        y_lengths = torch.tensor(y_lengths, dtype=torch.long)
        x_lengths = torch.tensor(x_lengths, dtype=torch.long)
        spks = torch.tensor(spks, dtype=torch.long) if self.n_spks > 1 else None

        return {
            "x": x,
            "x_tones": x_tones,
            "x_lengths": x_lengths,
            "y": y,
            "y_lengths": y_lengths,
            "spks": spks,
            "filepaths": filepaths,
            "x_texts": x_texts,
            "durations": durations if not torch.eq(durations, 0).all() else None,
        }
