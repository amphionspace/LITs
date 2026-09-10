import csv
import datetime as dt
import math
import random
from pathlib import Path
from typing import Any, Iterator, Optional

import torch
import lits.utils.monotonic_align as monotonic_align
from lits import utils
from lits.models.base import BaseLits
from lits.models.components.flow_matching import CFM, CFM_Causal
from lits.models.components.text_encoder import TextEncoder
from lits.text.bopomofo_utils import BOPOMOFO_TONES, split_rhyme_tone_token, zh354_duration_tokens
from lits.text.char_symbols.langs.ARPA import ARPA_TOKENS
from lits.utils.infer_duration_floor import apply_infer_duration_patches
from lits.text.char_symbols.symbol_inventories import lang2inventory
from lits.utils.model import (
    denormalize,
    duration_loss,
    fix_len_compatibility,
    generate_path,
    sequence_mask,
)

log = utils.get_pylogger(__name__)

class LITS(BaseLits):
    """
    LITS model: integrates text encoder, flow-matching decoder, and alignment for TTS.
    """
    def __init__(
        self,
        n_vocab: int,
        n_spks: int,
        spk_emb_dim: int,
        n_feats: int,
        encoder,
        decoder,
        cfm,
        data_statistics,
        out_size: int,
        optimizer=None,
        scheduler=None,
        prior_loss: bool = True,
        use_precomputed_durations=False,
        aux_loss_decay_start_step: int = -1,
        aux_loss_decay_end_step: int = -1,
        prior_encoder_lr: Optional[float] = None,
        duration_predictor_lr: Optional[float] = None,
        arpa_duration_stats_path: Optional[str] = None,
        arpa_duration_upper_stats_path: Optional[str] = None,
        arpa_duration_lower_column: str = "p01_ms",
        arpa_duration_upper_p95_column: str = "p95_ms",
        arpa_duration_upper_p99_column: str = "p99_ms",
        arpa_duration_upper_p99_scale: float = 2.0,
        arpa_duration_upper_p95_margin_ms: float = 200.0,
        arpa_duration_upper_min_ms: float = 240.0,
        arpa_duration_sample_rate: Optional[int] = None,
        arpa_duration_hop_length: Optional[int] = None,
        arpa_duration_cleaners: Optional[list[str]] = None,
        arpa_duration_spk_ids: Optional[list[int]] = None,
        zh_duration_stats_path: Optional[str] = None,
        zh_duration_upper_stats_path: Optional[str] = None,
        zh_duration_lower_column: str = "p01",
        zh_duration_upper_p95_column: str = "p95",
        zh_duration_upper_p99_column: str = "p99",
        zh_duration_upper_p99_scale: float = 1.5,
        zh_duration_upper_p95_margin_frames: float = 6.0,
        zh_duration_upper_min_frames: float = 8.0,
        zh_duration_min_count: int = 20,
        zh_duration_cleaners: Optional[list[str]] = None,
        duration_constrained_mas: bool = False,
        duration_floor_scale: float = 0.5,
        duration_floor_min_frames: int = 2,
        arpa_duration_floor_scale: Optional[float] = None,
        n_tones: int = 0,
        mas_n_tones: int = 0,
        tone_floor_frames: int = 0,
        tone_ceiling_frames: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.n_vocab = n_vocab
        self.n_tones = n_tones
        self.mas_n_tones = mas_n_tones if mas_n_tones > 0 else n_tones
        self.tone_floor_frames = tone_floor_frames
        self.tone_ceiling_frames = tone_ceiling_frames
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.n_feats = n_feats
        self.out_size = out_size
        self.prior_loss = prior_loss
        self.use_precomputed_durations = use_precomputed_durations
        self.duration_constrained_mas = duration_constrained_mas
        self.duration_floor_scale = duration_floor_scale
        self.duration_floor_min_frames = duration_floor_min_frames
        # English stats come from LJSpeech, not the training speakers, so their
        # p01 deserves a separate (more conservative) scale than the zh stats
        # measured on the actual training corpus. None = follow duration_floor_scale.
        self.arpa_duration_floor_scale = (
            duration_floor_scale if arpa_duration_floor_scale is None else arpa_duration_floor_scale
        )
        self.encoder_frozen = False
        self.register_buffer(
            "arpa_min_duration_frames_by_id",
            self._build_arpa_duration_floor_by_id(
                n_vocab=n_vocab,
                stats_path=arpa_duration_stats_path,
                lower_column=arpa_duration_lower_column,
                sample_rate=arpa_duration_sample_rate,
                hop_length=arpa_duration_hop_length,
                cleaners=arpa_duration_cleaners,
            ),
            persistent=False,
        )
        self.register_buffer(
            "arpa_max_duration_frames_by_id",
            self._build_arpa_duration_ceiling_by_id(
                n_vocab=n_vocab,
                stats_path=arpa_duration_upper_stats_path or arpa_duration_stats_path,
                p95_column=arpa_duration_upper_p95_column,
                p99_column=arpa_duration_upper_p99_column,
                p99_scale=arpa_duration_upper_p99_scale,
                p95_margin_ms=arpa_duration_upper_p95_margin_ms,
                min_upper_ms=arpa_duration_upper_min_ms,
                sample_rate=arpa_duration_sample_rate,
                hop_length=arpa_duration_hop_length,
                cleaners=arpa_duration_cleaners,
            ),
            persistent=False,
        )
        self.register_buffer(
            "arpa_duration_spk_id_filter",
            self._build_arpa_duration_spk_filter(arpa_duration_spk_ids),
            persistent=False,
        )
        self.register_buffer(
            "zh_min_duration_frames_by_id",
            self._build_zh_duration_floor_by_id(
                n_vocab=n_vocab,
                stats_path=zh_duration_stats_path if self.mas_n_tones == 0 else None,
                lower_column=zh_duration_lower_column,
                min_count=zh_duration_min_count,
                cleaners=zh_duration_cleaners,
            ),
            persistent=False,
        )
        self.register_buffer(
            "zh_max_duration_frames_by_id",
            self._build_zh_duration_ceiling_by_id(
                n_vocab=n_vocab,
                stats_path=(zh_duration_upper_stats_path or zh_duration_stats_path) if self.mas_n_tones == 0 else None,
                p95_column=zh_duration_upper_p95_column,
                p99_column=zh_duration_upper_p99_column,
                p99_scale=zh_duration_upper_p99_scale,
                p95_margin_frames=zh_duration_upper_p95_margin_frames,
                min_upper_frames=zh_duration_upper_min_frames,
                min_count=zh_duration_min_count,
                cleaners=zh_duration_cleaners,
            ),
            persistent=False,
        )
        # Tone-embedding / rhyme-body-tone paradigm: per-(rhyme, tone) stats drive a
        # 2D lookup keyed by (stem id, tone id).
        self.register_buffer(
            "zh_min_duration_frames_by_id_tone",
            self._build_zh_duration_floor_by_id_tone(
                n_vocab=n_vocab,
                n_tones=self.mas_n_tones,
                stats_path=zh_duration_stats_path if self.mas_n_tones > 0 else None,
                lower_column=zh_duration_lower_column,
                min_count=zh_duration_min_count,
                cleaners=zh_duration_cleaners,
            ),
            persistent=False,
        )
        self.register_buffer(
            "zh_max_duration_frames_by_id_tone",
            self._build_zh_duration_ceiling_by_id_tone(
                n_vocab=n_vocab,
                n_tones=self.mas_n_tones,
                stats_path=(zh_duration_upper_stats_path or zh_duration_stats_path) if self.mas_n_tones > 0 else None,
                p95_column=zh_duration_upper_p95_column,
                p99_column=zh_duration_upper_p99_column,
                p99_scale=zh_duration_upper_p99_scale,
                p95_margin_frames=zh_duration_upper_p95_margin_frames,
                min_upper_frames=zh_duration_upper_min_frames,
                min_count=zh_duration_min_count,
                cleaners=zh_duration_cleaners,
            ),
            persistent=False,
        )
        self.register_buffer(
            "tone_mark_token_ids",
            self._build_tone_mark_token_ids(n_vocab, zh_duration_cleaners),
            persistent=False,
        )
        
        if n_spks > 1:
            self.spk_emb = torch.nn.Embedding(n_spks, spk_emb_dim)
        self.encoder = TextEncoder(
            encoder.encoder_type,
            encoder.encoder_params,
            encoder.duration_predictor_params,
            n_vocab,
            n_spks,
            spk_emb_dim,
            n_tones=n_tones,
        )
        self.decoder = CFM_Causal(
            in_channels=2 * encoder.encoder_params.n_feats,
            out_channel=encoder.encoder_params.n_feats,
            cfm_params=cfm,
            decoder_params=decoder,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )
        # self.decoder = CFM(
        #     in_channels=2 * encoder.encoder_params.n_feats,
        #     out_channel=encoder.encoder_params.n_feats,
        #     cfm_params=cfm,
        #     decoder_params=decoder,
        #     n_spks=n_spks,
        #     spk_emb_dim=spk_emb_dim,
        # )
        self.update_data_statistics(data_statistics)

    @staticmethod
    def _build_arpa_duration_spk_filter(spk_ids: Optional[list[int]]) -> torch.Tensor:
        if spk_ids is None:
            return torch.empty(0, dtype=torch.long)
        return torch.tensor(sorted({int(spk_id) for spk_id in spk_ids}), dtype=torch.long)

    @staticmethod
    def _cleaner_model_key(cleaners) -> Optional[str]:
        if cleaners is None:
            return None
        if isinstance(cleaners, str):
            cleaner_names = [cleaners]
        else:
            cleaner_names = list(cleaners)
        cleaner_to_model_key = {
            "pinyin_direct_mixed_cleaners": "zh-en-direct",
            "pinyin_direct_mixed_rhyme_body_tone_cleaners": "zh-en-rhyme-body-tone",
            "en_zh_dict_mixed_cleaners": "zh-en-direct",
            "en_zh_dict_mixed_rhyme_body_tone_cleaners": "zh-en-rhyme-body-tone",
        }
        for cleaner_name in cleaner_names:
            model_key = cleaner_to_model_key.get(cleaner_name)
            if model_key is not None:
                return model_key
        return None

    @staticmethod
    def _load_arpa_duration_lower_ms(stats_path: str, lower_column: str) -> dict[str, float]:
        path = Path(stats_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"ARPA duration stats file not found: {path}")
        lower_ms_by_phone = {}
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if lower_column not in (reader.fieldnames or []):
                raise ValueError(
                    f"Column '{lower_column}' not found in ARPA duration stats file {path}"
                )
            for row in reader:
                phone = row["phone"].strip()
                lower_ms_by_phone[phone] = float(row[lower_column])
        if "AH0" in lower_ms_by_phone:
            for ax_phone in ("AX0", "AX1", "AX2"):
                lower_ms_by_phone.setdefault(ax_phone, lower_ms_by_phone["AH0"])
        return lower_ms_by_phone

    @staticmethod
    def _load_arpa_duration_upper_ms(
        stats_path: str,
        *,
        p95_column: str,
        p99_column: str,
        p99_scale: float,
        p95_margin_ms: float,
        min_upper_ms: float,
    ) -> dict[str, float]:
        path = Path(stats_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"ARPA duration stats file not found: {path}")
        upper_ms_by_phone = {}
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            missing_columns = [
                column for column in (p95_column, p99_column) if column not in fieldnames
            ]
            if missing_columns:
                raise ValueError(
                    f"Columns {missing_columns} not found in ARPA duration stats file {path}"
                )
            for row in reader:
                phone = row["phone"].strip()
                p95_ms = float(row[p95_column])
                p99_ms = float(row[p99_column])
                upper_ms_by_phone[phone] = max(
                    p99_scale * p99_ms,
                    p95_ms + p95_margin_ms,
                    min_upper_ms,
                )
        if "AH0" in upper_ms_by_phone:
            for ax_phone in ("AX0", "AX1", "AX2"):
                upper_ms_by_phone.setdefault(ax_phone, upper_ms_by_phone["AH0"])
        return upper_ms_by_phone

    @classmethod
    def _build_arpa_duration_floor_by_id(
        cls,
        *,
        n_vocab: int,
        stats_path: Optional[str],
        lower_column: str,
        sample_rate: Optional[int],
        hop_length: Optional[int],
        cleaners,
    ) -> torch.Tensor:
        floors = torch.zeros(n_vocab, dtype=torch.float32)
        if not stats_path:
            return floors
        if sample_rate is None or hop_length is None:
            raise ValueError("sample_rate and hop_length are required for ARPA duration floors")

        model_key = cls._cleaner_model_key(cleaners)
        if model_key is None:
            log.warning("ARPA duration floors enabled, but cleaners do not map to a known inventory")
            return floors
        inventory = lang2inventory.get(model_key)
        if inventory is None:
            log.warning("ARPA duration floors enabled, but inventory '%s' was not found", model_key)
            return floors

        lower_ms_by_phone = cls._load_arpa_duration_lower_ms(stats_path, lower_column)
        symbol_to_id = inventory["symbol_to_id"]
        ms_to_frames = float(sample_rate) / (1000.0 * float(hop_length))
        loaded = 0
        for token in ARPA_TOKENS:
            token_id = symbol_to_id.get(token)
            lower_ms = lower_ms_by_phone.get(token)
            if token_id is None or lower_ms is None or token_id >= n_vocab:
                continue
            floors[token_id] = max(1.0, lower_ms * ms_to_frames)
            loaded += 1
        log.info(
            "Loaded ARPA duration floors for %d tokens from %s (%s, inventory=%s)",
            loaded,
            stats_path,
            lower_column,
            model_key,
        )
        return floors

    @classmethod
    def _build_arpa_duration_ceiling_by_id(
        cls,
        *,
        n_vocab: int,
        stats_path: Optional[str],
        p95_column: str,
        p99_column: str,
        p99_scale: float,
        p95_margin_ms: float,
        min_upper_ms: float,
        sample_rate: Optional[int],
        hop_length: Optional[int],
        cleaners,
    ) -> torch.Tensor:
        ceilings = torch.zeros(n_vocab, dtype=torch.float32)
        if not stats_path:
            return ceilings
        if sample_rate is None or hop_length is None:
            raise ValueError("sample_rate and hop_length are required for ARPA duration ceilings")

        model_key = cls._cleaner_model_key(cleaners)
        if model_key is None:
            log.warning("ARPA duration ceilings enabled, but cleaners do not map to a known inventory")
            return ceilings
        inventory = lang2inventory.get(model_key)
        if inventory is None:
            log.warning("ARPA duration ceilings enabled, but inventory '%s' was not found", model_key)
            return ceilings

        upper_ms_by_phone = cls._load_arpa_duration_upper_ms(
            stats_path,
            p95_column=p95_column,
            p99_column=p99_column,
            p99_scale=p99_scale,
            p95_margin_ms=p95_margin_ms,
            min_upper_ms=min_upper_ms,
        )
        symbol_to_id = inventory["symbol_to_id"]
        ms_to_frames = float(sample_rate) / (1000.0 * float(hop_length))
        loaded = 0
        for token in ARPA_TOKENS:
            token_id = symbol_to_id.get(token)
            upper_ms = upper_ms_by_phone.get(token)
            if token_id is None or upper_ms is None or token_id >= n_vocab:
                continue
            ceilings[token_id] = max(1.0, upper_ms * ms_to_frames)
            loaded += 1
        log.info(
            "Loaded ARPA duration ceilings for %d tokens from %s "
            "(max(%s*%.3g, %s+%.3gms, %.3gms), inventory=%s)",
            loaded,
            stats_path,
            p99_column,
            p99_scale,
            p95_column,
            p95_margin_ms,
            min_upper_ms,
            model_key,
        )
        return ceilings

    @staticmethod
    def _build_tone_mark_token_ids(n_vocab: int, cleaners) -> torch.Tensor:
        model_key = LITS._cleaner_model_key(cleaners)
        if model_key is None:
            return torch.empty(0, dtype=torch.long)
        inventory = lang2inventory.get(model_key)
        if inventory is None:
            return torch.empty(0, dtype=torch.long)
        symbol_to_id = inventory["symbol_to_id"]
        ids = [
            symbol_to_id[mark]
            for mark in BOPOMOFO_TONES
            if mark in symbol_to_id and symbol_to_id[mark] < n_vocab
        ]
        return torch.tensor(ids, dtype=torch.long)

    def _tone_mark_mask(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """(B, T) bool mask for inline Bopomofo tone-mark token ids."""
        if self.tone_mark_token_ids.numel() == 0:
            return None
        tone_ids = self.tone_mark_token_ids.to(device=x.device)
        return (x.unsqueeze(-1) == tone_ids.view(1, 1, -1)).any(dim=-1)

    def _clamp_inference_tone_durations(
        self,
        w_ceil: torch.Tensor,
        x: torch.Tensor,
        x_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Clamp predicted duration (frames) for inline tone marks at inference."""
        if self.tone_floor_frames <= 0 and self.tone_ceiling_frames <= 0:
            return w_ceil
        is_tone_mark = self._tone_mark_mask(x)
        if is_tone_mark is None:
            return w_ceil
        min_frames = float(self.tone_floor_frames if self.tone_floor_frames > 0 else 1)
        out = w_ceil
        if self.tone_ceiling_frames > 0:
            out = torch.where(
                is_tone_mark,
                out.clamp(min=min_frames, max=float(self.tone_ceiling_frames)),
                out,
            )
        elif self.tone_floor_frames > 0:
            out = torch.where(
                is_tone_mark,
                torch.clamp(out, min=min_frames),
                out,
            )
        return out * x_mask

    @staticmethod
    def _zh_duration_model_keys(cleaners) -> list[str]:
        model_key = LITS._cleaner_model_key(cleaners)
        if model_key == "zh-en-rhyme-body-tone":
            return [model_key]
        return []

    @staticmethod
    def _load_zh_duration_lower_frames(
        stats_path: str,
        lower_column: str,
        *,
        min_count: int,
    ) -> dict[str, float]:
        path = Path(stats_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Zh duration stats file not found: {path}")
        delimiter = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
        lower_frames_by_token: dict[str, float] = {}
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            fieldnames = reader.fieldnames or []
            if "token_354" not in fieldnames:
                raise ValueError(f"Column 'token_354' not found in zh duration stats file {path}")
            if lower_column not in fieldnames:
                raise ValueError(f"Column '{lower_column}' not found in zh duration stats file {path}")
            count_column = "count" if "count" in fieldnames else None
            for row in reader:
                token = row["token_354"].strip()
                if not token:
                    continue
                if count_column is not None:
                    count_raw = row[count_column].strip()
                    if not count_raw or int(count_raw) < min_count:
                        continue
                lower_value = row[lower_column].strip()
                if not lower_value:
                    continue
                lower_frames_by_token[token] = float(lower_value)
        return lower_frames_by_token

    @staticmethod
    def _load_zh_duration_upper_frames(
        stats_path: str,
        *,
        p95_column: str,
        p99_column: str,
        p99_scale: float,
        p95_margin_frames: float,
        min_upper_frames: float,
        min_count: int,
    ) -> dict[str, float]:
        path = Path(stats_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Zh duration stats file not found: {path}")
        delimiter = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
        upper_frames_by_token: dict[str, float] = {}
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            fieldnames = reader.fieldnames or []
            missing_columns = [
                column for column in ("token_354", p95_column, p99_column) if column not in fieldnames
            ]
            if missing_columns:
                raise ValueError(
                    f"Columns {missing_columns} not found in zh duration stats file {path}"
                )
            count_column = "count" if "count" in fieldnames else None
            for row in reader:
                token = row["token_354"].strip()
                if not token:
                    continue
                if count_column is not None:
                    count_raw = row[count_column].strip()
                    if not count_raw or int(count_raw) < min_count:
                        continue
                p95_frames = float(row[p95_column])
                p99_frames = float(row[p99_column])
                upper_frames_by_token[token] = max(
                    p99_scale * p99_frames,
                    p95_frames + p95_margin_frames,
                    min_upper_frames,
                )
        return upper_frames_by_token

    @classmethod
    def _build_zh_duration_floor_by_id(
        cls,
        *,
        n_vocab: int,
        stats_path: Optional[str],
        lower_column: str,
        min_count: int,
        cleaners,
    ) -> torch.Tensor:
        floors = torch.zeros(n_vocab, dtype=torch.float32)
        if not stats_path:
            return floors

        model_keys = cls._zh_duration_model_keys(cleaners)
        if not model_keys:
            log.warning("Zh duration floors enabled, but cleaners do not map to a zh-en inventory")
            return floors

        lower_frames_by_token = cls._load_zh_duration_lower_frames(
            stats_path,
            lower_column,
            min_count=min_count,
        )
        loaded = 0
        for model_key in model_keys:
            inventory = lang2inventory.get(model_key)
            if inventory is None:
                log.warning("Zh duration floors enabled, but inventory '%s' was not found", model_key)
                continue
            symbol_to_id = inventory["symbol_to_id"]
            for token in zh354_duration_tokens():
                token_id = symbol_to_id.get(token)
                lower_frames = lower_frames_by_token.get(token)
                if token_id is None or lower_frames is None or token_id >= n_vocab:
                    continue
                floors[token_id] = max(1.0, lower_frames)
                loaded += 1
        log.info(
            "Loaded zh duration floors for %d tokens from %s (%s, min_count=%d)",
            loaded,
            stats_path,
            lower_column,
            min_count,
        )
        return floors

    @classmethod
    def _build_zh_duration_ceiling_by_id(
        cls,
        *,
        n_vocab: int,
        stats_path: Optional[str],
        p95_column: str,
        p99_column: str,
        p99_scale: float,
        p95_margin_frames: float,
        min_upper_frames: float,
        min_count: int,
        cleaners,
    ) -> torch.Tensor:
        ceilings = torch.zeros(n_vocab, dtype=torch.float32)
        if not stats_path:
            return ceilings

        model_keys = cls._zh_duration_model_keys(cleaners)
        if not model_keys:
            log.warning("Zh duration ceilings enabled, but cleaners do not map to a zh-en inventory")
            return ceilings

        upper_frames_by_token = cls._load_zh_duration_upper_frames(
            stats_path,
            p95_column=p95_column,
            p99_column=p99_column,
            p99_scale=p99_scale,
            p95_margin_frames=p95_margin_frames,
            min_upper_frames=min_upper_frames,
            min_count=min_count,
        )
        loaded = 0
        for model_key in model_keys:
            inventory = lang2inventory.get(model_key)
            if inventory is None:
                log.warning("Zh duration ceilings enabled, but inventory '%s' was not found", model_key)
                continue
            symbol_to_id = inventory["symbol_to_id"]
            for token in zh354_duration_tokens():
                token_id = symbol_to_id.get(token)
                upper_frames = upper_frames_by_token.get(token)
                if token_id is None or upper_frames is None or token_id >= n_vocab:
                    continue
                ceilings[token_id] = max(1.0, upper_frames)
                loaded += 1
        log.info(
            "Loaded zh duration ceilings for %d tokens from %s "
            "(max(%s*%.3g, %s+%.3g frames, %.3g frames), min_count=%d)",
            loaded,
            stats_path,
            p99_column,
            p99_scale,
            p95_column,
            p95_margin_frames,
            min_upper_frames,
            min_count,
        )
        return ceilings

    @classmethod
    def _fill_zh_frames_by_id_tone(
        cls,
        values_by_token: dict[str, float],
        *,
        n_vocab: int,
        n_tones: int,
        cleaners,
        label: str,
    ) -> torch.Tensor:
        """Fill a (n_vocab, n_tones+1) table from merged rhyme+tone stat keys.

        Each merged token (e.g. ㄣˊ) is split into (toneless stem, tone id) and
        the value lands at [stem_id, tone_id]; initials keep tone id 0."""
        table = torch.zeros(n_vocab, n_tones + 1, dtype=torch.float32)
        model_keys = cls._zh_duration_model_keys(cleaners)
        if not model_keys:
            log.warning("Zh duration %s (tone table) enabled, but cleaners do not map to a zh-en inventory", label)
            return table
        loaded = 0
        for model_key in model_keys:
            inventory = lang2inventory.get(model_key)
            if inventory is None:
                log.warning("Zh duration %s (tone table) enabled, but inventory '%s' was not found", label, model_key)
                continue
            symbol_to_id = inventory["symbol_to_id"]
            for token in zh354_duration_tokens():
                value = values_by_token.get(token)
                if value is None:
                    continue
                stem, tone_id = split_rhyme_tone_token(token)
                token_id = symbol_to_id.get(stem)
                if token_id is None or token_id >= n_vocab or tone_id > n_tones:
                    continue
                table[token_id, tone_id] = max(1.0, value)
                loaded += 1
        log.info("Loaded zh duration %s for %d (stem, tone) entries into the tone table", label, loaded)
        return table

    @classmethod
    def _build_zh_duration_floor_by_id_tone(
        cls,
        *,
        n_vocab: int,
        n_tones: int,
        stats_path: Optional[str],
        lower_column: str,
        min_count: int,
        cleaners,
    ) -> torch.Tensor:
        if not stats_path or n_tones <= 0:
            return torch.zeros(0, dtype=torch.float32)
        lower_frames_by_token = cls._load_zh_duration_lower_frames(
            stats_path,
            lower_column,
            min_count=min_count,
        )
        return cls._fill_zh_frames_by_id_tone(
            lower_frames_by_token,
            n_vocab=n_vocab,
            n_tones=n_tones,
            cleaners=cleaners,
            label="floors",
        )

    @classmethod
    def _build_zh_duration_ceiling_by_id_tone(
        cls,
        *,
        n_vocab: int,
        n_tones: int,
        stats_path: Optional[str],
        p95_column: str,
        p99_column: str,
        p99_scale: float,
        p95_margin_frames: float,
        min_upper_frames: float,
        min_count: int,
        cleaners,
    ) -> torch.Tensor:
        if not stats_path or n_tones <= 0:
            return torch.zeros(0, dtype=torch.float32)
        upper_frames_by_token = cls._load_zh_duration_upper_frames(
            stats_path,
            p95_column=p95_column,
            p99_column=p99_column,
            p99_scale=p99_scale,
            p95_margin_frames=p95_margin_frames,
            min_upper_frames=min_upper_frames,
            min_count=min_count,
        )
        return cls._fill_zh_frames_by_id_tone(
            upper_frames_by_token,
            n_vocab=n_vocab,
            n_tones=n_tones,
            cleaners=cleaners,
            label="ceilings",
        )

    def _zh_frames_lookup(
        self,
        table_1d: torch.Tensor,
        table_2d: torch.Tensor,
        x: torch.Tensor,
        x_tones: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Per-token zh bound lookup that works in both paradigms."""
        if table_2d.numel() > 0:
            if x_tones is None:
                return torch.zeros_like(x, dtype=table_2d.dtype)
            table_2d = table_2d.to(device=x.device)
            # Must stay out-of-place: .long() on an already-long tensor returns
            # the same object, and x_tones is saved by the tone embedding for
            # backward; an in-place clamp_ here would break autograd.
            tone_ids = torch.clamp(x_tones.long(), min=0, max=table_2d.size(1) - 1)
            return table_2d[x, tone_ids]
        return table_1d.to(device=x.device)[x]

    def _arpa_sample_filter(
        self,
        x: torch.Tensor,
        spk_ids: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """(B,) bool mask of samples the ARPA duration bounds apply to; None = all."""
        if self.arpa_duration_spk_id_filter.numel() == 0:
            return None
        spk_filter = self.arpa_duration_spk_id_filter.to(device=x.device)
        if spk_ids is None:
            return torch.any(spk_filter == 0).expand(x.size(0))
        return torch.any(
            spk_ids.to(device=x.device).long().unsqueeze(1) == spk_filter.unsqueeze(0),
            dim=1,
        )

    def _mas_floor_frames(
        self,
        x: torch.Tensor,
        spk_ids: Optional[torch.Tensor],
        x_tones: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Per-token duration floors (frames, B x T) for constrained MAS.

        Conservative transform: floor' = max(duration_floor_min_frames,
        duration_floor_scale * stats_floor) for bounded tokens, 0 elsewhere
        (0 keeps the implicit min-1 of classic MAS). Returns None when
        constrained MAS is disabled or no token in the batch is bounded."""
        if not self.duration_constrained_mas:
            return None
        arpa = self.arpa_min_duration_frames_by_id.to(device=x.device)[x]
        zh = self._zh_frames_lookup(
            self.zh_min_duration_frames_by_id,
            self.zh_min_duration_frames_by_id_tone,
            x,
            x_tones,
        )
        sample_filter = self._arpa_sample_filter(x, spk_ids)
        if sample_filter is not None:
            arpa = arpa * sample_filter.view(-1, 1).to(arpa.dtype)
        is_tone_mark = self._tone_mark_mask(x)
        bounded = (arpa > 0) | (zh > 0)
        if is_tone_mark is not None and self.tone_floor_frames > 0:
            bounded = bounded | is_tone_mark
        if not bool(bounded.any()):
            return None
        # Per-language scales applied before merging (a token is only ever
        # bounded by one of the two stat tables).
        scaled = torch.maximum(
            arpa * float(self.arpa_duration_floor_scale),
            zh * float(self.duration_floor_scale),
        ).clamp_min(float(self.duration_floor_min_frames))
        if is_tone_mark is not None and self.tone_floor_frames > 0:
            tone_floor = scaled.new_tensor(float(self.tone_floor_frames))
            scaled = torch.where(is_tone_mark, torch.maximum(scaled, tone_floor), scaled)
        return torch.where(bounded, scaled, torch.zeros_like(scaled))

    def _mas_ceiling_frames(
        self,
        x: torch.Tensor,
        spk_ids: Optional[torch.Tensor],
        x_tones: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Per-token duration ceilings (frames, B x T) for constrained MAS.

        Returns None when constrained MAS is disabled or no bounded token."""
        if not self.duration_constrained_mas:
            return None
        arpa = self.arpa_max_duration_frames_by_id.to(device=x.device)[x]
        zh = self._zh_frames_lookup(
            self.zh_max_duration_frames_by_id,
            self.zh_max_duration_frames_by_id_tone,
            x,
            x_tones,
        )
        sample_filter = self._arpa_sample_filter(x, spk_ids)
        if sample_filter is not None:
            arpa = arpa * sample_filter.view(-1, 1).to(arpa.dtype)
        is_tone_mark = self._tone_mark_mask(x)
        bounded = (arpa > 0) | (zh > 0)
        if is_tone_mark is not None and self.tone_ceiling_frames > 0:
            bounded = bounded | is_tone_mark
        if not bool(bounded.any()):
            return None
        merged = torch.maximum(arpa, zh)
        if is_tone_mark is not None and self.tone_ceiling_frames > 0:
            tone_ceiling = merged.new_tensor(float(self.tone_ceiling_frames))
            merged = torch.where(is_tone_mark, tone_ceiling, merged)
        return torch.where(bounded, merged, torch.zeros_like(merged))

    def _mask_duration_outliers(
        self,
        x: torch.Tensor,
        duration_targets: torch.Tensor,
        x_mask: torch.Tensor,
        spk_ids: Optional[torch.Tensor],
        x_tones: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return a mask that excludes unreliable phonetic duration targets
        from the duration predictor loss. The mask has the same shape as
        *duration_targets* (B, 1, T) with 1 = keep, 0 = exclude."""
        ones = torch.ones_like(x_mask)
        zero = duration_targets.new_tensor(0.0)

        def _lookup_bounds(
            min_frames_by_id: torch.Tensor,
            max_frames_by_id: torch.Tensor,
            min_frames_by_id_tone: Optional[torch.Tensor] = None,
            max_frames_by_id_tone: Optional[torch.Tensor] = None,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            use_tone_tables = (
                min_frames_by_id_tone is not None and min_frames_by_id_tone.numel() > 0
            ) or (
                max_frames_by_id_tone is not None and max_frames_by_id_tone.numel() > 0
            )
            if use_tone_tables:
                token_floors = self._zh_frames_lookup(
                    min_frames_by_id, min_frames_by_id_tone, x, x_tones
                ).to(dtype=duration_targets.dtype).unsqueeze(1)
                token_ceilings = self._zh_frames_lookup(
                    max_frames_by_id, max_frames_by_id_tone, x, x_tones
                ).to(dtype=duration_targets.dtype).unsqueeze(1)
                bounded = ((token_floors > 0) | (token_ceilings > 0)) & (x_mask > 0)
                return token_floors, token_ceilings, bounded
            has_floors = torch.count_nonzero(min_frames_by_id) > 0
            has_ceilings = torch.count_nonzero(max_frames_by_id) > 0
            if not has_floors and not has_ceilings:
                empty_bounds = torch.zeros_like(duration_targets)
                empty_mask = torch.zeros_like(x_mask, dtype=torch.bool)
                return empty_bounds, empty_bounds, empty_mask
            token_floors = min_frames_by_id.to(
                device=x.device,
                dtype=duration_targets.dtype,
            )[x].unsqueeze(1)
            token_ceilings = max_frames_by_id.to(
                device=x.device,
                dtype=duration_targets.dtype,
            )[x].unsqueeze(1)
            bounded = ((token_floors > 0) | (token_ceilings > 0)) & (x_mask > 0)
            return token_floors, token_ceilings, bounded

        # With constrained MAS the floors are enforced inside the alignment DP,
        # so floor-side targets are trustworthy and must not be masked (their
        # values may legitimately sit below the raw stats floor after the
        # conservative rescale). Ceiling-side masking is also retired once
        # ceiling-in-DP is active.
        skip_floor_mask = bool(self.duration_constrained_mas)
        skip_ceiling_mask = bool(self.duration_constrained_mas)

        def _build_outlier_masks(
            token_floors: torch.Tensor,
            token_ceilings: torch.Tensor,
            bounded_mask: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            if skip_floor_mask:
                too_short = torch.zeros_like(bounded_mask)
            else:
                too_short = bounded_mask & (token_floors > 0) & (duration_targets < token_floors)
            if skip_ceiling_mask:
                too_long = torch.zeros_like(bounded_mask)
            else:
                too_long = bounded_mask & (token_ceilings > 0) & (duration_targets > token_ceilings)
            return too_short, too_long, too_short | too_long

        def _summarize_mask(
            prefix: str,
            bounded_mask: torch.Tensor,
            too_short: torch.Tensor,
            too_long: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            masked = too_short | too_long
            bounded_count = bounded_mask.to(duration_targets.dtype).sum()
            denominator = bounded_count.clamp_min(1.0)
            return {
                f"{prefix}masked_token_ratio": masked.to(duration_targets.dtype).sum() / denominator,
                f"{prefix}masked_short_token_ratio": too_short.to(duration_targets.dtype).sum() / denominator,
                f"{prefix}masked_long_token_ratio": too_long.to(duration_targets.dtype).sum() / denominator,
            }

        arpa_floors, arpa_ceilings, arpa_bounded_mask = _lookup_bounds(
            self.arpa_min_duration_frames_by_id,
            self.arpa_max_duration_frames_by_id,
        )
        sample_matches = self._arpa_sample_filter(x, spk_ids)
        if sample_matches is not None:
            arpa_bounded_mask = arpa_bounded_mask & sample_matches.view(-1, 1, 1)

        zh_floors, zh_ceilings, zh_bounded_mask = _lookup_bounds(
            self.zh_min_duration_frames_by_id,
            self.zh_max_duration_frames_by_id,
            self.zh_min_duration_frames_by_id_tone,
            self.zh_max_duration_frames_by_id_tone,
        )
        arpa_too_short, arpa_too_long, arpa_masked = _build_outlier_masks(
            arpa_floors,
            arpa_ceilings,
            arpa_bounded_mask,
        )
        zh_too_short, zh_too_long, zh_masked = _build_outlier_masks(
            zh_floors,
            zh_ceilings,
            zh_bounded_mask,
        )
        bounded_mask = arpa_bounded_mask | zh_bounded_mask
        too_short = arpa_too_short | zh_too_short
        too_long = arpa_too_long | zh_too_long
        masked = arpa_masked | zh_masked
        stats = {}
        stats.update(_summarize_mask("", bounded_mask, too_short, too_long))
        stats.update(_summarize_mask("arpa_", arpa_bounded_mask, arpa_too_short, arpa_too_long))
        stats.update(_summarize_mask("zh_", zh_bounded_mask, zh_too_short, zh_too_long))
        dur_loss_mask = ones * (~masked).to(x_mask.dtype)
        return dur_loss_mask, stats

    def iter_prior_encoder_parameters(self) -> Iterator[torch.nn.Parameter]:
        """Parameters for mu_x / mu_y prior path (excludes proj_w and spk_emb)."""
        for name, param in self.encoder.named_parameters():
            if name.startswith("proj_w."):
                continue
            yield param

    def iter_duration_predictor_parameters(self) -> Iterator[torch.nn.Parameter]:
        yield from self.encoder.proj_w.parameters()

    def iter_spk_emb_parameters(self) -> Iterator[torch.nn.Parameter]:
        if self.n_spks > 1:
            yield from self.spk_emb.parameters()

    def iter_decoder_parameters(self) -> Iterator[torch.nn.Parameter]:
        yield from self.decoder.parameters()

    def freeze_encoder(self) -> None:
        """Freeze encoder + duration predictor + spk_emb; only decoder keeps training."""
        n_frozen = 0
        for param in self.iter_prior_encoder_parameters():
            param.requires_grad = False
            n_frozen += param.numel()
        for param in self.iter_duration_predictor_parameters():
            param.requires_grad = False
            n_frozen += param.numel()
        for param in self.iter_spk_emb_parameters():
            param.requires_grad = False
            n_frozen += param.numel()
        self.encoder_frozen = True
        log.info(
            f"Froze encoder + duration predictor + spk_emb ({n_frozen:,} parameters); "
            "only decoder remains trainable"
        )

    def _build_optimizer_param_groups(self) -> Optional[list[dict[str, Any]]]:
        opt_kwargs = dict(self.hparams.optimizer.keywords)
        base_lr = float(opt_kwargs.get("lr", 1e-4))
        weight_decay = float(opt_kwargs.get("weight_decay", 0.0))
        prior_lr = float(self.hparams.prior_encoder_lr) if self.hparams.prior_encoder_lr is not None else base_lr
        dur_lr = (
            float(self.hparams.duration_predictor_lr)
            if self.hparams.duration_predictor_lr is not None
            else base_lr
        )
        groups = [
            {
                "params": list(self.iter_prior_encoder_parameters()),
                "lr": prior_lr,
                "weight_decay": weight_decay,
                "name": "prior_encoder",
            },
            {
                "params": list(self.iter_duration_predictor_parameters()),
                "lr": dur_lr,
                "weight_decay": weight_decay,
                "name": "duration_predictor",
            },
            {
                "params": list(self.iter_decoder_parameters()),
                "lr": base_lr,
                "weight_decay": weight_decay,
                "name": "decoder",
            },
        ]
        spk_params = list(self.iter_spk_emb_parameters())
        if spk_params:
            groups.insert(2, {
                "params": spk_params,
                "lr": base_lr,
                "weight_decay": weight_decay,
                "name": "spk_emb",
            })
        return groups

    @torch.inference_mode()
    def synthesise(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        n_timesteps: int,
        temperature: float = 1.0,
        spks: torch.Tensor = None,
        length_scale: float = 1.0,
        x_tones: torch.Tensor = None,
    ) -> dict:
        """
        Generates mel-spectrogram from text.
        Returns encoder outputs, decoder outputs, alignment, denormalized mel, mel lengths, and RTF.
        Args:
            x: (batch_size, max_text_length)
            x_lengths: (batch_size,)
            n_timesteps: number of steps for reverse diffusion
            temperature: controls variance of terminal distribution
            spks: (batch_size,), optional
            length_scale: controls speech pace
        Returns:
            dict with keys: encoder_outputs, decoder_outputs, attn, mel, mel_lengths, rtf
        """
        t = dt.datetime.now()
        if self.n_spks > 1 and spks is not None:
            spks = self.spk_emb(spks.long())
        if spks is not None and len(spks.shape) == 1 and len(spks.shape) < len(x.shape):
            spks = spks.unsqueeze(0)
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spks, x_tones=x_tones)
        w = torch.exp(logw) * x_mask
        w_ceil = torch.ceil(w) * length_scale
        w_ceil = self._clamp_inference_tone_durations(w_ceil, x, x_mask)
        if getattr(self, "apply_infer_duration_patches", False):
            w_ceil = apply_infer_duration_patches(w_ceil, x, x_mask, n_vocab=self.n_vocab)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = y_lengths.max()
        y_max_length_ = fix_len_compatibility(y_max_length)
        y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)
        encoder_outputs = mu_y[:, :, :y_max_length]
        decoder_outputs = self.decoder(mu_y, y_mask, n_timesteps, True, temperature, spks)
        decoder_outputs = decoder_outputs[:, :, :y_max_length]
        t = (dt.datetime.now() - t).total_seconds()
        rtf = t * 22050 / (decoder_outputs.shape[-1] * 256)
        return {
            "encoder_outputs": encoder_outputs,
            "decoder_outputs": decoder_outputs,
            "attn": attn[:, :, :y_max_length],
            "mel": denormalize(decoder_outputs, self.mel_mean, self.mel_std),
            "mel_lengths": y_lengths,
            "rtf": rtf,
        }
    
    @torch.inference_mode()
    def get_hidden_mel(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        spks: torch.Tensor = None,
        length_scale: float = 1.0,
        x_tones: torch.Tensor = None,
    ) -> dict:
        """
        Generate hidden mel features and masks from text input.
        Args:
            x: (batch_size, max_text_length)
            x_lengths: (batch_size,)
            n_timesteps: number of steps for reverse diffusion
            temperature: controls variance of terminal distribution
            spks: (batch_size,), optional
            length_scale: controls speech pace
            streaming: unused, for interface compatibility
        Returns:
            dict with keys: mu_y, y_mask, y_max_length, spks
        """
        t = dt.datetime.now()
        if self.n_spks > 1 and spks is not None:
            spks = self.spk_emb(spks.long())
            if spks.dim() == 1:
                spks = spks.unsqueeze(0)
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spks, x_tones=x_tones)
        w = torch.exp(logw) * x_mask
        w_ceil = torch.ceil(w) * length_scale
        w_ceil = self._clamp_inference_tone_durations(w_ceil, x, x_mask)
        if getattr(self, "apply_infer_duration_patches", False):
            w_ceil = apply_infer_duration_patches(w_ceil, x, x_mask, n_vocab=self.n_vocab)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = y_lengths.max()
        y_max_length_ = fix_len_compatibility(y_max_length)
        y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)
        return {
            "mu_y": mu_y,
            "y_mask": y_mask,
            "y_max_length": y_max_length,
            "spks": spks,
        }

    def get_mel(
        self,
        mu_y: torch.Tensor,
        y_mask: torch.Tensor,
        n_timesteps: int,
        temperature: float,
        spks: torch.Tensor = None,
        finalize = True,
        streaming: bool = True,
        z: torch.Tensor = None,
        chunk_start: int = 0,
    ) -> torch.Tensor:
        """
        Generate denormalized mel-spectrogram from hidden features.
        Args:
            mu_y: (batch_size, n_feats, max_mel_length)
            y_mask: (batch_size, 1, max_mel_length)
            n_timesteps: number of steps for reverse diffusion
            temperature: controls variance of terminal distribution
            spks: (batch_size,), optional
            streaming: whether to use streaming mode in decoder
            z: optional pre-generated noise for consistent streaming chunks
            chunk_start: first mel frame index of the current streaming chunk
        Returns:
            mel: (batch_size, n_feats, max_mel_length)
        """
        decoder_outputs = self.decoder(
            mu_y,
            y_mask,
            n_timesteps,
            finalize,
            temperature,
            spks,
            streaming=streaming,
            z=z,
            chunk_start=chunk_start,
        )
        return denormalize(decoder_outputs, self.mel_mean, self.mel_std)

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        y: torch.Tensor,
        y_lengths: torch.Tensor,
        spks: torch.Tensor = None,
        out_size: int = None,
        cond=None,
        durations=None,
        x_tones: torch.Tensor = None,
    ):
        """
        Computes duration loss, prior loss, flow matching loss, and alignment.
        Args:
            x: (batch_size, max_text_length)
            x_lengths: (batch_size,)
            y: (batch_size, n_feats, max_mel_length)
            y_lengths: (batch_size,)
            spks: (batch_size,), optional
            out_size: segment length for decoder training
        Returns:
            dur_loss, prior_loss, diff_loss, attn, arpa_mask_stats
        """
        spk_ids = spks
        if self.n_spks > 1:
            spks = self.spk_emb(spks)
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spks, x_tones=x_tones)
        y_max_length = y.shape[-1]
        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        mas_floors = None
        mas_ceilings = None
        mas_stats: dict = {}
        if self.use_precomputed_durations:
            attn = generate_path(durations.squeeze(1), attn_mask.squeeze(1))
        else:
            with torch.no_grad():
                const = -0.5 * math.log(2 * math.pi) * self.n_feats
                factor = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype, device=mu_x.device)
                y_square = torch.matmul(factor.transpose(1, 2), y**2)
                y_mu_double = torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), y)
                mu_square = torch.sum(factor * (mu_x**2), 1).unsqueeze(-1)
                log_prior = y_square - y_mu_double + mu_square + const
                mas_floors = self._mas_floor_frames(x, spk_ids, x_tones)
                mas_ceilings = self._mas_ceiling_frames(x, spk_ids, x_tones)
                if mas_floors is not None or mas_ceilings is not None:
                    _zero = torch.zeros(x.size(0), x.size(1), device=x.device)
                    attn, mas_stats = monotonic_align.maximum_path_constrained(
                        log_prior,
                        attn_mask.squeeze(1),
                        mas_floors if mas_floors is not None else _zero,
                        mas_ceilings,
                    )
                else:
                    attn = monotonic_align.maximum_path(log_prior, attn_mask.squeeze(1))
                attn = attn.detach()
        duration_targets = torch.sum(attn.unsqueeze(1), -1)
        dur_loss_mask, duration_mask_stats = self._mask_duration_outliers(
            x,
            duration_targets,
            x_mask,
            spk_ids,
            x_tones,
        )
        if self.duration_constrained_mas and not self.use_precomputed_durations:
            # Every rank must log the exact same set of keys: with sync_dist
            # the epoch-end reduction runs one collective per metric, and a
            # rank whose batch has no bounded tokens (e.g. a batch with no bounded
            # tokens) would otherwise desynchronize DDP and crash NCCL.
            full_mas_stats = {
                "constrained_score_gap_per_frame_mean": 0.0,
                "constrained_score_gap_per_frame_max": 0.0,
                "floor_rescaled_sample_ratio": 0.0,
                "ceiling_rescaled_sample_ratio": 0.0,
                "ceiling_binding_token_ratio": 0.0,
                "infeasible_sample_ratio": 0.0,
                "floor_binding_token_ratio": 0.0,
                "sil_target_mean_frames": 0.0,
            }
            full_mas_stats.update(mas_stats)
            if mas_floors is not None:
                rounded_floors = torch.round(mas_floors)
                bounded = (rounded_floors > 0) & (x_mask.squeeze(1) > 0)
                binding = bounded & (duration_targets.squeeze(1) <= rounded_floors)
                full_mas_stats["floor_binding_token_ratio"] = (
                    binding.float().sum() / bounded.float().sum().clamp_min(1.0)
                )
            if mas_ceilings is not None:
                rounded_ceilings = torch.round(mas_ceilings)
                bounded = (rounded_ceilings > 0) & (x_mask.squeeze(1) > 0)
                binding = bounded & (duration_targets.squeeze(1) >= rounded_ceilings)
                full_mas_stats["ceiling_binding_token_ratio"] = (
                    binding.float().sum() / bounded.float().sum().clamp_min(1.0)
                )
            sil_mask = (x == 1) & (x_mask.squeeze(1) > 0)
            if bool(sil_mask.any()):
                full_mas_stats["sil_target_mean_frames"] = (
                    duration_targets.squeeze(1)[sil_mask].float().mean()
                )
            duration_mask_stats.update(full_mas_stats)
        logw_ = torch.log(1e-8 + duration_targets) * x_mask
        dur_loss = duration_loss(logw, logw_, x_lengths, mask=dur_loss_mask)
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)
        diff_loss, _ = self.decoder.compute_loss(x1=y, mask=y_mask, mu=mu_y, spks=spks, cond=cond)
        if self.prior_loss:
            prior_loss = torch.sum(0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask)
            prior_loss = prior_loss / (torch.sum(y_mask) * self.n_feats)
        else:
            prior_loss = 0
        return dur_loss, prior_loss, diff_loss, attn, duration_mask_stats
