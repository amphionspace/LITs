"""Inference-only duration patches (edit constants in this file).

Rules applied when ``--infer_duration_patches`` / ``INFER_DURATION_PATCHES=1``:
  English ARPAbet vowel floor — fix collapsed vowels (e.g. AW1 in "how").
  English affricate floor — fix collapsed JH/CH.
  Zh zero-initial single-final floor — fix collapsed interjections (e.g. ㄣ in "嗯？").
  Zh pause caps — shorten ``_`` / punctuation pauses in Bopomofo text.

Leading-silence / onset inflation is handled at training time via a leading
``<sil>`` token and constrained MAS, not inference patches.
"""

from __future__ import annotations

import os
from functools import lru_cache

import torch

from lits.text.char_symbols.langs.ARPA import ARPA_VOWEL_TOKENS
from lits.text.char_symbols.symbol_inventories import lang2inventory

# --- English vowel floor ---
# 0 = disable. Only ARPAbet vowels; consonants / punct / Bopomofo skipped.
INFER_EN_PHONE_DUR_COLLAPSE_MAX: float = 3.0
INFER_EN_PHONE_DUR_FLOOR: float = 5.0

# --- English affricate floor (JH / CH) ---
# Fix collapsed affricates (e.g. JH in EY1-JH-AE2) that sound like stops.
# 0 = disable floor. Applied whenever infer_duration_patches is on.
INFER_EN_AFFRICATE_DUR_COLLAPSE_MAX: float = float(
    os.environ.get("INFER_EN_AFFRICATE_DUR_COLLAPSE_MAX", "12")
)
INFER_EN_AFFRICATE_DUR_FLOOR: float = float(
    os.environ.get("INFER_EN_AFFRICATE_DUR_FLOOR", "0")
)
_EN_AFFRICATE_TOKENS = frozenset({"JH", "CH"})

# --- Zh zero-initial single-final floor (嗯 / 啊 / 哦 …) ---
# Only syllables shaped like ``ㄣ ˊ`` (one 韵母 + tone, no 声母).
# Multi-final syllables (``ㄨ ㄛ ˇ``) and normal ``声母+韵母`` are untouched.
INFER_ZH_FINAL_DUR_COLLAPSE_MAX: float = 1.0
INFER_ZH_FINAL_DUR_FLOOR: float = 5.0

# --- Zh pause caps (Bopomofo text) ---
# 0 = disable each rule separately.
INFER_ZH_WORD_SEP_DUR_CAP: float = 15.0          # 字 _ 字
INFER_ZH_PUNCT_ADJ_SEP_DUR_CAP: float = 30      # 字 _ ,  or  , _ 字
INFER_ZH_PUNCT_DUR_CAP: float = 3.0             # , ; : in zh context
# INFER_ZH_SENT_END_PUNCT_DUR_CAP: float = 4.0     # . ! ? in zh context

_ARPA_VOWEL_SET = frozenset(ARPA_VOWEL_TOKENS)
_BOPOMOFO_INITIALS = frozenset("ㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎㄏㄐㄑㄒㄓㄔㄕㄖㄗㄘㄙ")
_BOPOMOFO_FINALS = frozenset("ㄚㄛㄜㄝㄞㄟㄠㄡㄢㄣㄤㄥㄦㄧㄨㄩ")
_BOPOMOFO_TONES = frozenset("ˉˊˇˋ˙")
_BOPOMOFO_SYMBOLS = _BOPOMOFO_INITIALS | _BOPOMOFO_FINALS | _BOPOMOFO_TONES
_LIGHT_PUNCT_SYMBOLS = frozenset([",", ";", ":"])
_SENT_END_PUNCT_SYMBOLS = frozenset([".", "!", "?"])


def _inventory_for_vocab(n_vocab: int) -> dict | None:
    for payload in lang2inventory.values():
        if len(payload.get("symbols", [])) == n_vocab:
            return payload
    return None


@lru_cache(maxsize=8)
def english_arpa_affricate_token_ids(n_vocab: int) -> frozenset[int]:
    """JH/CH token ids for the symbol inventory matching *n_vocab*."""
    payload = _inventory_for_vocab(n_vocab)
    if payload is None:
        return frozenset()
    affricate_ids: set[int] = set()
    id_to_symbol = payload.get("id_to_symbol") or {}
    for token_id, symbol in id_to_symbol.items():
        if symbol in _EN_AFFRICATE_TOKENS:
            affricate_ids.add(int(token_id))
    return frozenset(affricate_ids)


@lru_cache(maxsize=8)
def english_arpa_vowel_token_ids(n_vocab: int) -> frozenset[int]:
    """Vowel token ids for the symbol inventory matching *n_vocab*."""
    payload = _inventory_for_vocab(n_vocab)
    if payload is None:
        return frozenset()
    vowel_ids: set[int] = set()
    id_to_symbol = payload.get("id_to_symbol") or {}
    for token_id, symbol in id_to_symbol.items():
        if symbol in _ARPA_VOWEL_SET:
            vowel_ids.add(int(token_id))
    return frozenset(vowel_ids)


@lru_cache(maxsize=8)
def bopomofo_initial_token_ids(n_vocab: int) -> frozenset[int]:
    """Bopomofo 声母 token ids."""
    payload = _inventory_for_vocab(n_vocab)
    if payload is None:
        return frozenset()
    ids: set[int] = set()
    symbol_to_id = payload.get("symbol_to_id") or {}
    for symbol in _BOPOMOFO_INITIALS:
        token_id = symbol_to_id.get(symbol)
        if token_id is not None:
            ids.add(int(token_id))
    return frozenset(ids)


@lru_cache(maxsize=8)
def bopomofo_tone_token_ids(n_vocab: int) -> frozenset[int]:
    """Bopomofo tone-mark token ids."""
    payload = _inventory_for_vocab(n_vocab)
    if payload is None:
        return frozenset()
    ids: set[int] = set()
    symbol_to_id = payload.get("symbol_to_id") or {}
    for symbol in _BOPOMOFO_TONES:
        token_id = symbol_to_id.get(symbol)
        if token_id is not None:
            ids.add(int(token_id))
    return frozenset(ids)


@lru_cache(maxsize=8)
def bopomofo_final_token_ids(n_vocab: int) -> frozenset[int]:
    """Bopomofo 韵母 token ids (zh-en-direct style inventories)."""
    payload = _inventory_for_vocab(n_vocab)
    if payload is None:
        return frozenset()
    ids: set[int] = set()
    symbol_to_id = payload.get("symbol_to_id") or {}
    for symbol in _BOPOMOFO_FINALS:
        token_id = symbol_to_id.get(symbol)
        if token_id is not None:
            ids.add(int(token_id))
    return frozenset(ids)


@lru_cache(maxsize=8)
def bopomofo_token_ids(n_vocab: int) -> frozenset[int]:
    """Bopomofo + tone-mark token ids (zh-en-direct style inventories)."""
    payload = _inventory_for_vocab(n_vocab)
    if payload is None:
        return frozenset()
    ids: set[int] = set()
    symbol_to_id = payload.get("symbol_to_id") or {}
    for symbol in _BOPOMOFO_SYMBOLS:
        token_id = symbol_to_id.get(symbol)
        if token_id is not None:
            ids.add(int(token_id))
    return frozenset(ids)


@lru_cache(maxsize=8)
def word_sep_token_id(n_vocab: int) -> int | None:
    payload = _inventory_for_vocab(n_vocab)
    if payload is None:
        return None
    symbol_to_id = payload.get("symbol_to_id") or {}
    if "_" not in symbol_to_id:
        return None
    return int(symbol_to_id["_"])


@lru_cache(maxsize=8)
def zh_punctuation_cap_by_token_id(n_vocab: int) -> dict[int, float]:
    """Per-token duration cap for punctuation symbols."""
    payload = _inventory_for_vocab(n_vocab)
    if payload is None:
        return {}
    caps: dict[int, float] = {}
    id_to_symbol = payload.get("id_to_symbol") or {}
    for token_id, symbol in id_to_symbol.items():
        if symbol in _LIGHT_PUNCT_SYMBOLS:
            caps[int(token_id)] = INFER_ZH_PUNCT_DUR_CAP
        # elif symbol in _SENT_END_PUNCT_SYMBOLS:
        #     caps[int(token_id)] = INFER_ZH_SENT_END_PUNCT_DUR_CAP
    return caps


def _text_mask(x_mask: torch.Tensor) -> torch.Tensor:
    return x_mask.squeeze(1).bool() if x_mask.dim() == 3 else x_mask.bool()


def _cap_tensor(
    w_ceil: torch.Tensor,
    durations: torch.Tensor,
    need: torch.Tensor,
    cap: float,
) -> torch.Tensor:
    if not need.any():
        return w_ceil
    return torch.where(need.unsqueeze(1), durations.new_full((), cap), w_ceil)


def _cap_tensor_per_position(
    w_ceil: torch.Tensor,
    durations: torch.Tensor,
    need: torch.Tensor,
    caps: torch.Tensor,
) -> torch.Tensor:
    if not need.any():
        return w_ceil
    return torch.where(need.unsqueeze(1), caps, w_ceil)


def apply_en_phone_duration_floor(
    w_ceil: torch.Tensor,
    x: torch.Tensor,
    x_mask: torch.Tensor,
    *,
    n_vocab: int,
) -> torch.Tensor:
    """Raise collapsed English vowel durations only."""
    floor = INFER_EN_PHONE_DUR_FLOOR
    collapse_max = INFER_EN_PHONE_DUR_COLLAPSE_MAX
    if floor <= 0 or collapse_max <= 0:
        return w_ceil

    vowel_ids = english_arpa_vowel_token_ids(n_vocab)
    if not vowel_ids:
        return w_ceil

    is_vowel = torch.zeros(n_vocab, dtype=torch.bool, device=x.device)
    for token_id in vowel_ids:
        is_vowel[token_id] = True

    text_mask = _text_mask(x_mask)
    en_vowel = is_vowel[x] & text_mask
    durations = w_ceil.squeeze(1)
    collapsed = en_vowel & (durations > 0) & (durations <= collapse_max)
    need_floor = collapsed & (durations < floor)
    if not need_floor.any():
        return w_ceil

    return torch.where(need_floor.unsqueeze(1), durations.new_full((), floor), w_ceil)


def apply_zh_bopomofo_final_duration_floor(
    w_ceil: torch.Tensor,
    x: torch.Tensor,
    x_mask: torch.Tensor,
    *,
    n_vocab: int,
) -> torch.Tensor:
    """Raise collapsed 韵母 only in zero-initial single-final syllables (嗯, 啊, …)."""
    floor = INFER_ZH_FINAL_DUR_FLOOR
    collapse_max = INFER_ZH_FINAL_DUR_COLLAPSE_MAX
    if floor <= 0 or collapse_max <= 0:
        return w_ceil

    final_ids = bopomofo_final_token_ids(n_vocab)
    tone_ids = bopomofo_tone_token_ids(n_vocab)
    sep_id = word_sep_token_id(n_vocab)
    if not final_ids or not tone_ids:
        return w_ceil

    is_final = torch.zeros(n_vocab, dtype=torch.bool, device=x.device)
    for token_id in final_ids:
        is_final[token_id] = True
    is_tone = torch.zeros(n_vocab, dtype=torch.bool, device=x.device)
    for token_id in tone_ids:
        is_tone[token_id] = True

    text_mask = _text_mask(x_mask)
    zh_final = is_final[x] & text_mask

    syllable_start = torch.zeros_like(text_mask)
    syllable_start[:, 0] = text_mask[:, 0]
    if sep_id is not None:
        prev_sep = torch.zeros_like(text_mask)
        prev_sep[:, 1:] = (x[:, :-1] == sep_id) & text_mask[:, :-1]
        syllable_start[:, 1:] = prev_sep[:, 1:] & text_mask[:, 1:]

    next_token = torch.zeros_like(x)
    next_token[:, :-1] = x[:, 1:]
    next_in_text = torch.zeros_like(text_mask)
    next_in_text[:, :-1] = text_mask[:, 1:]
    next_is_tone = is_tone[next_token] & next_in_text

    # ``ㄣ ˊ`` yes; ``ㄨ ㄛ ˇ`` / ``ㄋ ㄧ ˇ`` no
    single_final_syllable = zh_final & syllable_start & next_is_tone

    durations = w_ceil.squeeze(1)
    collapsed = single_final_syllable & (durations > 0) & (durations <= collapse_max)
    need_floor = collapsed & (durations < floor)
    if not need_floor.any():
        return w_ceil

    return torch.where(need_floor.unsqueeze(1), durations.new_full((), floor), w_ceil)


def apply_zh_word_sep_duration_cap(
    w_ceil: torch.Tensor,
    x: torch.Tensor,
    x_mask: torch.Tensor,
    *,
    n_vocab: int,
) -> torch.Tensor:
    """Cap zh ``_`` and punctuation pauses; English ``_`` / ARPA punct untouched."""
    sep_id = word_sep_token_id(n_vocab)
    bopo_ids = bopomofo_token_ids(n_vocab)
    punct_caps = zh_punctuation_cap_by_token_id(n_vocab)
    if sep_id is None or not bopo_ids:
        return w_ceil

    is_bopo = torch.zeros(n_vocab, dtype=torch.bool, device=x.device)
    for token_id in bopo_ids:
        is_bopo[token_id] = True

    is_punct = torch.zeros(n_vocab, dtype=torch.bool, device=x.device)
    punct_cap_table = torch.zeros(n_vocab, dtype=w_ceil.dtype, device=x.device)
    for token_id, cap in punct_caps.items():
        if token_id < n_vocab and cap > 0:
            is_punct[token_id] = True
            punct_cap_table[token_id] = cap

    text_mask = _text_mask(x_mask)
    bopo_at = is_bopo[x] & text_mask
    punct_at = is_punct[x] & text_mask
    is_sep = (x == sep_id) & text_mask

    prev_bopo = torch.zeros_like(bopo_at)
    prev_bopo[:, 1:] = bopo_at[:, :-1]
    next_bopo = torch.zeros_like(bopo_at)
    next_bopo[:, :-1] = bopo_at[:, 1:]
    prev_punct = torch.zeros_like(punct_at)
    prev_punct[:, 1:] = punct_at[:, :-1]
    next_punct = torch.zeros_like(punct_at)
    next_punct[:, :-1] = punct_at[:, 1:]

    durations = w_ceil.squeeze(1)

    # 字 _ 字
    if INFER_ZH_WORD_SEP_DUR_CAP > 0:
        zh_between = is_sep & prev_bopo & next_bopo
        need = zh_between & (durations > INFER_ZH_WORD_SEP_DUR_CAP)
        w_ceil = _cap_tensor(w_ceil, durations, need, INFER_ZH_WORD_SEP_DUR_CAP)
        durations = w_ceil.squeeze(1)

    # 字 _ ,  /  , _ 字
    if INFER_ZH_PUNCT_ADJ_SEP_DUR_CAP > 0:
        punct_adj_sep = is_sep & ((prev_bopo & next_punct) | (prev_punct & next_bopo))
        need = punct_adj_sep & (durations > INFER_ZH_PUNCT_ADJ_SEP_DUR_CAP)
        w_ceil = _cap_tensor(w_ceil, durations, need, INFER_ZH_PUNCT_ADJ_SEP_DUR_CAP)
        durations = w_ceil.squeeze(1)

    # punctuation in zh context (Bopomofo within one ``_`` hop, or direct neighbor)
    if punct_caps:
        prev_sep = torch.zeros_like(is_sep)
        prev_sep[:, 1:] = is_sep[:, :-1]
        next_sep = torch.zeros_like(is_sep)
        next_sep[:, :-1] = is_sep[:, 1:]
        prev2_bopo = torch.zeros_like(bopo_at)
        prev2_bopo[:, 2:] = bopo_at[:, :-2]
        next2_bopo = torch.zeros_like(bopo_at)
        next2_bopo[:, :-2] = bopo_at[:, 2:]
        zh_punct_ctx = punct_at & (
            prev_bopo
            | next_bopo
            | (prev_sep & prev2_bopo)
            | (next_sep & next2_bopo)
        )
        cap_at_pos = punct_cap_table[x]
        need = zh_punct_ctx & (durations > cap_at_pos) & (cap_at_pos > 0)
        w_ceil = _cap_tensor_per_position(w_ceil, durations, need, cap_at_pos)

    return w_ceil


def apply_en_affricate_duration_floor(
    w_ceil: torch.Tensor,
    x: torch.Tensor,
    x_mask: torch.Tensor,
    *,
    n_vocab: int,
    collapse_max: float | None = None,
    floor: float | None = None,
) -> torch.Tensor:
    """Raise collapsed English affricate (JH/CH) durations."""
    collapse_max = INFER_EN_AFFRICATE_DUR_COLLAPSE_MAX if collapse_max is None else collapse_max
    floor = INFER_EN_AFFRICATE_DUR_FLOOR if floor is None else floor
    if floor <= 0:
        return w_ceil

    affricate_ids = english_arpa_affricate_token_ids(n_vocab)
    if not affricate_ids:
        return w_ceil

    affricate_mask = torch.zeros_like(x, dtype=torch.bool)
    for token_id in affricate_ids:
        affricate_mask |= x == token_id
    affricate_mask &= x_mask.squeeze(1).bool()

    durations = w_ceil.squeeze(1)
    collapsed = affricate_mask & (durations > 0) & (durations <= collapse_max)
    need_floor = collapsed & (durations < floor)
    if not need_floor.any():
        return w_ceil
    return torch.where(need_floor.unsqueeze(1), durations.new_full((), floor), w_ceil)


def apply_infer_duration_patches(
    w_ceil: torch.Tensor,
    x: torch.Tensor,
    x_mask: torch.Tensor,
    *,
    n_vocab: int,
    scope: str | None = None,
) -> torch.Tensor:
    """Apply inference duration patches (en/zh floor/cap rules)."""
    del scope  # kept for call-site compatibility; no longer used
    w_ceil = apply_en_affricate_duration_floor(w_ceil, x, x_mask, n_vocab=n_vocab)
    w_ceil = apply_en_phone_duration_floor(w_ceil, x, x_mask, n_vocab=n_vocab)
    w_ceil = apply_zh_bopomofo_final_duration_floor(w_ceil, x, x_mask, n_vocab=n_vocab)
    w_ceil = apply_zh_word_sep_duration_cap(w_ceil, x, x_mask, n_vocab=n_vocab)
    return w_ceil
