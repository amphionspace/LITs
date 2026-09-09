"""Cleaner hook for LITs inference: Chinese hanzi lexicon + English CMUdict G2P."""

from __future__ import annotations

import sys
from pathlib import Path
import re

_PKG_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

from english_frontend import preprocess_english_input  # noqa: E402

_HANZI_BEFORE_ARPA_RE = re.compile(r"([\u4e00-\u9fff])(?=[A-Z])")
_ARPA_BEFORE_HANZI_RE = re.compile(r"([A-Z]{1,3}[012]?)(?=[\u4e00-\u9fff])")


def _is_pre_tokenized_zh_en_input(text: str) -> bool:
    """True when input already uses numbered pinyin and/or ARPAbet (training format).

    CMUdict G2P must be skipped for this path; otherwise tokens like ``wo3`` /
    ``AH0`` are misread as English words.
    """
    from english_frontend import _detach_trailing_punct, _is_english_word_token, is_arpa_phoneme
    from lits.text.language_cleaners import (
        _hanzi_char_re,
        _is_punctuation_token,
        _pinyin_syllable_re,
    )

    if _hanzi_char_re.search(text):
        return False

    has_pinyin_or_arpa = False
    has_raw_english = False
    for raw in text.split():
        if raw in ("_", "/", "|"):
            continue
        for piece in _detach_trailing_punct(raw):
            if not piece or _is_punctuation_token(piece):
                continue
            if _pinyin_syllable_re.match(piece) or is_arpa_phoneme(piece):
                has_pinyin_or_arpa = True
            elif _is_english_word_token(piece):
                has_raw_english = True

    return has_pinyin_or_arpa and not has_raw_english


def en_zh_dict_mixed_cleaners(text: str) -> str:
    """Hybrid cleaner: hanzi -> lexicon pinyin; English text -> CMUdict ARPAbet.

    Expects TN-normalized text from ``infer_e2e.sh`` for raw numbers, currency,
    plates, and similar spoken-form rules (unified ``data/zh`` TN profile).
    English words are resolved via CMUdict, supplement_lexicon, or letter-wise fallback.
    """
    return _en_zh_dict_mixed_cleaners_impl(text, rhyme_body_tone=False)


def en_zh_dict_mixed_rhyme_body_tone_cleaners(text: str) -> str:
    """Rhyme-body + inline tone mark paradigm (no syllable-boundary ``_``)."""
    return _en_zh_dict_mixed_cleaners_impl(text, rhyme_body_tone=True)


def _en_zh_dict_mixed_cleaners_impl(
    text: str,
    *,
    rhyme_body_tone: bool = False,
) -> str:
    if not text:
        return ""

    from english_frontend import is_arpabet_input, normalize_arpabet_input
    from lits.text.language_cleaners import (
        _hanzi_char_re,
        _hanzi_to_bopomofo_tokens,
        _pinyin_to_bopomofo_tokens,
    )

    if is_arpabet_input(text):
        normalized = normalize_arpabet_input(text, add_sentence_end=True)
        return _pinyin_to_bopomofo_tokens(
            normalized,
            rhyme_body_tone=rhyme_body_tone,
        )

    if _is_pre_tokenized_zh_en_input(text):
        return _pinyin_to_bopomofo_tokens(
            text,
            rhyme_body_tone=rhyme_body_tone,
        )

    arpa_line = preprocess_english_input(text)
    arpa_line = _HANZI_BEFORE_ARPA_RE.sub(r"\1 ", arpa_line)
    arpa_line = _ARPA_BEFORE_HANZI_RE.sub(r"\1 ", arpa_line)
    if _hanzi_char_re.search(arpa_line):
        return _hanzi_to_bopomofo_tokens(
            arpa_line,
            rhyme_body_tone=rhyme_body_tone,
        )
    return _pinyin_to_bopomofo_tokens(
        arpa_line,
        rhyme_body_tone=rhyme_body_tone,
    )
