"""Shared Bopomofo helpers for zh-en symbol sets and cleaners."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

BOPOMOFO_INITIALS = frozenset("ㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎㄏㄐㄑㄒㄓㄔㄕㄖㄗㄘㄙ")
BOPOMOFO_TONES = ("ˉ", "ˊ", "ˇ", "ˋ", "˙")
PINYIN_TONE_TO_BPMF = {"0": "˙", "5": "˙", "6": "ˊ", "1": "ˉ", "2": "ˊ", "3": "ˇ", "4": "ˋ"}

# Tone ids for the tone-embedding paradigm: 0 = no tone (initials, English,
# punctuation, blanks), 1..5 = the five Bopomofo tone marks in BOPOMOFO_TONES order.
TONE_MARK_TO_ID = {mark: idx + 1 for idx, mark in enumerate(BOPOMOFO_TONES)}
N_BOPOMOFO_TONES = len(BOPOMOFO_TONES)

_pinyin_2_bpmf_cache: dict[str, str] | None = None


def load_pinyin_2_bpmf() -> dict[str, str]:
    global _pinyin_2_bpmf_cache
    if _pinyin_2_bpmf_cache is not None:
        return _pinyin_2_bpmf_cache

    pinyin_file = REPO_ROOT / "lits" / "text" / "sources" / "pinyin_2_bpmf.txt"
    mapping: dict[str, str] = {}
    with open(pinyin_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pinyin, bpmf = line.split("\t")
            mapping[pinyin] = bpmf
    _pinyin_2_bpmf_cache = mapping
    return mapping


def split_bpmf_body(bpmf: str) -> tuple[str, str]:
    """Split a toneless Bopomofo syllable into (initial, rhyme_body).

    For whole-syllable zhuyin such as ㄓ/ㄔ/ㄕ/ㄖ/ㄗ/ㄘ/ㄙ, the rhyme body is
    the full syllable and the initial is empty so tone attaches to the nucleus.
    """
    idx = 0
    while idx < len(bpmf) and bpmf[idx] in BOPOMOFO_INITIALS:
        idx += 1
    initial = bpmf[:idx]
    rhyme = bpmf[idx:]
    if not rhyme:
        return "", bpmf
    return initial, rhyme


def collect_rhyme_bodies(pinyin_2_bpmf: dict[str, str] | None = None) -> list[str]:
    if pinyin_2_bpmf is None:
        pinyin_2_bpmf = load_pinyin_2_bpmf()
    bodies = {rhyme for _, rhyme in (split_bpmf_body(bpmf) for bpmf in pinyin_2_bpmf.values()) if rhyme}
    return sorted(bodies)


def bpmf_syllable_to_tokens(
    bpmf: str,
    tone_mark: str,
    *,
    rhyme_body_tone: bool = False,
) -> list[str]:
    """Convert one toneless Bopomofo syllable + tone mark to cleaner tokens."""
    initial, rhyme = split_bpmf_body(bpmf)
    if rhyme_body_tone:
        tokens: list[str] = []
        if initial:
            tokens.append(initial)
        if rhyme:
            tokens.append(rhyme)
        tokens.append(tone_mark)
        return tokens

    tokens = list(bpmf)
    tokens.append(tone_mark)
    return tokens
