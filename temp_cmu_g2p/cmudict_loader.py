"""Load CMUdict (0.7b or tab-separated) into a word -> phoneme list mapping."""

from __future__ import annotations

import re
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent
DEFAULT_CMUDICT_PATH = _PKG_ROOT / "data" / "cmudict-0.7b"


def _normalize_word_key(word: str) -> str:
    return word.strip().upper()


_VARIANT_SUFFIX_RE = re.compile(r"^(.*)\((\d+)\)$")


def _parse_word_key_and_variant(word: str) -> tuple[str, int]:
    """Split CMUdict ``WORD(N)`` notation into base key and 0-based variant index."""
    raw = word.strip()
    match = _VARIANT_SUFFIX_RE.match(raw)
    if match:
        return _normalize_word_key(match.group(1)), int(match.group(2))
    return _normalize_word_key(raw), 0


def _parse_phoneme_token(token: str) -> str:
    token = token.strip()
    if not token:
        raise ValueError("empty phoneme token")
    return token.upper()


def parse_cmudict_line(line: str) -> tuple[str, int, list[str]] | None:
    """Parse one CMUdict entry line. Returns (word_key, variant_idx, phonemes) or None."""
    stripped = line.strip()
    if not stripped or stripped.startswith(";;;"):
        return None

    if "\t" in stripped:
        word, phones_field = stripped.split("\t", 1)
    else:
        parts = stripped.split("  ", 1)
        if len(parts) != 2:
            parts = stripped.split(None, 1)
        if len(parts) != 2:
            return None
        word, phones_field = parts

    word_key, variant_idx = _parse_word_key_and_variant(word)
    phonemes = [_parse_phoneme_token(tok) for tok in phones_field.split()]
    if not phonemes:
        return None
    return word_key, variant_idx, phonemes


class CMUDict:
    """In-memory CMUdict with primary + numbered variant pronunciations."""

    def __init__(self, entries: dict[str, list[list[str]]]) -> None:
        self._entries = entries

    @classmethod
    def from_file(cls, path: str | Path) -> CMUDict:
        path = Path(path)
        raw: dict[str, list[list[str]]] = {}
        with path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                parsed = parse_cmudict_line(line)
                if parsed is None:
                    continue
                word_key, variant_idx, phonemes = parsed
                variants = raw.setdefault(word_key, [])
                while len(variants) <= variant_idx:
                    variants.append([])
                variants[variant_idx] = phonemes
        return cls(raw)

    def has(self, word: str) -> bool:
        return _normalize_word_key(word) in self._entries

    def lookup(self, word: str, variant: int = 0) -> list[str] | None:
        key = _normalize_word_key(word)
        variants = self._entries.get(key)
        if not variants:
            return None
        if variant < 0 or variant >= len(variants):
            variant = 0
        return list(variants[variant])

    def lookup_keys(self, word: str) -> list[str]:
        raw = word.strip()
        if not raw:
            return []

        keys: list[str] = []
        seen: set[str] = set()

        def add(candidate: str) -> None:
            key = _normalize_word_key(candidate)
            if key and key not in seen:
                seen.add(key)
                keys.append(key)

        add(raw)
        add(raw.upper())
        add(raw.lower())

        if raw[0] in {"'", "’"}:
            add(raw)
        elif raw[0].isalpha():
            add("'" + raw)

        if raw.endswith(("'s", "'S", "'s", "’s")):
            add(raw[:-2])
        if raw.endswith(("'", "’")):
            add(raw[:-1])

        return keys

    def lookup_word(self, word: str) -> list[str] | None:
        for key in self.lookup_keys(word):
            phonemes = self.lookup(key)
            if phonemes is not None:
                return phonemes
        return None

    def lookup_variants(self, word: str) -> list[list[str]] | None:
        key = _normalize_word_key(word)
        variants = self._entries.get(key)
        if not variants:
            return None
        return [list(phonemes) for phonemes in variants if phonemes]

    def __len__(self) -> int:
        return len(self._entries)
