"""Shared case-sensitive acronym rules (g2p_acronym_case)."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

PronunciationMode = Literal["word", "spell"]
CmudictAction = int | Literal["spell"] | None


def alpha_case_class(word: str) -> str | None:
    letters = [ch for ch in word if ch.isalpha()]
    if len(letters) < 2:
        return None
    if all(ch.isupper() for ch in letters):
        return "upper"
    if all(ch.islower() for ch in letters):
        return "lower"
    return None


@lru_cache(maxsize=1)
def _load_doc() -> dict[str, Any]:
    from lits.text.frontend_rules.loader import load_rules

    return load_rules("g2p_acronym_case", "en")


def get_case_entry(word: str) -> dict[str, Any] | None:
    return _load_doc().get("entries", {}).get(word.strip().upper())


def get_case_rule(word: str) -> dict[str, Any] | None:
    case_class = alpha_case_class(word)
    if case_class is None:
        return None
    entry = get_case_entry(word)
    if not entry:
        return None
    rule = entry.get(case_class)
    return rule if isinstance(rule, dict) else None


def get_case_pronunciation_mode(word: str) -> PronunciationMode | None:
    """Backend-neutral reading: whole word vs letter-by-letter."""
    rule = get_case_rule(word)
    if not rule:
        return None
    mode = rule.get("mode")
    if mode in ("word", "spell"):
        return mode
    return None


def resolve_cmudict_action(word: str, num_variants: int) -> CmudictAction:
    """Map shared rule to CMUdict variant index or spell fallback."""
    rule = get_case_rule(word)
    if not rule:
        return None

    cmudict = rule.get("cmudict", {})
    if isinstance(cmudict, dict) and "pick" in cmudict:
        pick = int(cmudict["pick"])
        if 0 <= pick < num_variants:
            return pick
        return None

    mode = rule.get("mode")
    if mode == "word":
        return 0 if num_variants > 0 else None
    if mode == "spell":
        if num_variants > 1:
            return 1
        return "spell"

    # Legacy CMUdict-only rules: top-level pick / mode spell.
    if rule.get("mode") == "spell":
        return "spell"
    if "pick" in rule:
        pick = int(rule["pick"])
        if 0 <= pick < num_variants:
            return pick
    return None
