"""Letter-dot abbreviation normalization (rules in frontend_rules)."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any


@lru_cache(maxsize=1)
def _abbrev_pattern() -> re.Pattern[str]:
    from lits.text.frontend_rules.loader import load_rules

    doc = load_rules("abbreviation", "en")
    raw = doc["resources"]["pattern_defs"]["letter_dot_abbrev"]
    return re.compile(raw)


def collapse_letter_dot_abbreviations(text: str) -> str:
    """``U.S.`` / ``U.S.A.`` / ``Ph.D.`` → ``US`` / ``USA`` / ``PhD`` (no internal periods)."""
    if not text:
        return text

    def repl(match: re.Match[str]) -> str:
        return "".join(ch for ch in match.group(0) if ch.isalpha())

    return _abbrev_pattern().sub(repl, text)
