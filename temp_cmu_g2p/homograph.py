"""Lightweight CMUdict homograph variant selection (rules in frontend_rules)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any


@dataclass(frozen=True)
class HomographContext:
    word: str
    prev: str | None = None
    next: str | None = None


def _norm_token(token: str | None) -> str:
    if not token:
        return ""
    return token.strip().lower().strip("'\"")


def _match_when(when: dict[str, Any], ctx: HomographContext, lists: dict[str, frozenset[str]]) -> bool:
    if not when:
        return True

    if when.get("next_starts_with_vowel"):
        nxt = _norm_token(ctx.next)
        if not nxt or nxt[0] not in "aeiou":
            return False

    if "prev_in" in when:
        prev = _norm_token(ctx.prev)
        if not prev or prev not in lists[when["prev_in"]]:
            return False

    if "prev_not_in" in when:
        prev = _norm_token(ctx.prev)
        if prev and prev in lists[when["prev_not_in"]]:
            return False

    if "next_in" in when:
        nxt = _norm_token(ctx.next)
        if not nxt or nxt not in lists[when["next_in"]]:
            return False

    if suffix := when.get("prev_suffix"):
        prev = _norm_token(ctx.prev)
        if not prev or not prev.endswith(suffix):
            return False

    if suffix := when.get("next_suffix"):
        nxt = _norm_token(ctx.next)
        if not nxt or not nxt.endswith(suffix):
            return False

    if pattern := when.get("prev_matches"):
        prev = _norm_token(ctx.prev)
        if not prev or not re.search(pattern, prev):
            return False

    if when.get("next_is_punct_or_end"):
        nxt = ctx.next
        if nxt is not None:
            nxt = nxt.strip()
            if nxt and not (len(nxt) == 1 and not nxt.isalnum()):
                return False

    return True


@lru_cache(maxsize=1)
def _load_homograph_doc() -> dict[str, Any]:
    from lits.text.frontend_rules.loader import load_rules

    return load_rules("g2p_homograph", "en")


@lru_cache(maxsize=1)
def _word_list_cache() -> dict[str, frozenset[str]]:
    doc = _load_homograph_doc()
    lists = doc.get("resources", {}).get("word_lists", {})
    return {name: frozenset(_norm_token(x) for x in items) for name, items in lists.items()}


def select_homograph_variant(
    word: str,
    num_variants: int,
    *,
    prev: str | None = None,
    next: str | None = None,
) -> int | None:
    """Return CMUdict 0-based variant index, or None to keep primary (0)."""
    if num_variants <= 1:
        return None

    doc = _load_homograph_doc()
    entry = doc.get("homographs", {}).get(word.strip().upper())
    if not entry:
        return None

    ctx = HomographContext(word=word, prev=prev, next=next)
    lists = _word_list_cache()

    for rule in entry.get("rules", []):
        when = rule.get("when", {})
        if not _match_when(when, ctx, lists):
            continue
        pick = int(rule["pick"])
        if 0 <= pick < num_variants:
            return pick
    return None
