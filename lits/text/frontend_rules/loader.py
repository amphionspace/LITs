"""Load frontend rule JSON documents."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

_RULES_ROOT = Path(__file__).resolve().parent / "rules"


@lru_cache(maxsize=None)
def load_rules(module: str, locale: str) -> dict[str, Any]:
    """Load a rule document for *module* (``punctuation`` | ``g2p_sandhi``) and *locale*."""
    path = _RULES_ROOT / module / f"{locale}.json"
    if not path.is_file():
        raise FileNotFoundError(f"frontend rules not found: {path}")
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    if doc.get("format") != "frontend_rules_v1":
        raise ValueError(f"unsupported rule format in {path}")
    if doc.get("module") != module:
        raise ValueError(f"module mismatch in {path}: expected {module!r}")
    return doc


def resolve_char_set(doc: dict[str, Any], name: str) -> frozenset[str]:
    raw = doc["resources"]["char_sets"][name]
    return frozenset(raw)


def resolve_char_map(doc: dict[str, Any], name: str) -> dict[str, str | None]:
    return doc["resources"]["char_maps"][name]


def resolve_constant(doc: dict[str, Any], name: str) -> str:
    return doc["resources"]["constants"][name]


def resolve_word_list(doc: dict[str, Any], name: str) -> frozenset[str]:
    return frozenset(doc["resources"]["word_lists"][name])
