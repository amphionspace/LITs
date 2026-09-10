"""Model-compatible bookend stages around the vendored normalizer.

Rule documents live in ``frontend/rules_v2/<locale>.full.json``.
"""

from __future__ import annotations

import copy
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from .ops.punctuation import apply_punctuation_op

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_RULES_V2_DIR = (
    _REPO_ROOT / "frontend" / "rules_v2"
)
_FALLBACK_LANG = "en"


def get_rules_v2_dir() -> Path:
    override = os.environ.get("TN_RULES_V2_DIR")
    if override:
        return Path(override)
    return _DEFAULT_RULES_V2_DIR


def _read_full_json(lang: str) -> dict[str, Any] | None:
    path = get_rules_v2_dir() / f"{lang}.full.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _merge_frontend_docs(base: dict[str, Any], extension: dict[str, Any]) -> dict[str, Any]:
    """Merge locale ``frontend`` onto ``en`` (resources + optional post stage inserts)."""
    merged = copy.deepcopy(base)
    base_resources = merged.setdefault("resources", {})
    ext_resources = extension.get("resources", {})

    for bucket in ("char_maps", "char_sets", "constants"):
        base_bucket = base_resources.setdefault(bucket, {})
        base_bucket.update(ext_resources.get(bucket, {}))

    if extension.get("pre_stages"):
        merged["pre_stages"] = extension["pre_stages"]

    if extension.get("post_stages"):
        merged["post_stages"] = extension["post_stages"]
    elif extension.get("post_stages_extra"):
        post = list(merged.get("post_stages", []))
        insert_idx = len(post)
        for idx, stage in enumerate(post):
            if stage.get("op") == "ensure_trailing_sentence_punct":
                insert_idx = idx
                break
        merged["post_stages"] = post[:insert_idx] + extension["post_stages_extra"] + post[insert_idx:]

    return merged


@lru_cache(maxsize=None)
def _load_frontend_doc(lang: str) -> dict[str, Any]:
    rules_dir = get_rules_v2_dir()
    en_doc = _read_full_json(_FALLBACK_LANG)
    if en_doc is None:
        raise FileNotFoundError(f"Missing {rules_dir}/{_FALLBACK_LANG}.full.json")

    en_frontend = en_doc.get("frontend")
    if not isinstance(en_frontend, dict) or not en_frontend.get("pre_stages"):
        raise FileNotFoundError(
            f"No frontend.pre_stages in {rules_dir}/{_FALLBACK_LANG}.full.json"
        )

    if lang == _FALLBACK_LANG:
        return en_frontend

    locale_doc = _read_full_json(lang)
    if locale_doc is None:
        return en_frontend

    locale_frontend = locale_doc.get("frontend")
    if not isinstance(locale_frontend, dict):
        return en_frontend

    if locale_frontend.get("pre_stages"):
        return locale_frontend

    return _merge_frontend_docs(en_frontend, locale_frontend)


def get_char_set(name: str, lang: str = _FALLBACK_LANG) -> frozenset[str]:
    resources = _load_frontend_doc(lang).get("resources", {})
    char_sets = resources.get("char_sets", {})
    raw = char_sets.get(name, "")
    if isinstance(raw, list):
        return frozenset(raw)
    return frozenset(str(raw))


def apply_tn_frontend_pre(text: str, lang: str) -> str:
    if not text:
        return text
    doc = _load_frontend_doc(lang)
    resources_doc = {"resources": doc.get("resources", {})}
    for stage in doc.get("pre_stages", []):
        text = apply_punctuation_op(
            stage["op"], text, resources_doc, stage.get("params")
        )
    return text


def apply_tn_frontend_post(text: str, lang: str) -> str:
    if not text:
        return text
    doc = _load_frontend_doc(lang)
    resources_doc = {"resources": doc.get("resources", {})}
    for stage in doc.get("post_stages", []):
        text = apply_punctuation_op(
            stage["op"], text, resources_doc, stage.get("params")
        )
    return text


def apply_tn_frontend_full(text: str, lang: str) -> str:
    """Pre + post (legacy ``common.json`` single-pass behavior)."""
    return apply_tn_frontend_post(apply_tn_frontend_pre(text, lang), lang)
