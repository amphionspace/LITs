"""Case-sensitive acronym pronunciation (CMUdict backend)."""

from __future__ import annotations

from lits.text.frontend_rules.ops.acronym_case import (
    CmudictAction,
    resolve_cmudict_action,
)


def select_acronym_case_action(word: str, num_variants: int) -> CmudictAction:
    """Return CMUdict variant index, ``spell``, or ``None`` to defer."""
    return resolve_cmudict_action(word, num_variants)
