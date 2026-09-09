#!/usr/bin/env python3
"""Merge CMUdict primary entries with temp_cmu_g2p supplement_lexicon.json.

Produces a tab-separated CMUdict-style file for the C++ en-zh-g2p backend.
Merge policy matches legacy ``CMUDictG2P``: keep CMUdict primary readings;
add supplement entries only when the uppercase key is absent from CMUdict.

Usage:
  python scripts/merge_cmudict_supplement.py
  python scripts/merge_cmudict_supplement.py --output path/to/merged.txt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CMUDICT = REPO_ROOT / "temp_cmu_g2p" / "data" / "cmudict-0.7b"
DEFAULT_SUPPLEMENT = REPO_ROOT / "temp_cmu_g2p" / "data" / "supplement_lexicon.json"
DEFAULT_OUTPUT = REPO_ROOT / "temp_cmu_g2p" / "data" / "cmudict-en-zh-merged.txt"

_VARIANT_SUFFIX_RE = re.compile(r"\(\d+\)$")


def _normalize_word_key(word: str) -> str:
    return word.strip().upper()


def _primary_word_key(word: str) -> str:
    return _normalize_word_key(_VARIANT_SUFFIX_RE.sub("", word.strip()))


def load_cmudict_primary_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(";;;"):
            continue
        if "\t" in stripped:
            word = stripped.split("\t", 1)[0]
        else:
            word = stripped.split(None, 1)[0]
        if word:
            keys.add(_primary_word_key(word))
    return keys


def load_cmudict_lines(path: Path) -> list[str]:
    out: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(";;;") or not line.strip():
            out.append(line)
            continue
        out.append(line.rstrip("\n"))
    return out


def parse_supplement_entries(path: Path) -> dict[str, list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_entries = data.get("entries", data)
    parsed: dict[str, list[str]] = {}
    for key, value in raw_entries.items():
        if isinstance(value, dict) and "phones" in value:
            phones = list(value["phones"])
        elif isinstance(value, list):
            phones = list(value)
        elif isinstance(value, str):
            phones = value.split()
        else:
            continue
        norm_key = _normalize_word_key(key)
        parsed[norm_key] = [tok.strip().upper() for tok in phones if tok.strip()]
    return parsed


def merge_cmudict_supplement(
    cmudict_path: Path,
    supplement_path: Path,
    output_path: Path,
) -> dict[str, int]:
    if not cmudict_path.is_file():
        raise FileNotFoundError(f"CMUdict not found: {cmudict_path}")
    if not supplement_path.is_file():
        raise FileNotFoundError(f"Supplement lexicon not found: {supplement_path}")

    cmudict_keys = load_cmudict_primary_keys(cmudict_path)
    supplement = parse_supplement_entries(supplement_path)

    lines = load_cmudict_lines(cmudict_path)
    added = 0
    skipped_existing = 0
    supplement_lines: list[str] = []
    for key in sorted(supplement):
        phones = supplement[key]
        if not phones:
            continue
        if key in cmudict_keys:
            skipped_existing += 1
            continue
        supplement_lines.append(f"{key}\t{' '.join(phones)}")
        added += 1

    if lines and not lines[-1].startswith(";;; SUPPLEMENT"):
        lines.append(";;; SUPPLEMENT entries from supplement_lexicon.json (OOV only)")
    lines.extend(supplement_lines)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "cmudict_keys": len(cmudict_keys),
        "supplement_total": len(supplement),
        "added": added,
        "skipped_existing": skipped_existing,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge CMUdict + supplement_lexicon.json")
    p.add_argument("--cmudict", type=Path, default=DEFAULT_CMUDICT)
    p.add_argument("--supplement", type=Path, default=DEFAULT_SUPPLEMENT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    stats = merge_cmudict_supplement(args.cmudict, args.supplement, args.output)
    print(
        f"[merge_cmudict_supplement] wrote {args.output} "
        f"(cmudict={stats['cmudict_keys']}, supplement={stats['supplement_total']}, "
        f"added={stats['added']}, skipped_existing={stats['skipped_existing']})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
