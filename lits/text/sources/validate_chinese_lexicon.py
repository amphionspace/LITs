#!/usr/bin/env python3
"""Validate chinese_lexicon.txt / user_dict.txt pinyin entries.

Run after editing lexicon files (not during inference):

    python lits/text/sources/validate_chinese_lexicon.py
    python lits/text/sources/validate_chinese_lexicon.py --lexicon /path/to/chinese_lexicon.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCES_DIR = Path(__file__).resolve().parent
DEFAULT_LEXICON = SOURCES_DIR / "chinese_lexicon.txt"
DEFAULT_USER_DICT = SOURCES_DIR / "user_dict.txt"


def _load_pinyin_validator():
  """Lightweight frontend shell: only pinyin→bopomofo tables, no lexicon load."""
  if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
  from lits.text.g2p.mandarin import Frontend_chinese

  resource_path = REPO_ROOT / "lits" / "text"
  frontend = Frontend_chinese.__new__(Frontend_chinese)
  frontend.tone_dict = {"0": "˙", "5": "˙", "1": "", "2": "ˊ", "3": "ˇ", "4": "ˋ"}
  frontend.pinyin_2_bopomofo_dict = {}
  pinyin_2_bpmf = resource_path / "sources" / "pinyin_2_bpmf.txt"
  with open(pinyin_2_bpmf, "r", encoding="utf-8") as f:
    for line in f:
      pinyin, bopomofo = line.strip().split("\t")
      frontend.pinyin_2_bopomofo_dict[pinyin] = bopomofo
  return frontend


def validate_lexicon_file(path: Path, frontend) -> tuple[int, int, list[str]]:
    """Return (valid_rows, invalid_rows, sample_errors)."""
    if not path.is_file():
        raise FileNotFoundError(path)

    valid = 0
    invalid = 0
    errors: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                invalid += 1
                if len(errors) < 20:
                    errors.append(f"{path.name}:{line_no}: expected 'word<TAB>pinyin', got {line!r}")
                continue
            pinyin = parts[-1]
            if frontend.check_pinyin(pinyin):
                valid += 1
            else:
                invalid += 1
                if len(errors) < 20:
                    errors.append(f"{path.name}:{line_no}: invalid pinyin {pinyin!r} for {parts[0]!r}")
    return valid, invalid, errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Chinese lexicon pinyin entries.")
    parser.add_argument("--lexicon", type=Path, default=DEFAULT_LEXICON)
    parser.add_argument("--user-dict", type=Path, default=DEFAULT_USER_DICT)
    parser.add_argument("--skip-user-dict", action="store_true")
    args = parser.parse_args()

    print("Loading pinyin tables for validation...")
    frontend = _load_pinyin_validator()

    total_valid = 0
    total_invalid = 0
    all_errors: list[str] = []

    for label, path in [("chinese_lexicon", args.lexicon)]:
        print(f"Validating {path} ...")
        valid, invalid, errors = validate_lexicon_file(path, frontend)
        total_valid += valid
        total_invalid += invalid
        all_errors.extend(errors)
        print(f"  {label}: valid={valid} invalid={invalid}")

    if not args.skip_user_dict and args.user_dict.is_file():
        print(f"Validating {args.user_dict} ...")
        valid, invalid, errors = validate_lexicon_file(args.user_dict, frontend)
        total_valid += valid
        total_invalid += invalid
        all_errors.extend(errors)
        print(f"  user_dict: valid={valid} invalid={invalid}")

    if all_errors:
        print("\nSample errors:")
        for err in all_errors:
            print(f"  {err}")

    print(f"\nTotal: valid={total_valid} invalid={total_invalid}")
    return 1 if total_invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())
