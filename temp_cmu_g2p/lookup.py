#!/usr/bin/env python3
"""CLI: English text or ARPAbet -> slash-delimited ARPAbet for TTS."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

from cmudict_loader import DEFAULT_CMUDICT_PATH  # noqa: E402
from english_frontend import preprocess_english_input  # noqa: E402
from g2p_engine import CMUDictG2P, DEFAULT_SUPPLEMENT_PATH  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Look up English text via CMUdict, e.g. hello -> "HH AH0 L OW1 / ."',
    )
    parser.add_argument("text", help="English word/sentence or ARPAbet input")
    parser.add_argument("--cmudict", type=Path, default=DEFAULT_CMUDICT_PATH)
    parser.add_argument("--supplement", type=Path, default=DEFAULT_SUPPLEMENT_PATH)
    args = parser.parse_args()

    if not args.cmudict.is_file():
        print(f"CMUdict not found: {args.cmudict}", file=sys.stderr)
        return 1

    engine = CMUDictG2P.from_paths(args.cmudict, args.supplement)
    print(preprocess_english_input(args.text, engine))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
