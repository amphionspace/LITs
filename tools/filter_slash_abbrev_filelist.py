#!/usr/bin/env python3
"""Filter filelist lines whose text contains slash-tokenized letter abbreviations.

In slash-separated training text, abbreviations like ``н.э.`` or ``в.п.`` appear as::

    н / . / э / .
    в / . / п / .

This script scans an input filelist (``<fields>|...|<text>`` per line), detects
such patterns in the trailing text field, and writes matching lines to an output
filelist. With ``--text_only``, it writes only the matching text field, one text
per line.

Detection rule
--------------
At least two Cyrillic letters each followed by a dot token, separated by ``/``::

    <letter> / . / <letter> / . [/ <letter> / . ...]

Examples that match::

    ... между / н / . / э / . / и / до / н / . / э / . / снижает ...
    ... в / . / п / . / кочубей ...
    ... а / . / х / . / бенкендорф ...

Examples that do not match::

    ... слово / . / конец / .          (not letter-dot pairs)
    ... а / . / чарторыйский             (single-letter abbrev only)

Usage
-----
  python filter_slash_abbrev_filelist.py \\
      --input_txt data/filelists/ru_train.txt \\
      --output_txt data/filelists/ru_train_slash_abbrev.txt

  python filter_slash_abbrev_filelist.py \\
      --input_txt ru.txt --output_txt ru_abbrev.txt --verbose

  python filter_slash_abbrev_filelist.py \\
      --input_txt ru.txt --output_txt ru_abbrev_text_only.txt --text_only
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_CYRILLIC_LETTER = r"[а-яёА-ЯЁ]"
_SLASH = r"\s*/\s*"
_DOT = r"\."

# н / . / э / .  (and longer chains: п / . / а / . / ...)
SLASH_LETTER_DOT_ABBREV_RE = re.compile(
    _CYRILLIC_LETTER
    + _SLASH
    + _DOT
    + r"(?:"
    + _SLASH
    + _CYRILLIC_LETTER
    + _SLASH
    + _DOT
    + r")+",
    re.UNICODE,
)


def text_from_filelist_line(line: str) -> str:
    """Return trailing text field (last ``|``-separated segment)."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return ""
    return stripped.rsplit("|", 1)[-1]


def has_slash_letter_dot_abbrev(text: str) -> bool:
    """True if text contains ``letter / . / letter / .`` (Cyrillic, >=2 letters)."""
    return bool(SLASH_LETTER_DOT_ABBREV_RE.search(text))


def find_slash_letter_dot_abbrevs(text: str) -> list[str]:
    """Return non-overlapping matched substrings (for verbose logging)."""
    return [m.group(0) for m in SLASH_LETTER_DOT_ABBREV_RE.finditer(text)]


def filter_filelist(
    input_path: Path,
    output_path: Path,
    *,
    text_only: bool = False,
    verbose: bool = False,
) -> dict[str, int]:
    matched_lines: list[str] = []
    stats = {"read": 0, "matched": 0, "skipped_empty": 0}

    for raw_line in input_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        stats["read"] += 1
        text = text_from_filelist_line(raw_line)
        if not text:
            stats["skipped_empty"] += 1
            continue
        if has_slash_letter_dot_abbrev(text):
            matched_lines.append(text if text_only else raw_line.rstrip("\n"))
            stats["matched"] += 1
            if verbose:
                patterns = find_slash_letter_dot_abbrevs(text)
                preview = raw_line[:120] + ("..." if len(raw_line) > 120 else "")
                print(f"[match] {preview}", file=sys.stderr)
                for pat in patterns:
                    print(f"        abbrev: {pat}", file=sys.stderr)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(matched_lines) + ("\n" if matched_lines else ""),
        encoding="utf-8",
    )
    return stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract filelist lines with slash-tokenized Cyrillic letter abbreviations.",
    )
    p.add_argument(
        "--input_txt",
        type=Path,
        required=True,
        help="Input filelist; text is the last |-separated field on each line",
    )
    p.add_argument(
        "--output_txt",
        type=Path,
        required=True,
        help="Output path for matching filelist lines, or text-only lines with --text_only",
    )
    p.add_argument(
        "--text_only",
        action="store_true",
        help="Write only the trailing text field for each match, not the full filelist line",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print each matched line and detected abbrev substring to stderr",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input_txt.is_file():
        print(f"Input not found: {args.input_txt}", file=sys.stderr)
        return 1

    stats = filter_filelist(
        args.input_txt,
        args.output_txt,
        text_only=args.text_only,
        verbose=args.verbose,
    )
    print(
        f"[filter_slash_abbrev] read={stats['read']} matched={stats['matched']} "
        f"skipped_empty={stats['skipped_empty']} -> {args.output_txt}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
