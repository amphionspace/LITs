#!/usr/bin/env python3
"""Align Chinese text (with prosody #N markers) to pinyin and insert matching punctuation."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

RHYTHM_RE = re.compile(r"#\d+")
# Pinyin is always lowercase, e.g. zhe4, nar4, renr2.
PINYIN_SYLLABLE_RE = re.compile(r"^[a-z]+[0-9]$")
PINYIN_ER_SYLLABLE_RE = re.compile(r"^er[0-9]$")
PINYIN_R_TAIL_RE = re.compile(r"r[0-9]$")

PUNCT_SET = frozenset(
    "，。！？、；：""''（）【】《》…—·,.!?;:\"'()[]<>"
)

# Common characters before erhua 儿 (non-exhaustive; rules A/B handle most cases).
ERHUA_PREV_CHARS = frozenset(
    "人事物意味价法影声点块瓣根劲湾弯鸡气鸭孩圈馅片调丝条绳"
)


def is_han(ch: str) -> bool:
    return len(ch) == 1 and "\u4e00" <= ch <= "\u9fff"


def strip_rhythm(text: str) -> str:
    return RHYTHM_RE.sub("", text).strip()


def validate_pinyins(pinyins: list[str]) -> None:
    for syl in pinyins:
        if not PINYIN_SYLLABLE_RE.match(syl):
            raise ValueError(f"expected lowercase pinyin syllable, got {syl!r}")


def count_remaining_han(text: str, start: int) -> int:
    n = 0
    for ch in text[start:]:
        if is_han(ch):
            n += 1
    return n


def has_erhua_r_tail(syl: str) -> bool:
    """Erhua tail like nar4/wanr1/qir4; not a standalone er syllable (er2, er5)."""
    if PINYIN_ER_SYLLABLE_RE.match(syl):
        return False
    return bool(PINYIN_R_TAIL_RE.search(syl))


def should_merge_erhua(
    prev_han: str | None,
    pi: int,
    pinyins: list[str],
    text: str,
    pos: int,
) -> bool:
    if prev_han is None:
        return False

    # e.g. 傻样儿+而: yangr4 then 儿 — merge 儿 even though next syllable is 而's er2.
    if pi > 0 and has_erhua_r_tail(pinyins[pi - 1]):
        return True

    # Next syllable is standalone er* for this 儿 → 儿子, 鸟儿, etc.
    if pi < len(pinyins) and PINYIN_ER_SYLLABLE_RE.match(pinyins[pi]):
        return False

    if prev_han in ERHUA_PREV_CHARS and pi >= len(pinyins):
        return True

    # Counting fallback: skipping this 儿 balances han vs pinyin counts.
    remaining_han = count_remaining_han(text, pos)
    remaining_py = len(pinyins) - pi
    if remaining_han - 1 == remaining_py:
        return True

    return False


def align_pinyin_to_text(
    text: str,
    pinyins: list[str],
) -> list[str]:
    cleaned = strip_rhythm(text)
    out: list[str] = []
    pi = 0
    prev_han: str | None = None

    for pos, ch in enumerate(cleaned):
        if ch in PUNCT_SET:
            out.append(ch)
            continue

        if not is_han(ch):
            # Non-Chinese symbols are not expected in this format; skip without using pinyin.
            continue

        if ch == "儿" and should_merge_erhua(prev_han, pi, pinyins, cleaned, pos):
            prev_han = ch
            continue

        if pi >= len(pinyins):
            raise ValueError(
                f"pinyin exhausted at char {ch!r} (pos {pos}), pi={pi}, len={len(pinyins)}"
            )

        out.append(pinyins[pi])
        pi += 1
        prev_han = ch

    if pi != len(pinyins):
        raise ValueError(f"unused pinyin: {len(pinyins) - pi} syllables left (pi={pi})")

    return out


def parse_text_line(line: str) -> tuple[str, str]:
    line = line.rstrip("\n")
    if "\t" not in line:
        raise ValueError(f"expected tab-separated id and text: {line[:80]!r}")
    sample_id, text = line.split("\t", 1)
    return sample_id.strip(), text


def process_file(
    input_path: Path,
    output_path: Path,
    error_log_path: Path,
) -> tuple[int, int]:
    ok = 0
    failed = 0
    out_lines: list[str] = []
    errors: list[str] = []

    with input_path.open(encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        raw = lines[i]
        if not raw.strip():
            i += 1
            continue

        if not re.match(r"^\d", raw):
            i += 1
            continue

        sample_id, text = parse_text_line(raw)
        i += 1
        if i >= len(lines):
            errors.append(f"{sample_id}\tmissing pinyin line after text")
            failed += 1
            break

        pinyin_line = lines[i].strip()
        i += 1

        if not pinyin_line:
            errors.append(f"{sample_id}\tempty pinyin line")
            failed += 1
            continue

        pinyins = pinyin_line.split()
        try:
            validate_pinyins(pinyins)
            tokens = align_pinyin_to_text(text, pinyins)
            pinyin_with_punct = " ".join(tokens)
        except ValueError as exc:
            errors.append(
                f"{sample_id}\t{exc}\ttext={strip_rhythm(text)!r}\tpinyin={pinyin_line!r}"
            )
            failed += 1
            continue

        out_lines.append(f"{sample_id}\t{text}\n")
        out_lines.append(f"\t{pinyin_with_punct}\n")
        ok += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(out_lines), encoding="utf-8")

    error_log_path.parent.mkdir(parents=True, exist_ok=True)
    if errors:
        error_log_path.write_text("\n".join(errors) + "\n", encoding="utf-8")
    elif error_log_path.exists():
        error_log_path.unlink()

    return ok, failed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Insert Chinese punctuation into pinyin lines (prosody text pairs)."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input file: alternating id+text line and indented pinyin line",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output path (same format as input)",
    )
    parser.add_argument(
        "--error-log",
        type=Path,
        default=None,
        help="Log alignment failures (default: <output>.align_errors.log)",
    )
    args = parser.parse_args()

    error_log = args.error_log
    if error_log is None:
        error_log = args.output.with_suffix(args.output.suffix + ".align_errors.log")

    ok, failed = process_file(args.input, args.output, error_log)
    print(f"ok={ok} failed={failed} -> {args.output}", file=sys.stderr)
    if failed:
        print(f"errors logged to {error_log}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
