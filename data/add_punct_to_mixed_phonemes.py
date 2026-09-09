#!/usr/bin/env python3
"""Insert Chinese punctuation into mixed zh-en phoneme lines (pinyin + ARPAbet + /).

Reads the same two-line format as add_punct_to_pinyin.py:
  000001\t汉字#1与#1English#2，标点#4。
  \t ni3 hao3 / DH IH1 S / HH AH0 L OW1

Only punctuation is taken from the text line (#N rhythm markers are dropped).
Pinyin syllables, ARPAbet tokens, and ``/`` (or ``|``, ``_``) boundaries are copied
in order; Chinese characters consume pinyin, Latin English spans consume ARPAbet
blocks (with optional ``/`` between words).
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path
from typing import Iterator

_DATA_DIR = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "add_punct_to_pinyin", _DATA_DIR / "add_punct_to_pinyin.py"
)
_add_punct = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_add_punct)

PUNCT_SET = _add_punct.PUNCT_SET
strip_rhythm = _add_punct.strip_rhythm
is_han = _add_punct.is_han
parse_text_line = _add_punct.parse_text_line
count_remaining_han = _add_punct.count_remaining_han
has_erhua_r_tail = _add_punct.has_erhua_r_tail
ERHUA_PREV_CHARS = _add_punct.ERHUA_PREV_CHARS
PINYIN_ER_SYLLABLE_RE = _add_punct.PINYIN_ER_SYLLABLE_RE

PINYIN_SYLLABLE_RE = re.compile(r"^[a-z]+[0-9]$")
ARPABET_TOKEN_RE = re.compile(r"^[A-Z]{1,3}[012]?$")
BOUNDARY_TOKENS = frozenset({"/", "|", "_"})
_EN_WORD_RE = re.compile(r"[A-Za-z]+(?:[''-][A-Za-z]+)*")


def is_pinyin_token(token: str) -> bool:
    return bool(PINYIN_SYLLABLE_RE.match(token))


def is_arpabet_token(token: str) -> bool:
    return bool(ARPABET_TOKEN_RE.match(token))


def is_boundary_token(token: str) -> bool:
    return token in BOUNDARY_TOKENS


def iter_text_units(cleaned: str) -> Iterator[tuple[str, str]]:
    """Yield (kind, value): punct | han | en."""
    i = 0
    n = len(cleaned)
    while i < n:
        ch = cleaned[i]
        if ch in PUNCT_SET:
            yield ("punct", ch)
            i += 1
            continue
        if is_han(ch):
            yield ("han", ch)
            i += 1
            continue
        m = _EN_WORD_RE.match(cleaned, i)
        if m:
            yield ("en", m.group())
            i = m.end()
            continue
        i += 1


def _skip_boundaries(tokens: list[str], ti: int, out: list[str]) -> int:
    while ti < len(tokens) and is_boundary_token(tokens[ti]):
        out.append(tokens[ti])
        ti += 1
    return ti


def _next_pinyin_index(tokens: list[str], ti: int) -> int | None:
    while ti < len(tokens):
        if is_pinyin_token(tokens[ti]):
            return ti
        if is_boundary_token(tokens[ti]):
            ti += 1
            continue
        return None
    return None


def _prev_pinyin_token(tokens: list[str], ti: int) -> str | None:
    j = ti - 1
    while j >= 0:
        if is_pinyin_token(tokens[j]):
            return tokens[j]
        if is_boundary_token(tokens[j]):
            j -= 1
            continue
        return None
    return None


def _count_remaining_pinyin(tokens: list[str], ti: int) -> int:
    return sum(1 for tok in tokens[ti:] if is_pinyin_token(tok))


def should_merge_erhua_mixed(
    prev_han: str | None,
    ti: int,
    tokens: list[str],
    text: str,
    char_pos: int,
) -> bool:
    if prev_han is None:
        return False

    prev_syl = _prev_pinyin_token(tokens, ti)
    if prev_syl and has_erhua_r_tail(prev_syl):
        return True

    nxt_i = _next_pinyin_index(tokens, ti)
    if nxt_i is not None and PINYIN_ER_SYLLABLE_RE.match(tokens[nxt_i]):
        return False

    if prev_han in ERHUA_PREV_CHARS and nxt_i is None:
        return True

    remaining_han = count_remaining_han(text, char_pos)
    remaining_py = _count_remaining_pinyin(tokens, ti)
    if remaining_han - 1 == remaining_py:
        return True

    return False


def align_mixed_phonemes_to_text(text: str, tokens: list[str]) -> list[str]:
    cleaned = strip_rhythm(text)
    out: list[str] = []
    ti = 0
    prev_han: str | None = None
    char_pos = 0

    for kind, value in iter_text_units(cleaned):
        if kind == "punct":
            out.append(value)
            char_pos += len(value)
            continue

        if kind == "en":
            ti = _skip_boundaries(tokens, ti, out)
            while ti < len(tokens):
                if is_pinyin_token(tokens[ti]):
                    break
                if is_arpabet_token(tokens[ti]) or is_boundary_token(tokens[ti]):
                    out.append(tokens[ti])
                    ti += 1
                    continue
                raise ValueError(
                    f"expected ARPAbet or boundary for English {value!r}, got {tokens[ti]!r}"
                )
            char_pos += len(value)
            continue

        # kind == "han"
        ch = value
        if ch == "儿" and should_merge_erhua_mixed(
            prev_han, ti, tokens, cleaned, char_pos
        ):
            prev_han = ch
            char_pos += 1
            continue

        ti = _skip_boundaries(tokens, ti, out)
        if ti >= len(tokens):
            raise ValueError(f"phoneme exhausted at han {ch!r}")
        if not is_pinyin_token(tokens[ti]):
            raise ValueError(
                f"expected pinyin for {ch!r}, got {tokens[ti]!r} (ti={ti})"
            )
        out.append(tokens[ti])
        ti += 1
        prev_han = ch
        char_pos += 1

    if ti != len(tokens):
        rest = " ".join(tokens[ti:])
        raise ValueError(f"unused phoneme tokens: {rest!r}")

    return out


def validate_mixed_tokens(tokens: list[str]) -> None:
    for tok in tokens:
        if (
            is_pinyin_token(tok)
            or is_arpabet_token(tok)
            or is_boundary_token(tok)
        ):
            continue
        raise ValueError(f"unexpected phoneme token {tok!r}")


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
            errors.append(f"{sample_id}\tmissing phoneme line after text")
            failed += 1
            break

        phoneme_line = lines[i].strip()
        i += 1

        if not phoneme_line:
            errors.append(f"{sample_id}\tempty phoneme line")
            failed += 1
            continue

        tokens = phoneme_line.split()
        try:
            validate_mixed_tokens(tokens)
            result = align_mixed_phonemes_to_text(text, tokens)
            phoneme_with_punct = " ".join(result)
        except ValueError as exc:
            errors.append(
                f"{sample_id}\t{exc}\ttext={strip_rhythm(text)!r}\tphoneme={phoneme_line!r}"
            )
            failed += 1
            continue

        out_lines.append(f"{sample_id}\t{text}\n")
        out_lines.append(f"\t{phoneme_with_punct}\n")
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
        description="Insert Chinese punctuation into mixed pinyin+ARPAbet phoneme lines."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input: alternating id+text line and indented phoneme line",
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
