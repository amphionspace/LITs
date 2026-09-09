#!/usr/bin/env python3
"""Compare CMOS English text against ARPA reference using the local English frontend."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PKG_DIR = _REPO_ROOT / "temp_cmu_g2p"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

from english_frontend import (  # noqa: E402
    _lookup_english_token_phonemes,
    _neighbor_word_tokens,
    _prefer_letter_name_in_english,
    get_default_g2p,
    parse_arpa_word_units,
    tokenize_english_text,
)


def _phones_for_token(
    g2p,
    tokens: list[tuple[str, str]],
    idx: int,
) -> list[str]:
    kind, surface = tokens[idx]
    if kind == "punct":
        if surface == "-":
            prev_word = idx > 0 and tokens[idx - 1][0] == "word"
            next_word = idx + 1 < len(tokens) and tokens[idx + 1][0] == "word"
            if prev_word and next_word:
                return []
        return [surface]

    prev_word, next_word = _neighbor_word_tokens(tokens, idx)
    return _lookup_english_token_phonemes(
        g2p,
        surface,
        prefer_letter_name=_prefer_letter_name_in_english(surface, idx, tokens),
        prev_word=prev_word,
        next_word=next_word,
    )


def _unit_to_phones(unit: str) -> list[str]:
    if len(unit) == 1 and not unit[0].isalnum():
        return [unit]
    return unit.split()


def _consume_arpa_units(
    arpa_units: list[str],
    start: int,
    target_phones: list[str],
) -> tuple[int, list[str]] | None:
    """Match target phonemes to one or more consecutive ARPA units."""
    if not target_phones:
        return start, []

    collected: list[str] = []
    idx = start
    while idx < len(arpa_units) and len(collected) < len(target_phones):
        collected.extend(_unit_to_phones(arpa_units[idx]))
        if collected == target_phones:
            return idx + 1, collected
        if len(collected) > len(target_phones):
            return None
        idx += 1
    return None


def _format_phones(phones: list[str]) -> str:
    return " ".join(phones)


def _format_arpa_span(units: list[str], start: int, end: int) -> str:
    return " / ".join(units[start:end])


def compare_lines(
    line_no: int,
    text: str,
    arpa_line: str,
    g2p,
) -> list[str]:
    """Return log lines for mismatches on this sentence pair."""
    tokens = tokenize_english_text(text)
    arpa_units = parse_arpa_word_units(arpa_line)

    logs: list[str] = []
    arpa_idx = 0

    for tok_idx, (kind, surface) in enumerate(tokens):
        dict_phones = _phones_for_token(g2p, tokens, tok_idx)
        if not dict_phones:
            continue

        matched = _consume_arpa_units(arpa_units, arpa_idx, dict_phones)
        if matched is None:
            if arpa_idx >= len(arpa_units):
                logs.append(
                    f"line {line_no}: missing arpa for {kind} {surface!r}: "
                    f"{_format_phones(dict_phones)}"
                )
                break

            next_idx = arpa_idx + 1
            arpa_phones = _unit_to_phones(arpa_units[arpa_idx])
            arpa_span = arpa_units[arpa_idx]
            dict_span = _format_phones(dict_phones)
            logs.append(f"line {line_no}: {kind} {surface!r}")
            logs.append(f"  dict: {dict_span}")
            logs.append(f"  arpa: {arpa_span}")
            arpa_idx = next_idx
            continue

        next_idx, arpa_phones = matched
        arpa_span = _format_arpa_span(arpa_units, arpa_idx, next_idx)
        dict_span = _format_phones(dict_phones)

        if arpa_phones != dict_phones:
            logs.append(f"line {line_no}: {kind} {surface!r}")
            logs.append(f"  dict: {dict_span}")
            logs.append(f"  arpa: {arpa_span}")

        arpa_idx = next_idx

    if arpa_idx < len(arpa_units):
        extra = arpa_units[arpa_idx:]
        logs.append(
            f"line {line_no}: unused arpa units after token walk: "
            f"{' / '.join(extra)}"
        )

    return logs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Log dictionary lookup mismatches vs CMOS ARPA reference.",
    )
    parser.add_argument(
        "--words",
        type=Path,
        default=Path(__file__).with_name("cmos_words.txt"),
    )
    parser.add_argument(
        "--arpa",
        type=Path,
        default=Path(__file__).with_name("cmos_words_arpa.txt"),
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path(__file__).with_name("cmos_dict_vs_arpa_mismatches.log"),
    )
    args = parser.parse_args()

    words_lines = args.words.read_text(encoding="utf-8").splitlines()
    arpa_lines = args.arpa.read_text(encoding="utf-8").splitlines()
    if len(words_lines) != len(arpa_lines):
        print(
            f"Line count mismatch: {len(words_lines)} vs {len(arpa_lines)}",
            file=sys.stderr,
        )
        return 1

    g2p = get_default_g2p()
    all_logs: list[str] = []
    mismatch_lines = 0

    for line_no, (text, arpa_line) in enumerate(zip(words_lines, arpa_lines), start=1):
        text = text.strip()
        arpa_line = arpa_line.strip()
        if not text and not arpa_line:
            continue
        line_logs = compare_lines(line_no, text, arpa_line, g2p)
        if line_logs:
            mismatch_lines += 1
            all_logs.extend(line_logs)
            all_logs.append("")

    header = [
        f"words: {args.words}",
        f"arpa:  {args.arpa}",
        f"total lines: {len(words_lines)}",
        f"lines with mismatches: {mismatch_lines}",
        "",
    ]
    args.log.write_text("\n".join(header + all_logs), encoding="utf-8")
    print(f"Wrote {args.log} ({mismatch_lines} lines with mismatches)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
