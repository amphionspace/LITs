#!/usr/bin/env python3
"""Compare legacy Python cleaners vs C++ tts_cli G2P frontend.

Two comparison modes:
  e2e   - raw text -> legacy TN+cleaner vs cpp TN+G2P (full pipeline)
  g2p   - same legacy TN text -> legacy cleaner vs cpp G2P (isolates G2P)

Usage:
  python scripts/compare_frontend_parity.py \\
    --input_txt test/fixtures/en_zh_parity_corpus.txt \\
    --model_lang en-zh-dict \\
    --output_tsv test/golden/en_zh_dict_parity.tsv
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from infer_e2e import (  # noqa: E402
    MODEL_TN_LANGS,
    TtsCliEngine,
    _tn_lang_for_line,
    read_plain_lines,
)
from lits.text import phonemes_to_sequence_with_tones, text_to_sequence  # noqa: E402

MODEL2CLEANER = {
    "en-zh-dict": "en_zh_dict_mixed_rhyme_body_tone_cleaners",
    "en-zh": "pinyin_direct_mixed_rhyme_body_tone_cleaners",
    "en-zh-dict-rhyme-body-tone": "en_zh_dict_mixed_rhyme_body_tone_cleaners",
    "en-zh-rhyme-body-tone": "pinyin_direct_mixed_rhyme_body_tone_cleaners",
}


@dataclass
class LineResult:
    line_no: int
    raw: str
    legacy_tn: str
    cpp_tn: str
    legacy_cleaned: str
    cpp_cleaned: str
    legacy_token_ids: list[int]
    cpp_token_ids: list[int]
    legacy_error: str | None = None
    cpp_error: str | None = None

    @property
    def tn_match(self) -> bool:
        return self.legacy_tn == self.cpp_tn

    @property
    def g2p_match(self) -> bool:
        if self.legacy_error or self.cpp_error:
            return False
        return self.legacy_cleaned == self.cpp_cleaned

    @property
    def e2e_match(self) -> bool:
        return self.g2p_match and self.tn_match

    @property
    def token_ids_match(self) -> bool:
        if self.legacy_error or self.cpp_error:
            return False
        return self.legacy_token_ids == self.cpp_token_ids


def _first_token_diff(a: str, b: str) -> str:
    a_toks = a.split()
    b_toks = b.split()
    n = min(len(a_toks), len(b_toks))
    for i in range(n):
        if a_toks[i] != b_toks[i]:
            return f"idx={i} legacy={a_toks[i]!r} cpp={b_toks[i]!r}"
    if len(a_toks) != len(b_toks):
        return f"len legacy={len(a_toks)} cpp={len(b_toks)}"
    return ""


def _tokenize_relaxed(cleaned: str) -> list[str]:
    """Ignore ``_`` boundary markers when comparing token streams."""
    return [tok for tok in cleaned.replace(" _ ", " ").split() if tok and tok != "_"]


def _relaxed_g2p_match(legacy_cleaned: str, cpp_cleaned: str) -> bool:
    return _tokenize_relaxed(legacy_cleaned) == _tokenize_relaxed(cpp_cleaned)


def _legacy_tn_text(raw: str, model_lang: str, tn: TtsCliEngine, tn_primary: str | None) -> str:
    del model_lang
    lang = _tn_lang_for_line(raw, tn_primary)
    return tn.tn_line(raw, lang)


def _cpp_tn_text(raw: str, model_lang: str, tts_cli: TtsCliEngine, tn_primary: str | None) -> str:
    del model_lang
    lang = _tn_lang_for_line(raw, tn_primary)
    return tts_cli.tn_line(raw, lang)


def _legacy_cleaned(tn_text: str, cleaner: str) -> tuple[str, list[int]]:
    token_ids, cleaned = text_to_sequence(tn_text, [cleaner])
    return cleaned, token_ids


def _cpp_cleaned(tn_text: str, cleaner: str, tts_cli: TtsCliEngine) -> tuple[str, list[int]]:
    phonemes = tts_cli.g2p_line(tn_text)
    token_ids, _, _ = phonemes_to_sequence_with_tones(phonemes, [cleaner])
    return phonemes, token_ids


def process_line(
    line_no: int,
    raw: str,
    *,
    model_lang: str,
    cleaner: str,
    tn: TtsCliEngine,
    tts_cli: TtsCliEngine,
    tn_primary: str | None,
    g2p_tn_source: str,
) -> LineResult:
    legacy_error: str | None = None
    cpp_error: str | None = None
    legacy_tn = ""
    cpp_tn = ""
    legacy_cleaned = ""
    cpp_cleaned = ""
    legacy_token_ids: list[int] = []
    cpp_token_ids: list[int] = []

    try:
        legacy_tn = _legacy_tn_text(raw, model_lang, tn, tn_primary)
    except Exception as exc:
        legacy_error = str(exc)

    try:
        cpp_tn = _cpp_tn_text(raw, model_lang, tts_cli, tn_primary)
    except Exception as exc:
        cpp_error = str(exc)

    g2p_tn = legacy_tn if g2p_tn_source == "legacy" else cpp_tn
    if g2p_tn_source == "legacy" and legacy_error:
        g2p_tn = ""
    if g2p_tn_source == "cpp" and cpp_error:
        g2p_tn = ""

    if g2p_tn and legacy_error is None:
        try:
            legacy_cleaned, legacy_token_ids = _legacy_cleaned(g2p_tn, cleaner)
        except Exception as exc:
            legacy_error = str(exc)

    if g2p_tn and cpp_error is None:
        try:
            cpp_cleaned, cpp_token_ids = _cpp_cleaned(g2p_tn, cleaner, tts_cli)
        except Exception as exc:
            cpp_error = str(exc)

    return LineResult(
        line_no=line_no,
        raw=raw,
        legacy_tn=legacy_tn,
        cpp_tn=cpp_tn,
        legacy_cleaned=legacy_cleaned,
        cpp_cleaned=cpp_cleaned,
        legacy_token_ids=legacy_token_ids,
        cpp_token_ids=cpp_token_ids,
        legacy_error=legacy_error,
        cpp_error=cpp_error,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare legacy vs cpp en-zh frontend parity")
    p.add_argument("--input_txt", type=Path, required=True)
    p.add_argument("--output_tsv", type=Path, required=True)
    p.add_argument("--model_lang", required=True, choices=sorted(MODEL_TN_LANGS))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--tn_primary", default=None)
    p.add_argument(
        "--tn_bin_dir",
        type=Path,
        default=REPO_ROOT / "e2e_infer" / "bin",
    )
    p.add_argument(
        "--tn_data_root",
        type=Path,
        default=REPO_ROOT / "Transsion_Multilingual_Text_Normalization_for_TTS" / "data",
    )
    p.add_argument(
        "--show_mismatches",
        type=int,
        default=20,
        help="Print up to N mismatch details to stderr (0=none)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cleaner = MODEL2CLEANER.get(args.model_lang)
    if not cleaner:
        print(f"No cleaner for model_lang={args.model_lang!r}", file=sys.stderr)
        return 1

    tts_cli = TtsCliEngine(args.tn_bin_dir, data_root=args.tn_data_root)
    if not tts_cli.is_g2p_available():
        print("cpp frontend not ready (rebuild tts_cli with en-zh-g2p)", file=sys.stderr)
        return 1

    tn = TtsCliEngine(args.tn_bin_dir, data_root=args.tn_data_root)
    lines = read_plain_lines(args.input_txt, args.limit)

    e2e_full: list[LineResult] = []
    g2p_results: list[LineResult] = []
    try:
        for idx, raw in enumerate(lines, start=1):
            g2p_row = process_line(
                idx,
                raw,
                model_lang=args.model_lang,
                cleaner=cleaner,
                tn=tn,
                tts_cli=tts_cli,
                tn_primary=args.tn_primary,
                g2p_tn_source="legacy",
            )
            g2p_results.append(g2p_row)

            e2e_row = LineResult(
                line_no=idx,
                raw=raw,
                legacy_tn=g2p_row.legacy_tn,
                cpp_tn=g2p_row.cpp_tn,
                legacy_cleaned="",
                cpp_cleaned="",
                legacy_token_ids=[],
                cpp_token_ids=[],
                legacy_error=g2p_row.legacy_error,
                cpp_error=g2p_row.cpp_error,
            )
            if not g2p_row.legacy_error and g2p_row.legacy_tn:
                try:
                    e2e_row.legacy_cleaned, e2e_row.legacy_token_ids = _legacy_cleaned(
                        g2p_row.legacy_tn, cleaner
                    )
                except Exception as exc:
                    e2e_row.legacy_error = str(exc)
            if not g2p_row.cpp_error and g2p_row.cpp_tn:
                try:
                    e2e_row.cpp_cleaned, e2e_row.cpp_token_ids = _cpp_cleaned(
                        g2p_row.cpp_tn, cleaner, tts_cli
                    )
                except Exception as exc:
                    e2e_row.cpp_error = str(exc)
            e2e_full.append(e2e_row)
    finally:
        tn.close()

    header = (
        "line_no\traw\ttn_match\tg2p_on_legacy_tn\t"
        "e2e_match\tlegacy_error\tcpp_error\t"
        "legacy_tn\tcpp_tn\tlegacy_cleaned\tcpp_cleaned\tfirst_diff\n"
    )
    rows: list[str] = []
    e2e_ok = g2p_ok = tn_ok = relaxed_ok = 0
    errors = 0

    for e2e, g2p in zip(e2e_full, g2p_results):
        if e2e.legacy_error or e2e.cpp_error:
            errors += 1
        if e2e.tn_match:
            tn_ok += 1
        if g2p.g2p_match:
            g2p_ok += 1
        if e2e.g2p_match:
            e2e_ok += 1
        if (
            not g2p.legacy_error
            and not g2p.cpp_error
            and _relaxed_g2p_match(g2p.legacy_cleaned, g2p.cpp_cleaned)
        ):
            relaxed_ok += 1

        diff = ""
        if not g2p.g2p_match and not g2p.legacy_error and not g2p.cpp_error:
            diff = _first_token_diff(g2p.legacy_cleaned, g2p.cpp_cleaned)

        rows.append(
            "\t".join(
                [
                    str(e2e.line_no),
                    e2e.raw.replace("\t", " "),
                    "1" if e2e.tn_match else "0",
                    "1" if g2p.g2p_match else "0",
                    "1" if e2e.g2p_match else "0",
                    e2e.legacy_error or "",
                    e2e.cpp_error or "",
                    e2e.legacy_tn.replace("\t", " "),
                    e2e.cpp_tn.replace("\t", " "),
                    g2p.legacy_cleaned.replace("\t", " "),
                    g2p.cpp_cleaned.replace("\t", " "),
                    diff,
                ]
            )
        )

    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    args.output_tsv.write_text(header + "\n".join(rows) + "\n", encoding="utf-8")

    total = len(lines)
    print(
        f"[parity] lines={total} "
        f"tn_match={tn_ok}/{total} "
        f"g2p_on_legacy_tn={g2p_ok}/{total} "
        f"g2p_relaxed={relaxed_ok}/{total} "
        f"e2e_match={e2e_ok}/{total} "
        f"errors={errors}"
    )
    print(f"[parity] report: {args.output_tsv}")

    if args.show_mismatches:
        shown = 0
        for e2e, g2p in zip(e2e_full, g2p_results):
            if shown >= args.show_mismatches:
                break
            if e2e.legacy_error or e2e.cpp_error:
                print(
                    f"[mismatch] line {e2e.line_no} ERROR "
                    f"legacy={e2e.legacy_error!r} cpp={e2e.cpp_error!r}",
                    file=sys.stderr,
                )
                shown += 1
                continue
            issues: list[str] = []
            if not e2e.tn_match:
                issues.append("tn")
            if not g2p.g2p_match:
                issues.append("g2p")
            if not issues:
                continue
            print(f"[mismatch] line {e2e.line_no} ({', '.join(issues)})", file=sys.stderr)
            print(f"  raw: {e2e.raw!r}", file=sys.stderr)
            if not e2e.tn_match:
                print(f"  legacy_tn: {e2e.legacy_tn!r}", file=sys.stderr)
                print(f"  cpp_tn:    {e2e.cpp_tn!r}", file=sys.stderr)
            if not g2p.g2p_match:
                print(f"  legacy: {g2p.legacy_cleaned!r}", file=sys.stderr)
                print(f"  cpp:    {g2p.cpp_cleaned!r}", file=sys.stderr)
                diff = _first_token_diff(g2p.legacy_cleaned, g2p.cpp_cleaned)
                if diff:
                    print(f"  first_diff: {diff}", file=sys.stderr)
            shown += 1

    return 1 if (g2p_ok < total or errors) else 0


if __name__ == "__main__":
    raise SystemExit(main())
