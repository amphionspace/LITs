#!/usr/bin/env python3
"""Convert raw Chinese-English text in a training filelist to phoneme tokens.

The input columns are preserved and only the final text column is replaced.
Supported LITs rows include ``wav|text``, ``wav|speaker|text``, and
``wav|speaker|start|end|text``.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLEANER = "en_zh_dict_mixed_rhyme_body_tone_cleaners"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a raw-text LITs filelist to a phoneme filelist."
    )
    parser.add_argument("input", type=Path, help="Raw-text training filelist")
    parser.add_argument("output", type=Path, help="Output phoneme filelist")
    parser.add_argument("--cleaner", default=DEFAULT_CLEANER)
    parser.add_argument(
        "--run-tn",
        action="store_true",
        help="Run the vendored C++ zh/en text normalizer before Python G2P",
    )
    parser.add_argument(
        "--tn-bin",
        type=Path,
        default=REPO_ROOT / "e2e_infer" / "bin" / "tts_cli",
    )
    parser.add_argument(
        "--tn-data-root", type=Path, default=REPO_ROOT / "frontend" / "data"
    )
    parser.add_argument("--cmudict", type=Path, help="External English CMUdict")
    parser.add_argument("--en-supplement", type=Path, help="English OOV JSON lexicon")
    parser.add_argument("--zh-lexicon", type=Path, help="Chinese word<TAB>pinyin lexicon")
    parser.add_argument("--zh-user-dict", type=Path, help="Chinese override dictionary")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preview", type=int, default=3, help="Rows to print")
    return parser.parse_args()


def _set_optional_path(env_name: str, value: Path | None) -> None:
    if value is None:
        return
    path = value.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{env_name} file not found: {path}")
    os.environ[env_name] = str(path)


def _tn_profile(text: str) -> str:
    has_zh = bool(re.search(r"[\u4e00-\u9fff]", text))
    has_en = bool(re.search(r"[A-Za-z]", text))
    return "en" if has_en and not has_zh else "zh"


def _run_tn(text: str, binary: Path, data_root: Path) -> str:
    profile = _tn_profile(text)
    resource_dir = data_root / profile
    if not binary.is_file():
        raise FileNotFoundError(
            f"TN binary not found: {binary}; run bash install_e2e_tn.sh first"
        )
    if not (resource_dir / "config.json").is_file():
        raise FileNotFoundError(f"TN profile not found: {resource_dir}")
    proc = subprocess.run(
        [str(binary), "--data", str(resource_dir)],
        input=text + "\n",
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"TN failed for profile={profile}: {proc.stderr.strip() or proc.stdout.strip()}"
        )
    normalized = proc.stdout.strip()
    if not normalized:
        raise ValueError(f"TN produced empty text for {text!r}")
    return normalized


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"input filelist not found: {input_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"output exists (pass --overwrite): {output_path}")
    if input_path == output_path:
        raise ValueError("input and output must be different files")

    _set_optional_path("LITS_CMUDICT", args.cmudict)
    _set_optional_path("LITS_EN_SUPPLEMENT", args.en_supplement)
    _set_optional_path("LITS_ZH_LEXICON", args.zh_lexicon)
    _set_optional_path("LITS_ZH_USER_DICT", args.zh_user_dict)

    tn_bin = args.tn_bin.expanduser().resolve()
    tn_data_root = args.tn_data_root.expanduser().resolve()

    sys.path.insert(0, str(REPO_ROOT))
    from lits.text import _clean_text  # pylint: disable=import-outside-toplevel

    output_rows: list[str] = []
    previews: list[tuple[int, str, str]] = []
    for line_no, raw in enumerate(input_path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        columns = raw.strip().split("|")
        if len(columns) not in (2, 3, 4, 5):
            raise ValueError(
                f"line {line_no}: expected 2, 3, 4, or 5 columns, got {len(columns)}"
            )
        text = columns[-1].strip()
        if not text:
            raise ValueError(f"line {line_no}: empty text")
        normalized = _run_tn(text, tn_bin, tn_data_root) if args.run_tn else text
        phonemes = _clean_text(normalized, [args.cleaner]).strip()
        if not phonemes:
            raise ValueError(f"line {line_no}: G2P produced no phonemes for {text!r}")
        columns[-1] = phonemes
        output_rows.append("|".join(columns))
        if len(previews) < max(0, args.preview):
            previews.append((line_no, text, normalized, phonemes))

    if not output_rows:
        raise ValueError("input filelist contains no data rows")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(output_rows) + "\n", encoding="utf-8")

    print(f"wrote {len(output_rows)} rows: {output_path}")
    for line_no, text, normalized, phonemes in previews:
        print(f"[{line_no}] text:       {text}")
        if args.run_tn:
            print(f"[{line_no}] normalized: {normalized}")
        print(f"[{line_no}] phonemes:   {phonemes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
