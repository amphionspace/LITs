#!/usr/bin/env python3
"""Preflight: verify filelist texts survive cleaners without <unk> tokens.

Reads a data YAML (e.g. configs/data/en-zh_24k_pinyin.yaml), runs train/valid
filelists through the configured cleaners, and reports rows whose cleaned tokens
are missing from the model symbol table.


Usage:
  python preflight_filelist.py configs/data/en-zh_24k_pinyin.yaml
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def _load_yaml_mapping(path: Path) -> dict:
    try:
        import yaml  # type: ignore

        with path.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except ImportError:
        raw = _load_flat_data_yaml(path)
    if not isinstance(raw, dict):
        raise ValueError(f"expected mapping in {path}")
    return raw


def _load_flat_data_yaml(path: Path) -> dict:
    """Minimal parser for flat configs/data/*.yaml (no PyYAML required)."""
    data: dict = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("-") or ":" not in stripped:
            continue
        key, _, val = stripped.partition(":")
        key = key.strip()
        val = val.split("#", 1)[0].strip()
        if val.startswith("[") and val.endswith("]"):
            data[key] = [
                item.strip().strip("'\"")
                for item in val[1:-1].split(",")
                if item.strip()
            ]
        elif (val.startswith('"') and val.endswith('"')) or (
            val.startswith("'") and val.endswith("'")
        ):
            data[key] = val[1:-1]
        elif re.fullmatch(r"-?\d+", val):
            data[key] = int(val)
        elif val.lower() in ("true", "false"):
            data[key] = val.lower() == "true"
        else:
            data[key] = val
    return data

sys.path.insert(0, str(REPO_ROOT))

from lits.text import (  # noqa: E402
    _CLEANER_TO_MODEL_KEY,
    _TOKEN_SEQUENCE_CLEANERS,
    _activate_model_symbols,
    _clean_text,
)
from lits.text.char_symbols.symbol_inventories import lang2inventory  # noqa: E402


def parse_filelist(filelist_path: Path, split_char: str = "|") -> list[list[str]]:
    with filelist_path.open(encoding="utf-8") as f:
        return [line.strip().split(split_char) for line in f if line.strip()]


@dataclass
class Issue:
    split: str
    line_no: int
    filepath: str
    text: str
    cleaned: str
    unknown_tokens: list[str]


def _resolve_path(path_str: str, repo_root: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _load_data_config(yaml_path: Path) -> dict:
    raw = _load_yaml_mapping(yaml_path)
    if "data" in raw and isinstance(raw["data"], dict):
        return raw["data"]
    return raw


def _parse_text_column(row: list[str], n_spks: int) -> tuple[str, str]:
    """Return (wav_path, text) using the same rules as TextMelDataset._parse_entry."""
    filepath = row[0]
    if n_spks > 1:
        if len(row) == 3:
            text = row[2]
        elif len(row) == 5:
            text = row[4]
        else:
            raise ValueError(f"invalid multi-speaker row ({len(row)} cols): {row!r}")
    else:
        if len(row) == 2:
            text = row[1]
        elif len(row) == 3:
            text = row[2]
        elif len(row) == 4:
            text = row[3]
        elif len(row) == 5:
            text = row[4]
        else:
            raise ValueError(f"invalid single-speaker row ({len(row)} cols): {row!r}")
    return filepath, text


def _resolve_model_key(cleaner_names: list[str]) -> str:
    for name in cleaner_names:
        if name in _CLEANER_TO_MODEL_KEY:
            return _CLEANER_TO_MODEL_KEY[name]
    return next(iter(lang2inventory.keys()))


def _unknown_tokens(cleaned: str, cleaner_names: list[str], symbol_to_id: dict[str, int]) -> list[str]:
    token_level = any(name in _TOKEN_SEQUENCE_CLEANERS for name in cleaner_names)
    if token_level:
        units = cleaned.split()
    else:
        units = list(cleaned)
    seen: set[str] = set()
    unknown: list[str] = []
    for unit in units:
        if not unit or unit in seen:
            continue
        if unit not in symbol_to_id:
            unknown.append(unit)
            seen.add(unit)
    return unknown


def _collect_text_rows(
    split: str,
    filelist_path: Path,
    n_spks: int,
) -> tuple[list[tuple[int, str, str]], list[str]]:
    rows_out: list[tuple[int, str, str]] = []
    parse_errors: list[str] = []
    rows = parse_filelist(filelist_path)

    for line_no, row in enumerate(rows, start=1):
        if not row or not any(cell.strip() for cell in row):
            continue
        try:
            filepath, text = _parse_text_column(row, n_spks)
        except ValueError as exc:
            parse_errors.append(f"{split}\tline {line_no}\t{exc}\t{row!r}")
            continue
        rows_out.append((line_no, filepath, text))
    return rows_out, parse_errors


def _check_filelist(
    split: str,
    text_rows: list[tuple[int, str, str]],
    cleaners: list[str],
    symbol_to_id: dict[str, int],
) -> list[Issue]:
    issues: list[Issue] = []

    for line_no, filepath, text in text_rows:
        try:
            cleaned = _clean_text(text, cleaners)
            unknown = _unknown_tokens(cleaned, cleaners, symbol_to_id)
        except Exception as exc:  # noqa: BLE001 — report cleaner failures as issues
            issues.append(
                Issue(
                    split=split,
                    line_no=line_no,
                    filepath=filepath,
                    text=text,
                    cleaned="",
                    unknown_tokens=[f"<cleaner error: {exc}>"],
                )
            )
            continue

        if unknown:
            issues.append(
                Issue(
                    split=split,
                    line_no=line_no,
                    filepath=filepath,
                    text=text,
                    cleaned=cleaned,
                    unknown_tokens=unknown,
                )
            )

    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight filelist cleaner / symbol check.")
    parser.add_argument("yaml", type=Path, help="Data config YAML (train/valid paths + cleaners).")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repo root for relative filelist paths (default: script directory).",
    )
    parser.add_argument(
        "--max-report",
        type=int,
        default=100,
        help="Max issue lines to print per split (default: 100).",
    )
    args = parser.parse_args()

    yaml_path = args.yaml.resolve()
    if not yaml_path.is_file():
        print(f"config not found: {yaml_path}", file=sys.stderr)
        return 2

    cfg = _load_data_config(yaml_path)
    train_key = "train_filelist_path"
    valid_key = "valid_filelist_path"
    for key in (train_key, valid_key, "cleaners"):
        if key not in cfg:
            print(f"missing required key {key!r} in {yaml_path}", file=sys.stderr)
            return 2

    cleaners = list(cfg["cleaners"])
    n_spks = int(cfg.get("n_spks", 2))
    model_key = _resolve_model_key(cleaners)
    _activate_model_symbols(model_key)
    symbol_to_id = lang2inventory[model_key]["symbol_to_id"]

    print(f"config:     {yaml_path}")
    print(f"cleaners:   {cleaners}")
    print(f"model_key:  {model_key}  (vocab size {len(symbol_to_id)})")
    print()

    all_issues: list[Issue] = []
    all_parse_errors: list[str] = []
    totals: dict[str, int] = {}

    for split, rel_path in (("train", cfg[train_key]), ("valid", cfg[valid_key])):
        filelist_path = _resolve_path(rel_path, args.repo_root)
        if not filelist_path.is_file():
            print(f"[{split}] MISSING filelist: {filelist_path}", file=sys.stderr)
            return 2

        text_rows, parse_errors = _collect_text_rows(split, filelist_path, n_spks)
        all_parse_errors.extend(parse_errors)

        issues = _check_filelist(split, text_rows, cleaners, symbol_to_id)
        totals[split] = len(text_rows)
        all_issues.extend(issues)
        print(f"[{split}] {filelist_path}")
        print(
            f"         rows={len(text_rows)}  issues={len(issues)}  "
            f"parse_errors={len(parse_errors)}"
        )

    print()
    if all_parse_errors:
        print("=== Parse errors ===")
        for line in all_parse_errors[: args.max_report]:
            print(line)
        if len(all_parse_errors) > args.max_report:
            print(f"... and {len(all_parse_errors) - args.max_report} more parse errors")
        print()

    if not all_issues and not all_parse_errors:
        print("OK: no unknown tokens after cleaning.")
        return 0

    if all_issues:
        print("=== Unknown tokens (after cleaners) ===")
        shown = 0
        for issue in all_issues:
            if shown >= args.max_report:
                break
            preview = issue.text[:120].replace("\n", " ")
            print(
                f"{issue.split}\tline {issue.line_no}\tunk={issue.unknown_tokens!r}\n"
                f"  path: {issue.filepath}\n"
                f"  text: {preview}\n"
                f"  cleaned: {issue.cleaned[:160]}{'...' if len(issue.cleaned) > 160 else ''}"
            )
            shown += 1
        if len(all_issues) > args.max_report:
            print(f"... and {len(all_issues) - args.max_report} more rows with <unk>")

    print()
    print(
        f"FAIL: {len(all_issues)} row(s) with unknown tokens, "
        f"{len(all_parse_errors)} parse error(s)."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
