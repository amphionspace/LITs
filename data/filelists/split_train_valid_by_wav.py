#!/usr/bin/env python3
"""Split filelist(s) into train/valid by wav, keeping all text versions together.

When one utterance has multiple rows (no-diac / partial / full), every row sharing
the same ``wav`` path is assigned to the same split.

Supports a paired filelist (e.g. with-diac) aligned 1:1 on ``wav`` — the same split
is applied to both.

Examples:
  # Merged filelist (multi-row per wav)
  python data/filelists/split_train_valid_by_wav.py \\
    --input data/filelists/all_mixed.txt \\
    --train-output data/filelists/train_mixed.txt \\
    --valid-output data/filelists/valid_mixed.txt \\
    --ratio 0.9 --seed 42 --shuffle

  # Split no-diac + with-diac before merge
  python data/filelists/split_train_valid_by_wav.py \\
    --input data/filelists/all_no_diac.txt \\
    --paired-input data/filelists/all_with_diac.txt \\
    --train-output data/filelists/train_no_diac.txt \\
    --valid-output data/filelists/valid_no_diac.txt \\
    --paired-train-output data/filelists/train_with_diac.txt \\
    --paired-valid-output data/filelists/valid_with_diac.txt \\
    --ratio 0.9 --seed 42
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split filelist by wav so all versions of an utterance stay in one split."
    )
    parser.add_argument("--input", type=Path, required=True, help="Primary filelist (wav|spk|text).")
    parser.add_argument("--train-output", type=Path, required=True, help="Train split output path.")
    parser.add_argument("--valid-output", type=Path, required=True, help="Valid split output path.")
    parser.add_argument(
        "--paired-input",
        type=Path,
        default=None,
        help="Optional aligned filelist with the same wav keys (typically one row per wav).",
    )
    parser.add_argument(
        "--paired-train-output",
        type=Path,
        default=None,
        help="Train output for --paired-input (required when --paired-input is set).",
    )
    parser.add_argument(
        "--paired-valid-output",
        type=Path,
        default=None,
        help="Valid output for --paired-input (required when --paired-input is set).",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.9,
        help="Fraction of wav groups assigned to train (default: 0.9).",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for group shuffling.")
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle rows within each output file (groups are still not split).",
    )
    return parser.parse_args()


def _load_groups(path: Path) -> tuple[dict[str, list[str]], list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    wav_order: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split("|")
            if len(parts) < 3:
                raise ValueError(f"{path}:{line_no}: expected wav|spk|text, got {len(parts)} fields")
            wav = parts[0]
            if not wav:
                raise ValueError(f"{path}:{line_no}: empty wav path")
            if wav not in groups:
                wav_order.append(wav)
            groups[wav].append(stripped)
    if not wav_order:
        raise ValueError(f"{path}: empty filelist")
    return groups, wav_order


def _load_paired_rows(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split("|")
            if len(parts) < 3:
                raise ValueError(f"{path}:{line_no}: expected wav|spk|text, got {len(parts)} fields")
            wav = parts[0]
            if not wav:
                raise ValueError(f"{path}:{line_no}: empty wav path")
            if wav in rows:
                raise ValueError(f"{path}:{line_no}: duplicate wav path {wav!r}")
            rows[wav] = stripped
    if not rows:
        raise ValueError(f"{path}: empty filelist")
    return rows


def _split_wav_keys(wav_keys: list[str], *, ratio: float, seed: int) -> set[str]:
    if not 0.0 < ratio < 1.0:
        raise ValueError(f"--ratio must be in (0, 1), got {ratio}")
    rng = random.Random(seed)
    shuffled = list(wav_keys)
    rng.shuffle(shuffled)
    n_train = int(len(shuffled) * ratio)
    if n_train <= 0:
        raise ValueError(f"--ratio {ratio} yields 0 train groups for {len(shuffled)} utterances")
    if n_train >= len(shuffled):
        raise ValueError(
            f"--ratio {ratio} yields all {len(shuffled)} groups in train; valid would be empty"
        )
    return set(shuffled[:n_train])


def _collect_lines(groups: dict[str, list[str]], wav_keys: list[str]) -> list[str]:
    lines: list[str] = []
    for wav in wav_keys:
        lines.extend(groups[wav])
    return lines


def _write_lines(path: Path, lines: list[str], *, shuffle: bool, seed: int) -> None:
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def split_filelists(
    primary_groups: dict[str, list[str]],
    wav_order: list[str],
    paired_rows: dict[str, str] | None,
    *,
    ratio: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    primary_wavs = set(wav_order)
    if paired_rows is not None:
        paired_wavs = set(paired_rows)
        only_primary = sorted(primary_wavs - paired_wavs)
        only_paired = sorted(paired_wavs - primary_wavs)
        if only_primary or only_paired:
            msg = []
            if only_primary:
                msg.append(f"only in --input ({len(only_primary)}): {only_primary[:3]}")
            if only_paired:
                msg.append(f"only in --paired-input ({len(only_paired)}): {only_paired[:3]}")
            raise ValueError("wav keys do not match between inputs:\n  " + "\n  ".join(msg))

    train_wavs = _split_wav_keys(wav_order, ratio=ratio, seed=seed)
    valid_wavs = primary_wavs - train_wavs
    return train_wavs, valid_wavs


def main() -> int:
    args = parse_args()

    if args.paired_input is not None:
        if args.paired_train_output is None or args.paired_valid_output is None:
            print(
                "error: --paired-train-output and --paired-valid-output are required "
                "when --paired-input is set",
                file=sys.stderr,
            )
            return 2
    elif args.paired_train_output is not None or args.paired_valid_output is not None:
        print("error: paired outputs require --paired-input", file=sys.stderr)
        return 2

    primary_groups, wav_order = _load_groups(args.input.resolve())
    paired_rows = _load_paired_rows(args.paired_input.resolve()) if args.paired_input else None

    train_wavs, valid_wavs = split_filelists(
        primary_groups,
        wav_order,
        paired_rows,
        ratio=args.ratio,
        seed=args.seed,
    )

    train_order = [wav for wav in wav_order if wav in train_wavs]
    valid_order = [wav for wav in wav_order if wav in valid_wavs]

    train_lines = _collect_lines(primary_groups, train_order)
    valid_lines = _collect_lines(primary_groups, valid_order)
    _write_lines(args.train_output, train_lines, shuffle=args.shuffle, seed=args.seed)
    _write_lines(args.valid_output, valid_lines, shuffle=args.shuffle, seed=args.seed + 1)

    paired_train_lines: list[str] = []
    paired_valid_lines: list[str] = []
    if paired_rows is not None:
        paired_train_lines = [paired_rows[wav] for wav in train_order]
        paired_valid_lines = [paired_rows[wav] for wav in valid_order]
        _write_lines(args.paired_train_output, paired_train_lines, shuffle=args.shuffle, seed=args.seed)
        _write_lines(args.paired_valid_output, paired_valid_lines, shuffle=args.shuffle, seed=args.seed + 1)

    multi_version_wavs = sum(1 for wav in wav_order if len(primary_groups[wav]) > 1)
    train_path = args.train_output.resolve()
    valid_path = args.valid_output.resolve()
    print(f"input:            {args.input.resolve()}")
    if args.paired_input:
        print(f"paired input:     {args.paired_input.resolve()}")
    print(f"utterances:       {len(wav_order)}  (multi-row: {multi_version_wavs})")
    print(f"ratio:            {args.ratio}  seed={args.seed}")
    print(
        f"train groups:     {len(train_wavs)}  rows={len(train_lines)}  -> {train_path}"
    )
    print(
        f"valid groups:     {len(valid_wavs)}  rows={len(valid_lines)}  -> {valid_path}"
    )
    if paired_rows is not None:
        print(
            f"paired train:     {len(paired_train_lines)} rows -> "
            f"{args.paired_train_output.resolve()}"
        )
        print(
            f"paired valid:     {len(paired_valid_lines)} rows -> "
            f"{args.paired_valid_output.resolve()}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
