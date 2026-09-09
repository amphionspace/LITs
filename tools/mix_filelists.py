#!/usr/bin/env python3
"""Mix filelists from multiple groups without spk_idx collisions.

Each group may label speakers from 0 internally. Groups are processed in the
order given on the command line:

  * **First group** keeps its original spk_idx (e.g. Arabic 0–5 → 0–5).
  * **Later groups** are shifted to start after the first group's max id
    (e.g. English 0–7 → 6–13).

Supported line formats::

    <wav_path>|<spk_idx>|<text>
    <wav_path>|<spk_idx>|<start>|<end>|<text>

Usage
-----
  # Arabic already split; English merged then randomly split
  python mix_filelists.py \\
      --presplit-group ar ar_train.txt ar_val.txt \\
      --group en LJSpeech.txt LJS_synth_all_arpa.txt en_40k.txt \\
      --output-prefix en-zh_mix24k \\
      --output-dir data/filelists

  # all groups randomly split
  python mix_filelists.py \\
      --group ar arabic_all.txt \\
      --group en LJSpeech.txt LJS_synth_all_arpa.txt \\
      --output-prefix en-zh_mix24k
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FilelistLine:
    raw: str
    wav_path: str
    spk_idx: int
    tail: str  # text, or "start|end|text"

    @classmethod
    def parse(cls, line: str) -> FilelistLine | None:
        line = line.strip()
        if not line or line.startswith("#"):
            return None
        parts = line.split("|")
        if len(parts) < 3:
            return None
        try:
            spk_idx = int(parts[1])
        except ValueError:
            return None
        return cls(
            raw=line,
            wav_path=parts[0],
            spk_idx=spk_idx,
            tail="|".join(parts[2:]),
        )

    def with_spk_idx(self, spk_idx: int) -> str:
        return f"{self.wav_path}|{spk_idx}|{self.tail}"


def read_filelist(path: Path) -> list[FilelistLine]:
    lines: list[FilelistLine] = []
    with path.open(encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            item = FilelistLine.parse(raw)
            if item is None:
                if raw.strip() and not raw.strip().startswith("#"):
                    raise ValueError(f"{path}:{lineno}: invalid line format")
                continue
            lines.append(item)
    return lines


def compute_group_mapping(
    spk_ids: set[int], next_start: int, *, preserve_ids: bool
) -> tuple[dict[int, int], int]:
    """Map local speaker ids to non-overlapping global ids.

    The first group keeps its original ids (e.g. Arabic 0–5 stays 0–5).
    Later groups are shifted to begin at ``next_start`` (e.g. English 0→6).
    """
    sorted_ids = sorted(spk_ids)
    if preserve_ids:
        mapping = {spk: spk for spk in sorted_ids}
        return mapping, max(sorted_ids) + 1

    min_spk = sorted_ids[0]
    mapping = {old: next_start + (old - min_spk) for old in sorted_ids}
    return mapping, next_start + len(sorted_ids)


def remap_group(
    lines: list[FilelistLine], mapping: dict[int, int]
) -> list[str]:
    remapped: list[str] = []
    for line in lines:
        if line.spk_idx not in mapping:
            raise ValueError(f"speaker id {line.spk_idx} missing from mapping")
        remapped.append(line.with_spk_idx(mapping[line.spk_idx]))
    return remapped


def split_train_val(
    lines: list[str], train_ratio: float, rng: random.Random
) -> tuple[list[str], list[str]]:
    shuffled = lines[:]
    rng.shuffle(shuffled)
    split = int(len(shuffled) * train_ratio)
    return shuffled[:split], shuffled[split:]


def print_mapping(group_name: str, mapping: dict[int, int]) -> None:
    pairs = ", ".join(f"{old}->{new}" for old, new in sorted(mapping.items()))
    print(f"  {group_name}: {pairs}")


def process_presplit_group(
    group_name: str,
    train_paths: list[Path],
    val_paths: list[Path],
    *,
    group_idx: int,
    next_start: int,
    train_list: list[str],
    val_list: list[str],
) -> int:
    train_lines: list[FilelistLine] = []
    val_lines: list[FilelistLine] = []
    for path in train_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        train_lines.extend(read_filelist(path))
    for path in val_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        val_lines.extend(read_filelist(path))

    if not train_lines and not val_lines:
        raise ValueError(f"presplit group '{group_name}' is empty")

    spk_ids = {line.spk_idx for line in train_lines + val_lines}
    mapping, next_start = compute_group_mapping(
        spk_ids, next_start, preserve_ids=(group_idx == 0)
    )
    print_mapping(group_name, mapping)

    train_list.extend(remap_group(train_lines, mapping))
    val_list.extend(remap_group(val_lines, mapping))
    print(
        f"  {group_name} presplit: train={len(train_lines)}, val={len(val_lines)}"
    )
    return next_start


def process_split_group(
    group_name: str,
    paths: list[Path],
    *,
    group_idx: int,
    next_start: int,
    train_ratio: float,
    rng: random.Random,
    train_list: list[str],
    val_list: list[str],
) -> int:
    if not paths:
        raise ValueError(f"group '{group_name}' has no filelists")

    group_lines: list[FilelistLine] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        group_lines.extend(read_filelist(path))

    spk_ids = {line.spk_idx for line in group_lines}
    mapping, next_start = compute_group_mapping(
        spk_ids, next_start, preserve_ids=(group_idx == 0)
    )
    print_mapping(group_name, mapping)

    remapped = remap_group(group_lines, mapping)
    train_part, val_part = split_train_val(remapped, train_ratio, rng)
    train_list.extend(train_part)
    val_list.extend(val_part)
    print(
        f"  {group_name} split: total={len(group_lines)}, "
        f"train={len(train_part)}, val={len(val_part)}"
    )
    return next_start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        action="append",
        nargs="+",
        metavar=("NAME", "FILE"),
        default=[],
        help="Group to randomly split: name plus one or more filelists",
    )
    parser.add_argument(
        "--presplit-group",
        action="append",
        nargs=3,
        metavar=("NAME", "TRAIN", "VAL"),
        default=[],
        help=(
            "Group with fixed train/val files; speaker remap still applies. "
            "Processed before --group entries, in listed order."
        ),
    )
    parser.add_argument(
        "--presplit-train",
        type=Path,
        nargs="*",
        default=[],
        help="Extra training lines appended as-is (no speaker remap)",
    )
    parser.add_argument(
        "--presplit-val",
        type=Path,
        nargs="*",
        default=[],
        help="Extra validation lines appended as-is (no speaker remap)",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        required=True,
        help="Output filename prefix, e.g. en-zh_mix24k",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/filelists"),
        help="Output directory (default: data/filelists)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Extra prefix for output files, e.g. transsion_",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Train/val split ratio for --group files (default: 0.9)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    if not args.group and not args.presplit_group:
        parser.error("need at least one --presplit-group or --group")

    rng = random.Random(args.seed)

    train_list: list[str] = []
    val_list: list[str] = []
    next_start = 0
    group_idx = 0

    print("Speaker remapping:")
    for name, train_file, val_file in args.presplit_group:
        next_start = process_presplit_group(
            name,
            [Path(train_file)],
            [Path(val_file)],
            group_idx=group_idx,
            next_start=next_start,
            train_list=train_list,
            val_list=val_list,
        )
        group_idx += 1

    for group_spec in args.group:
        group_name = group_spec[0]
        paths = [Path(p) for p in group_spec[1:]]
        next_start = process_split_group(
            group_name,
            paths,
            group_idx=group_idx,
            next_start=next_start,
            train_ratio=args.train_ratio,
            rng=rng,
            train_list=train_list,
            val_list=val_list,
        )
        group_idx += 1

    for path in args.presplit_train:
        if not path.is_file():
            raise FileNotFoundError(path)
        train_list.extend(line.raw for line in read_filelist(path))

    for path in args.presplit_val:
        if not path.is_file():
            raise FileNotFoundError(path)
        val_list.extend(line.raw for line in read_filelist(path))

    rng.shuffle(train_list)
    rng.shuffle(val_list)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / f"{args.prefix}{args.output_prefix}_training.txt"
    val_path = args.output_dir / f"{args.prefix}{args.output_prefix}_validation.txt"

    with train_path.open("w", encoding="utf-8") as f:
        for line in train_list:
            f.write(line + "\n")

    with val_path.open("w", encoding="utf-8") as f:
        for line in val_list:
            f.write(line + "\n")

    all_spks = sorted({int(line.split("|", 3)[1]) for line in train_list + val_list})
    print()
    print(f"Wrote {train_path} ({len(train_list)} lines)")
    print(f"Wrote {val_path} ({len(val_list)} lines)")
    print(f"Global speaker ids: {all_spks} (n_spks={len(all_spks)})")


if __name__ == "__main__":
    main()
