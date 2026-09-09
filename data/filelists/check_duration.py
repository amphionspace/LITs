#!/usr/bin/env python3
"""Filter filelist entries longer than a duration threshold."""

from __future__ import annotations

import argparse
import wave
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path

from tqdm import tqdm

LONG_THRESHOLD = 30.0  # seconds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a filelist with utterances longer than the threshold removed."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input filelist path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output filelist path. Defaults to <input_stem>_max30s<suffix> beside input.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=LONG_THRESHOLD,
        help=f"Drop entries longer than this many seconds (default: {LONG_THRESHOLD}).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional path to write removed long entries.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Parallel workers (default: min(16, cpu_count()); use 1 for sequential).",
    )
    return parser.parse_args()


def get_duration(wav_path: str) -> float | None:
    try:
        with wave.open(wav_path, "rb") as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return None


def _check_line(args: tuple[str, float]) -> tuple[str, str]:
    """Return (status, raw_line); status is kept|long|missing|error."""
    raw, threshold = args
    wav_path = raw.strip().split("|", 1)[0]

    if not Path(wav_path).exists():
        return "missing", raw

    dur = get_duration(wav_path)
    if dur is None:
        return "error", raw

    if dur > threshold:
        return "long", raw

    return "kept", raw


def _collect_results(
    results: list[tuple[str, str]],
) -> tuple[list[str], list[str], dict[str, int]]:
    kept: list[str] = []
    removed: list[str] = []
    stats = defaultdict(int)

    for status, raw in results:
        stats["total"] += 1
        if status == "missing":
            stats["missing"] += 1
        elif status == "error":
            stats["errors"] += 1
        elif status == "long":
            stats["long"] += 1
            removed.append(raw)
        else:
            stats["kept"] += 1
            kept.append(raw)

    return kept, removed, stats


def filter_filelist(
    filelist_path: Path,
    threshold: float = LONG_THRESHOLD,
    jobs: int | None = None,
) -> tuple[list[str], list[str], dict[str, int]]:
    with open(filelist_path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]

    work = [(raw, threshold) for raw in lines]
    if not work:
        return [], [], defaultdict(int)

    workers = jobs if jobs is not None else min(16, cpu_count())
    if workers <= 1:
        results = [
            _check_line(item)
            for item in tqdm(work, desc="Checking durations", unit="line")
        ]
    else:
        with Pool(processes=workers) as pool:
            results = list(
                tqdm(
                    pool.imap(_check_line, work, chunksize=256),
                    total=len(work),
                    desc="Checking durations",
                    unit="line",
                )
            )

    return _collect_results(results)


def default_output_path(input_path: Path, threshold: float) -> Path:
    suffix = input_path.suffix or ".txt"
    stem = input_path.stem
    threshold_label = str(int(threshold)) if threshold.is_integer() else str(threshold).replace(".", "p")
    return input_path.with_name(f"{stem}_max{threshold_label}s{suffix}")


def print_summary(name: str, stats: dict[str, int], threshold: float, output_path: Path) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {name}")
    print(f"{'=' * 70}")
    print(f"  Total lines:      {stats['total']}")
    print(f"  Kept (<= {threshold:g}s): {stats['kept']}")
    print(f"  Removed (> {threshold:g}s): {stats['long']}")
    print(f"  Missing wav:      {stats['missing']}")
    print(f"  Read errors:      {stats['errors']}")
    print(f"  Output:           {output_path}")


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    if not input_path.is_file():
        raise SystemExit(f"[ERROR] Input file not found: {input_path}")

    output_path = args.output.resolve() if args.output else default_output_path(input_path, args.threshold)
    kept, removed, stats = filter_filelist(
        input_path,
        threshold=args.threshold,
        jobs=args.jobs,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        if kept:
            f.write("\n".join(kept) + "\n")

    if args.report and removed:
        report_path = args.report.resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(removed) + "\n")
        print(f"Removed entries saved to: {report_path}")

    print_summary(input_path.name, stats, args.threshold, output_path)


if __name__ == "__main__":
    main()
