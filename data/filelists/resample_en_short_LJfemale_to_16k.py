#!/usr/bin/env python3
"""Resample en_short_LJfemale wavs from 24 kHz to 16 kHz."""

from __future__ import annotations

import argparse
from multiprocessing import Pool, cpu_count
from pathlib import Path

import librosa
import soundfile as sf
from tqdm import tqdm

DEFAULT_INPUT_DIR = Path("/chenmingjie/xingwen/dataset/en_short_LJfemale")
DEFAULT_OUTPUT_DIR = Path("/chenmingjie/xingwen/dataset/en_short_LJfemale_16k")
DEFAULT_SOURCE_SR = 24000
DEFAULT_TARGET_SR = 16000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Downsample en_short_LJfemale wav files from 24 kHz to 16 kHz."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Source wav directory (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output wav directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--source-sr",
        type=int,
        default=DEFAULT_SOURCE_SR,
        help=f"Expected source sample rate (default: {DEFAULT_SOURCE_SR})",
    )
    parser.add_argument(
        "--target-sr",
        type=int,
        default=DEFAULT_TARGET_SR,
        help=f"Target sample rate (default: {DEFAULT_TARGET_SR})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, cpu_count()),
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    return parser.parse_args()


def collect_wav_files(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    wav_files = sorted(input_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No wav files found in: {input_dir}")
    return wav_files


def process_one(task: tuple[Path, Path, int, int, bool]) -> str | None:
    src_path, dst_path, source_sr, target_sr, overwrite = task
    try:
        if dst_path.exists() and not overwrite:
            return None

        dst_path.parent.mkdir(parents=True, exist_ok=True)

        audio, sr = librosa.load(src_path, sr=None, mono=True)
        if sr != source_sr:
            return f"Unexpected sample rate {sr} (expected {source_sr}): {src_path}"

        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

        sf.write(dst_path, audio, target_sr, subtype="PCM_16")
        return None
    except Exception as exc:  # pylint: disable=broad-except
        return f"Failed: {src_path} | {exc}"


def main() -> None:
    args = parse_args()
    wav_files = collect_wav_files(args.input_dir)
    tasks = [
        (
            src_path,
            args.output_dir / src_path.name,
            args.source_sr,
            args.target_sr,
            args.overwrite,
        )
        for src_path in wav_files
    ]

    print(f"Input dir : {args.input_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Resample  : {args.source_sr} Hz -> {args.target_sr} Hz")
    print(f"Files     : {len(tasks)}")
    print(f"Workers   : {args.workers}")

    errors: list[str] = []
    with Pool(processes=args.workers) as pool:
        for result in tqdm(
            pool.imap_unordered(process_one, tasks),
            total=len(tasks),
            desc="Resampling",
        ):
            if result is not None:
                errors.append(result)

    print(f"Done. Errors: {len(errors)}")
    if errors:
        err_file = args.output_dir / "resample_errors.txt"
        err_file.write_text("\n".join(errors) + "\n", encoding="utf-8")
        print(f"Error log: {err_file}")


if __name__ == "__main__":
    main()
