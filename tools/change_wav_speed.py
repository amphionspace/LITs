#!/usr/bin/env python3
"""Change playback speed of all wav files in a folder.

Usage:
    python tools/change_wav_speed.py \
        --input_dir /path/to/wavs \
        --output_dir /path/to/output \
        --speed 1.2 \
        --workers 8

Speed > 1.0 speeds up (shorter duration), < 1.0 slows down.
By default pitch/tone is preserved (sox tempo). Original files are not modified.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

SOX_BIN = shutil.which("sox")
FFMPEG_BIN = shutil.which("ffmpeg")


def _atempo_filter_chain(speed: float) -> str:
    """Build ffmpeg atempo filter chain (each atempo node only supports 0.5-2.0)."""
    if speed <= 0:
        raise ValueError(f"speed must be > 0, got {speed}")

    filters: list[str] = []
    remaining = speed
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    filters.append(f"atempo={remaining:.8f}".rstrip("0").rstrip("."))
    return ",".join(filters)


def _change_speed_sox(src_path: str, dst_path: str, speed: float) -> None:
    if SOX_BIN is None:
        raise RuntimeError("sox not found in PATH")
    subprocess.run(
        [SOX_BIN, src_path, dst_path, "tempo", f"{speed:g}"],
        check=True,
        capture_output=True,
        text=True,
    )


def _change_speed_ffmpeg(src_path: str, dst_path: str, speed: float) -> None:
    if FFMPEG_BIN is None:
        raise RuntimeError("ffmpeg not found in PATH")
    subprocess.run(
        [
            FFMPEG_BIN,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            src_path,
            "-filter:a",
            _atempo_filter_chain(speed),
            dst_path,
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def _change_speed_librosa(src_path: str, dst_path: str, speed: float) -> None:
    audio, sr = librosa.load(src_path, sr=None, mono=False)
    if audio.ndim == 1:
        channels = [audio]
    else:
        channels = [audio[i] for i in range(audio.shape[0])]

    out_channels = [librosa.effects.time_stretch(ch, rate=speed) for ch in channels]
    if len(out_channels) == 1:
        audio_out = out_channels[0]
    else:
        audio_out = np.stack(out_channels, axis=0)
    sf.write(dst_path, audio_out.T if audio_out.ndim > 1 else audio_out, sr)


def _change_speed_resample(src_path: str, dst_path: str, speed: float) -> None:
    audio, sr = librosa.load(src_path, sr=None, mono=False)
    if audio.ndim == 1:
        channels = [audio]
    else:
        channels = [audio[i] for i in range(audio.shape[0])]

    target_sr = max(1, int(round(sr * speed)))
    out_channels = [librosa.resample(ch, orig_sr=sr, target_sr=target_sr) for ch in channels]
    if len(out_channels) == 1:
        audio_out = out_channels[0]
    else:
        audio_out = np.stack(out_channels, axis=0)
    sf.write(dst_path, audio_out.T if audio_out.ndim > 1 else audio_out, sr)


def change_speed_one(
    src_path: str,
    dst_path: str,
    speed: float,
    method: str,
) -> str:
    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)

    if speed == 1.0:
        shutil.copy2(src_path, dst_path)
        return dst_path

    if method == "tempo":
        if SOX_BIN is not None:
            _change_speed_sox(src_path, dst_path, speed)
        elif FFMPEG_BIN is not None:
            _change_speed_ffmpeg(src_path, dst_path, speed)
        else:
            _change_speed_librosa(src_path, dst_path, speed)
    elif method == "sox":
        _change_speed_sox(src_path, dst_path, speed)
    elif method == "ffmpeg":
        _change_speed_ffmpeg(src_path, dst_path, speed)
    elif method == "librosa":
        _change_speed_librosa(src_path, dst_path, speed)
    elif method == "resample":
        _change_speed_resample(src_path, dst_path, speed)
    else:
        raise ValueError(f"Unknown method: {method}")

    return dst_path


def collect_wav_files(input_dir: Path, recursive: bool) -> list[Path]:
    if recursive:
        return sorted(p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".wav")
    return sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".wav")


def _resolve_backend(method: str) -> str:
    if method != "tempo":
        return method
    if SOX_BIN is not None:
        return "sox"
    if FFMPEG_BIN is not None:
        return "ffmpeg"
    return "librosa"


def main() -> None:
    parser = argparse.ArgumentParser(description="Change speed of wav files in a folder")
    parser.add_argument("--input_dir", type=str, required=True, help="Folder containing input wav files")
    parser.add_argument("--output_dir", type=str, required=True, help="Folder for speed-adjusted wav files")
    parser.add_argument(
        "--speed",
        type=float,
        required=True,
        help="Speed factor (>1 faster, <1 slower, 1 unchanged)",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=("tempo", "sox", "ffmpeg", "librosa", "resample"),
        default="tempo",
        help="tempo: keep pitch/tone (default, uses sox/ffmpeg); "
        "resample: change pitch with speed; librosa: phase-vocoder fallback",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process wav files in subdirectories and preserve relative paths",
    )
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    args = parser.parse_args()

    if args.speed <= 0:
        raise SystemExit("--speed must be > 0")

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    wav_files = collect_wav_files(input_dir, args.recursive)
    if not wav_files:
        raise SystemExit(f"No wav files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    backend = _resolve_backend(args.method)
    tasks: list[tuple[str, str, float, str]] = []
    for src in wav_files:
        rel = src.relative_to(input_dir)
        dst = output_dir / rel
        tasks.append((str(src), str(dst), args.speed, args.method))

    print(
        f"Changing speed of {len(tasks)} wav files to {args.speed}x "
        f"(method={args.method}, backend={backend}) with {args.workers} workers..."
    )
    print(f"Input : {input_dir}")
    print(f"Output: {output_dir}")

    done = 0
    failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(change_speed_one, src, dst, speed, method): src
            for src, dst, speed, method in tasks
        }
        for future in as_completed(futures):
            src = futures[future]
            try:
                future.result()
                done += 1
            except Exception as exc:
                print(f"[FAILED] {src}: {exc}")
                failed += 1
            if (done + failed) % 100 == 0:
                print(f"  Progress: {done + failed}/{len(tasks)} (failed: {failed})")

    print(f"\nDone. Succeeded: {done}, Failed: {failed}")


if __name__ == "__main__":
    main()
