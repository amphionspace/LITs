"""Resample all wav files referenced in a filelist to a target sample rate.

Usage:
    python tools/resample_filelist.py \
        --filelist data/filelists/en-zh_pinyin_training.txt \
        --target_sr 16000 \
        --output_dir data/resampled_16k \
        --workers 16

The script reads the filelist (first column = wav path), resamples each file
to target_sr, and writes the output to output_dir preserving the relative
directory structure. A new filelist with updated paths is written next to the
original (suffixed with _resampled).
"""

import argparse
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import soundfile as sf
import numpy as np

try:
    import librosa
    USE_LIBROSA = True
except ImportError:
    import torchaudio
    USE_LIBROSA = False


def resample_one(src_path: str, dst_path: str, target_sr: int) -> str:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    if USE_LIBROSA:
        audio, sr = librosa.load(src_path, sr=None, mono=False)
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sf.write(dst_path, audio.T if audio.ndim > 1 else audio, target_sr)
    else:
        waveform, sr = torchaudio.load(src_path)
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        torchaudio.save(dst_path, waveform, target_sr)

    return dst_path


def main():
    parser = argparse.ArgumentParser(description="Resample wavs from a filelist")
    parser.add_argument("--filelist", type=str, required=True, help="Input filelist (pipe-separated, first col is wav path)")
    parser.add_argument("--target_sr", type=int, required=True, help="Target sample rate")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for resampled wavs")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    args = parser.parse_args()

    filelist_path = Path(args.filelist)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(filelist_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    tasks = []
    new_lines = []
    for line in lines:
        cols = line.split("|")
        src_wav = cols[0]
        src_path = Path(src_wav)
        dst_path = str(output_dir / src_path.name)
        cols[0] = dst_path
        new_lines.append("|".join(cols))
        tasks.append((src_wav, dst_path, args.target_sr))

    print(f"Resampling {len(tasks)} files to {args.target_sr}Hz with {args.workers} workers...")

    done = 0
    failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(resample_one, s, d, sr): s for s, d, sr in tasks}
        for future in as_completed(futures):
            src = futures[future]
            try:
                future.result()
                done += 1
            except Exception as e:
                print(f"[FAILED] {src}: {e}")
                failed += 1
            if (done + failed) % 500 == 0:
                print(f"  Progress: {done + failed}/{len(tasks)} (failed: {failed})")

    out_filelist = filelist_path.with_name(filelist_path.stem + "_resampled.txt")
    with open(out_filelist, "w", encoding="utf-8") as f:
        for line in new_lines:
            f.write(line + "\n")

    print(f"\nDone. Resampled: {done}, Failed: {failed}")
    print(f"New filelist: {out_filelist}")


if __name__ == "__main__":
    main()
