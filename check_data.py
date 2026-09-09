"""Check training data for common issues that can cause NaN / loss spikes."""

import argparse
import os
import sys
import math
import numpy as np
import soundfile as sf
import torch
from collections import Counter, defaultdict
from pathlib import Path

def mel_spectrogram_check(audio, n_fft, n_mels, sr, hop_length, win_length, f_min, f_max):
    """Minimal mel computation to check for NaN/Inf."""
    import torchaudio
    mel_fn = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=n_fft, n_mels=n_mels,
        hop_length=hop_length, win_length=win_length,
        f_min=f_min, f_max=f_max, center=False,
    )
    mel = mel_fn(audio)
    mel = torch.clamp(mel, min=1e-5).log()
    return mel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filelist", help="Path to training filelist")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--n_mels", type=int, default=88)
    parser.add_argument("--hop_length", type=int, default=256)
    parser.add_argument("--win_length", type=int, default=1024)
    parser.add_argument("--f_min", type=float, default=0)
    parser.add_argument("--f_max", type=float, default=8000)
    parser.add_argument("--mel_mean", type=float, default=-5.391725063323975)
    parser.add_argument("--mel_std", type=float, default=2.453462839126587)
    parser.add_argument("--max_dur", type=float, default=30.0, help="Flag audio longer than this (sec)")
    parser.add_argument("--min_dur", type=float, default=0.3, help="Flag audio shorter than this (sec)")
    args = parser.parse_args()

    with open(args.filelist) as f:
        lines = f.read().strip().split("\n")

    print(f"Total entries: {len(lines)}")
    print("=" * 70)

    issues = defaultdict(list)
    durations = []
    spk_counts = Counter()
    mel_stats = []

    for i, line in enumerate(lines):
        parts = line.strip().split("|")
        if len(parts) < 3:
            issues["bad_format"].append((i, line[:100]))
            continue

        wav_path, spk_str, text = parts[0], parts[1], parts[2]
        spk_counts[spk_str] += 1

        # 1. Check file exists
        if not os.path.isfile(wav_path):
            issues["missing_file"].append((i, wav_path))
            continue

        # 2. Check audio
        try:
            info = sf.info(wav_path)
        except Exception as e:
            issues["unreadable_file"].append((i, wav_path, str(e)))
            continue

        # Sample rate
        if info.samplerate != args.sr:
            issues["wrong_sr"].append((i, wav_path, info.samplerate))

        # Duration
        dur = info.frames / info.samplerate
        durations.append((i, wav_path, dur))
        if dur > args.max_dur:
            issues["too_long"].append((i, wav_path, f"{dur:.1f}s"))
        if dur < args.min_dur:
            issues["too_short"].append((i, wav_path, f"{dur:.3f}s"))
        if info.frames == 0:
            issues["empty_audio"].append((i, wav_path))
            continue

        # 3. Check text
        if len(text.strip()) == 0:
            issues["empty_text"].append((i, wav_path))

        # 4. Check mel for NaN/Inf (sample every 100th)
        if i % 100 == 0:
            try:
                audio_np, sr = sf.read(wav_path, always_2d=True, dtype="float32")
                audio = torch.from_numpy(audio_np.T)

                if torch.isnan(audio).any() or torch.isinf(audio).any():
                    issues["nan_audio"].append((i, wav_path))
                    continue

                if audio.abs().max() < 1e-6:
                    issues["silent_audio"].append((i, wav_path))
                    continue

                mel = mel_spectrogram_check(
                    audio, args.n_fft, args.n_mels, args.sr,
                    args.hop_length, args.win_length, args.f_min, args.f_max
                )

                norm_mel = (mel - args.mel_mean) / args.mel_std

                if torch.isnan(norm_mel).any():
                    issues["nan_mel"].append((i, wav_path))
                if torch.isinf(norm_mel).any():
                    issues["inf_mel"].append((i, wav_path))

                mel_max = norm_mel.max().item()
                mel_min = norm_mel.min().item()
                if abs(mel_max) > 20 or abs(mel_min) > 20:
                    issues["extreme_mel"].append((i, wav_path, f"min={mel_min:.1f}, max={mel_max:.1f}"))

            except Exception as e:
                issues["mel_error"].append((i, wav_path, str(e)))

        if (i + 1) % 5000 == 0:
            print(f"  Checked {i+1}/{len(lines)}...")

    # Print results
    print(f"\n{'=' * 70}")
    print("DATA CHECK RESULTS")
    print(f"{'=' * 70}")

    print(f"\nTotal entries: {len(lines)}")
    print(f"Speaker distribution: {dict(spk_counts)}")

    if durations:
        durs_only = [d for _, _, d in durations]
        print(f"\nDuration stats:")
        print(f"  min:    {min(durs_only):.3f}s")
        print(f"  max:    {max(durs_only):.3f}s")
        print(f"  mean:   {np.mean(durs_only):.3f}s")
        print(f"  median: {np.median(durs_only):.3f}s")
        print(f"  total:  {sum(durs_only)/3600:.1f}h")

    if not issues:
        print("\nNo issues found!")
    else:
        print(f"\n{'!' * 70}")
        print("ISSUES FOUND:")
        print(f"{'!' * 70}")
        for issue_type, entries in sorted(issues.items()):
            print(f"\n[{issue_type}] ({len(entries)} entries)")
            for entry in entries[:10]:
                print(f"  {entry}")
            if len(entries) > 10:
                print(f"  ... and {len(entries)-10} more")

    total_issues = sum(len(v) for v in issues.values())
    print(f"\nTotal issues: {total_issues}")
    return 1 if total_issues > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
