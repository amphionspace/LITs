#!/usr/bin/env python3
"""Re-aggregate MFA duration stats excluding first/last phones (or tokens).

Produces *_cleaned* outputs for ceiling-in-DP retrain and a delta report
against the baseline CSV.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MAPPING_TSV = REPO_ROOT / "mfa_zh354_duration/mapping/mfa_zh354_token_mapping.tsv"

DEFAULT_SAMPLE_RATE = 24000
DEFAULT_HOP = 384
SKIP_PHONE_LABELS = frozenset({"sil", "sp", "spn", ""})


@dataclass
class PhoneInterval:
    label: str
    start: float
    end: float

    @property
    def dur_ms(self) -> float:
        return (self.end - self.start) * 1000.0

    @property
    def dur_frames(self) -> float:
        return (self.end - self.start) * DEFAULT_SAMPLE_RATE / DEFAULT_HOP


def parse_textgrid_phones(path: Path) -> list[PhoneInterval]:
    try:
        import textgrid
    except ImportError as exc:
        raise SystemExit("pip install textgrid") from exc

    tg = textgrid.TextGrid.fromFile(str(path))
    phones: list[PhoneInterval] = []
    for tier in tg:
        if "phone" in tier.name.lower():
            for interval in tier:
                label = interval.mark.strip()
                if not label:
                    continue
                phones.append(
                    PhoneInterval(label, float(interval.minTime), float(interval.maxTime))
                )
            break
    return phones


def middle_phone_durations(phones: list[PhoneInterval]) -> list[tuple[str, float, float]]:
    """Return (label, dur_ms, dur_frames) for interior phones only."""
    if len(phones) <= 2:
        return []
    out: list[tuple[str, float, float]] = []
    for phone in phones[1:-1]:
        if phone.label.lower() in SKIP_PHONE_LABELS:
            continue
        out.append((phone.label, phone.dur_ms, phone.dur_frames))
    return out


def percentile_row(values_ms: np.ndarray, values_frames: np.ndarray) -> dict[str, float]:
    if values_ms.size == 0:
        return {}
    pct = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    ms_pct = np.percentile(values_ms, pct)
    frame_pct = np.percentile(values_frames, [5, 50, 95])
    return {
        "count": int(values_ms.size),
        "mean_ms": float(values_ms.mean()),
        "std_ms": float(values_ms.std()),
        "min_ms": float(values_ms.min()),
        "p01_ms": float(ms_pct[0]),
        "p05_ms": float(ms_pct[1]),
        "p10_ms": float(ms_pct[2]),
        "p25_ms": float(ms_pct[3]),
        "p50_ms": float(ms_pct[4]),
        "p75_ms": float(ms_pct[5]),
        "p90_ms": float(ms_pct[6]),
        "p95_ms": float(ms_pct[7]),
        "p99_ms": float(ms_pct[8]),
        "max_ms": float(values_ms.max()),
        "mean_frames_16ms": float(values_frames.mean()),
        "p05_frames_16ms": float(frame_pct[0]),
        "p50_frames_16ms": float(frame_pct[1]),
        "p95_frames_16ms": float(frame_pct[2]),
        "lt_1_frame_ratio": float((values_frames < 1.0).mean()),
        "lt_2_frames_ratio": float((values_frames < 2.0).mean()),
        "ge_10_frames_ratio": float((values_frames >= 10.0).mean()),
    }


def aggregate_arpa_from_textgrids(align_dir: Path) -> dict[str, dict[str, float]]:
    by_phone_ms: dict[str, list[float]] = defaultdict(list)
    by_phone_frames: dict[str, list[float]] = defaultdict(list)
    tg_paths = sorted(align_dir.glob("*.TextGrid"))
    if not tg_paths:
        raise FileNotFoundError(f"No TextGrid files found in {align_dir}")
    for tg_path in tg_paths:
        phones = parse_textgrid_phones(tg_path)
        for label, dur_ms, dur_frames in middle_phone_durations(phones):
            by_phone_ms[label].append(dur_ms)
            by_phone_frames[label].append(dur_frames)
    stats: dict[str, dict[str, float]] = {}
    for phone in sorted(by_phone_ms):
        row = percentile_row(
            np.asarray(by_phone_ms[phone], dtype=np.float64),
            np.asarray(by_phone_frames[phone], dtype=np.float64),
        )
        if row:
            stats[phone] = row
    return stats


def write_arpa_csv(path: Path, stats: dict[str, dict[str, float]]) -> None:
    fieldnames = [
        "phone",
        "count",
        "mean_ms",
        "std_ms",
        "min_ms",
        "p01_ms",
        "p05_ms",
        "p10_ms",
        "p25_ms",
        "p50_ms",
        "p75_ms",
        "p90_ms",
        "p95_ms",
        "p99_ms",
        "max_ms",
        "mean_frames_16ms",
        "p05_frames_16ms",
        "p50_frames_16ms",
        "p95_frames_16ms",
        "lt_1_frame_ratio",
        "lt_2_frames_ratio",
        "ge_10_frames_ratio",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for phone, row in sorted(stats.items()):
            writer.writerow({"phone": phone, **row})


def load_arpa_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return {row["phone"]: row for row in csv.DictReader(f)}


def write_delta_report(
    path: Path,
    baseline: dict[str, dict[str, str]],
    cleaned: dict[str, dict[str, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "phone",
        "count_baseline",
        "count_cleaned",
        "p95_ms_baseline",
        "p95_ms_cleaned",
        "p95_ms_delta_pct",
        "p99_ms_baseline",
        "p99_ms_cleaned",
        "p99_ms_delta_pct",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        phones = sorted(set(baseline) | set(cleaned))
        for phone in phones:
            base = baseline.get(phone, {})
            clean = cleaned.get(phone, {})
            if not base or not clean:
                continue
            p95_b = float(base.get("p95_ms", 0) or 0)
            p95_c = float(clean.get("p95_ms", 0) or 0)
            p99_b = float(base.get("p99_ms", 0) or 0)
            p99_c = float(clean.get("p99_ms", 0) or 0)
            writer.writerow(
                {
                    "phone": phone,
                    "count_baseline": base.get("count", ""),
                    "count_cleaned": clean.get("count", ""),
                    "p95_ms_baseline": p95_b,
                    "p95_ms_cleaned": p95_c,
                    "p95_ms_delta_pct": ((p95_c - p95_b) / p95_b * 100.0) if p95_b else 0.0,
                    "p99_ms_baseline": p99_b,
                    "p99_ms_cleaned": p99_c,
                    "p99_ms_delta_pct": ((p99_c - p99_b) / p99_b * 100.0) if p99_b else 0.0,
                }
            )


def aggregate_zh_from_textgrids(
    align_dir: Path,
    data_txt: Path,
    mapping_tsv: Path,
    cleaner: str,
) -> dict[str, dict[str, float]]:
    sys.path.insert(0, str(REPO_ROOT / "mfa_zh354_duration"))
    from map import load_mapping_tsv
    from run import assign_token_durations, parse_textgrid_phones, zh_tokens_from_text

    mapping = load_mapping_tsv(mapping_tsv)
    by_token: dict[str, list[float]] = defaultdict(list)
    utterances: dict[str, str] = {}
    with data_txt.open(encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) < 3:
                continue
            utt_id = Path(parts[0]).stem
            utterances[utt_id] = parts[2].strip()

    for utt_id, text in utterances.items():
        tg_path = align_dir / f"{utt_id}.TextGrid"
        if not tg_path.is_file():
            continue
        phones = parse_textgrid_phones(tg_path)
        tokens = zh_tokens_from_text(text)
        assigns = assign_token_durations(phones, tokens, mapping)
        if len(assigns) <= 2:
            continue
        for tok, dur_frames in assigns[1:-1]:
            by_token[tok].append(float(dur_frames))

    stats: dict[str, dict[str, float]] = {}
    for token, values in sorted(by_token.items()):
        arr = np.asarray(values, dtype=np.float64)
        pct = np.percentile(arr, [1, 5, 50, 95, 99])
        stats[token] = {
            "count": int(arr.size),
            "mean_frames": float(arr.mean()),
            "std_frames": float(arr.std()),
            "p01": float(pct[0]),
            "p5": float(pct[1]),
            "p50": float(pct[2]),
            "p95": float(pct[3]),
            "p99": float(pct[4]),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
    return stats


def write_zh_tsv(path: Path, stats: dict[str, dict[str, float]], baseline_path: Path | None) -> None:
    # Preserve extra columns from baseline when available.
    baseline_rows: dict[str, dict[str, str]] = {}
    if baseline_path and baseline_path.is_file():
        with baseline_path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            baseline_rows = {row["token_354"]: row for row in reader}

    fieldnames = [
        "token_354",
        "token_type",
        "mfa_phones",
        "count",
        "mean_frames",
        "std_frames",
        "p01",
        "p5",
        "p50",
        "p95",
        "p99",
        "min",
        "max",
        "suggested_floor",
        "suggested_cap",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for token, row in sorted(stats.items()):
            base = baseline_rows.get(token, {})
            writer.writerow(
                {
                    "token_354": token,
                    "token_type": base.get("token_type", ""),
                    "mfa_phones": base.get("mfa_phones", ""),
                    "count": row["count"],
                    "mean_frames": f"{row['mean_frames']:.2f}",
                    "std_frames": f"{row['std_frames']:.2f}",
                    "p01": f"{row['p01']:.1f}",
                    "p5": f"{row['p5']:.1f}",
                    "p50": f"{row['p50']:.1f}",
                    "p95": f"{row['p95']:.1f}",
                    "p99": f"{row['p99']:.1f}",
                    "min": f"{row['min']:.1f}",
                    "max": f"{row['max']:.1f}",
                    "suggested_floor": base.get("suggested_floor", ""),
                    "suggested_cap": base.get("suggested_cap", ""),
                }
            )


def write_zh_delta_report(
    path: Path,
    baseline_path: Path,
    cleaned: dict[str, dict[str, float]],
) -> None:
    with baseline_path.open(encoding="utf-8") as f:
        baseline = {row["token_354"]: row for row in csv.DictReader(f, delimiter="\t")}
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "token_354",
        "p95_baseline",
        "p95_cleaned",
        "p95_delta_pct",
        "p99_baseline",
        "p99_cleaned",
        "p99_delta_pct",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for token in sorted(set(baseline) | set(cleaned)):
            base = baseline.get(token, {})
            clean = cleaned.get(token)
            if not base or not clean:
                continue
            p95_b = float(base.get("p95", 0) or 0)
            p95_c = float(clean["p95"])
            p99_b = float(base.get("p99", 0) or 0)
            p99_c = float(clean["p99"])
            writer.writerow(
                {
                    "token_354": token,
                    "p95_baseline": p95_b,
                    "p95_cleaned": p95_c,
                    "p95_delta_pct": ((p95_c - p95_b) / p95_b * 100.0) if p95_b else 0.0,
                    "p99_baseline": p99_b,
                    "p99_cleaned": p99_c,
                    "p99_delta_pct": ((p99_c - p99_b) / p99_b * 100.0) if p99_b else 0.0,
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean MFA duration stats (exclude first/last units)")
    parser.add_argument("--format", choices=("arpa", "zh"), required=True)
    parser.add_argument("--align-dir", type=Path, required=True, help="Directory of MFA TextGrid files")
    parser.add_argument("--output", type=Path, required=True, help="Cleaned stats output path")
    parser.add_argument("--baseline", type=Path, required=True, help="Original stats CSV/TSV for delta report")
    parser.add_argument("--delta-report", type=Path, default=None)
    parser.add_argument("--data-txt", type=Path, default=None, help="Utterance list (zh format only)")
    parser.add_argument("--mapping-tsv", type=Path, default=DEFAULT_MAPPING_TSV)
    args = parser.parse_args()

    delta_path = args.delta_report or args.output.with_name(args.output.stem + "_delta_report.csv")

    if args.format == "arpa":
        cleaned = aggregate_arpa_from_textgrids(args.align_dir)
        write_arpa_csv(args.output, cleaned)
        baseline = load_arpa_csv(args.baseline)
        write_delta_report(delta_path, baseline, cleaned)
    else:
        if args.data_txt is None:
            parser.error("--data-txt is required for zh format")
        cleaned = aggregate_zh_from_textgrids(
            args.align_dir, args.data_txt, args.mapping_tsv, cleaner=""
        )
        write_zh_tsv(args.output, cleaned, args.baseline)
        write_zh_delta_report(delta_path, args.baseline, cleaned)

    print(f"Wrote cleaned stats: {args.output}")
    print(f"Wrote delta report: {delta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
