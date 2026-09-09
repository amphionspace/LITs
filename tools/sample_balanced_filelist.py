#!/usr/bin/env python3
"""Sample a balanced King + Shahenda filelist from original and synthetic pools.

Input line format: ``<wav_path>|<spk_id>|<arpa_sequence>``

Speakers (by ``spk_id``):
  * King/VC           -> 0
  * Shahenda/shahenda -> 1

Sampling policy
---------------
* 2 speakers, 10k lines each (20k total by default).
* Per speaker: short 20%, medium 40%, long 40%.
* For each bucket, take from the **original** pool first; only use **synthetic**
  lines when the original pool cannot fill the bucket quota.

Length buckets (by slash count in arpa field)
---------------------------------------------
  short  : <= 3 words
  medium : 4–8 words
  long   : > 8 words

Usage
-----
  python sample_balanced_filelist.py \\
      --input /path/to/king_and_shahenda_original.txt \\
      --synth /path/to/king_shahenda_filelist_arpa.txt \\
      --output /path/to/balanced_20k.txt
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Bucket = Literal["short", "medium", "long"]

SPEAKERS: dict[str, int] = {
    "King/VC": 0,
    "Shahenda/shahenda": 1,
}

SPK_ID_TO_NAME: dict[str, str] = {str(spk_id): name for name, spk_id in SPEAKERS.items()}
SPEAKER_ORDER: tuple[str, ...] = tuple(SPEAKERS.keys())
BUCKET_ORDER: tuple[Bucket, ...] = ("short", "medium", "long")
DEFAULT_BUCKET_RATIOS: dict[Bucket, float] = {
    "short": 0.3,
    "medium": 0.3,
    "long": 0.4,
}


@dataclass(frozen=True)
class FilelistLine:
    raw: str
    wav_path: str
    spk_id: str
    text: str

    @classmethod
    def parse(cls, line: str) -> FilelistLine | None:
        line = line.strip()
        if not line or line.startswith("#"):
            return None
        parts = line.split("|", 2)
        if len(parts) != 3:
            return None
        return cls(raw=line, wav_path=parts[0], spk_id=parts[1], text=parts[2])


def word_count(text: str) -> int:
    """Word count = number of ``/`` in text (one slash per word, incl. last word)."""
    return text.count("/")


def classify_bucket(text: str) -> Bucket:
    n = word_count(text)
    if n <= 3:
        return "short"
    if n <= 8:
        return "medium"
    return "long"


def identify_speaker(spk_id: str) -> str | None:
    return SPK_ID_TO_NAME.get(spk_id.strip())


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


def bucketize(lines: list[FilelistLine]) -> dict[Bucket, list[FilelistLine]]:
    buckets: dict[Bucket, list[FilelistLine]] = {b: [] for b in BUCKET_ORDER}
    for line in lines:
        buckets[classify_bucket(line.text)].append(line)
    return buckets


def allocate_proportional_counts(
    target: int, ratios: dict[Bucket, float]
) -> dict[Bucket, int]:
    """Largest-remainder method so bucket counts sum exactly to target."""
    raw = {b: target * ratios[b] for b in BUCKET_ORDER}
    base = {b: int(raw[b]) for b in raw}
    remainder = target - sum(base.values())
    fractions = sorted(
        ((raw[b] - base[b], b) for b in raw),
        reverse=True,
    )
    for _, bucket in fractions[:remainder]:
        base[bucket] += 1
    return base  # type: ignore[return-value]


def group_speakers(lines: list[FilelistLine]) -> tuple[dict[str, list[FilelistLine]], list[FilelistLine]]:
    by_speaker: dict[str, list[FilelistLine]] = defaultdict(list)
    unmatched: list[FilelistLine] = []
    for line in lines:
        spk = identify_speaker(line.spk_id)
        if spk is None:
            unmatched.append(line)
        else:
            by_speaker[spk].append(line)
    return by_speaker, unmatched


def print_bucket_stats(
    name: str,
    buckets: dict[Bucket, list[FilelistLine]],
    *,
    total: int | None = None,
) -> None:
    total = total if total is not None else sum(len(buckets[b]) for b in BUCKET_ORDER)
    print(f"SPK: {name}, total {total}")
    if total == 0:
        return
    for bucket in BUCKET_ORDER:
        cnt = len(buckets[bucket])
        pct = 100.0 * cnt / total
        print(f"        {bucket:6s} proportion: {pct:5.2f}%")


def print_speaker_stats(name: str, lines: list[FilelistLine]) -> None:
    print_bucket_stats(name, bucketize(lines), total=len(lines))


def sample_bucket_with_priority(
    original_buckets: dict[Bucket, list[FilelistLine]],
    synth_buckets: dict[Bucket, list[FilelistLine]],
    target_counts: dict[Bucket, int],
    rng: random.Random,
    *,
    speaker: str,
) -> tuple[list[FilelistLine], dict[Bucket, tuple[int, int]]]:
    """Sample each bucket from original first, then synthetic."""
    selected: list[FilelistLine] = []
    source_counts: dict[Bucket, tuple[int, int]] = {}

    for bucket in BUCKET_ORDER:
        need = target_counts[bucket]
        orig_pool = original_buckets[bucket][:]
        synth_pool = synth_buckets[bucket][:]
        rng.shuffle(orig_pool)
        rng.shuffle(synth_pool)

        from_orig = min(need, len(orig_pool))
        selected.extend(orig_pool[:from_orig])
        still_need = need - from_orig

        from_synth = min(still_need, len(synth_pool))
        selected.extend(synth_pool[:from_synth])
        source_counts[bucket] = (from_orig, from_synth)

        if from_orig + from_synth < need:
            raise ValueError(
                f"{speaker} {bucket}: need {need}, "
                f"original has {len(original_buckets[bucket])}, "
                f"synthetic has {len(synth_buckets[bucket])} "
                f"(took {from_orig} original + {from_synth} synthetic)"
            )

    rng.shuffle(selected)
    return selected, source_counts


def sample_speaker(
    spk_name: str,
    original_lines: list[FilelistLine],
    synth_lines: list[FilelistLine],
    target: int,
    ratios: dict[Bucket, float],
    rng: random.Random,
) -> list[FilelistLine]:
    orig_buckets = bucketize(original_lines)
    synth_buckets = bucketize(synth_lines)
    target_counts = allocate_proportional_counts(target, ratios)

    print(f"\n{spk_name} original pool:")
    print_bucket_stats(f"{spk_name} (original)", orig_buckets)
    print(f"{spk_name} synthetic pool:")
    print_bucket_stats(f"{spk_name} (synthetic)", synth_buckets)
    print(
        f"{spk_name} target={target}, bucket quotas: "
        + ", ".join(f"{b}={target_counts[b]}" for b in BUCKET_ORDER)
    )

    selected, source_counts = sample_bucket_with_priority(
        orig_buckets,
        synth_buckets,
        target_counts,
        rng,
        speaker=spk_name,
    )
    for bucket in BUCKET_ORDER:
        orig_n, synth_n = source_counts[bucket]
        print(
            f"        {bucket:6s} sampled: {orig_n} original + {synth_n} synthetic "
            f"(quota {target_counts[bucket]})"
        )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Original filelist (King spk_id=0, Shahenda spk_id=1)",
    )
    parser.add_argument(
        "--synth",
        type=Path,
        required=True,
        help="Synthetic filelist used when original data is insufficient",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output balanced filelist",
    )
    parser.add_argument(
        "--target-per-spk",
        type=int,
        default=15000,
        help="Target lines per speaker (default: 10000)",
    )
    parser.add_argument(
        "--short-ratio",
        type=float,
        default=DEFAULT_BUCKET_RATIOS["short"],
        help="Target short proportion (default: 0.20)",
    )
    parser.add_argument(
        "--medium-ratio",
        type=float,
        default=DEFAULT_BUCKET_RATIOS["medium"],
        help="Target medium proportion (default: 0.45)",
    )
    parser.add_argument(
        "--long-ratio",
        type=float,
        default=DEFAULT_BUCKET_RATIOS["long"],
        help="Target long proportion (default: 0.35)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    ratios: dict[Bucket, float] = {
        "short": args.short_ratio,
        "medium": args.medium_ratio,
        "long": args.long_ratio,
    }
    ratio_sum = sum(ratios.values())
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(
            f"bucket ratios must sum to 1.0, got {ratio_sum:.6f} "
            f"(short={ratios['short']}, medium={ratios['medium']}, long={ratios['long']})"
        )

    rng = random.Random(args.seed)

    original_lines = read_filelist(args.input)
    synth_lines = read_filelist(args.synth)
    orig_by_speaker, orig_unmatched = group_speakers(original_lines)
    synth_by_speaker, synth_unmatched = group_speakers(synth_lines)

    if orig_unmatched:
        unknown_ids = sorted({line.spk_id for line in orig_unmatched})
        print(
            f"WARNING: {len(orig_unmatched)} original lines skipped "
            f"(unknown spk_id: {', '.join(unknown_ids)})"
        )
    if synth_unmatched:
        unknown_ids = sorted({line.spk_id for line in synth_unmatched})
        print(
            f"WARNING: {len(synth_unmatched)} synthetic lines skipped "
            f"(unknown spk_id: {', '.join(unknown_ids)})"
        )

    output_by_speaker: dict[str, list[FilelistLine]] = {}
    for spk_name in SPEAKER_ORDER:
        orig_pool = orig_by_speaker.get(spk_name, [])
        synth_pool = synth_by_speaker.get(spk_name, [])
        if not orig_pool and not synth_pool:
            raise ValueError(f"no lines found for speaker {spk_name} (spk_id={SPEAKERS[spk_name]})")
        output_by_speaker[spk_name] = sample_speaker(
            spk_name,
            orig_pool,
            synth_pool,
            args.target_per_spk,
            ratios,
            rng,
        )

    all_selected: list[FilelistLine] = []
    for spk_name in SPEAKER_ORDER:
        all_selected.extend(output_by_speaker[spk_name])

    rng.shuffle(all_selected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for line in all_selected:
            f.write(line.raw + "\n")

    print()
    print(f"Wrote {len(all_selected)} lines to {args.output}")
    print(f"Target per speaker: {args.target_per_spk}")
    print(
        "Target bucket mix: "
        + ", ".join(f"{b}={ratios[b]:.0%}" for b in BUCKET_ORDER)
    )
    print()
    for spk_name in SPEAKER_ORDER:
        print_speaker_stats(spk_name, output_by_speaker[spk_name])
        print()


if __name__ == "__main__":
    main()
