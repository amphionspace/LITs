#!/usr/bin/env python3
"""MFA align Chinese corpus and aggregate duration stats for 354 zh tokens.

See README.md in this directory for setup and usage.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PIPELINE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PIPELINE_ROOT))

from lits.text import text_to_sequence
from lits.text.bopomofo_utils import BOPOMOFO_INITIALS
from map import (
    DEFAULT_MAPPING_TSV,
    TokenMapping,
    _is_mfa_nucleus,
    build_all_mappings,
    contextual_mfa_phones,
    load_mapping_tsv,
    mfa_phone_base,
    mfa_phones_compatible,
    validate_354_coverage,
    write_mapping_tsv,
)

DEFAULT_DATA_TXT = Path("/chenmingjie/xingwen/dataset/Dataset/zh/BZNSYP/zh_data.txt")
DEFAULT_WAV_DIR = Path("/chenmingjie/xingwen/dataset/Dataset/zh/BZNSYP/wav_24k")
DEFAULT_MFA_DICT = Path("/chenmingjie/xingwen/tools/MFA/pretrained_models/dictionary/mandarin_china_mfa.dict")
DEFAULT_MFA_ACOUSTIC = Path("/chenmingjie/xingwen/tools/MFA/pretrained_models/acoustic/mandarin_mfa.zip")
DEFAULT_RUN_NAME = "bznsyp_10k"

MEL_HOP = 384
SAMPLE_RATE = 24000
CLEANER = "en_zh_dict_mixed_rhyme_tone_cleaners"
PUNCT = frozenset(",.;:!?…—\"'()[]「」")


@dataclass
class PhoneInterval:
    label: str
    start: float
    end: float

    @property
    def dur_sec(self) -> float:
        return self.end - self.start

    @property
    def dur_frames(self) -> float:
        return self.dur_sec * SAMPLE_RATE / MEL_HOP


def load_utterances(data_txt: Path, wav_dir: Path, limit: int) -> list[tuple[str, Path, str]]:
    rows: list[tuple[str, Path, str]] = []
    with data_txt.open(encoding="utf-8") as f:
        for line in f:
            if len(rows) >= limit:
                break
            parts = line.strip().split("|")
            if len(parts) < 3:
                continue
            wav_path = Path(parts[0])
            if not wav_path.is_file():
                wav_path = wav_dir / wav_path.name
            if not wav_path.is_file():
                continue
            utt_id = wav_path.stem
            text = parts[2].strip()
            rows.append((utt_id, wav_path, text))
    return rows


def prepare_mfa_corpus(corpus_dir: Path, utterances: list[tuple[str, Path, str]]) -> None:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    for utt_id, wav_path, text in utterances:
        target_wav = corpus_dir / f"{utt_id}.wav"
        if not target_wav.exists():
            target_wav.symlink_to(wav_path.resolve())
        spaced = " ".join(ch for ch in text if not ch.isspace())
        (corpus_dir / f"{utt_id}.lab").write_text(spaced, encoding="utf-8")


def run_mfa_align(
    corpus_dir: Path,
    output_dir: Path,
    num_jobs: int,
    *,
    mfa_dict: Path,
    mfa_acoustic: Path,
    single_speaker: bool = False,
    temporary_directory: Path | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "mfa",
        "align",
        str(corpus_dir),
        str(mfa_dict),
        str(mfa_acoustic),
        str(output_dir),
        "--clean",
        "--num_jobs",
        str(num_jobs),
        "--overwrite",
    ]
    if single_speaker:
        cmd.append("--single_speaker")
    if temporary_directory is not None:
        cmd.extend(["--temporary_directory", str(temporary_directory)])
    print("[mfa] running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_textgrid_phones(path: Path) -> list[PhoneInterval]:
    try:
        import textgrid
    except ImportError:
        raise SystemExit("pip install textgrid") from None

    tg = textgrid.TextGrid.fromFile(str(path))
    phones: list[PhoneInterval] = []
    for tier in tg:
        if "phone" in tier.name.lower():
            for interval in tier:
                label = interval.mark.strip()
                if not label or label.lower() in {"sil", "sp", "spn"}:
                    continue
                phones.append(PhoneInterval(label, float(interval.minTime), float(interval.maxTime)))
            break
    return phones


def zh_tokens_from_text(text: str) -> list[str]:
    _, cleaned = text_to_sequence(text, [CLEANER])
    return [t for t in cleaned.split() if t != "_"]


def expand_to_mfa_phones(tokens: list[str], mapping: dict[str, TokenMapping]) -> list[tuple[str, tuple[str, ...]]]:
    seq: list[tuple[str, tuple[str, ...]]] = []
    prev: str | None = None
    for tok in tokens:
        if tok in PUNCT or tok not in mapping:
            prev = tok
            continue
        seq.append((tok, contextual_mfa_phones(tok, prev, mapping)))
        prev = tok
    return seq


# Onset phones that may or may not appear in the alignment depending on the
# dictionary variant the aligner picked (我 = o / ʔ o / w o). Consumed when
# present at the current position, skipped without failing otherwise.
_OPTIONAL_ONSET_BASES = frozenset({"ʔ", "w", "j", "ɥ"})

# Optional coda phones: erhua/er syllables are aligned as "o" or "o ɻ"
# depending on the dict variant, so a trailing ɻ must not fail the token.
_OPTIONAL_CODA_BASES = frozenset({"ɻ"})


def assign_token_durations(
    phones: list[PhoneInterval],
    tokens: list[str],
    mapping: dict[str, TokenMapping],
    *,
    lookahead: int = 6,
) -> list[tuple[str, float]]:
    expected = expand_to_mfa_phones(tokens, mapping)
    if not expected:
        return []
    assignments: list[tuple[str, float]] = []
    pi = 0
    for tok, exp_phones in expected:
        trial_pi = pi
        dur = 0.0
        matched = 0
        failed = False
        blocking_nucleus = -1
        for idx, exp_ph in enumerate(exp_phones):
            # Optional onset (never the nucleus/last phone) or optional coda
            # (only after something already matched): consume if present at the
            # current position, otherwise move on without failing.
            base = mfa_phone_base(exp_ph)
            is_optional_onset = idx < len(exp_phones) - 1 and base in _OPTIONAL_ONSET_BASES
            is_optional_coda = (
                idx == len(exp_phones) - 1 and matched > 0 and base in _OPTIONAL_CODA_BASES
            )
            if is_optional_onset or is_optional_coda:
                if trial_pi < len(phones) and mfa_phones_compatible(phones[trial_pi].label, exp_ph):
                    dur += phones[trial_pi].dur_frames
                    trial_pi += 1
                    matched += 1
                continue
            found = False
            limit = min(len(phones), trial_pi + lookahead)
            for j in range(trial_pi, limit):
                if mfa_phones_compatible(phones[j].label, exp_ph):
                    dur += phones[j].dur_frames
                    trial_pi = j + 1
                    matched += 1
                    found = True
                    break
                if _is_mfa_nucleus(phones[j].label):
                    # Never scan past another syllable's nucleus: the expected
                    # phone is genuinely absent here, and matching a later
                    # occurrence would steal duration from a following token.
                    blocking_nucleus = j
                    break
            if not found:
                failed = True
                break
        if not failed and matched > 0 and dur > 0:
            assignments.append((tok, dur))
            pi = trial_pi
        elif failed:
            # Resync so a single variant mismatch doesn't cascade into failures
            # of all following tokens: commit whatever this token consumed, and
            # if nothing was consumed, skip the mismatched nucleus that blocked
            # the scan (it is this token's actual realization).
            if matched > 0:
                pi = trial_pi
            elif blocking_nucleus >= 0:
                pi = blocking_nucleus + 1
    return assignments


def aggregate_stats(assignments: list[tuple[str, float]]) -> dict[str, dict]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for tok, dur in assignments:
        if dur > 0:
            buckets[tok].append(dur)
    out: dict[str, dict] = {}
    for tok, vals in buckets.items():
        arr = np.array(vals)
        out[tok] = {
            "count": int(len(vals)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "p01": float(np.percentile(arr, 1)),
            "p5": float(np.percentile(arr, 5)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }
    return out


def loose_duration_cap_frames(
    s: dict,
    *,
    p99_scale: float = 1.5,
    p95_margin_frames: float = 6.0,
    min_upper_frames: float = 8.0,
) -> float:
    return max(p99_scale * s["p99"], s["p95"] + p95_margin_frames, min_upper_frames)


def write_stats_tsv(path: Path, stats: dict[str, dict], mapping: dict[str, TokenMapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow([
            "token_354", "token_type", "mfa_phones", "count",
            "mean_frames", "std_frames", "p01", "p5", "p50", "p95", "p99", "min", "max",
            "suggested_floor", "suggested_cap",
        ])
        for tok in sorted(stats, key=lambda t: (mapping.get(t, TokenMapping(t, "", (), "", "", "")).token_type, t)):
            s = stats[tok]
            m = mapping.get(tok)
            typ = m.token_type if m else ""
            mfa = m.mfa_phones_str if m else ""
            floor = max(1.0, s["p01"]) if s["count"] >= 20 else ""
            cap = loose_duration_cap_frames(s) if s["count"] >= 20 else ""
            w.writerow([
                tok, typ, mfa, s["count"],
                f"{s['mean']:.2f}", f"{s['std']:.2f}",
                f"{s['p01']:.1f}", f"{s['p5']:.1f}", f"{s['p50']:.1f}",
                f"{s['p95']:.1f}", f"{s['p99']:.1f}",
                f"{s['min']:.1f}", f"{s['max']:.1f}",
                f"{floor:.1f}" if floor != "" else "",
                f"{cap:.1f}" if cap != "" else "",
            ])


def resolve_out_dir(run_name: str | None, out_dir: Path | None) -> Path:
    if out_dir is not None:
        return out_dir
    name = run_name or DEFAULT_RUN_NAME
    return PIPELINE_ROOT / "runs" / name


def main() -> int:
    p = argparse.ArgumentParser(description="MFA-based 354 zh token duration stats pipeline")
    p.add_argument("--data-txt", type=Path, default=DEFAULT_DATA_TXT,
                   help="Utterance list: wav_path|speaker|text (pipe-separated)")
    p.add_argument("--wav-dir", type=Path, default=DEFAULT_WAV_DIR,
                   help="Fallback directory when wav_path in data-txt is not found")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--run-name", type=str, default=None,
                   help=f"Output under runs/<name>/ (default: {DEFAULT_RUN_NAME})")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Explicit output dir (overrides --run-name)")
    p.add_argument("--mapping-tsv", type=Path, default=DEFAULT_MAPPING_TSV)
    p.add_argument("--mfa-dict", type=Path, default=DEFAULT_MFA_DICT)
    p.add_argument("--mfa-acoustic", type=Path, default=DEFAULT_MFA_ACOUSTIC)
    p.add_argument("--rebuild-mapping", action="store_true")
    p.add_argument("--align", action="store_true", help="Run MFA align (slow)")
    p.add_argument("--stats", action="store_true", help="Aggregate durations from TextGrids")
    p.add_argument("--num-jobs", type=int, default=3)
    p.add_argument(
        "--single-speaker",
        action="store_true",
        help="Pass --single_speaker to MFA so one-speaker corpora can still split across jobs.",
    )
    p.add_argument(
        "--temporary-directory",
        type=Path,
        default=None,
        help="Temporary directory for MFA sqlite/cache files (default: MFA global profile temp dir).",
    )
    args = p.parse_args()

    if not args.align and not args.stats and not args.rebuild_mapping:
        p.error("Specify at least one of: --align, --stats, --rebuild-mapping")

    if args.rebuild_mapping or not args.mapping_tsv.is_file():
        mappings = build_all_mappings(args.mfa_dict)
        report = validate_354_coverage(mappings)
        if report["missing"]:
            print("Missing:", report["missing"], file=sys.stderr)
            return 1
        write_mapping_tsv(args.mapping_tsv, mappings)
        print(f"Wrote mapping: {args.mapping_tsv} ({len(mappings)} rows)")

    out_dir = resolve_out_dir(args.run_name, args.out_dir)
    mapping = load_mapping_tsv(args.mapping_tsv)
    utterances = load_utterances(args.data_txt, args.wav_dir, args.limit)
    print(f"Loaded {len(utterances)} utterances")
    print(f"Output: {out_dir}")

    corpus_dir = out_dir / "mfa_corpus"
    align_dir = out_dir / "mfa_align"

    if args.align:
        prepare_mfa_corpus(corpus_dir, utterances)
        run_mfa_align(
            corpus_dir, align_dir, args.num_jobs,
            mfa_dict=args.mfa_dict,
            mfa_acoustic=args.mfa_acoustic,
            single_speaker=args.single_speaker,
            temporary_directory=args.temporary_directory,
        )

    if args.stats:
        all_assignments: list[tuple[str, float]] = []
        matched_utts = 0
        for utt_id, _, text in utterances:
            tg_path = align_dir / f"{utt_id}.TextGrid"
            if not tg_path.is_file():
                continue
            phones = parse_textgrid_phones(tg_path)
            tokens = zh_tokens_from_text(text)
            assigns = assign_token_durations(phones, tokens, mapping)
            if assigns:
                matched_utts += 1
                all_assignments.extend(assigns)

        stats = aggregate_stats(all_assignments)
        stats_path = out_dir / "zh354_duration_stats.tsv"
        write_stats_tsv(stats_path, stats, mapping)

        init_stats = {k: v for k, v in stats.items() if k in BOPOMOFO_INITIALS}
        rhyme_stats = {k: v for k, v in stats.items() if k not in BOPOMOFO_INITIALS}

        summary = {
            "data_txt": str(args.data_txt),
            "wav_dir": str(args.wav_dir),
            "limit": args.limit,
            "utterances": len(utterances),
            "matched_utterances": matched_utts,
            "token_assignments": len(all_assignments),
            "tokens_with_stats": len(stats),
            "initials_with_stats": len(init_stats),
            "rhyme_tone_with_stats": len(rhyme_stats),
            "mel_hop": MEL_HOP,
            "sample_rate": SAMPLE_RATE,
            "mfa_dict": str(args.mfa_dict),
            "mfa_acoustic": str(args.mfa_acoustic),
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        print(f"Wrote {stats_path}")

        print("\n=== Initial duration (p5/p50/p95 frames) ===")
        for tok in sorted(init_stats, key=lambda t: -init_stats[t]["count"])[:10]:
            s = init_stats[tok]
            print(f"  {tok}: n={s['count']} p5={s['p5']:.1f} p50={s['p50']:.1f} p95={s['p95']:.1f}")

        print("\n=== Rhyme+tone (top 10 by count) ===")
        for tok in sorted(rhyme_stats, key=lambda t: -rhyme_stats[t]["count"])[:10]:
            s = rhyme_stats[tok]
            print(f"  {tok}: n={s['count']} p5={s['p5']:.1f} p50={s['p50']:.1f} p95={s['p95']:.1f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
