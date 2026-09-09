#!/usr/bin/env python3
"""Find spectral badcases by comparing synthesized audio against GT audio.

This tool is designed for offline test sets where each item has a ground-truth
wav, a synthesized wav, and optionally token/word spans.  The utterance-level
metrics use DTW so the two wavs do not need to have the same length.  If token
spans are supplied, the same DTW path is also used to attribute spectral errors
back to token positions.

Manifest formats:
  CSV/TSV/JSONL with columns:
    utt_id, gt_wav, synth_wav, text, language, speaker,
    gt_spans, synth_spans, prior_mel

  Plain pipe format:
    gt_wav|synth_wav|text

Span file formats:
  CSV/TSV/JSONL with columns:
    index, token, start_frame, end_frame
  or:
    index, token, start_sec, end_sec

The token TSV produced by tools/visualize_token_mel.py is accepted as a synth
span file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np
import soundfile as sf
from scipy.fftpack import dct
from scipy.signal import resample_poly
from tqdm.auto import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_MEL_BASIS: dict[tuple[Any, ...], np.ndarray] = {}


def hz_to_mel_slaney(freqs: np.ndarray) -> np.ndarray:
    freqs = np.asarray(freqs, dtype=np.float64)
    f_sp = 200.0 / 3
    mels = freqs / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0
    log_t = freqs >= min_log_hz
    mels[log_t] = min_log_mel + np.log(freqs[log_t] / min_log_hz) / logstep
    return mels


def mel_to_hz_slaney(mels: np.ndarray) -> np.ndarray:
    mels = np.asarray(mels, dtype=np.float64)
    f_sp = 200.0 / 3
    freqs = f_sp * mels
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0
    log_t = mels >= min_log_mel
    freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
    return freqs


def build_mel_basis(
    *,
    sampling_rate: int,
    n_fft: int,
    num_mels: int,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    """Approximate librosa.filters.mel(..., htk=False, norm='slaney')."""
    mel_min = hz_to_mel_slaney(np.asarray([fmin]))[0]
    mel_max = hz_to_mel_slaney(np.asarray([fmax]))[0]
    mel_points = np.linspace(mel_min, mel_max, num_mels + 2)
    hz_points = mel_to_hz_slaney(mel_points)
    fft_freqs = np.linspace(0.0, float(sampling_rate) / 2, 1 + n_fft // 2)

    fdiff = np.diff(hz_points)
    ramps = hz_points[:, None] - fft_freqs[None, :]
    weights = np.zeros((num_mels, fft_freqs.size), dtype=np.float32)
    for idx in range(num_mels):
        lower = -ramps[idx] / fdiff[idx]
        upper = ramps[idx + 2] / fdiff[idx + 1]
        weights[idx] = np.maximum(0.0, np.minimum(lower, upper))

    enorm = 2.0 / np.maximum(hz_points[2 : num_mels + 2] - hz_points[:num_mels], 1e-8)
    weights *= enorm[:, np.newaxis]
    return weights.astype(np.float32, copy=False)


def get_mel_basis(
    *,
    fmax: float,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    fmin: float,
) -> np.ndarray:
    key = (n_fft, num_mels, sampling_rate, float(fmin), float(fmax))
    if key not in _MEL_BASIS:
        _MEL_BASIS[key] = build_mel_basis(
            sampling_rate=sampling_rate,
            n_fft=n_fft,
            num_mels=num_mels,
            fmin=fmin,
            fmax=fmax,
        )
    return _MEL_BASIS[key]


def padded_hann_window(n_fft: int, win_size: int) -> np.ndarray:
    window = np.hanning(win_size).astype(np.float32)
    if win_size == n_fft:
        return window
    if win_size > n_fft:
        raise ValueError(f"win_size={win_size} cannot exceed n_fft={n_fft}")
    left = (n_fft - win_size) // 2
    right = n_fft - win_size - left
    return np.pad(window, (left, right), mode="constant")


def frame_audio(y: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    if y.size < frame_size:
        y = np.pad(y, (0, frame_size - y.size), mode="constant")
    n_frames = 1 + (y.size - frame_size) // hop_size
    shape = (n_frames, frame_size)
    strides = (y.strides[0] * hop_size, y.strides[0])
    return np.lib.stride_tricks.as_strided(y, shape=shape, strides=strides)


def mel_spectrogram_eval(
    y: np.ndarray,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    mel_basis = get_mel_basis(
        fmax=fmax,
        n_fft=n_fft,
        num_mels=num_mels,
        sampling_rate=sampling_rate,
        fmin=fmin,
    )
    pad = int((n_fft - hop_size) / 2)
    y = np.pad(y.astype(np.float32, copy=False), (pad, pad), mode="reflect")
    frames = frame_audio(y, n_fft, hop_size)
    window = padded_hann_window(n_fft, win_size)
    spec = np.fft.rfft(frames * window[None, :], n=n_fft, axis=1)
    mag = np.sqrt(np.abs(spec) ** 2 + 1e-9).T
    mel = mel_basis @ mag
    return np.log(np.clip(mel, 1e-5, None)).astype(np.float32, copy=False)


@dataclass
class EvalItem:
    utt_id: str
    gt_wav: Path
    synth_wav: Path
    text: str = ""
    language: str = ""
    speaker: str = ""
    gt_spans: Path | None = None
    synth_spans: Path | None = None
    prior_mel: Path | None = None


@dataclass
class Span:
    index: int
    token: str
    start_frame: int
    end_frame: int
    word: str = ""
    category: str = ""

    @property
    def frames(self) -> int:
        return max(0, self.end_frame - self.start_frame)


@dataclass
class DtwResult:
    total_cost: float
    normalized_cost: float
    path_a: np.ndarray
    path_b: np.ndarray

    @property
    def path_length(self) -> int:
        return int(self.path_a.size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--sample_rate", type=int, default=24000)
    parser.add_argument("--hop_length", type=int, default=384)
    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--n_mels", type=int, default=100)
    parser.add_argument("--win_length", type=int, default=1536)
    parser.add_argument("--f_min", type=float, default=0.0)
    parser.add_argument("--f_max", type=float, default=12000.0)
    parser.add_argument("--device", type=str, default=None, help="Ignored; kept for CLI compatibility.")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--dtw_metric", choices=("l1", "l2", "cosine"), default="l1")
    parser.add_argument(
        "--dtw_band",
        type=int,
        default=0,
        help="Sakoe-Chiba band in frames. 0 means full DTW.",
    )
    parser.add_argument(
        "--max_dtw_cells",
        type=int,
        default=25_000_000,
        help="Refuse full DTW above this many cells unless --dtw_band is set.",
    )
    parser.add_argument("--mcep_dim", type=int, default=13)
    parser.add_argument("--no_cache_mels", action="store_true")
    parser.add_argument("--mel_cache_dir", type=Path, default=None)
    parser.add_argument(
        "--calibration_token_csv",
        type=Path,
        default=None,
        help="Optional token_metrics.csv from a good checkpoint for robust z-score baselines.",
    )
    parser.add_argument("--min_group_count", type=int, default=8)
    parser.add_argument("--z_threshold", type=float, default=3.0)
    parser.add_argument("--duration_ok_low", type=float, default=0.75)
    parser.add_argument("--duration_ok_high", type=float, default=1.33)
    parser.add_argument("--duration_short", type=float, default=0.60)
    parser.add_argument("--duration_long", type=float, default=1.80)
    parser.add_argument("--low_iou_threshold", type=float, default=0.25)
    parser.add_argument("--top_k_visualizations", type=int, default=20)
    parser.add_argument("--figure_dpi", type=int, default=150)
    return parser.parse_args()


def safe_stem(value: str) -> str:
    stem = Path(value).stem or value
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stem)
    return safe.strip("._-") or "sample"


def resolve_path(value: str | None, base_dir: Path) -> Path | None:
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path


def first_present(row: dict[str, Any], names: Iterable[str], default: str = "") -> str:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return default


def sniff_delimiter_from_sample(first: str) -> str:
    if "\t" in first:
        return "\t"
    if "," in first:
        return ","
    return "|"


def read_tabular(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        lines = f.readlines()
    if not lines:
        return []

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict[str, str]] = []
        for line_idx, line in enumerate(lines, start=1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if not isinstance(data, dict):
                raise ValueError(f"{path}:{line_idx}: expected a JSON object")
            rows.append({str(k): "" if v is None else str(v) for k, v in data.items()})
        return rows

    sample = lines[0]
    delimiter = sniff_delimiter_from_sample(sample)
    if delimiter == "|" and not any(name in sample for name in ("gt_wav", "synth_wav", "syn_wav")):
        rows = []
        for line_idx, line in enumerate(lines, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 2:
                raise ValueError(f"{path}:{line_idx}: expected gt_wav|synth_wav|text")
            rows.append(
                {
                    "gt_wav": parts[0],
                    "synth_wav": parts[1],
                    "text": parts[2] if len(parts) >= 3 else "",
                }
            )
        return rows

    reader = csv.DictReader(io.StringIO("".join(lines)), delimiter=delimiter)
    return [dict(row) for row in reader]


def read_manifest(path: Path, max_samples: int | None) -> list[EvalItem]:
    base_dir = path.resolve().parent
    items: list[EvalItem] = []
    for idx, row in enumerate(read_tabular(path)):
        gt_wav = resolve_path(first_present(row, ("gt_wav", "ref_wav", "reference_wav")), base_dir)
        synth_wav = resolve_path(
            first_present(row, ("synth_wav", "syn_wav", "generated_wav", "pred_wav")), base_dir
        )
        if gt_wav is None or synth_wav is None:
            raise ValueError(f"{path}: row {idx + 1} must include gt_wav and synth_wav")
        utt_id = first_present(row, ("utt_id", "id", "filepath"), "")
        if not utt_id:
            utt_id = f"{idx:05d}_{safe_stem(gt_wav.name)}"
        item = EvalItem(
            utt_id=utt_id,
            gt_wav=gt_wav,
            synth_wav=synth_wav,
            text=first_present(row, ("text", "normalized_text", "reference_text"), ""),
            language=first_present(row, ("language", "lang"), ""),
            speaker=first_present(row, ("speaker", "spk", "spk_id"), ""),
            gt_spans=resolve_path(first_present(row, ("gt_spans", "gt_span_file", "ref_spans"), ""), base_dir),
            synth_spans=resolve_path(
                first_present(row, ("synth_spans", "syn_spans", "pred_spans", "token_spans"), ""),
                base_dir,
            ),
            prior_mel=resolve_path(first_present(row, ("prior_mel", "mu_y", "mu_y_npy"), ""), base_dir),
        )
        items.append(item)
        if max_samples is not None and len(items) >= max_samples:
            break
    return items


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def mel_cache_key(path: Path, args: argparse.Namespace) -> str:
    stat = path.stat()
    payload = {
        "path": str(path.resolve()),
        "mtime_ns": stat.st_mtime_ns,
        "size": stat.st_size,
        "sample_rate": args.sample_rate,
        "hop_length": args.hop_length,
        "n_fft": args.n_fft,
        "n_mels": args.n_mels,
        "win_length": args.win_length,
        "f_min": args.f_min,
        "f_max": args.f_max,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def load_audio(path: Path, sample_rate: int) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = np.mean(wav, axis=1)
    if sr != sample_rate:
        divisor = math.gcd(int(sr), int(sample_rate))
        wav = resample_poly(wav, sample_rate // divisor, sr // divisor).astype(np.float32, copy=False)
    if wav.size == 0:
        raise ValueError(f"Empty audio: {path}")
    peak = float(np.max(np.abs(wav)))
    if peak > 1.0:
        wav = wav / peak
    return wav.astype(np.float32, copy=False)


def compute_log_mel(path: Path, args: argparse.Namespace) -> np.ndarray:
    wav = load_audio(path, args.sample_rate)
    return mel_spectrogram_eval(
        wav,
        n_fft=args.n_fft,
        num_mels=args.n_mels,
        sampling_rate=args.sample_rate,
        hop_size=args.hop_length,
        win_size=args.win_length,
        fmin=args.f_min,
        fmax=args.f_max,
    )


def load_or_compute_mel(
    path: Path,
    *,
    args: argparse.Namespace,
    cache_dir: Path,
) -> np.ndarray:
    if not args.no_cache_mels:
        cache_path = cache_dir / f"{mel_cache_key(path, args)}.npy"
        if cache_path.is_file():
            return np.load(cache_path)
        mel = compute_log_mel(path, args)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, mel)
        return mel
    return compute_log_mel(path, args)


def normalize_mel_shape(mel: np.ndarray, n_mels: int, source: Path) -> np.ndarray:
    if mel.ndim != 2:
        raise ValueError(f"{source}: expected a 2D mel array, got shape {mel.shape}")
    if mel.shape[0] == n_mels:
        return mel.astype(np.float32, copy=False)
    if mel.shape[1] == n_mels:
        return mel.T.astype(np.float32, copy=False)
    if mel.shape[0] > n_mels:
        return mel[:n_mels].astype(np.float32, copy=False)
    raise ValueError(f"{source}: cannot interpret mel shape {mel.shape} for n_mels={n_mels}")


def load_prior_mel(path: Path | None, args: argparse.Namespace) -> np.ndarray | None:
    if path is None:
        return None
    require_file(path, "prior_mel")
    return normalize_mel_shape(np.load(path), args.n_mels, path)


def row_to_span(row: dict[str, str], sample_rate: int, hop_length: int, row_idx: int) -> Span:
    index_raw = first_present(row, ("index", "idx", "token_index", "pos"), str(row_idx))
    token = first_present(row, ("token", "symbol", "phone", "word"), "")
    word = first_present(row, ("word", "text"), "")
    category = first_present(row, ("category", "type"), "")

    start_frame_raw = first_present(row, ("start_frame", "start", "begin_frame"), "")
    end_frame_raw = first_present(row, ("end_frame", "end", "stop_frame"), "")
    if start_frame_raw and end_frame_raw:
        start_frame = int(round(float(start_frame_raw)))
        end_frame = int(round(float(end_frame_raw)))
    else:
        start_sec_raw = first_present(row, ("start_sec", "begin_sec", "start_time"), "")
        end_sec_raw = first_present(row, ("end_sec", "stop_sec", "end_time"), "")
        if start_sec_raw and end_sec_raw:
            start_frame = int(round(float(start_sec_raw) * sample_rate / hop_length))
            end_frame = int(round(float(end_sec_raw) * sample_rate / hop_length))
        else:
            start_ms_raw = first_present(row, ("start_ms", "begin_ms"), "")
            end_ms_raw = first_present(row, ("end_ms", "stop_ms"), "")
            if not start_ms_raw or not end_ms_raw:
                raise ValueError(f"span row {row_idx + 1}: missing frame/sec/ms boundaries")
            start_frame = int(round(float(start_ms_raw) * sample_rate / hop_length / 1000.0))
            end_frame = int(round(float(end_ms_raw) * sample_rate / hop_length / 1000.0))

    return Span(
        index=int(float(index_raw)),
        token=token,
        start_frame=max(0, start_frame),
        end_frame=max(0, end_frame),
        word=word,
        category=category,
    )


def read_spans(path: Path | None, sample_rate: int, hop_length: int) -> list[Span]:
    if path is None:
        return []
    require_file(path, "span file")
    spans = [row_to_span(row, sample_rate, hop_length, idx) for idx, row in enumerate(read_tabular(path))]
    spans = [span for span in spans if span.end_frame > span.start_frame]
    return sorted(spans, key=lambda span: (span.start_frame, span.end_frame, span.index))


def frame_cost_vector(x: np.ndarray, y: np.ndarray, metric: str) -> np.ndarray:
    if metric == "l1":
        return np.mean(np.abs(y - x[:, None]), axis=0)
    if metric == "l2":
        diff = y - x[:, None]
        return np.sqrt(np.mean(diff * diff, axis=0))
    x_norm = np.linalg.norm(x) + 1e-8
    y_norm = np.linalg.norm(y, axis=0) + 1e-8
    return 1.0 - np.sum(y * x[:, None], axis=0) / (x_norm * y_norm)


def dtw_path(a: np.ndarray, b: np.ndarray, metric: str, band: int, max_cells: int) -> DtwResult:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("DTW inputs must be shaped [features, frames]")
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"DTW feature dims differ: {a.shape[0]} vs {b.shape[0]}")
    n = int(a.shape[1])
    m = int(b.shape[1])
    if n == 0 or m == 0:
        raise ValueError(f"Cannot DTW empty sequence: {n} x {m}")
    if band <= 0 and n * m > max_cells:
        raise ValueError(
            f"Full DTW would use {n * m:,} cells. Set --dtw_band or increase --max_dtw_cells."
        )

    acc = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    back = np.zeros((n, m), dtype=np.uint8)
    acc[0, 0] = 0.0

    for i in range(1, n + 1):
        if band > 0:
            center = int(round((i - 1) * (m - 1) / max(n - 1, 1))) + 1
            j_start = max(1, center - band)
            j_end = min(m, center + band) + 1
        else:
            j_start = 1
            j_end = m + 1
        costs = frame_cost_vector(a[:, i - 1], b[:, j_start - 1 : j_end - 1], metric)
        for offset, j in enumerate(range(j_start, j_end)):
            diag = acc[i - 1, j - 1]
            up = acc[i - 1, j]
            left = acc[i, j - 1]
            move = int(np.argmin((diag, up, left)))
            acc[i, j] = costs[offset] + (diag, up, left)[move]
            back[i - 1, j - 1] = move

    if not np.isfinite(acc[n, m]):
        raise ValueError(f"DTW failed to find a path. Try a wider --dtw_band; current band={band}.")

    path_a: list[int] = []
    path_b: list[int] = []
    i = n
    j = m
    while i > 0 and j > 0:
        path_a.append(i - 1)
        path_b.append(j - 1)
        move = int(back[i - 1, j - 1])
        if move == 0:
            i -= 1
            j -= 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
    path_a.reverse()
    path_b.reverse()
    path_a_arr = np.asarray(path_a, dtype=np.int32)
    path_b_arr = np.asarray(path_b, dtype=np.int32)
    return DtwResult(
        total_cost=float(acc[n, m]),
        normalized_cost=float(acc[n, m] / max(len(path_a), 1)),
        path_a=path_a_arr,
        path_b=path_b_arr,
    )


def path_metric(a: np.ndarray, b: np.ndarray, path_a: np.ndarray, path_b: np.ndarray, metric: str) -> float:
    if path_a.size == 0:
        return float("nan")
    a_path = a[:, path_a]
    b_path = b[:, path_b]
    if metric == "l1":
        return float(np.mean(np.abs(a_path - b_path)))
    if metric == "l2":
        diff = a_path - b_path
        return float(np.mean(np.sqrt(np.mean(diff * diff, axis=0))))
    denom = (np.linalg.norm(a_path, axis=0) * np.linalg.norm(b_path, axis=0)) + 1e-8
    return float(np.mean(1.0 - np.sum(a_path * b_path, axis=0) / denom))


def mfcc_from_logmel(mel: np.ndarray, dim: int) -> np.ndarray:
    coeffs = dct(mel, type=2, axis=0, norm="ortho")
    end = min(coeffs.shape[0], dim + 1)
    return coeffs[1:end]


def mcd_on_path(gt_mel: np.ndarray, synth_mel: np.ndarray, path_a: np.ndarray, path_b: np.ndarray, dim: int) -> float:
    if path_a.size == 0:
        return float("nan")
    gt_mfcc = mfcc_from_logmel(gt_mel, dim)
    synth_mfcc = mfcc_from_logmel(synth_mel, dim)
    diff = gt_mfcc[:, path_a] - synth_mfcc[:, path_b]
    dist = np.sqrt(np.sum(diff * diff, axis=0))
    return float((10.0 / math.log(10.0)) * math.sqrt(2.0) * np.mean(dist))


def energy_mean(mel: np.ndarray) -> float:
    if mel.size == 0:
        return float("nan")
    return float(np.mean(mel))


def clip_span(span: Span, total_frames: int) -> tuple[int, int]:
    start = min(max(0, span.start_frame), total_frames)
    end = min(max(start, span.end_frame), total_frames)
    return start, end


def span_slice(mel: np.ndarray, span: Span) -> np.ndarray:
    start, end = clip_span(span, mel.shape[1])
    return mel[:, start:end]


def span_iou(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return float(inter / union) if union > 0 else float("nan")


def map_gt_span_with_path(span: Span, dtw: DtwResult, gt_mel: np.ndarray, synth_mel: np.ndarray) -> dict[str, Any]:
    start, end = clip_span(span, gt_mel.shape[1])
    mask = (dtw.path_a >= start) & (dtw.path_a < end)
    if not np.any(mask):
        return {
            "global_dtw_cost": float("nan"),
            "mapped_synth_start_frame": "",
            "mapped_synth_end_frame": "",
            "mapped_synth_frames": "",
            "global_coverage_ratio": float("nan"),
        }
    gt_idx = dtw.path_a[mask]
    synth_idx = dtw.path_b[mask]
    mapped_start = int(np.min(synth_idx))
    mapped_end = int(np.max(synth_idx)) + 1
    costs = np.mean(np.abs(gt_mel[:, gt_idx] - synth_mel[:, synth_idx]), axis=0)
    mapped_frames = mapped_end - mapped_start
    return {
        "global_dtw_cost": float(np.mean(costs)),
        "mapped_synth_start_frame": mapped_start,
        "mapped_synth_end_frame": mapped_end,
        "mapped_synth_frames": mapped_frames,
        "global_coverage_ratio": float(mapped_frames / max(end - start, 1)),
    }


def pair_spans(gt_spans: list[Span], synth_spans: list[Span]) -> list[tuple[Span, Span | None]]:
    if not gt_spans:
        return []
    if not synth_spans:
        return [(span, None) for span in gt_spans]
    synth_by_index = {span.index: span for span in synth_spans}
    if len(synth_by_index) == len(synth_spans):
        return [(gt_span, synth_by_index.get(gt_span.index)) for gt_span in gt_spans]
    pairs: list[tuple[Span, Span | None]] = []
    for pos, gt_span in enumerate(gt_spans):
        synth_span = synth_spans[pos] if pos < len(synth_spans) else None
        pairs.append((gt_span, synth_span))
    return pairs


def local_dtw_metrics(
    gt_mel: np.ndarray,
    synth_mel: np.ndarray,
    gt_span: Span,
    synth_span: Span,
    args: argparse.Namespace,
) -> dict[str, float]:
    gt_slice = span_slice(gt_mel, gt_span)
    synth_slice = span_slice(synth_mel, synth_span)
    if gt_slice.shape[1] == 0 or synth_slice.shape[1] == 0:
        return {
            "local_mel_dtw": float("nan"),
            "local_mcd": float("nan"),
            "local_path_length": float("nan"),
            "energy_delta": float("nan"),
            "energy_abs_delta": float("nan"),
        }
    local = dtw_path(
        gt_slice,
        synth_slice,
        metric=args.dtw_metric,
        band=args.dtw_band,
        max_cells=args.max_dtw_cells,
    )
    energy_delta = energy_mean(synth_slice) - energy_mean(gt_slice)
    return {
        "local_mel_dtw": local.normalized_cost,
        "local_mcd": mcd_on_path(gt_slice, synth_slice, local.path_a, local.path_b, args.mcep_dim),
        "local_path_length": float(local.path_length),
        "energy_delta": float(energy_delta),
        "energy_abs_delta": float(abs(energy_delta)),
    }


def token_metrics_for_item(
    item: EvalItem,
    gt_mel: np.ndarray,
    synth_mel: np.ndarray,
    prior_mel: np.ndarray | None,
    gt_spans: list[Span],
    synth_spans: list[Span],
    dtw: DtwResult,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for gt_span, synth_span in pair_spans(gt_spans, synth_spans):
        gt_start, gt_end = clip_span(gt_span, gt_mel.shape[1])
        row: dict[str, Any] = {
            "utt_id": item.utt_id,
            "language": item.language,
            "speaker": item.speaker,
            "token_index": gt_span.index,
            "token": gt_span.token,
            "word": gt_span.word,
            "category": gt_span.category,
            "gt_start_frame": gt_start,
            "gt_end_frame": gt_end,
            "gt_frames": gt_end - gt_start,
            "synth_start_frame": "",
            "synth_end_frame": "",
            "synth_frames": "",
            "duration_ratio": float("nan"),
            "span_iou": float("nan"),
            "prior_mel_dtw": float("nan"),
        }
        row.update(map_gt_span_with_path(gt_span, dtw, gt_mel, synth_mel))

        if synth_span is not None:
            synth_start, synth_end = clip_span(synth_span, synth_mel.shape[1])
            synth_frames = synth_end - synth_start
            row.update(
                {
                    "synth_start_frame": synth_start,
                    "synth_end_frame": synth_end,
                    "synth_frames": synth_frames,
                    "duration_ratio": float(synth_frames / max(gt_end - gt_start, 1)),
                }
            )
            if isinstance(row["mapped_synth_start_frame"], int):
                row["span_iou"] = span_iou(
                    int(row["mapped_synth_start_frame"]),
                    int(row["mapped_synth_end_frame"]),
                    synth_start,
                    synth_end,
                )
            row.update(local_dtw_metrics(gt_mel, synth_mel, gt_span, synth_span, args))

            if prior_mel is not None:
                prior_slice = span_slice(prior_mel, synth_span)
                gt_slice = span_slice(gt_mel, gt_span)
                if prior_slice.shape[1] > 0 and gt_slice.shape[1] > 0:
                    prior_dtw = dtw_path(
                        gt_slice,
                        prior_slice,
                        metric=args.dtw_metric,
                        band=args.dtw_band,
                        max_cells=args.max_dtw_cells,
                    )
                    row["prior_mel_dtw"] = prior_dtw.normalized_cost
        else:
            row.update(
                {
                    "local_mel_dtw": float("nan"),
                    "local_mcd": float("nan"),
                    "local_path_length": float("nan"),
                    "energy_delta": float("nan"),
                    "energy_abs_delta": float("nan"),
                }
            )

        rows.append(row)
    return rows


def numeric(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def duration_bucket(frames: float) -> str:
    if not np.isfinite(frames):
        return "unknown"
    if frames <= 2:
        return "1-2"
    if frames <= 5:
        return "3-5"
    if frames <= 10:
        return "6-10"
    if frames <= 20:
        return "11-20"
    return "21+"


def robust_center_scale(values: list[float]) -> tuple[float, float]:
    arr = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    scale = 1.4826 * mad
    if scale < 1e-8:
        std = float(np.std(arr))
        scale = std if std >= 1e-8 else 1.0
    return median, scale


def load_calibration_rows(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    require_file(path, "calibration_token_csv")
    return [dict(row) for row in read_tabular(path)]


def choose_reference_values(
    row: dict[str, Any],
    reference: list[dict[str, Any]],
    metric: str,
    min_group_count: int,
) -> list[float]:
    token = str(row.get("token", ""))
    category = str(row.get("category", ""))
    bucket = duration_bucket(numeric(row, "gt_frames"))
    candidates = [
        lambda r: str(r.get("token", "")) == token and duration_bucket(numeric(r, "gt_frames")) == bucket,
        lambda r: str(r.get("token", "")) == token,
        lambda r: str(r.get("category", "")) == category and duration_bucket(numeric(r, "gt_frames")) == bucket,
        lambda r: str(r.get("category", "")) == category,
        lambda r: True,
    ]
    for predicate in candidates:
        values = [numeric(ref_row, metric) for ref_row in reference if predicate(ref_row)]
        values = [value for value in values if np.isfinite(value)]
        if len(values) >= min_group_count:
            return values
    return [numeric(ref_row, metric) for ref_row in reference if np.isfinite(numeric(ref_row, metric))]


def add_anomaly_scores(
    rows: list[dict[str, Any]],
    calibration_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    if not rows:
        return
    reference = calibration_rows or rows
    metrics = ("local_mel_dtw", "global_dtw_cost", "prior_mel_dtw", "energy_abs_delta")
    for row in rows:
        z_values: list[float] = []
        for metric in metrics:
            value = numeric(row, metric)
            values = choose_reference_values(row, reference, metric, args.min_group_count)
            center, scale = robust_center_scale(values)
            z_key = f"{metric}_z"
            if np.isfinite(value) and np.isfinite(center) and np.isfinite(scale):
                z = (value - center) / max(scale, 1e-8)
                row[z_key] = float(z)
                z_values.append(float(z))
            else:
                row[z_key] = float("nan")

        acoustic_z = max([z for z in z_values if np.isfinite(z)], default=float("nan"))
        row["acoustic_z"] = acoustic_z
        duration_ratio = numeric(row, "duration_ratio")
        span_iou_value = numeric(row, "span_iou")
        coverage_ratio = numeric(row, "global_coverage_ratio")
        reasons: list[str] = []

        duration_ok = (
            np.isfinite(duration_ratio)
            and args.duration_ok_low <= duration_ratio <= args.duration_ok_high
        )
        if np.isfinite(acoustic_z) and acoustic_z >= args.z_threshold:
            reasons.append("content_mismatch_duration_ok" if duration_ok else "spectral_outlier")
        if np.isfinite(duration_ratio) and duration_ratio < args.duration_short:
            reasons.append("short_duration")
        if np.isfinite(duration_ratio) and duration_ratio > args.duration_long:
            reasons.append("long_duration")
        if np.isfinite(span_iou_value) and span_iou_value < args.low_iou_threshold:
            reasons.append("low_span_iou")
        if np.isfinite(coverage_ratio) and coverage_ratio < args.duration_short:
            reasons.append("low_global_coverage")
        if numeric(row, "prior_mel_dtw_z") >= args.z_threshold:
            reasons.append("prior_mismatch")

        duration_penalty = 0.0
        if np.isfinite(duration_ratio):
            duration_penalty = abs(math.log(max(duration_ratio, 1e-8)))
        row["anomaly_score"] = float(max(acoustic_z if np.isfinite(acoustic_z) else 0.0, duration_penalty))
        row["flag_reason"] = ";".join(dict.fromkeys(reasons))


def summarize_utterance_tokens(utterance_rows: list[dict[str, Any]], token_rows: list[dict[str, Any]]) -> None:
    tokens_by_utt: dict[str, list[dict[str, Any]]] = {}
    for row in token_rows:
        tokens_by_utt.setdefault(str(row["utt_id"]), []).append(row)

    for utt_row in utterance_rows:
        utt_tokens = tokens_by_utt.get(str(utt_row["utt_id"]), [])
        if not utt_tokens:
            utt_row["max_token_anomaly_score"] = ""
            utt_row["num_flagged_tokens"] = 0
            utt_row["top_anomalous_tokens"] = ""
            continue
        sorted_tokens = sorted(utt_tokens, key=lambda row: numeric(row, "anomaly_score"), reverse=True)
        utt_row["max_token_anomaly_score"] = numeric(sorted_tokens[0], "anomaly_score")
        utt_row["num_flagged_tokens"] = sum(1 for row in utt_tokens if str(row.get("flag_reason", "")))
        pieces = []
        for row in sorted_tokens[:5]:
            label = str(row.get("token") or row.get("word") or row.get("token_index"))
            score = numeric(row, "anomaly_score")
            reason = str(row.get("flag_reason", ""))
            pieces.append(f"{label}:{score:.2f}:{reason}" if np.isfinite(score) else label)
        utt_row["top_anomalous_tokens"] = " ".join(pieces)


TOKEN_FIELDNAMES = [
    "utt_id",
    "language",
    "speaker",
    "token_index",
    "token",
    "word",
    "category",
    "gt_start_frame",
    "gt_end_frame",
    "gt_frames",
    "synth_start_frame",
    "synth_end_frame",
    "synth_frames",
    "duration_ratio",
    "local_mel_dtw",
    "local_mcd",
    "local_path_length",
    "global_dtw_cost",
    "mapped_synth_start_frame",
    "mapped_synth_end_frame",
    "mapped_synth_frames",
    "global_coverage_ratio",
    "span_iou",
    "prior_mel_dtw",
    "energy_delta",
    "energy_abs_delta",
    "local_mel_dtw_z",
    "global_dtw_cost_z",
    "prior_mel_dtw_z",
    "energy_abs_delta_z",
    "acoustic_z",
    "anomaly_score",
    "flag_reason",
]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if not rows and fieldnames is None:
        return
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            cleaned = {}
            for key in fieldnames:
                value = row.get(key, "")
                if isinstance(value, float) and not np.isfinite(value):
                    value = ""
                cleaned[key] = value
            writer.writerow(cleaned)


def write_run_summary(path: Path, utterance_rows: list[dict[str, Any]], token_rows: list[dict[str, Any]]) -> None:
    flagged = [row for row in token_rows if str(row.get("flag_reason", ""))]
    utterance_scores = [numeric(row, "utterance_mel_dtw") for row in utterance_rows]
    token_scores = [numeric(row, "anomaly_score") for row in token_rows]
    summary = {
        "num_utterances": len(utterance_rows),
        "num_tokens": len(token_rows),
        "num_flagged_tokens": len(flagged),
        "mean_utterance_mel_dtw": np.nanmean(utterance_scores) if utterance_scores else float("nan"),
        "p95_utterance_mel_dtw": np.nanpercentile(utterance_scores, 95) if utterance_scores else float("nan"),
        "mean_token_anomaly_score": np.nanmean(token_scores) if token_scores else float("nan"),
        "p95_token_anomaly_score": np.nanpercentile(token_scores, 95) if token_scores else float("nan"),
    }
    with path.open("w", encoding="utf-8") as f:
        for key, value in summary.items():
            if isinstance(value, float) and not np.isfinite(value):
                value = ""
            f.write(f"{key}\t{value}\n")


def draw_spans(
    ax: plt.Axes,
    spans: list[Span],
    *,
    hop_length: int,
    sample_rate: int,
    color: str,
    max_labels: int = 60,
) -> None:
    show_labels = len(spans) <= max_labels
    for span in spans:
        start = span.start_frame * hop_length / sample_rate
        end = span.end_frame * hop_length / sample_rate
        ax.axvline(start, color=color, linewidth=0.45, alpha=0.45)
        if show_labels and end > start:
            label = span.token or span.word or str(span.index)
            ax.text(
                0.5 * (start + end),
                0.98,
                label,
                transform=ax.get_xaxis_transform(),
                rotation=90,
                ha="center",
                va="top",
                fontsize=6,
                color=color,
            )
    if spans:
        ax.axvline(spans[-1].end_frame * hop_length / sample_rate, color=color, linewidth=0.45, alpha=0.45)


def highlight_token(ax: plt.Axes, start_frame: int, end_frame: int, args: argparse.Namespace, color: str) -> None:
    start = start_frame * args.hop_length / args.sample_rate
    end = end_frame * args.hop_length / args.sample_rate
    if end > start:
        ax.axvspan(start, end, color=color, alpha=0.18)


def plot_badcase(
    output_path: Path,
    item: EvalItem,
    gt_mel: np.ndarray,
    synth_mel: np.ndarray,
    prior_mel: np.ndarray | None,
    gt_spans: list[Span],
    synth_spans: list[Span],
    token_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    n_rows = 3 if prior_mel is not None else 2
    duration_sec = max(gt_mel.shape[1], synth_mel.shape[1]) * args.hop_length / args.sample_rate
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(max(12, duration_sec * 2.0), 3.2 * n_rows),
        sharex=False,
        constrained_layout=True,
    )
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])

    gt_extent = [0, gt_mel.shape[1] * args.hop_length / args.sample_rate, 0, gt_mel.shape[0]]
    synth_extent = [0, synth_mel.shape[1] * args.hop_length / args.sample_rate, 0, synth_mel.shape[0]]
    axes[0].imshow(gt_mel, origin="lower", aspect="auto", extent=gt_extent, interpolation="nearest")
    axes[0].set_title("GT log-mel")
    axes[0].set_ylabel("mel")
    draw_spans(axes[0], gt_spans, hop_length=args.hop_length, sample_rate=args.sample_rate, color="white")

    axes[1].imshow(synth_mel, origin="lower", aspect="auto", extent=synth_extent, interpolation="nearest")
    axes[1].set_title("Synth log-mel")
    axes[1].set_ylabel("mel")
    draw_spans(axes[1], synth_spans, hop_length=args.hop_length, sample_rate=args.sample_rate, color="white")

    if prior_mel is not None:
        prior_extent = [0, prior_mel.shape[1] * args.hop_length / args.sample_rate, 0, prior_mel.shape[0]]
        axes[2].imshow(prior_mel, origin="lower", aspect="auto", extent=prior_extent, interpolation="nearest")
        axes[2].set_title("Prior / mu_y")
        axes[2].set_ylabel("mel")
        draw_spans(axes[2], synth_spans, hop_length=args.hop_length, sample_rate=args.sample_rate, color="white")

    for row in sorted(token_rows, key=lambda value: numeric(value, "anomaly_score"), reverse=True)[:3]:
        gt_start = int(numeric(row, "gt_start_frame")) if np.isfinite(numeric(row, "gt_start_frame")) else None
        gt_end = int(numeric(row, "gt_end_frame")) if np.isfinite(numeric(row, "gt_end_frame")) else None
        synth_start = numeric(row, "synth_start_frame")
        synth_end = numeric(row, "synth_end_frame")
        if gt_start is not None and gt_end is not None:
            highlight_token(axes[0], gt_start, gt_end, args, "tab:red")
        if np.isfinite(synth_start) and np.isfinite(synth_end):
            highlight_token(axes[1], int(synth_start), int(synth_end), args, "tab:red")
            if prior_mel is not None:
                highlight_token(axes[2], int(synth_start), int(synth_end), args, "tab:red")

    text = item.text[:220] + ("..." if len(item.text) > 220 else "")
    top = " ".join(
        f"{row.get('token') or row.get('word')}:{numeric(row, 'anomaly_score'):.2f}"
        for row in sorted(token_rows, key=lambda value: numeric(value, "anomaly_score"), reverse=True)[:5]
    )
    fig.suptitle(f"{item.utt_id}\n{text}\nTop tokens: {top}", fontsize=9)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.figure_dpi)
    plt.close(fig)


def process_item(
    item: EvalItem,
    args: argparse.Namespace,
    cache_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    require_file(item.gt_wav, "gt_wav")
    require_file(item.synth_wav, "synth_wav")
    gt_mel = load_or_compute_mel(item.gt_wav, args=args, cache_dir=cache_dir)
    synth_mel = load_or_compute_mel(item.synth_wav, args=args, cache_dir=cache_dir)
    prior_mel = load_prior_mel(item.prior_mel, args)
    gt_spans = read_spans(item.gt_spans, args.sample_rate, args.hop_length)
    synth_spans = read_spans(item.synth_spans, args.sample_rate, args.hop_length)

    dtw = dtw_path(gt_mel, synth_mel, metric=args.dtw_metric, band=args.dtw_band, max_cells=args.max_dtw_cells)
    utterance_row: dict[str, Any] = {
        "utt_id": item.utt_id,
        "gt_wav": str(item.gt_wav),
        "synth_wav": str(item.synth_wav),
        "language": item.language,
        "speaker": item.speaker,
        "text": item.text,
        "gt_frames": int(gt_mel.shape[1]),
        "synth_frames": int(synth_mel.shape[1]),
        "duration_ratio": float(synth_mel.shape[1] / max(gt_mel.shape[1], 1)),
        "dtw_path_length": dtw.path_length,
        "utterance_mel_dtw": dtw.normalized_cost,
        "utterance_mel_l1_on_path": path_metric(gt_mel, synth_mel, dtw.path_a, dtw.path_b, "l1"),
        "utterance_mel_l2_on_path": path_metric(gt_mel, synth_mel, dtw.path_a, dtw.path_b, "l2"),
        "utterance_mel_cosine_on_path": path_metric(gt_mel, synth_mel, dtw.path_a, dtw.path_b, "cosine"),
        "utterance_mcd": mcd_on_path(gt_mel, synth_mel, dtw.path_a, dtw.path_b, args.mcep_dim),
        "has_gt_spans": bool(gt_spans),
        "has_synth_spans": bool(synth_spans),
        "has_prior_mel": prior_mel is not None,
    }
    token_rows = token_metrics_for_item(
        item,
        gt_mel,
        synth_mel,
        prior_mel,
        gt_spans,
        synth_spans,
        dtw,
        args,
    )
    payload = {
        "item": item,
        "gt_mel": gt_mel,
        "synth_mel": synth_mel,
        "prior_mel": prior_mel,
        "gt_spans": gt_spans,
        "synth_spans": synth_spans,
    }
    return utterance_row, token_rows, payload


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.mel_cache_dir or (args.output_dir / "mel_cache")

    items = read_manifest(args.manifest, args.max_samples)
    if not items:
        raise ValueError(f"No items found in {args.manifest}")

    utterance_rows: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    viz_payloads: dict[str, dict[str, Any]] = {}
    for item in tqdm(items, desc="spectral_eval"):
        utt_row, item_token_rows, payload = process_item(item, args, cache_dir)
        utterance_rows.append(utt_row)
        token_rows.extend(item_token_rows)
        if args.top_k_visualizations > 0:
            viz_payloads[item.utt_id] = payload

    add_anomaly_scores(token_rows, load_calibration_rows(args.calibration_token_csv), args)
    summarize_utterance_tokens(utterance_rows, token_rows)

    write_csv(args.output_dir / "utterance_metrics.csv", utterance_rows)
    write_csv(args.output_dir / "token_metrics.csv", token_rows, fieldnames=TOKEN_FIELDNAMES)
    flagged_rows = [row for row in token_rows if str(row.get("flag_reason", ""))]
    write_csv(
        args.output_dir / "flagged_tokens.csv",
        sorted(flagged_rows, key=lambda row: numeric(row, "anomaly_score"), reverse=True),
        fieldnames=TOKEN_FIELDNAMES,
    )
    write_run_summary(args.output_dir / "run_summary.txt", utterance_rows, token_rows)

    if args.top_k_visualizations > 0:
        tokens_by_utt: dict[str, list[dict[str, Any]]] = {}
        for row in token_rows:
            tokens_by_utt.setdefault(str(row["utt_id"]), []).append(row)
        ranked = sorted(
            utterance_rows,
            key=lambda row: (
                numeric(row, "max_token_anomaly_score")
                if np.isfinite(numeric(row, "max_token_anomaly_score"))
                else numeric(row, "utterance_mel_dtw")
            ),
            reverse=True,
        )
        for rank, utt_row in enumerate(ranked[: args.top_k_visualizations], start=1):
            utt_id = str(utt_row["utt_id"])
            payload = viz_payloads.get(utt_id)
            if payload is None:
                continue
            plot_badcase(
                args.output_dir / "badcase_figures" / f"{rank:03d}_{safe_stem(utt_id)}.png",
                payload["item"],
                payload["gt_mel"],
                payload["synth_mel"],
                payload["prior_mel"],
                payload["gt_spans"],
                payload["synth_spans"],
                tokens_by_utt.get(utt_id, []),
                args,
            )

    print(f"Wrote reports to {args.output_dir}")


if __name__ == "__main__":
    main()
