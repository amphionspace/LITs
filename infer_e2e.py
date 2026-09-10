#!/usr/bin/env python3
"""E2E inference: C++ TN/G2P -> JSON Text2Id -> acoustic model -> Vocos.

No Python text cleaners or legacy frontend bookends are loaded. Each normalized
line runs G2P/Text2Id once; optional long-text slicing reuses those numeric IDs.
The model process consumes the prepared token/tone IDs directly.
"""

from __future__ import annotations

import atexit
import argparse
from dataclasses import dataclass
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
from lits.infer_wav_merge import merge_chunks_from_manifest  # noqa: E402
from lits.runtime.infer_chunking import ChunkPlanEntry  # noqa: E402
from lits.runtime.text2id import EncodedPhonemes, Text2Id, write_jsonl  # noqa: E402

DEFAULT_TN_BIN_DIR = REPO_ROOT / "e2e_infer" / "bin"
DEFAULT_TN_DATA_ROOT = (
    REPO_ROOT / "frontend" / "data"
)

SUPPORTED_MODEL_LANGS = (
    "en-zh",
    "en-zh-dict",
    "en-zh-rhyme-body-tone",
    "en-zh-dict-rhyme-body-tone",
)
G2P_PROFILE = "en-zh-g2p"

MODEL_TN_LANGS: dict[str, tuple[str, str]] = {
    lang: ("zh", "en") for lang in SUPPORTED_MODEL_LANGS
}

_SCRIPT_DETECTORS = {
    "zh": re.compile(r"[\u4e00-\u9fff]"),
    "en": re.compile(r"[A-Za-z]"),
}

_STRONG_PHONEME_BOUNDARIES = frozenset({".", ";", "!", "?", "…"})
_WEAK_PHONEME_BOUNDARIES = frozenset({",", ":"})
_TONE_MARK_BOUNDARIES = frozenset({"ˉ", "ˊ", "ˇ", "ˋ", "˙"})


class TtsCliEngine:
    """``tts_cli`` subprocess wrapper."""

    def __init__(self, bin_dir: Path, data_root: Path = DEFAULT_TN_DATA_ROOT):
        self.bin_dir = Path(bin_dir)
        self.data_root = Path(data_root)
        self.cli_path = self.bin_dir / "tts_cli"
        self._processes: dict[str, subprocess.Popen[str]] = {}
        if not self.cli_path.is_file():
            raise FileNotFoundError(f"tts_cli not found: {self.cli_path}")
        if not os.access(self.cli_path, os.X_OK):
            raise PermissionError(f"tts_cli is not executable: {self.cli_path}")
        atexit.register(self.close)

    def _subprocess_env(self) -> dict[str, str]:
        env = os.environ.copy()
        icu_lib = os.environ.get("E2E_ICU_LIB_DIR", "").strip()
        if not icu_lib:
            icu_root = os.environ.get("ICU_ROOT", "").strip()
            if icu_root:
                icu_lib = str(Path(icu_root) / "lib")
        icu_env_sh = self.bin_dir / "icu_env.sh"
        if icu_env_sh.is_file():
            for line in icu_env_sh.read_text(encoding="utf-8").splitlines():
                if line.startswith("export LD_LIBRARY_PATH="):
                    value = line.split("=", 1)[1].strip().strip('"').strip("'")
                    env["LD_LIBRARY_PATH"] = value
                    break
        elif icu_lib and Path(icu_lib).is_dir():
            env["LD_LIBRARY_PATH"] = icu_lib
        return env

    def _profile_dir(self, profile: str) -> Path:
        resource_dir = self.data_root / profile
        if not (resource_dir / "config.json").is_file():
            raise FileNotFoundError(f"TN profile not found: {resource_dir}")
        return resource_dir

    def normalize(self, profile: str, text: str) -> str:
        text = text.strip()
        if not text:
            return ""
        data_dir = self._profile_dir(profile)
        proc = self._processes.get(profile)
        if proc is None or proc.poll() is not None:
            proc = subprocess.Popen(
                [str(self.cli_path), "--data", str(data_dir)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=self._subprocess_env(),
            )
            self._processes[profile] = proc

        assert proc.stdin is not None
        assert proc.stdout is not None
        try:
            proc.stdin.write(text + "\n")
            proc.stdin.flush()
            out = proc.stdout.readline()
        except (BrokenPipeError, OSError) as exc:
            detail = ""
            if proc.stderr is not None:
                detail = proc.stderr.read().strip()
            raise RuntimeError(
                f"tts_cli --data {data_dir.name} failed: {detail or exc}"
            ) from exc
        out = out.strip()
        if not out:
            detail = ""
            if proc.poll() is not None and proc.stderr is not None:
                detail = proc.stderr.read().strip()
            raise RuntimeError(
                f"tts_cli --data {data_dir.name} returned empty output for {text[:120]!r}"
                + (f": {detail}" if detail else "")
            )
        return out

    def close(self) -> None:
        for proc in self._processes.values():
            if proc.poll() is not None:
                continue
            if proc.stdin is not None:
                proc.stdin.close()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.terminate()
                proc.wait(timeout=2)
        self._processes.clear()

    def tn_line(self, text: str, lang: str) -> str:
        return self.normalize(lang, text)

    def g2p_line(self, tn_text: str) -> str:
        return self.normalize(G2P_PROFILE, tn_text)

    def is_g2p_available(self) -> bool:
        return (self.data_root / G2P_PROFILE / "config.json").is_file()


@dataclass(frozen=True)
class FrontendArtifacts:
    """The single frontend result consumed by diagnostics and model inference."""

    rows: list[EncodedPhonemes]
    chunk_plan: list[ChunkPlanEntry]
    synth_input_path: Path
    phonemes_path: Path
    token_ids_path: Path
    chunk_manifest_path: Path | None
    stats: dict[str, int]


def _slice_encoded(
    encoded: EncodedPhonemes,
    tokens: tuple[str, ...],
    start: int,
    end: int,
) -> EncodedPhonemes:
    tone_ids = None if encoded.tone_ids is None else encoded.tone_ids[start:end]
    return EncodedPhonemes(
        phonemes=" ".join(tokens[start:end]),
        token_ids=encoded.token_ids[start:end],
        tone_ids=tone_ids,
    )


def _insert_blank_ids(encoded: EncodedPhonemes, blank_id: int) -> EncodedPhonemes:
    token_ids = [blank_id]
    for token_id in encoded.token_ids:
        token_ids.extend((token_id, blank_id))
    tone_ids = None
    if encoded.tone_ids is not None:
        tone_ids = [0]
        for tone_id in encoded.tone_ids:
            tone_ids.extend((tone_id, 0))
    return EncodedPhonemes(encoded.phonemes, token_ids, tone_ids)


def _preferred_cut(
    tokens: tuple[str, ...],
    start: int,
    limit: int,
) -> tuple[int, str]:
    """Choose a safe phoneme boundary without invoking the frontend again."""

    def candidates_for(boundaries: frozenset[str], *, absorb_separator: bool) -> list[int]:
        positions: list[int] = []
        for index in range(start, limit):
            if tokens[index] not in boundaries:
                continue
            if tokens[index] == "_" and index == start:
                continue
            position = index + 1
            if absorb_separator and position < len(tokens) and tokens[position] == "_":
                absorbed = position + 1
                if absorbed <= limit:
                    position = absorbed
            if position <= limit:
                positions.append(position)
        return positions

    priorities = (
        ("strong", candidates_for(_STRONG_PHONEME_BOUNDARIES, absorb_separator=True)),
        ("weak", candidates_for(_WEAK_PHONEME_BOUNDARIES, absorb_separator=True)),
        ("word", candidates_for(frozenset({"_"}), absorb_separator=False)),
        ("tone", candidates_for(_TONE_MARK_BOUNDARIES, absorb_separator=False)),
    )
    for kind, positions in priorities:
        if positions:
            return max(positions), kind
    preview = " ".join(tokens[start : min(limit + 3, len(tokens))])
    raise ValueError(
        "cannot split encoded phonemes safely within max_infer_tokens; "
        "increase the token budget (no punctuation, word, or complete tone boundary "
        f"before the limit near: {preview!r})"
    )


def split_encoded_phonemes(
    encoded: EncodedPhonemes,
    max_tokens: int,
    *,
    add_blank: bool,
    blank_id: int,
) -> tuple[list[EncodedPhonemes], bool]:
    """Split one already-encoded utterance; never call TN, G2P, or Text2Id."""

    tokens = tuple(encoded.phonemes.split())
    if not tokens or len(tokens) != len(encoded.token_ids):
        raise ValueError(
            "frontend phoneme/token alignment mismatch: "
            f"phonemes={len(tokens)} token_ids={len(encoded.token_ids)}"
        )
    if encoded.tone_ids is not None and len(encoded.tone_ids) != len(tokens):
        raise ValueError(
            "frontend tone/token alignment mismatch: "
            f"tone_ids={len(encoded.tone_ids)} token_ids={len(tokens)}"
        )

    if max_tokens <= 0:
        row = _insert_blank_ids(encoded, blank_id) if add_blank else encoded
        return [row], False

    raw_budget = (max_tokens - 1) // 2 if add_blank else max_tokens
    if raw_budget <= 0:
        raise ValueError(
            f"max_infer_tokens={max_tokens} is too small when add_blank={add_blank}"
        )

    raw_chunks: list[EncodedPhonemes] = []
    start = 0
    while start < len(tokens):
        limit = min(start + raw_budget, len(tokens))
        if limit == len(tokens):
            end = limit
        else:
            end, _ = _preferred_cut(tokens, start, limit)
        if end <= start:
            raise RuntimeError(f"invalid encoded chunk boundary: start={start} end={end}")
        raw_chunks.append(_slice_encoded(encoded, tokens, start, end))
        start = end

    chunks = [
        _insert_blank_ids(chunk, blank_id) if add_blank else chunk
        for chunk in raw_chunks
    ]
    oversized = [len(chunk.token_ids) for chunk in chunks if len(chunk.token_ids) > max_tokens]
    if oversized:
        raise RuntimeError(
            f"encoded chunk exceeds max_infer_tokens={max_tokens}: {oversized}"
        )
    return chunks, False


def _tn_lang_for_line(text: str, tn_lang: str | None = None) -> str:
    if tn_lang is not None:
        if tn_lang not in ("zh", "en"):
            raise ValueError(f"--tn_primary must be zh or en, got {tn_lang!r}")
        return tn_lang
    if _SCRIPT_DETECTORS["zh"].search(text):
        return "zh"
    return "en"


def read_plain_lines(path: Path, limit: int = 0) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if limit > 0:
        lines = lines[:limit]
    return [raw.strip() for raw in lines if raw.strip()]


def write_phonemes_file(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def build_normalized_txt(
    input_path: Path,
    output_path: Path,
    *,
    tts_cli: TtsCliEngine,
    limit: int,
    tn_primary: str | None = None,
) -> dict:
    texts = read_plain_lines(input_path, limit)
    out_rows: list[str] = []
    stats: dict = {"input_lines": len(texts), "written": 0, "tn_lang_counts": {}}

    print(f"[e2e] TN: normalizing {len(texts)} line(s) from {input_path} ...", flush=True)
    for idx, text in enumerate(texts, start=1):
        lang = _tn_lang_for_line(text, tn_primary)
        stats["tn_lang_counts"][lang] = stats["tn_lang_counts"].get(lang, 0) + 1
        out_rows.append(tts_cli.tn_line(text, lang))
        stats["written"] += 1
        if idx == 1 or idx % 50 == 0 or idx == len(texts):
            print(f"[e2e] TN progress: {idx}/{len(texts)}", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(out_rows) + ("\n" if out_rows else ""), encoding="utf-8")
    return stats


def build_frontend_artifacts(
    tn_text_path: Path,
    phonemes_path: Path,
    token_ids_path: Path,
    *,
    tts_cli: TtsCliEngine,
    text2id: Text2Id,
    limit: int = 0,
    add_blank: bool = False,
    prepend_sil: bool = True,
    max_tokens: int = 0,
    chunk_manifest_path: Path | None = None,
) -> FrontendArtifacts:
    """Run G2P/Text2Id once per TN line, then only slice the encoded result."""

    lines = read_plain_lines(tn_text_path, limit)
    if not lines:
        raise ValueError(f"No non-empty lines in {tn_text_path}")

    rows: list[EncodedPhonemes] = []
    plan: list[ChunkPlanEntry] = []
    synth_line = 0
    for orig_line, tn_text in enumerate(lines, start=1):
        # This is the only production call site for G2P and Text2Id.
        encoded = text2id.encode(
            tts_cli.g2p_line(tn_text),
            add_blank=False,
            prepend_sil=prepend_sil,
        )
        chunks, had_hard_split = split_encoded_phonemes(
            encoded,
            max_tokens,
            add_blank=add_blank,
            blank_id=text2id.blank_id,
        )
        n_chunks = len(chunks)
        for chunk_idx, chunk in enumerate(chunks, start=1):
            synth_line += 1
            rows.append(chunk)
            plan.append(
                ChunkPlanEntry(
                    orig_line=orig_line,
                    chunk_idx=chunk_idx,
                    n_chunks=n_chunks,
                    synth_line=synth_line,
                    text=chunk.phonemes,
                    n_tokens=len(chunk.token_ids),
                    had_hard_split=had_hard_split,
                    source_text=tn_text,
                )
            )

    write_phonemes_file(phonemes_path, [row.phonemes for row in rows])
    write_jsonl(token_ids_path, rows)

    manifest_path: Path | None = None
    synth_input_path = tn_text_path
    if max_tokens > 0:
        # The acoustic model consumes numeric IDs. This aligned phoneme file is
        # only the non-empty line carrier required by inference_stream.py.
        synth_input_path = phonemes_path
        manifest_path = chunk_manifest_path or (phonemes_path.parent / "chunk_manifest.tsv")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        header = (
            "synth_line\torig_line\tchunk_idx\tn_chunks\tn_tokens\t"
            "had_hard_split\ttext\torig_text\n"
        )
        manifest_rows = [
            f"{entry.synth_line}\t{entry.orig_line}\t{entry.chunk_idx}\t"
            f"{entry.n_chunks}\t{entry.n_tokens}\t{int(entry.had_hard_split)}\t"
            f"{entry.text}\t{entry.source_text or ''}"
            for entry in plan
        ]
        manifest_path.write_text(
            header + "\n".join(manifest_rows) + "\n",
            encoding="utf-8",
        )

    stats = {
        "input_lines": len(lines),
        "written": len(rows),
        "synth_chunks": len(plan),
        "split_orig_lines": len({e.orig_line for e in plan if e.n_chunks > 1}),
        "hard_split_orig_lines": len({e.orig_line for e in plan if e.had_hard_split}),
        "max_tokens": max_tokens,
    }
    return FrontendArtifacts(
        rows=rows,
        chunk_plan=plan,
        synth_input_path=synth_input_path,
        phonemes_path=phonemes_path,
        token_ids_path=token_ids_path,
        chunk_manifest_path=manifest_path,
        stats=stats,
    )


def _build_inference_cmd(
    args: argparse.Namespace,
    input_txt: Path,
    output_dir: Path,
    output_txt: Path,
    *,
    limit: int = 0,
    chunk_manifest: Path | None = None,
    merged_output_dir: Path | None = None,
    token_ids_jsonl: Path,
    frontend_n_vocab: int,
) -> list[str]:
    cmd = [
        sys.executable,
        str(args.inference_script),
        "--model_lang",
        args.model_lang,
        "--checkpoint",
        args.checkpoint,
        "--input_txt",
        str(input_txt),
        "--spk_id",
        str(args.spk_id),
        "--output_dir",
        str(output_dir),
        "--output_txt",
        str(output_txt),
        "--token_ids_jsonl",
        str(token_ids_jsonl),
        "--frontend_n_vocab",
        str(frontend_n_vocab),
        "--vocos_checkpoint",
        args.vocos_checkpoint,
        "--vocos_root",
        args.vocos_root,
        "--output_sample_rate",
        str(args.output_sample_rate),
        "--n_timesteps",
        str(args.n_timesteps),
        "--length_scale",
        str(args.length_scale),
        "--temperature",
        str(args.temperature),
        "--chunk_size",
        str(args.chunk_size),
        "--mel_cache_len",
        str(args.mel_cache_len),
        "--pre_lookahead_len",
        str(args.pre_lookahead_len),
        "--num_decoding_left_chunks",
        str(args.num_decoding_left_chunks),
    ]
    if args.t_grid is not None:
        cmd.extend(["--t_grid", args.t_grid])
    if args.decoder_left_frames is not None:
        cmd.extend(["--decoder_left_frames", str(args.decoder_left_frames)])
    if args.encoder_num_decoding_left_chunks is not None:
        cmd.extend(["--encoder_num_decoding_left_chunks", str(args.encoder_num_decoding_left_chunks)])
    if args.noise_seed is not None:
        cmd.extend(["--noise_seed", str(args.noise_seed)])
    if args.fp16:
        cmd.append("--fp16")
    if args.infer_duration_patches:
        cmd.append("--infer_duration_patches")
    if args.non_streaming:
        cmd.append("--non_streaming")
    if args.no_waveform_crossfade:
        cmd.append("--no_waveform_crossfade")
    if args.vocoder_mode != "chunked":
        cmd.extend(["--vocoder_mode", args.vocoder_mode])
    if limit > 0:
        cmd.extend(["--limit", str(limit)])
    if chunk_manifest is not None:
        cmd.extend(["--chunk_manifest", str(chunk_manifest)])
    if merged_output_dir is not None:
        cmd.extend(["--merged_output_dir", str(merged_output_dir)])
    if args.chunk_silence_ms is not None:
        cmd.extend(["--chunk_silence_ms", str(args.chunk_silence_ms)])
    if args.keep_chunk_wavs:
        cmd.append("--keep_chunk_wavs")
    return cmd


def run_chunked_synth(
    args: argparse.Namespace,
    output_dir: Path,
    output_txt: Path,
    *,
    artifacts: FrontendArtifacts,
    text2id: Text2Id,
) -> int:
    chunk_wav_dir = output_dir / "_chunk_wavs"
    manifest_path = artifacts.chunk_manifest_path
    if manifest_path is None:
        print("[e2e] ERROR: chunk manifest missing", file=sys.stderr)
        return 1
    chunk_stats = artifacts.stats
    print(
        "[e2e] chunking: "
        f"orig_lines={chunk_stats['input_lines']} "
        f"synth_chunks={chunk_stats['synth_chunks']} "
        f"split_lines={chunk_stats['split_orig_lines']} "
        f"hard_split_lines={chunk_stats['hard_split_orig_lines']} "
        f"max_tokens={chunk_stats['max_tokens']}"
    )
    print(f"[e2e] chunk manifest: {manifest_path}")
    print(f"[e2e] synth input (aligned phonemes): {artifacts.synth_input_path}")

    cmd = _build_inference_cmd(
        args,
        artifacts.synth_input_path,
        chunk_wav_dir,
        output_txt,
        chunk_manifest=manifest_path,
        merged_output_dir=output_dir,
        token_ids_jsonl=artifacts.token_ids_path,
        frontend_n_vocab=text2id.n_vocab,
    )
    print(f"[e2e] Running: {' '.join(cmd)}")
    rc = subprocess.call(cmd)
    if rc != 0:
        return rc

    if not args.keep_chunk_wavs and chunk_wav_dir.is_dir():
        try:
            chunk_wav_dir.rmdir()
        except OSError:
            pass
    print(f"[e2e] merged utterance wav(s) under {output_dir}")
    return rc


def run_dry_run(
    artifacts: FrontendArtifacts,
    output_txt: Path,
    stages_path: Path | None,
) -> dict:
    out_rows: list[str] = []
    stage_rows: list[str] = []
    stats = {"written": len(artifacts.rows), "errors": 0}

    for entry, encoded in zip(artifacts.chunk_plan, artifacts.rows, strict=True):
        out_rows.append(encoded.phonemes)
        stage_rows.append(
            f"{entry.synth_line}\t{entry.orig_line}\t"
            f"{entry.chunk_idx}/{entry.n_chunks}\t{entry.source_text or ''}\t"
            f"{encoded.phonemes}\t{len(encoded.token_ids)}"
        )

    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_txt.write_text("\n".join(out_rows) + ("\n" if out_rows else ""), encoding="utf-8")
    if stages_path is not None:
        header = "wav_id\tline_no\tchunk\ttn_text\tcleaned\tn_tokens\n"
        stages_path.write_text(header + "\n".join(stage_rows) + ("\n" if stage_rows else ""), encoding="utf-8")
    return stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="E2E TTS: C++ TN/G2P -> JSON Text2Id -> LITs -> Vocos",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model_lang", required=True, choices=list(SUPPORTED_MODEL_LANGS))
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--input_txt", default=None)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--output_txt", required=True)
    p.add_argument("--tn_bin_dir", type=Path, default=DEFAULT_TN_BIN_DIR)
    p.add_argument("--tn_data_root", type=Path, default=DEFAULT_TN_DATA_ROOT)
    p.add_argument("--tn_primary", default=None, choices=["zh", "en"])
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--spk_id", type=int, default=None)
    p.add_argument("--skip_tn", action="store_true")
    p.add_argument("--skip_synth", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--dry_run_txt", type=Path, default=None)
    p.add_argument("--dry_run_stages", type=Path, default=None)
    p.add_argument("--keep_normalized", action="store_true")
    p.add_argument("--normalized_txt", type=Path, default=None)
    p.add_argument(
        "--max_infer_tokens",
        type=int,
        default=int(os.environ.get("MAX_INFER_TOKENS", "256")),
    )
    p.add_argument("--chunk_silence_ms", type=int, default=int(os.environ.get("CHUNK_SILENCE_MS", "150")))
    p.add_argument("--keep_chunk_wavs", action="store_true")
    p.add_argument("--merge_chunks_only", action="store_true")
    p.add_argument("--inference_script", type=Path, default=REPO_ROOT / "inference_stream.py")
    p.add_argument(
        "--vocos_checkpoint",
        default=str(REPO_ROOT / "vocos" / "generator.ckpt"),
        help="24 kHz Vocos generator checkpoint",
    )
    p.add_argument("--vocos_root", default=str(REPO_ROOT))
    p.add_argument("--output_sample_rate", type=int, default=24000)
    p.add_argument("--n_timesteps", type=int, default=10)
    p.add_argument("--t_grid", type=str, default=None)
    p.add_argument("--length_scale", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=0.667)
    p.add_argument("--noise_seed", type=int, default=None)
    p.add_argument("--chunk_size", type=int, default=50)
    p.add_argument("--mel_cache_len", type=int, default=8)
    p.add_argument("--pre_lookahead_len", type=int, default=3)
    p.add_argument("--num_decoding_left_chunks", type=int, default=-1)
    p.add_argument("--decoder_left_frames", type=int, default=None)
    p.add_argument("--encoder_num_decoding_left_chunks", type=int, default=None)
    p.add_argument("--add_blank", action="store_true", default=False)
    p.add_argument(
        "--no_prepend_sil",
        action="store_true",
        help="Do not prepend sentence-initial <sil> (legacy checkpoints only)",
    )
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--infer_duration_patches", action="store_true")
    p.add_argument("--non_streaming", action="store_true")
    p.add_argument("--no_waveform_crossfade", action="store_true")
    p.add_argument("--vocoder_mode", type=str, default="chunked", choices=["chunked", "full"])
    p.add_argument("--phonemes_txt", type=Path, default=None)
    p.add_argument(
        "--text2id_config",
        type=Path,
        default=None,
        help="Versioned JSON token inventory (defaults to the selected G2P profile)",
    )
    p.add_argument("--token_ids_jsonl", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_txt = Path(args.output_txt)

    if args.merge_chunks_only:
        return merge_chunks_from_manifest(
            output_dir,
            output_txt,
            sample_rate=args.output_sample_rate,
            silence_ms=args.chunk_silence_ms,
            delete_chunks=not args.keep_chunk_wavs,
        )

    if not args.input_txt:
        print("[e2e] ERROR: --input_txt is required", file=sys.stderr)
        return 1
    input_path = Path(args.input_txt)
    if not input_path.is_file():
        print(f"Input not found: {input_path}", file=sys.stderr)
        return 1

    if args.dry_run:
        args.skip_synth = True

    normalized_path = args.normalized_txt or (output_dir / "normalized.txt")
    phonemes_path = args.phonemes_txt or (output_dir / "phonemes.txt")
    token_ids_path = args.token_ids_jsonl or (output_dir / "token_ids.jsonl")

    if not args.tn_data_root.is_dir():
        print(f"TN data root not found: {args.tn_data_root}", file=sys.stderr)
        return 1

    try:
        tts_cli = TtsCliEngine(args.tn_bin_dir, data_root=args.tn_data_root)
    except (FileNotFoundError, PermissionError) as exc:
        print(f"[e2e] ERROR: {exc}", file=sys.stderr)
        return 1

    if not tts_cli.is_g2p_available():
        print(
            "[e2e] ERROR: data/en-zh-g2p missing. Rebuild: bash install_e2e_tn.sh",
            file=sys.stderr,
        )
        return 1

    text2id_config = args.text2id_config or (
        args.tn_data_root / G2P_PROFILE / "model_tokens.json"
    )
    try:
        text2id = Text2Id(text2id_config)
    except (OSError, ValueError) as exc:
        print(f"[e2e] ERROR: invalid Text2Id config: {exc}", file=sys.stderr)
        return 1

    print(
        "[e2e] frontend: tts_cli TN -> en-zh-g2p -> JSON Text2Id "
        f"({text2id_config})"
    )

    if args.skip_tn:
        tn_text_path = input_path
        print(f"[e2e] SKIP_TN: {tn_text_path}")
    else:
        stats = build_normalized_txt(
            input_path,
            normalized_path,
            tts_cli=tts_cli,
            limit=args.limit,
            tn_primary=args.tn_primary,
        )
        if stats["written"] == 0:
            print("[e2e] ERROR: no lines after TN", file=sys.stderr)
            return 2
        counts = stats.get("tn_lang_counts") or {}
        summary = (
            f"tn_lang={args.tn_primary}"
            if args.tn_primary
            else "tn_lang=auto(" + ",".join(f"{k}={v}" for k, v in sorted(counts.items())) + ")"
        )
        print(f"[e2e] TN done -> {normalized_path} ({summary})")
        tn_text_path = normalized_path

    try:
        artifacts = build_frontend_artifacts(
            tn_text_path,
            phonemes_path,
            token_ids_path,
            tts_cli=tts_cli,
            text2id=text2id,
            limit=args.limit if args.skip_tn else 0,
            add_blank=args.add_blank,
            prepend_sil=not args.no_prepend_sil,
            max_tokens=args.max_infer_tokens,
            chunk_manifest_path=output_dir / "chunk_manifest.tsv",
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[e2e] ERROR: frontend preparation failed: {exc}", file=sys.stderr)
        return 2
    print(
        f"[e2e] G2P/Text2Id single pass -> {phonemes_path}; IDs -> {token_ids_path} "
        f"(input_lines={artifacts.stats['input_lines']} "
        f"model_rows={artifacts.stats['written']})"
    )

    if args.keep_normalized and not args.skip_tn:
        print(f"[e2e] normalized: {normalized_path}")

    if args.dry_run:
        stats = run_dry_run(
            artifacts,
            args.dry_run_txt or output_txt,
            args.dry_run_stages or (output_dir / "dry_run_stages.tsv"),
        )
        print(f"[dry_run] written={stats['written']} errors={stats['errors']}")
        return 2 if stats["errors"] else 0

    if args.skip_synth:
        return 0

    if args.spk_id is None or not args.checkpoint:
        print("[e2e] ERROR: --spk_id and --checkpoint required for synthesis", file=sys.stderr)
        return 1
    if not args.inference_script.is_file():
        print(f"Inference script not found: {args.inference_script}", file=sys.stderr)
        return 1

    if args.max_infer_tokens > 0:
        return run_chunked_synth(
            args,
            output_dir,
            output_txt,
            artifacts=artifacts,
            text2id=text2id,
        )

    cmd = _build_inference_cmd(
        args,
        tn_text_path,
        output_dir,
        output_txt,
        limit=args.limit,
        token_ids_jsonl=artifacts.token_ids_path,
        frontend_n_vocab=text2id.n_vocab,
    )
    print(f"[e2e] Running: {' '.join(cmd)}")
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
