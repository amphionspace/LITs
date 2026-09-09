"""Merge per-chunk inference wavs into per-utterance outputs."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

from lits.runtime.infer_chunking import ChunkPlanEntry


def merge_chunk_wavs(
    plan: list[ChunkPlanEntry],
    chunk_wav_dir: Path,
    output_dir: Path,
    *,
    sample_rate: int,
    silence_ms: int,
    delete_chunks: bool = True,
    chunk_wav_path: Callable[[ChunkPlanEntry], Path] | None = None,
) -> list[str]:
    """Concatenate per-chunk wavs into one wav per original input line."""
    import numpy as np
    import soundfile as sf

    if chunk_wav_path is None:
        chunk_wav_path = lambda entry: chunk_wav_dir / f"{entry.synth_line}.wav"

    output_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[int, list[ChunkPlanEntry]] = {}
    for entry in plan:
        grouped.setdefault(entry.orig_line, []).append(entry)

    meta_lines: list[str] = []
    silence_samples = int(sample_rate * silence_ms / 1000) if silence_ms > 0 else 0
    silence = (
        np.zeros(silence_samples, dtype=np.float32) if silence_samples > 0 else None
    )

    for orig_line in sorted(grouped):
        entries = sorted(grouped[orig_line], key=lambda e: e.chunk_idx)
        audio_parts: list[np.ndarray] = []
        merged_text_parts: list[str] = []
        chunk_paths: list[Path] = []
        for entry in entries:
            chunk_wav = chunk_wav_path(entry)
            if not chunk_wav.is_file():
                raise FileNotFoundError(f"Missing chunk wav: {chunk_wav}")
            audio, sr = sf.read(str(chunk_wav), dtype="float32", always_2d=False)
            if sr != sample_rate:
                raise ValueError(
                    f"Sample rate mismatch in {chunk_wav}: got {sr}, expected {sample_rate}"
                )
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if audio_parts and silence is not None:
                audio_parts.append(silence)
            audio_parts.append(audio)
            merged_text_parts.append(entry.text)
            chunk_paths.append(chunk_wav)

        merged = np.concatenate(audio_parts) if len(audio_parts) > 1 else audio_parts[0]
        out_wav = output_dir / f"{orig_line}.wav"
        sf.write(str(out_wav), merged, sample_rate, subtype="PCM_24")
        source_text = next(
            (entry.source_text for entry in entries if entry.source_text is not None),
            None,
        )
        merged_text = source_text if source_text is not None else " ".join(merged_text_parts)
        meta_lines.append(f"{out_wav}|{merged_text}")

        if delete_chunks:
            for chunk_wav in chunk_paths:
                chunk_wav.unlink(missing_ok=True)

    return meta_lines


def load_chunk_manifest(path: Path) -> list[ChunkPlanEntry]:
    rows = path.read_text(encoding="utf-8").splitlines()
    if len(rows) <= 1:
        return []
    has_source_text = rows[0].split("\t")[-1] == "orig_text"
    plan: list[ChunkPlanEntry] = []
    for row in rows[1:]:
        if not row.strip():
            continue
        parts = row.split("\t", 7 if has_source_text else 6)
        expected_fields = 8 if has_source_text else 7
        if len(parts) != expected_fields:
            raise ValueError(f"Bad manifest row in {path}: {row!r}")
        synth_line, orig_line, chunk_idx, n_chunks, n_tokens, had_hard, text = parts[:7]
        source_text = parts[7] if has_source_text else None
        plan.append(
            ChunkPlanEntry(
                orig_line=int(orig_line),
                chunk_idx=int(chunk_idx),
                n_chunks=int(n_chunks),
                synth_line=int(synth_line),
                text=text,
                n_tokens=int(n_tokens),
                had_hard_split=bool(int(had_hard)),
                source_text=source_text,
            )
        )
    return plan


def build_chunk_plan_index(
    plan: list[ChunkPlanEntry],
) -> tuple[dict[int, ChunkPlanEntry], dict[int, list[ChunkPlanEntry]]]:
    by_synth_line: dict[int, ChunkPlanEntry] = {}
    by_orig_line: dict[int, list[ChunkPlanEntry]] = {}
    for entry in plan:
        by_synth_line[entry.synth_line] = entry
        by_orig_line.setdefault(entry.orig_line, []).append(entry)
    for entries in by_orig_line.values():
        entries.sort(key=lambda e: e.chunk_idx)
    return by_synth_line, by_orig_line


def _detect_per_line_chunk_layout(chunk_wav_root: Path) -> bool:
    if not chunk_wav_root.is_dir():
        return False
    for child in chunk_wav_root.iterdir():
        if child.is_dir() and (child / "1.wav").is_file():
            return True
    return False


def merge_chunks_from_manifest(
    output_dir: Path,
    output_txt: Path,
    *,
    sample_rate: int,
    silence_ms: int,
    delete_chunks: bool = True,
) -> int:
    """Merge existing chunk wavs listed in chunk_manifest.tsv (skip incomplete lines)."""
    manifest_path = output_dir / "chunk_manifest.tsv"
    if not manifest_path.is_file():
        print(f"[e2e] ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    plan = load_chunk_manifest(manifest_path)
    if not plan:
        print(f"[e2e] ERROR: empty manifest: {manifest_path}", file=sys.stderr)
        return 1

    chunk_wav_root = output_dir / "_chunk_wavs"
    per_line = _detect_per_line_chunk_layout(chunk_wav_root)

    grouped: dict[int, list[ChunkPlanEntry]] = {}
    for entry in plan:
        grouped.setdefault(entry.orig_line, []).append(entry)

    meta_lines: list[str] = []
    merged_lines = 0
    skipped_lines = 0

    for orig_line in sorted(grouped):
        entries = sorted(grouped[orig_line], key=lambda e: e.chunk_idx)
        if per_line:
            line_chunk_dir = chunk_wav_root / str(orig_line)

            def _path(e: ChunkPlanEntry, _dir=line_chunk_dir) -> Path:
                return _dir / f"{e.chunk_idx}.wav"
        else:
            line_chunk_dir = chunk_wav_root

            def _path(e: ChunkPlanEntry, _root=chunk_wav_root) -> Path:
                return _root / f"{e.synth_line}.wav"

        missing = [_path(e) for e in entries if not _path(e).is_file()]
        if missing:
            skipped_lines += 1
            print(
                f"[e2e] skip line {orig_line}: missing {len(missing)}/{len(entries)} chunk wav(s)",
                file=sys.stderr,
            )
            continue

        try:
            line_meta = merge_chunk_wavs(
                entries,
                line_chunk_dir,
                output_dir,
                sample_rate=sample_rate,
                silence_ms=silence_ms,
                delete_chunks=delete_chunks,
                chunk_wav_path=_path,
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"[e2e] ERROR: failed to merge line {orig_line}: {exc}", file=sys.stderr)
            return 1

        meta_lines.extend(line_meta)
        merged_lines += 1
        out_wav = output_dir / f"{orig_line}.wav"
        print(f"[e2e] merged line {orig_line} -> {out_wav}")

        if per_line and delete_chunks:
            shutil.rmtree(line_chunk_dir, ignore_errors=True)

    if delete_chunks and chunk_wav_root.is_dir():
        try:
            chunk_wav_root.rmdir()
        except OSError:
            pass

    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_txt.write_text("\n".join(meta_lines) + ("\n" if meta_lines else ""), encoding="utf-8")
    print(
        f"[e2e] merge-only: {merged_lines} utterance(s) merged, "
        f"{skipped_lines} incomplete line(s) skipped"
    )
    if delete_chunks:
        print("[e2e] deleted merged chunk wav(s)")
    print(f"[e2e] output list: {output_txt}")
    return 0 if merged_lines > 0 or skipped_lines > 0 else 1
