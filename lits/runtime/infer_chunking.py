"""Legacy post-TN text chunking helpers kept for compatibility.

The production English-Chinese path no longer calls these text-probing helpers;
it performs one G2P/Text2Id pass and slices the resulting phoneme/ID rows in
``infer_e2e.py``.

Split priority:
  1. Strong punctuation (. ; ! ? … and CJK equivalents)
  2. Weak punctuation (, ： etc.)
  3. Hard word-boundary split when a segment still exceeds the limit

Intended to run on TN-normalized plain text (no embedded newlines).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

# TN output should not contain line breaks; do not split on \n here.
STRONG_SPLIT_RE = re.compile(r"(?<=[.!?…;。！？；])\s*")
WEAK_SPLIT_RE = re.compile(r"(?<=[,，、:：])\s*")


@dataclass(frozen=True)
class ChunkBoundary:
    kind: str  # "strong" | "weak" | "hard"


@dataclass(frozen=True)
class ChunkResult:
    chunks: tuple[str, ...]
    boundaries: tuple[ChunkBoundary, ...]

    @property
    def needed_split(self) -> bool:
        return len(self.chunks) > 1


def _split_nonempty_parts(text: str, pattern: re.Pattern[str]) -> list[str]:
    return [part for part in pattern.split(text) if part and part.strip()]


def chunk_tn_text(
    text: str,
    max_tokens: int,
    count_tokens: Callable[[str], int],
    *,
    min_merge_tokens: int = 32,
) -> ChunkResult:
    """Split *text* so every chunk has at most *max_tokens* (by *count_tokens*)."""
    text = text.strip()
    if not text:
        return ChunkResult(chunks=(), boundaries=())
    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be positive, got {max_tokens}")
    if count_tokens(text) <= max_tokens:
        return ChunkResult(chunks=(text,), boundaries=())

    strong_parts = _split_nonempty_parts(text, STRONG_SPLIT_RE)
    if len(strong_parts) > 1:
        strong_parts = _merge_short_parts(strong_parts, min_merge_tokens, count_tokens)
        if all(count_tokens(part) <= max_tokens for part in strong_parts):
            return ChunkResult(
                chunks=tuple(strong_parts),
                boundaries=tuple(ChunkBoundary("strong") for _ in strong_parts[:-1]),
            )

    final_chunks: list[str] = []
    boundaries: list[ChunkBoundary] = []
    parts_iter = strong_parts if len(strong_parts) > 1 else [text]
    for part in parts_iter:
        if count_tokens(part) <= max_tokens:
            if final_chunks:
                boundaries.append(
                    ChunkBoundary("strong" if len(strong_parts) > 1 else "weak")
                )
            final_chunks.append(part)
            continue

        weak_parts = _split_nonempty_parts(part, WEAK_SPLIT_RE)
        weak_parts = _merge_short_parts(weak_parts, min_merge_tokens, count_tokens)
        if len(weak_parts) > 1 and all(count_tokens(wp) <= max_tokens for wp in weak_parts):
            if final_chunks:
                boundaries.append(
                    ChunkBoundary("strong" if len(strong_parts) > 1 else "weak")
                )
            for wi, wp in enumerate(weak_parts):
                if wi > 0:
                    boundaries.append(ChunkBoundary("weak"))
                final_chunks.append(wp)
            continue

        sub_chunks: list[str] = []
        sub_boundaries: list[ChunkBoundary] = []
        weak_iter = weak_parts if len(weak_parts) > 1 else [part]
        for wi, wp in enumerate(weak_iter):
            if count_tokens(wp) <= max_tokens:
                if sub_chunks:
                    sub_boundaries.append(ChunkBoundary("weak"))
                sub_chunks.append(wp)
                continue
            hard_parts = _hard_split(wp, max_tokens, count_tokens)
            if sub_chunks:
                sub_boundaries.append(ChunkBoundary("weak"))
            sub_chunks.extend(hard_parts)
            sub_boundaries.extend(
                ChunkBoundary("hard") for _ in hard_parts[1:]
            )

        if final_chunks and sub_chunks:
            boundaries.append(
                ChunkBoundary("strong" if len(strong_parts) > 1 else "weak")
            )
        final_chunks.extend(sub_chunks)
        boundaries.extend(sub_boundaries)

    if not final_chunks:
        hard_parts = _hard_split(text, max_tokens, count_tokens)
        return ChunkResult(
            chunks=tuple(hard_parts),
            boundaries=tuple(ChunkBoundary("hard") for _ in hard_parts[:-1]),
        )

    _assert_chunks_within_budget(final_chunks, max_tokens, count_tokens)
    return ChunkResult(chunks=tuple(final_chunks), boundaries=tuple(boundaries))


def _merge_short_parts(
    parts: list[str],
    min_tokens: int,
    count_tokens: Callable[[str], int],
) -> list[str]:
    if not parts:
        return parts
    merged: list[str] = []
    buf = parts[0]
    for nxt in parts[1:]:
        if count_tokens(buf) < min_tokens:
            buf = f"{buf} {nxt}".strip()
        else:
            merged.append(buf)
            buf = nxt
    merged.append(buf)
    return merged


def _hard_split(
    text: str,
    max_tokens: int,
    count_tokens: Callable[[str], int],
) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    buf: list[str] = []
    for word in words:
        candidate = " ".join(buf + [word]).strip()
        if buf and count_tokens(candidate) > max_tokens:
            chunks.append(" ".join(buf))
            buf = [word]
        else:
            buf.append(word)
    if buf:
        chunks.append(" ".join(buf))

    fixed: list[str] = []
    for chunk in chunks:
        if count_tokens(chunk) <= max_tokens:
            fixed.append(chunk)
            continue
        words = chunk.split()
        if len(words) <= 1:
            # Single lexeme longer than the budget: keep it (cannot split safely).
            fixed.append(chunk)
            continue
        mid = len(chunk) // 2
        cut = chunk.rfind(" ", 0, mid)
        if cut <= 0:
            cut = mid
        fixed.extend(_hard_split(chunk[:cut], max_tokens, count_tokens))
        fixed.extend(_hard_split(chunk[cut:], max_tokens, count_tokens))
    return [c.strip() for c in fixed if c.strip()]


def _assert_chunks_within_budget(
    chunks: list[str],
    max_tokens: int,
    count_tokens: Callable[[str], int],
) -> None:
    for chunk in chunks:
        n = count_tokens(chunk)
        if n > max_tokens:
            raise RuntimeError(
                f"chunking bug: segment has {n} tokens (max={max_tokens}): {chunk[:80]!r}"
            )


@dataclass(frozen=True)
class ChunkPlanEntry:
    orig_line: int
    chunk_idx: int
    n_chunks: int
    synth_line: int
    text: str
    n_tokens: int
    had_hard_split: bool
    source_text: str | None = None


def plan_infer_chunks_for_line(
    text: str,
    orig_line: int,
    max_tokens: int,
    count_tokens: Callable[[str], int],
    *,
    min_merge_tokens: int = 32,
) -> list[ChunkPlanEntry]:
    """Expand one TN-normalized utterance into synthesis chunks (local synth_line 1..n)."""
    result = chunk_tn_text(
        text,
        max_tokens,
        count_tokens,
        min_merge_tokens=min_merge_tokens,
    )
    chunks = result.chunks if result.chunks else (text.strip(),)
    n_chunks = len(chunks)
    had_hard = any(b.kind == "hard" for b in result.boundaries)
    plan: list[ChunkPlanEntry] = []
    for chunk_idx, chunk_text in enumerate(chunks, start=1):
        plan.append(
            ChunkPlanEntry(
                orig_line=orig_line,
                chunk_idx=chunk_idx,
                n_chunks=n_chunks,
                synth_line=chunk_idx,
                text=chunk_text,
                n_tokens=count_tokens(chunk_text),
                had_hard_split=had_hard,
            )
        )
    return plan


def plan_infer_chunks(
    lines: list[str],
    max_tokens: int,
    count_tokens: Callable[[str], int],
    *,
    min_merge_tokens: int = 32,
) -> list[ChunkPlanEntry]:
    """Expand TN-normalized utterances into synthesis chunks."""
    plan: list[ChunkPlanEntry] = []
    synth_line = 0
    for orig_line, text in enumerate(lines, start=1):
        line_plan = plan_infer_chunks_for_line(
            text,
            orig_line,
            max_tokens,
            count_tokens,
            min_merge_tokens=min_merge_tokens,
        )
        for entry in line_plan:
            synth_line += 1
            plan.append(
                ChunkPlanEntry(
                    orig_line=entry.orig_line,
                    chunk_idx=entry.chunk_idx,
                    n_chunks=entry.n_chunks,
                    synth_line=synth_line,
                    text=entry.text,
                    n_tokens=entry.n_tokens,
                    had_hard_split=entry.had_hard_split,
                    source_text=entry.source_text,
                )
            )
    return plan
