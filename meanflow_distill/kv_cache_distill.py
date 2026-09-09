"""Chunked KV-cache ODE helpers for distillation (matches causal_kv inference)."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, List, Sequence, Tuple

import torch
from torch.nn.parallel import DistributedDataParallel


@contextmanager
def _use_unwrapped_estimator(decoder) -> Iterator[None]:
    """DDP wraps the estimator; solve_euler needs forward_streaming on the module."""
    estimator = decoder.estimator
    if isinstance(estimator, DistributedDataParallel):
        wrapped = estimator
        decoder.estimator = wrapped.module
        try:
            yield
        finally:
            decoder.estimator = wrapped
    else:
        yield


def compute_chunk_starts(total_frames: int, chunk_size: int) -> List[int]:
    pad = total_frames % chunk_size
    if total_frames - pad > 0:
        return list(range(0, total_frames - pad, chunk_size))
    return [0]


def configure_decoder_streaming_context(
    model,
    *,
    decoder_left_frames: int,
    static_chunk_size: int | None = None,
) -> None:
    """Align decoder attention/KV limits with streaming inference."""
    decoder = model.decoder
    decoder.decoder_left_frames = decoder_left_frames
    if hasattr(decoder, "estimator"):
        estimator = decoder.estimator
        if hasattr(estimator, "base"):
            estimator = estimator.base
        estimator.decoder_left_frames = decoder_left_frames
        if static_chunk_size is not None:
            estimator.static_chunk_size = static_chunk_size


def apply_euler_step_chunked(
    decoder,
    x: torch.Tensor,
    mu: torch.Tensor,
    mask: torch.Tensor,
    spks: torch.Tensor,
    r: torch.Tensor,
    t_end: torch.Tensor,
    dt: torch.Tensor,
    *,
    chunk_size: int,
    pre_lookahead_len: int = 0,
) -> torch.Tensor:
    """One Euler step over an utterance using the inference KV-cache path."""
    total_len = mu.shape[-1]
    chunk_starts = compute_chunk_starts(total_len, chunk_size)
    r_scalar = r.reshape(-1)[0]
    t_end_scalar = t_end.reshape(-1)[0]
    t_span = torch.stack([r_scalar.reshape(()), t_end_scalar.reshape(())]).to(
        device=x.device, dtype=x.dtype
    )

    decoder._decoder_caches = None
    x_next = x.clone()

    with _use_unwrapped_estimator(decoder):
        for j, start_idx in enumerate(chunk_starts):
            is_last = j == len(chunk_starts) - 1
            if not is_last:
                end_idx = min(start_idx + chunk_size + pre_lookahead_len, total_len)
            else:
                end_idx = total_len

            if not is_last and pre_lookahead_len > 0:
                effective_end = end_idx - pre_lookahead_len
            else:
                effective_end = end_idx

            mu_chunk = mu[:, :, :effective_end]
            mask_chunk = mask[:, :, :effective_end]
            x_chunk = x[:, :, :effective_end]

            out = decoder.solve_euler(
                x_chunk,
                t_span=t_span,
                mu=mu_chunk,
                mask=mask_chunk,
                spks=spks,
                cond=None,
                streaming=True,
                chunk_start=start_idx,
                use_kv_cache=True,
            )
            x_next[:, :, start_idx:effective_end] = out[:, :, start_idx:effective_end]

    return x_next


def euler_step_kv_cache(
    decoder,
    x: torch.Tensor,
    mu: torch.Tensor,
    mask: torch.Tensor,
    spks: torch.Tensor,
    r_val: float,
    t_val: float,
    *,
    chunk_size: int,
    pre_lookahead_len: int = 0,
) -> torch.Tensor:
    """One Euler interval via chunked KV cache (matches causal_kv inference)."""
    r = torch.full((x.shape[0],), r_val, device=x.device, dtype=x.dtype)
    t_end = torch.full((x.shape[0],), t_val, device=x.device, dtype=x.dtype)
    dt = t_val - r_val
    return apply_euler_step_chunked(
        decoder,
        x,
        mu,
        mask,
        spks,
        r,
        t_end,
        dt,
        chunk_size=chunk_size,
        pre_lookahead_len=pre_lookahead_len,
    )


def trajectory_kv_cache(
    model,
    mu: torch.Tensor,
    mask: torch.Tensor,
    spks: torch.Tensor,
    z: torch.Tensor,
    t_grid: Sequence[float],
    *,
    chunk_size: int,
    pre_lookahead_len: int = 0,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Run ODE with chunked KV cache over *t_grid* (matches causal_kv inference)."""
    decoder = model.decoder
    x = z
    states = [x]
    velocities = []

    for i in range(len(t_grid) - 1):
        r_val = float(t_grid[i])
        t_val = float(t_grid[i + 1])
        dt = t_val - r_val
        x_prev = x
        x = euler_step_kv_cache(
            decoder,
            x,
            mu,
            mask,
            spks,
            r_val,
            t_val,
            chunk_size=chunk_size,
            pre_lookahead_len=pre_lookahead_len,
        )
        v = (x - x_prev) / dt
        states.append(x)
        velocities.append(v)

    return states, velocities


def student_trajectory_kv_cache(
    student,
    mu: torch.Tensor,
    mask: torch.Tensor,
    spks: torch.Tensor,
    z: torch.Tensor,
    student_t_grid: Sequence[float],
    *,
    chunk_size: int,
    pre_lookahead_len: int = 0,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Run student ODE with chunked KV cache (matches causal_kv inference)."""
    return trajectory_kv_cache(
        student,
        mu,
        mask,
        spks,
        z,
        student_t_grid,
        chunk_size=chunk_size,
        pre_lookahead_len=pre_lookahead_len,
    )


def teacher_trajectory_kv_cache(
    teacher,
    mu: torch.Tensor,
    mask: torch.Tensor,
    spks: torch.Tensor,
    z: torch.Tensor,
    teacher_steps: int,
    *,
    chunk_size: int,
    pre_lookahead_len: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Teacher uniform-grid trajectory via chunked KV cache (deployment-aligned)."""
    t_grid = [i / teacher_steps for i in range(teacher_steps + 1)]
    states, velocities = trajectory_kv_cache(
        teacher,
        mu,
        mask,
        spks,
        z,
        t_grid,
        chunk_size=chunk_size,
        pre_lookahead_len=pre_lookahead_len,
    )
    return torch.stack(states, dim=0), torch.stack(velocities, dim=0)
