import numpy as np
import torch

from lits.utils.monotonic_align.core import maximum_path_c, maximum_path_constrained_c


def maximum_path(value, mask):
    """Cython optimised version.
    value: [b, t_x, t_y]
    mask: [b, t_x, t_y]
    """
    value = value * mask
    device = value.device
    dtype = value.dtype
    value = value.data.cpu().numpy().astype(np.float32)
    path = np.zeros_like(value).astype(np.int32)
    mask = mask.data.cpu().numpy()

    t_x_max = mask.sum(1)[:, 0].astype(np.int32)
    t_y_max = mask.sum(2)[:, 0].astype(np.int32)
    maximum_path_c(path, value, t_x_max, t_y_max)
    return torch.from_numpy(path).to(device=device, dtype=dtype)


def _rescale_floors_inplace(floors: np.ndarray, t_xs: np.ndarray, t_ys: np.ndarray) -> np.ndarray:
    """Per-sample feasibility: ensure sum(max(floor, 1)) <= t_y by proportional
    down-scaling (floors stay >= 1). Returns a bool array marking rescaled samples."""
    rescaled = np.zeros(floors.shape[0], dtype=bool)
    for i in range(floors.shape[0]):
        t_x = int(t_xs[i])
        t_y = int(t_ys[i])
        if t_x <= 0 or t_y < t_x:
            # Degenerate sample; the kernel falls back to unconstrained MAS.
            continue
        row = np.maximum(floors[i, :t_x], 1)
        total = int(row.sum())
        if total <= t_y:
            floors[i, :t_x] = row
            continue
        rescaled[i] = True
        row = np.maximum(1, np.floor(row.astype(np.float64) * (t_y / total))).astype(floors.dtype)
        excess = int(row.sum()) - t_y
        while excess > 0:
            j = int(row.argmax())
            if row[j] <= 1:
                break
            dec = min(excess, int(row[j]) - 1)
            row[j] -= dec
            excess -= dec
        floors[i, :t_x] = row
    return rescaled


def _rescale_ceilings_inplace(
    ceilings: np.ndarray,
    t_xs: np.ndarray,
    t_ys: np.ndarray,
) -> np.ndarray:
    """Per-sample feasibility: unbounded tokens (ceiling <= 0) become t_y; if the
    sum of ceilings is still below t_y, proportionally scale bounded ceilings up."""
    rescaled = np.zeros(ceilings.shape[0], dtype=bool)
    for i in range(ceilings.shape[0]):
        t_x = int(t_xs[i])
        t_y = int(t_ys[i])
        if t_x <= 0 or t_y < t_x:
            continue
        row = ceilings[i, :t_x].copy()
        bounded_mask = row > 0
        row[~bounded_mask] = t_y
        total = int(row.sum())
        if total >= t_y:
            ceilings[i, :t_x] = row
            continue
        rescaled[i] = True
        bounded_total = int(row[bounded_mask].sum()) if bounded_mask.any() else 0
        if bounded_total <= 0:
            ceilings[i, :t_x] = row
            continue
        scale = float(t_y) / float(bounded_total)
        row[bounded_mask] = np.minimum(
            t_y,
            np.maximum(1, np.ceil(row[bounded_mask].astype(np.float64) * scale)).astype(ceilings.dtype),
        )
        ceilings[i, :t_x] = row
    return rescaled


def _reconcile_floor_ceiling_inplace(floors: np.ndarray, ceilings: np.ndarray, t_xs: np.ndarray) -> None:
    """If floor > ceiling for a bounded token, raise ceiling to floor."""
    for i in range(floors.shape[0]):
        t_x = int(t_xs[i])
        if t_x <= 0:
            continue
        for j in range(t_x):
            if ceilings[i, j] > 0 and floors[i, j] > ceilings[i, j]:
                ceilings[i, j] = floors[i, j]


def maximum_path_constrained(value, mask, floors, ceilings=None):
    """Monotonic alignment with per-token minimum and maximum durations (frames).

    value: [b, t_x, t_y] log-prior
    mask: [b, t_x, t_y]
    floors: [b, t_x] minimum frames per token; values <= 0 mean unconstrained
        (which is still the implicit min-1 of classic MAS).
    ceilings: optional [b, t_x] maximum frames per token; values <= 0 mean
        unbounded. When None, all tokens are treated as unbounded.

    Returns (path, stats): the constrained path is adopted unconditionally
    (feasibility is guaranteed by proportional floor/ceiling rescaling); the
    constrained-vs-free score gap is returned purely for observability.
    """
    value = value * mask
    device = value.device
    dtype = value.dtype
    value = value.data.cpu().numpy().astype(np.float32)
    path = np.zeros_like(value).astype(np.int32)
    mask = mask.data.cpu().numpy()

    t_x_max = mask.sum(1)[:, 0].astype(np.int32)
    t_y_max = mask.sum(2)[:, 0].astype(np.int32)

    floors_np = np.ascontiguousarray(
        np.rint(floors.detach().cpu().numpy()).astype(np.int32)
    )
    if floors_np.shape[1] < value.shape[1]:
        floors_np = np.pad(floors_np, ((0, 0), (0, value.shape[1] - floors_np.shape[1])))

    if ceilings is None:
        ceilings_np = np.zeros_like(floors_np, dtype=np.int32)
    else:
        ceilings_np = np.ascontiguousarray(
            np.rint(ceilings.detach().cpu().numpy()).astype(np.int32)
        )
        if ceilings_np.shape[1] < value.shape[1]:
            ceilings_np = np.pad(
                ceilings_np, ((0, 0), (0, value.shape[1] - ceilings_np.shape[1]))
            )

    floor_rescaled = _rescale_floors_inplace(floors_np, t_x_max, t_y_max)
    ceiling_rescaled = _rescale_ceilings_inplace(ceilings_np, t_x_max, t_y_max)
    _reconcile_floor_ceiling_inplace(floors_np, ceilings_np, t_x_max)

    scores_constrained = np.zeros(value.shape[0], dtype=np.float32)
    scores_free = np.zeros(value.shape[0], dtype=np.float32)
    maximum_path_constrained_c(
        path, value, floors_np, ceilings_np, t_x_max, t_y_max, scores_constrained, scores_free
    )

    frames = np.maximum(t_y_max.astype(np.float64), 1.0)
    gap_per_frame = (scores_free.astype(np.float64) - scores_constrained.astype(np.float64)) / frames
    feasible = t_y_max >= t_x_max
    gap_valid = gap_per_frame[feasible] if feasible.any() else np.zeros(1)

    bounded_ceiling = ceilings_np > 0
    token_frames = path.sum(axis=2)
    bounded_token_frames = token_frames * bounded_ceiling
    bounded_token_count = bounded_ceiling.sum()
    binding_count = (
        (bounded_token_frames > 0)
        & (bounded_token_frames == ceilings_np)
    ).sum()
    stats = {
        "constrained_score_gap_per_frame_mean": float(gap_valid.mean()),
        "constrained_score_gap_per_frame_max": float(gap_valid.max()),
        "floor_rescaled_sample_ratio": float(floor_rescaled.mean()),
        "ceiling_rescaled_sample_ratio": float(ceiling_rescaled.mean()),
        "ceiling_binding_token_ratio": float(binding_count / max(bounded_token_count, 1)),
        "infeasible_sample_ratio": float((~feasible).mean()),
    }
    return torch.from_numpy(path).to(device=device, dtype=dtype), stats
