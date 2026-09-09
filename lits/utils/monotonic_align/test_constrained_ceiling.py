"""Toy tests for ceiling-in-DP constrained MAS.

Run from repo root:
    python lits/utils/monotonic_align/test_constrained_ceiling.py
"""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch

_PKG_DIR = Path(__file__).resolve().parent


def _load_maximum_path_constrained():
    """Import wrapper without pulling in lits.utils (and lightning)."""
    sys.path.insert(0, str(_PKG_DIR))
    import core  # noqa: WPS433 — compiled extension in this directory

    for name in ("lits", "lits.utils", "lits.utils.monotonic_align"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    sys.modules["lits.utils.monotonic_align.core"] = core

    init_path = _PKG_DIR / "__init__.py"
    spec = importlib.util.spec_from_file_location("lits.utils.monotonic_align", init_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["lits.utils.monotonic_align"] = mod
    spec.loader.exec_module(mod)
    return mod.maximum_path_constrained


maximum_path_constrained = _load_maximum_path_constrained()


def _toy_value(t_x: int, t_y: int, sil_frames: int) -> torch.Tensor:
    value = torch.full((1, t_x, t_y), -5.0, dtype=torch.float32)
    value[0, 0, :sil_frames] = 5.0
    for tok in range(1, t_x):
        value[0, tok, sil_frames:] = 5.0
    return value


def _mask(t_x: int, t_y: int) -> torch.Tensor:
    return torch.ones((1, t_x, t_y), dtype=torch.float32)


def _durations(path: np.ndarray) -> list[int]:
    return [int(path[0, tok].sum()) for tok in range(path.shape[1])]


def test_sil_absorbs_leading_frames():
  # token 0 = <sil> (unbounded), tokens 1-3 bounded with floor=2 ceiling=5.
    t_x, t_y, sil_frames = 4, 15, 5
    value = _toy_value(t_x, t_y, sil_frames)
    floors = torch.tensor([[0, 2, 2, 2]], dtype=torch.float32)
    ceilings = torch.tensor([[0, 5, 5, 5]], dtype=torch.float32)
    path, _ = maximum_path_constrained(value, _mask(t_x, t_y), floors, ceilings)
    durs = _durations(path.numpy())
    assert durs[0] >= sil_frames, f"expected <sil> to absorb leading frames, got {durs}"
    assert all(2 <= d <= 5 for d in durs[1:]), f"speech tokens out of bounds: {durs}"


def test_zero_ceilings_match_floor_only():
    t_x, t_y, sil_frames = 4, 15, 5
    value = _toy_value(t_x, t_y, sil_frames)
    floors = torch.tensor([[0, 2, 2, 2]], dtype=torch.float32)
    zero_ceil = torch.zeros((1, t_x), dtype=torch.float32)
    path_a, _ = maximum_path_constrained(value, _mask(t_x, t_y), floors, zero_ceil)
    path_b, _ = maximum_path_constrained(value, _mask(t_x, t_y), floors, None)
    np.testing.assert_array_equal(path_a.numpy(), path_b.numpy())


def test_tight_ceiling_still_covers_all_frames():
    t_x, t_y = 3, 10
    value = torch.randn(1, t_x, t_y)
    floors = torch.tensor([[1, 2, 2]], dtype=torch.float32)
    ceilings = torch.tensor([[0, 4, 4]], dtype=torch.float32)
    path, stats = maximum_path_constrained(value, _mask(t_x, t_y), floors, ceilings)
    assert int(path.sum()) == t_y
    assert stats["infeasible_sample_ratio"] == 0.0


if __name__ == "__main__":
    test_sil_absorbs_leading_frames()
    test_zero_ceilings_match_floor_only()
    test_tight_ceiling_still_covers_all_frames()
    print("all tests passed")
