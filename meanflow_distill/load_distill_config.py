#!/usr/bin/env python3
"""Load distill YAML config and emit bash variable assignments."""

from __future__ import annotations

import argparse
import os
import shlex
from pathlib import Path
from typing import Any

import yaml

PATH_KEYS = frozenset(
    {
        "teacher_ckpt",
        "train_manifest",
        "val_manifest",
        "resume",
        "output_dir",
    }
)

SCALAR_KEYS = (
    "cuda_visible_devices",
    "teacher_ckpt",
    "train_manifest",
    "val_manifest",
    "cleaner",
    "default_spk",
    "student_steps",
    "student_t_grid",
    "teacher_steps",
    "temperature",
    "batch_size",
    "max_steps",
    "lr",
    "save_every",
    "val_every",
    "val_batches",
    "log_every",
    "precision",
    "output_suffix",
    "output_dir",
    "resume",
    "streaming",
    "mu_streaming",
    "teacher_decoder_streaming",
    "decoder_streaming",
    "kv_cache_distill",
    "distill_chunk_size",
    "decoder_left_frames",
    "pre_lookahead_len",
)


def resolve_path(value: Any, repo_root: Path) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    path = Path(text)
    if not path.is_absolute():
        path = repo_root / path
    return str(path.resolve())


def merged_value(key: str, cfg: dict[str, Any], repo_root: Path) -> str:
    env_key = key.upper()
    if env_key in os.environ:
        raw = os.environ[env_key]
    else:
        raw = cfg.get(key)

    if raw is None:
        return ""

    if key in PATH_KEYS:
        return resolve_path(raw, repo_root)

    return str(raw)


def emit_shell_assignments(cfg: dict[str, Any], repo_root: Path) -> None:
    for key in SCALAR_KEYS:
        value = merged_value(key, cfg, repo_root)
        print(f"{key.upper()}={shlex.quote(value)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}

    emit_shell_assignments(cfg, args.repo_root.resolve())


if __name__ == "__main__":
    main()
