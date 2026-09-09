"""Strip a training Lightning checkpoint down to inference-only weights.

Training checkpoints include optimizer states, LR schedulers, and loop metadata
that are not needed for inference. This script keeps only the model weights
(`state_dict`), constructor args (`hyper_parameters`), Lightning version
metadata (`pytorch-lightning_version`), and basic metadata so the output can
be loaded with ``LITS.load_from_checkpoint()`` like a full ckpt.

Usage:
    python tools/export_inference_ckpt.py \\
        --input /path/to/training.ckpt \\
        --output /path/to/inference.ckpt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
import torch


def _format_size(num_bytes: int) -> str:
    if num_bytes >= 1 << 30:
        return f"{num_bytes / (1 << 30):.2f} GiB"
    if num_bytes >= 1 << 20:
        return f"{num_bytes / (1 << 20):.1f} MiB"
    if num_bytes >= 1 << 10:
        return f"{num_bytes / (1 << 10):.1f} KiB"
    return f"{num_bytes} B"


def export_inference_ckpt(input_path: Path, output_path: Path) -> dict:
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Expected a dict checkpoint, got {type(checkpoint).__name__}")

    if "state_dict" not in checkpoint:
        raise KeyError(f"No 'state_dict' found in {input_path}")

    if "hyper_parameters" not in checkpoint:
        raise KeyError(
            f"No 'hyper_parameters' found in {input_path}; "
            "cannot rebuild LITS without constructor args"
        )

    inference_ckpt = {
        "state_dict": checkpoint["state_dict"],
        "hyper_parameters": checkpoint["hyper_parameters"],
        "pytorch-lightning_version": checkpoint.get(
            "pytorch-lightning_version", L.__version__
        ),
        "epoch": checkpoint.get("epoch", 0),
        "global_step": checkpoint.get("global_step", 0),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(inference_ckpt, output_path)

    dropped = sorted(k for k in checkpoint if k not in inference_ckpt)
    return {
        "input_path": input_path,
        "output_path": output_path,
        "input_size": input_path.stat().st_size,
        "output_size": output_path.stat().st_size,
        "num_params_tensors": len(inference_ckpt["state_dict"]),
        "dropped_keys": dropped,
        "epoch": inference_ckpt["epoch"],
        "global_step": inference_ckpt["global_step"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export an inference-only LITS checkpoint from a training checkpoint.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the training Lightning checkpoint (.ckpt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the inference-only checkpoint (.ckpt)",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Input checkpoint not found: {args.input}")

    info = export_inference_ckpt(args.input, args.output)

    print(f"Input:  {info['input_path']} ({_format_size(info['input_size'])})")
    print(f"Output: {info['output_path']} ({_format_size(info['output_size'])})")
    print(
        f"Saved {info['num_params_tensors']} tensors "
        f"(epoch={info['epoch']}, global_step={info['global_step']})"
    )
    if info["dropped_keys"]:
        print("Dropped keys:", ", ".join(info["dropped_keys"]))
    else:
        print("Input already contained only inference keys; copied model weights.")


if __name__ == "__main__":
    main()
