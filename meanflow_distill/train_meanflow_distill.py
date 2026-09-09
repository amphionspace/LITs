#!/usr/bin/env python3
"""Standalone mean-flow distillation for LITS CFM_Causal decoder.

This script intentionally lives outside the transsion_lits training package and
does not edit its code. It imports LITS modules through --lits-root.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from streaming_flags import add_streaming_args, resolve_streaming_flags, streaming_config_summary, warn_streaming_config


def add_lits_root(lits_root: Path) -> None:
    lits_root = lits_root.resolve()
    if str(lits_root) not in sys.path:
        sys.path.insert(0, str(lits_root))


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.dim() < pred.dim():
        mask = mask.unsqueeze(1)
    mask = mask.to(dtype=pred.dtype, device=pred.device)
    denom = torch.sum(mask) * pred.shape[1]
    denom = torch.clamp(denom, min=1.0)
    return torch.sum(((pred - target) ** 2) * mask) / denom


def parse_manifest_line(line: str, default_spk: int) -> Tuple[int, str]:
    """Parse a manifest line into (speaker_id, text).

    Supported formats:
        text
        spk_id|text
        wav_path|text                    (wav_path ignored)
        wav_path|spk_id|text             (wav_path ignored)
        wav_path|spk_id|start|end|text   (wav_path, start, end ignored)
    """
    parts = line.rstrip("\n").split("|")
    if len(parts) == 1:
        return default_spk, parts[0]
    if len(parts) == 2:
        if parts[0].strip().isdigit():
            return int(parts[0]), parts[1]
        return default_spk, parts[1]
    if len(parts) >= 3:
        spk = int(parts[1]) if parts[1].strip().isdigit() else default_spk
        text_parts = parts[2:]
        while text_parts and _is_float(text_parts[0]):
            text_parts = text_parts[1:]
        return spk, "|".join(text_parts) if text_parts else parts[-1]
    return default_spk, line.strip()


def _is_float(s: str) -> bool:
    try:
        float(s.strip())
        return True
    except ValueError:
        return False


class TextPromptDataset(Dataset):
    def __init__(
        self,
        manifest: Path,
        cleaner: str,
        default_spk: int,
        add_blank: bool,
        max_text_len: int,
    ):
        from lits.text import text_to_sequence
        from lits.utils.utils import intersperse

        self.rows = []
        self.cleaner = cleaner
        self.add_blank = add_blank
        self.max_text_len = max_text_len
        for line_no, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            spk, text = parse_manifest_line(line, default_spk)
            token_ids, cleaned = text_to_sequence(text, [cleaner])
            if add_blank:
                token_ids = intersperse(token_ids, 0)
            if not token_ids:
                raise ValueError(f"empty token sequence at {manifest}:{line_no}: {text!r}")
            if len(token_ids) > max_text_len:
                token_ids = token_ids[:max_text_len]
            self.rows.append(
                {
                    "spk": spk,
                    "text": text,
                    "cleaned": cleaned,
                    "tokens": torch.tensor(token_ids, dtype=torch.long),
                }
            )
        if not self.rows:
            raise ValueError(f"manifest is empty: {manifest}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        return self.rows[idx]


def collate_prompts(batch: Sequence[dict]) -> dict:
    lengths = torch.tensor([len(item["tokens"]) for item in batch], dtype=torch.long)
    max_len = int(lengths.max().item())
    tokens = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, item in enumerate(batch):
        tokens[i, : len(item["tokens"])] = item["tokens"]
    return {
        "x": tokens,
        "x_lengths": lengths,
        "spks": torch.tensor([item["spk"] for item in batch], dtype=torch.long),
        "text": [item["text"] for item in batch],
        "cleaned": [item["cleaned"] for item in batch],
    }


def cycle(loader: Iterable[dict]):
    while True:
        for batch in loader:
            yield batch


def freeze_all(module: torch.nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def unwrap_ddp(module: torch.nn.Module) -> torch.nn.Module:
    return module.module if isinstance(module, DistributedDataParallel) else module


def set_trainable_decoder(student: torch.nn.Module, train_encoder_in_decoder: bool) -> List[torch.nn.Parameter]:
    freeze_all(student)
    estimator = unwrap_ddp(student.decoder.estimator)
    for param in estimator.parameters():
        param.requires_grad = True
    if train_encoder_in_decoder and hasattr(student.decoder, "encoder"):
        for param in student.decoder.encoder.parameters():
            param.requires_grad = True
    return [param for param in student.parameters() if param.requires_grad]


def prepare_decoder_condition(model, x, x_lengths, spks, streaming: bool):
    hidden = model.get_hidden_mel(x=x, x_lengths=x_lengths, spks=spks)
    mu_y = hidden["mu_y"]
    y_mask = hidden["y_mask"]
    spk_emb = hidden["spks"]
    mu_dec = model.decoder.encoder(mu_y, y_mask, streaming=streaming)
    return mu_dec, y_mask, spk_emb


@torch.no_grad()
def teacher_trajectory(
    teacher,
    mu: torch.Tensor,
    mask: torch.Tensor,
    spks: torch.Tensor,
    z: torch.Tensor,
    teacher_steps: int,
    streaming: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    t_grid = torch.linspace(0.0, 1.0, teacher_steps + 1, device=z.device, dtype=z.dtype)
    x = z
    states = [x]
    velocities = []
    for i in range(teacher_steps):
        t = t_grid[i].expand(z.shape[0])
        v = teacher.decoder.estimator(x, mask, mu, t, spks, None, streaming=streaming)
        dt = t_grid[i + 1] - t_grid[i]
        x = x + dt * v
        states.append(x)
        velocities.append(v)
    return torch.stack(states, dim=0), torch.stack(velocities, dim=0)


def student_trajectory(
    student,
    mu: torch.Tensor,
    mask: torch.Tensor,
    spks: torch.Tensor,
    z: torch.Tensor,
    student_steps: int,
    streaming: bool,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    t_grid = torch.linspace(0.0, 1.0, student_steps + 1, device=z.device, dtype=z.dtype)
    x = z
    states = [x]
    velocities = []
    for i in range(student_steps):
        t = t_grid[i].expand(z.shape[0])
        v = student.decoder.estimator(x, mask, mu, t, spks, None, streaming=streaming)
        dt = t_grid[i + 1] - t_grid[i]
        x = x + dt * v
        states.append(x)
        velocities.append(v)
    return states, velocities


def teacher_boundaries(states: torch.Tensor, teacher_steps: int, student_steps: int) -> List[torch.Tensor]:
    if teacher_steps % student_steps != 0:
        raise ValueError("--teacher-steps must be divisible by --student-steps")
    stride = teacher_steps // student_steps
    return [states[i * stride] for i in range(student_steps + 1)]


def distill_loss(
    student,
    teacher,
    batch: dict,
    args,
    device: torch.device,
) -> Tuple[torch.Tensor, dict]:
    x = batch["x"].to(device)
    x_lengths = batch["x_lengths"].to(device)
    spks = batch["spks"].to(device)

    with torch.no_grad():
        mu_teacher, y_mask, spk_emb = prepare_decoder_condition(
            teacher, x, x_lengths, spks, streaming=args.mu_streaming
        )
        mu_teacher = mu_teacher.detach().clone()
        y_mask = y_mask.detach().clone()
        spk_emb = spk_emb.detach().clone() if spk_emb is not None else None
        z = torch.randn_like(mu_teacher) * args.temperature
        t_states, _ = teacher_trajectory(
            teacher=teacher,
            mu=mu_teacher,
            mask=y_mask,
            spks=spk_emb,
            z=z,
            teacher_steps=args.teacher_steps,
            streaming=args.teacher_decoder_streaming,
        )
        t_states = t_states.detach().clone()
        t_bounds = teacher_boundaries(t_states, args.teacher_steps, args.student_steps)

    if args.train_decoder_encoder:
        raise NotImplementedError(
            "--train-decoder-encoder is intentionally disabled for the first distiller. "
            "The stable target is decoder estimator distillation with the teacher's "
            "encoded decoder condition held fixed."
        )
    mu_student = mu_teacher.detach()
    s_states, _ = student_trajectory(
        student=student,
        mu=mu_student,
        mask=y_mask,
        spks=spk_emb,
        z=z.detach(),
        student_steps=args.student_steps,
        streaming=args.decoder_streaming,
    )

    endpoint = masked_mse(s_states[-1], t_bounds[-1], y_mask)
    traj = torch.stack(
        [masked_mse(s_states[i], t_bounds[i].detach(), y_mask) for i in range(1, args.student_steps + 1)]
    ).mean()

    dt = 1.0 / args.student_steps
    mean_flow_terms = []
    for i in range(args.student_steps):
        t_i = torch.full((x.shape[0],), i / args.student_steps, device=device, dtype=mu_student.dtype)
        pred_v = student.decoder.estimator(
            t_bounds[i].detach(), y_mask, mu_student, t_i, spk_emb, None, streaming=args.decoder_streaming
        )
        target_v = (t_bounds[i + 1] - t_bounds[i]).detach() / dt
        mean_flow_terms.append(masked_mse(pred_v, target_v, y_mask))
    mean_flow = torch.stack(mean_flow_terms).mean()

    total = (
        args.endpoint_weight * endpoint
        + args.trajectory_weight * traj
        + args.mean_flow_weight * mean_flow
    )
    return total, {
        "loss_total": float(total.detach().cpu()),
        "loss_endpoint": float(endpoint.detach().cpu()),
        "loss_trajectory": float(traj.detach().cpu()),
        "loss_mean_flow": float(mean_flow.detach().cpu()),
        "mel_frames": int(y_mask.sum().detach().cpu().item()),
    }


@dataclass
class DistillMetadata:
    teacher_ckpt: str
    student_steps: int
    teacher_steps: int
    cleaner: str
    manifest: str
    target: str
    created_at: float
    global_step: int


def save_student_checkpoint(student, output_dir: Path, args, step: int, metrics: dict) -> Path:
    ckpt_path = output_dir / "checkpoints" / f"student_step_{step:07d}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = DistillMetadata(
        teacher_ckpt=str(args.teacher_ckpt),
        student_steps=args.student_steps,
        teacher_steps=args.teacher_steps,
        cleaner=args.cleaner,
        manifest=str(args.manifest),
        target="lits-vocos-24k-ceshi decoder mean-flow 4-step",
        created_at=time.time(),
        global_step=step,
    )
    was_ddp = isinstance(student.decoder.estimator, DistributedDataParallel)
    ddp_estimator = student.decoder.estimator if was_ddp else None
    if was_ddp:
        student.decoder.estimator = ddp_estimator.module
    payload = {
        "state_dict": student.state_dict(),
        "metadata": asdict(metadata),
        "metrics": metrics,
    }
    if was_ddp:
        student.decoder.estimator = ddp_estimator
    torch.save(payload, ckpt_path)
    return ckpt_path


def setup_distributed(args) -> Tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        dist.init_process_group(backend=args.dist_backend)
    return distributed, rank, local_rank, world_size


def cleanup_distributed(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def reduce_metrics(metrics: dict, device: torch.device, world_size: int) -> dict:
    if world_size == 1:
        return metrics
    keys = sorted(key for key, value in metrics.items() if isinstance(value, (int, float)))
    values = torch.tensor([float(metrics[key]) for key in keys], device=device)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values = values / world_size
    reduced = dict(metrics)
    for key, value in zip(keys, values.detach().cpu().tolist()):
        reduced[key] = value
    return reduced


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lits-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--teacher-ckpt", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cleaner", default="pinyin_direct_mixed_rhyme_body_tone_cleaners")
    parser.add_argument("--default-spk", type=int, default=0)
    parser.add_argument("--add-blank", action="store_true")
    parser.add_argument("--max-text-len", type=int, default=256)
    parser.add_argument("--student-steps", type=int, default=4)
    parser.add_argument("--teacher-steps", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.667)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["fp32", "amp"], default="amp")
    add_streaming_args(parser)
    parser.add_argument("--train-decoder-encoder", action="store_true")
    parser.add_argument("--endpoint-weight", type=float, default=1.0)
    parser.add_argument("--trajectory-weight", type=float, default=0.5)
    parser.add_argument("--mean-flow-weight", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--dist-backend", choices=["gloo", "nccl"], default="gloo")
    return parser


def jsonable_args(args) -> dict:
    out = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def main() -> None:
    args = build_arg_parser().parse_args()
    resolve_streaming_flags(args)
    warn_streaming_config(args)
    add_lits_root(args.lits_root)

    from lits.models.lits import LITS

    distributed, rank, local_rank, world_size = setup_distributed(args)
    try:
        if args.teacher_steps % args.student_steps != 0:
            raise ValueError("--teacher-steps must be divisible by --student-steps")

        random.seed(args.seed + rank)
        torch.manual_seed(args.seed + rank)
        torch.set_float32_matmul_precision("high")
        if torch.cuda.is_available() and args.device.startswith("cuda"):
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")

        if rank == 0:
            args.output_dir.mkdir(parents=True, exist_ok=True)
        if distributed:
            dist.barrier()

        dataset = TextPromptDataset(
            manifest=args.manifest,
            cleaner=args.cleaner,
            default_spk=args.default_spk,
            add_blank=args.add_blank,
            max_text_len=args.max_text_len,
        )
        sampler = (
            DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=args.seed,
                drop_last=False,
            )
            if distributed
            else None
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=args.num_workers,
            collate_fn=collate_prompts,
            drop_last=False,
        )
        batches = cycle(loader)

        teacher = LITS.load_from_checkpoint(args.teacher_ckpt, map_location=device).to(device).eval()
        student = LITS.load_from_checkpoint(args.teacher_ckpt, map_location=device).to(device).train()
        freeze_all(teacher)
        trainable = set_trainable_decoder(student, train_encoder_in_decoder=args.train_decoder_encoder)
        if distributed:
            student.decoder.estimator = DistributedDataParallel(
                student.decoder.estimator,
                device_ids=[local_rank] if device.type == "cuda" else None,
                output_device=local_rank if device.type == "cuda" else None,
                find_unused_parameters=False,
            )
            trainable = [param for param in student.decoder.estimator.parameters() if param.requires_grad]

        optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
        scaler = torch.cuda.amp.GradScaler(enabled=(args.precision == "amp" and device.type == "cuda"))
        writer = SummaryWriter(str(args.output_dir / "tensorboard")) if rank == 0 else None

        metadata = {
            "teacher_ckpt": str(args.teacher_ckpt),
            "manifest": str(args.manifest),
            "dataset_size": len(dataset),
            "student_steps": args.student_steps,
            "teacher_steps": args.teacher_steps,
            "cleaner": args.cleaner,
            "trainable_params": sum(p.numel() for p in trainable),
            "total_params": sum(p.numel() for p in student.parameters()),
            "distributed": distributed,
            "world_size": world_size,
            "dist_backend": args.dist_backend if distributed else None,
            "args": jsonable_args(args),
        }
        if rank == 0:
            (args.output_dir / "metadata.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)

        last_metrics = {}
        for step in range(1, args.max_steps + 1):
            if sampler is not None and (step - 1) % max(1, len(loader)) == 0:
                sampler.set_epoch((step - 1) // max(1, len(loader)))
            batch = next(batches)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(args.precision == "amp" and device.type == "cuda")):
                loss, metrics = distill_loss(student, teacher, batch, args, device)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            metrics["grad_norm"] = float(grad_norm.detach().cpu())
            metrics = reduce_metrics(metrics, device, world_size)
            last_metrics = metrics
            if rank == 0:
                for key, value in metrics.items():
                    writer.add_scalar(key, value, step)
                writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], step)

            if rank == 0 and (step == 1 or step % args.log_every == 0):
                print(
                    "step={step} loss={loss_total:.6f} endpoint={loss_endpoint:.6f} "
                    "traj={loss_trajectory:.6f} mean_flow={loss_mean_flow:.6f} grad={grad_norm:.4f}".format(
                        step=step, **metrics
                    ),
                    flush=True,
                )

            if rank == 0 and step % args.save_every == 0:
                ckpt_path = save_student_checkpoint(student, args.output_dir, args, step, metrics)
                print(f"saved {ckpt_path}", flush=True)

        if rank == 0:
            ckpt_path = save_student_checkpoint(student, args.output_dir, args, args.max_steps, last_metrics)
            writer.close()
            print(f"done; final checkpoint: {ckpt_path}", flush=True)
    finally:
        cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
