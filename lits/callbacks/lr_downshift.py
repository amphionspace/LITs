"""Downshift per-param-group learning rates when a validation metric plateaus.

Soft alternative to FreezeEncoderCallback: instead of hard-freezing the
encoder (which previously hurt low-resource-language WER when triggered too
early), multiply the lr of the target param groups by ``factor`` once the
monitored metric has not improved on its historical best for ``patience``
validation epochs.

With ``rewind_to_best=True`` the callback additionally rolls the model (and
optimizer moments) back to the checkpoint saved by the ModelCheckpoint
instance that monitors ``rewind_monitor`` (e.g. the best-val-prior ckpt)
before applying the downshift, so training continues from the best weights
instead of the already-degraded ones. Current learning rates are preserved
across the optimizer-state reload (only the target groups get the factor).

Param-group names come from ``LITS._build_optimizer_param_groups``:
"prior_encoder", "duration_predictor", "spk_emb", "decoder".
"""

from __future__ import annotations

import os
from typing import Any, Iterable, Optional

import torch
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from lits.utils import pylogger

log = pylogger.get_pylogger(__name__)


class LRDownshiftOnPlateau(Callback):
    def __init__(
        self,
        enabled: bool = True,
        monitor: str = "sub_loss/val_prior_loss",
        groups: Iterable[str] = ("prior_encoder",),
        factor: float = 0.1,
        patience: int = 5,
        min_delta: float = 0.0,
        max_downshifts: int = 2,
        rewind_to_best: bool = False,
        rewind_monitor: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.enabled = enabled
        self.monitor = monitor
        self.groups = list(groups)
        self.factor = float(factor)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.max_downshifts = int(max_downshifts)
        self.rewind_to_best = rewind_to_best
        self.rewind_monitor = rewind_monitor or monitor
        self.best_metric: Optional[float] = None
        self.epochs_no_improve: int = 0
        self.num_downshifts: int = 0

    @property
    def state_key(self) -> str:
        # Multiple instances of this callback coexist (encoder / duration
        # predictor); Lightning requires a unique key per stateful callback
        # for checkpointing.
        return self._generate_state_key(monitor=self.monitor, groups=tuple(self.groups))

    @staticmethod
    def _metric_value(metrics: dict, key: str) -> Optional[float]:
        if key not in metrics:
            return None
        value = metrics[key]
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().item()
        return float(value)

    def _apply_downshift(self, trainer: Trainer) -> None:
        applied: dict[str, float] = {}
        for optimizer in trainer.optimizers:
            for group in optimizer.param_groups:
                name = group.get("name")
                if name in self.groups:
                    group["lr"] = group["lr"] * self.factor
                    if "initial_lr" in group:
                        group["initial_lr"] = group["initial_lr"] * self.factor
                    applied[name] = group["lr"]
        if applied:
            log.info(
                "LRDownshiftOnPlateau: %s plateaued (best=%.6f, patience=%d) -> new lr %s",
                self.monitor,
                self.best_metric if self.best_metric is not None else float("nan"),
                self.patience,
                applied,
            )
        else:
            log.warning(
                "LRDownshiftOnPlateau: no param group matched %s; nothing downshifted",
                self.groups,
            )

    def _best_ckpt_path(self, trainer: Trainer, pl_module: LightningModule) -> str:
        path = ""
        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint) and cb.monitor == self.rewind_monitor:
                path = cb.best_model_path or ""
                break
        if trainer.world_size > 1:
            payload = [path if trainer.is_global_zero else ""]
            torch.distributed.broadcast_object_list(payload, src=0)
            path = payload[0]
        return path

    def _rewind(self, trainer: Trainer, pl_module: LightningModule) -> bool:
        """Load model weights and optimizer state from the best checkpoint,
        keeping the current learning rates. Returns True on success."""
        path = self._best_ckpt_path(trainer, pl_module)
        if not path or not os.path.exists(path):
            log.warning(
                "LRDownshiftOnPlateau: no checkpoint found for monitor %s; "
                "downshifting without rewind",
                self.rewind_monitor,
            )
            return False
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        pl_module.load_state_dict(ckpt["state_dict"])
        for optimizer, opt_state in zip(trainer.optimizers, ckpt.get("optimizer_states", [])):
            current_lrs = [
                (group["lr"], group.get("initial_lr")) for group in optimizer.param_groups
            ]
            optimizer.load_state_dict(opt_state)
            for group, (lr, initial_lr) in zip(optimizer.param_groups, current_lrs):
                group["lr"] = lr
                if initial_lr is not None:
                    group["initial_lr"] = initial_lr
        log.info("LRDownshiftOnPlateau: rewound model/optimizer to %s", path)
        return True

    def _sync_state(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.world_size <= 1:
            return
        payload = torch.tensor(
            [
                float(self.num_downshifts),
                float(self.epochs_no_improve),
                self.best_metric if self.best_metric is not None else float("inf"),
            ],
            device=pl_module.device,
        )
        torch.distributed.broadcast(payload, src=0)
        self.num_downshifts = int(payload[0].item())
        self.epochs_no_improve = int(payload[1].item())
        best = float(payload[2].item())
        self.best_metric = None if best == float("inf") else best

    def _log_status(self, trainer: Trainer) -> None:
        if trainer.logger is None:
            return
        metrics = {
            "lr_downshift/num_downshifts": float(self.num_downshifts),
            "lr_downshift/epochs_no_improve": float(self.epochs_no_improve),
        }
        if self.best_metric is not None:
            metrics["lr_downshift/best_metric"] = self.best_metric
        trainer.logger.log_metrics(metrics, step=trainer.global_step)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self.enabled or trainer.sanity_checking:
            return
        metric = self._metric_value(trainer.callback_metrics, self.monitor)
        if metric is None:
            log.warning("LRDownshiftOnPlateau: missing %s; skipping update", self.monitor)
            return

        downshift_now = False
        if trainer.is_global_zero:
            if self.best_metric is None or metric < self.best_metric - self.min_delta:
                self.best_metric = metric
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1
                if (
                    self.epochs_no_improve >= self.patience
                    and self.num_downshifts < self.max_downshifts
                ):
                    downshift_now = True

        if trainer.world_size > 1:
            flag = torch.tensor([float(downshift_now)], device=pl_module.device)
            torch.distributed.broadcast(flag, src=0)
            downshift_now = bool(flag.item())

        # Rewind and lr change must run on every rank so weights/optimizers stay in sync.
        if downshift_now:
            rewound = self._rewind(trainer, pl_module) if self.rewind_to_best else False
            self._apply_downshift(trainer)
            if trainer.is_global_zero:
                self.num_downshifts += 1
                self.epochs_no_improve = 0
                if not rewound:
                    # Without a rewind the historical best may never be reached
                    # again once the lr is reduced, so judge further degradation
                    # against the current (post-plateau) level instead.
                    self.best_metric = metric
                # With a rewind we are back at the best weights, so the
                # historical best remains the right baseline.

        self._sync_state(trainer, pl_module)
        if trainer.is_global_zero:
            self._log_status(trainer)

    def state_dict(self) -> dict[str, Any]:
        return {
            "best_metric": self.best_metric,
            "epochs_no_improve": self.epochs_no_improve,
            "num_downshifts": self.num_downshifts,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.best_metric = state_dict.get("best_metric")
        self.epochs_no_improve = int(state_dict.get("epochs_no_improve", 0))
        self.num_downshifts = int(state_dict.get("num_downshifts", 0))
