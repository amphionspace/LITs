"""Freeze encoder + duration predictor + spk_emb when validation prior loss plateaus.

After freeze, only the decoder (diffusion) continues training.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.utilities import rank_zero_only

from lits.utils import pylogger

log = pylogger.get_pylogger(__name__)


class FreezeEncoderCallback(Callback):
    """Freeze all non-decoder parameters after val prior loss stops improving.

    Frozen components: text encoder (emb, prenet, transformer, proj_m),
    duration predictor (proj_w), and speaker embedding (spk_emb).
    Only the decoder remains trainable.
    """

    def __init__(
        self,
        enabled: bool = True,
        monitor: str = "sub_loss/val_prior_loss",
        patience: int = 15,
        min_steps: int = 0,
        min_delta: float = 0.0,
    ) -> None:
        super().__init__()
        self.enabled = enabled
        self.monitor = monitor
        self.patience = int(patience)
        self.min_steps = int(min_steps)
        self.min_delta = float(min_delta)
        self.best_metric: Optional[float] = None
        self.epochs_no_improve: int = 0
        self.frozen: bool = False

    @staticmethod
    def _metric_value(metrics: dict, key: str) -> Optional[float]:
        if key not in metrics:
            return None
        value = metrics[key]
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().item()
        return float(value)

    def _freeze(self, pl_module: LightningModule) -> None:
        if not hasattr(pl_module, "freeze_encoder"):
            log.warning("FreezeEncoderCallback: model has no freeze_encoder(); skipping")
            return
        pl_module.freeze_encoder()
        self.frozen = True

    def _sync_state(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.world_size <= 1:
            return
        payload = torch.tensor(
            [
                float(self.frozen),
                float(self.epochs_no_improve),
                self.best_metric if self.best_metric is not None else -1.0,
            ],
            device=pl_module.device,
        )
        torch.distributed.broadcast(payload, src=0)
        self.frozen = bool(payload[0].item())
        self.epochs_no_improve = int(payload[1].item())
        best = float(payload[2].item())
        self.best_metric = None if best < 0.0 else best
        if self.frozen:
            self._freeze(pl_module)

    @rank_zero_only
    def _log_status(self, trainer: Trainer) -> None:
        metrics = {
            "freeze_encoder/frozen": float(self.frozen),
            "freeze_encoder/epochs_no_improve": float(self.epochs_no_improve),
        }
        if self.best_metric is not None:
            metrics["freeze_encoder/best_metric"] = self.best_metric
        trainer.logger.log_metrics(metrics, step=trainer.global_step)

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self.enabled:
            return
        self._sync_state(trainer, pl_module)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self.enabled or self.frozen:
            return

        metric = self._metric_value(trainer.callback_metrics, self.monitor)
        if metric is None:
            log.warning(
                f"FreezeEncoderCallback: missing {self.monitor}; skipping update"
            )
            return

        if trainer.is_global_zero:
            if self.best_metric is None or metric < self.best_metric - self.min_delta:
                self.best_metric = metric
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1

            if trainer.global_step >= self.min_steps and self.epochs_no_improve >= self.patience:
                log.info(
                    f"FreezeEncoderCallback: plateau on {self.monitor} "
                    f"(metric={metric:.6f}, best={self.best_metric:.6f}, "
                    f"patience={self.patience}) → freezing encoder + dur predictor + spk_emb"
                )
                self._freeze(pl_module)

        self._sync_state(trainer, pl_module)
        if trainer.is_global_zero:
            self._log_status(trainer)

    def state_dict(self) -> dict[str, Any]:
        return {
            "frozen": self.frozen,
            "best_metric": self.best_metric,
            "epochs_no_improve": self.epochs_no_improve,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.frozen = bool(state_dict.get("frozen", False))
        self.best_metric = state_dict.get("best_metric")
        self.epochs_no_improve = int(state_dict.get("epochs_no_improve", 0))
