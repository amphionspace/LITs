"""Validation-driven schedule for duration and prior loss weights (independent)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

import torch
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.utilities import rank_zero_only

from lits.utils import pylogger

log = pylogger.get_pylogger(__name__)

Stage = Literal["full", "decay", "done"]


@dataclass
class _LossTrack:
    name: str
    monitor: str
    patience: int
    decay_epochs: int
    decay_val_stride: int
    min_weight: float
    stage: Stage = "full"
    best_metric: Optional[float] = None
    epochs_no_improve: int = 0
    decay_steps: int = 0
    val_epochs_in_decay: int = 0
    current_weight: float = 1.0

    def _apply_decay_step(self) -> None:
        self.decay_steps += 1
        progress = min(1.0, self.decay_steps / self.decay_epochs)
        self.current_weight = 1.0 - progress * (1.0 - self.min_weight)
        if self.decay_steps >= self.decay_epochs:
            self.stage = "done"
            log.info(
                f"Aux loss schedule [{self.name}]: decay finished "
                f"(weight={self.current_weight:.4f})"
            )

    def advance_after_min_steps(
        self,
        trainer: Trainer,
        metric: float,
        min_steps: int,
    ) -> None:
        if self.stage == "done":
            return

        if self.stage == "full":
            if self.best_metric is None or metric < self.best_metric:
                self.best_metric = metric
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1

            if trainer.global_step >= min_steps and self.epochs_no_improve >= self.patience:
                self.stage = "decay"
                self.decay_steps = 0
                self.val_epochs_in_decay = 0
                remaining = max(0, self.decay_epochs - 1) * self.decay_val_stride
                log.info(
                    f"Aux loss schedule [{self.name}]: plateau detected "
                    f"(metric={metric:.6f}, best={self.best_metric:.6f}, "
                    f"patience={self.patience}) → {self.decay_epochs} decay steps "
                    f"(drop now, then every {self.decay_val_stride} val epoch(s); "
                    f"{remaining} val epochs after first drop)"
                )

        if self.stage == "decay":
            if self.decay_steps == 0:
                # First drop on plateau trigger; also handles resume when decay_steps==0.
                self._apply_decay_step()
                self.val_epochs_in_decay = 0
                return

            self.val_epochs_in_decay += 1
            if self.val_epochs_in_decay % self.decay_val_stride != 0:
                return

            self._apply_decay_step()

    def to_payload(self) -> list[float]:
        stage_id = {"full": 0.0, "decay": 1.0, "done": 2.0}[self.stage]
        best = self.best_metric if self.best_metric is not None else 0.0
        return [
            stage_id,
            self.current_weight,
            float(self.epochs_no_improve),
            float(self.decay_steps),
            best,
            float(self.val_epochs_in_decay),
        ]

    def load_payload(self, payload: list[float]) -> None:
        stage_id = int(payload[0].item() if isinstance(payload[0], torch.Tensor) else payload[0])
        self.stage = ("full", "decay", "done")[stage_id]
        self.current_weight = float(payload[1])
        self.epochs_no_improve = int(payload[2])
        self.decay_steps = int(payload[3])
        self.best_metric = float(payload[4])
        self.val_epochs_in_decay = int(payload[5]) if len(payload) > 5 else 0


class AuxLossScheduleCallback(Callback):
    """Independently decay dur/prior loss weights after each val metric plateaus.

    Plateau is detected when val loss fails to beat the running best for ``patience``
    validation epochs (any strict decrease vs best resets the counter).

    ``dur_min_weight`` is the floor for duration (typically 0).
    ``prior_min_weight`` is the floor for prior (keep > 0 so encoder stays anchored to mel).
    """

    DEFAULT_PRIOR_MIN_WEIGHT = 0.3

    def __init__(
        self,
        enabled: bool = True,
        dur_monitor: str = "sub_loss/val_dur_loss",
        prior_monitor: str = "sub_loss/val_prior_loss",
        schedule_prior: bool = True,
        min_steps: int = 0,
        patience: int = 10,
        dur_patience: Optional[int] = None,
        prior_patience: Optional[int] = None,
        decay_epochs: int = 8,
        dur_decay_epochs: Optional[int] = None,
        prior_decay_epochs: Optional[int] = None,
        decay_val_stride: int = 1,
        dur_decay_val_stride: Optional[int] = None,
        prior_decay_val_stride: Optional[int] = None,
        dur_min_weight: float = 0.0,
        prior_min_weight: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.enabled = enabled
        self.schedule_prior = schedule_prior
        self.min_steps = int(min_steps)

        dur_floor = float(dur_min_weight)

        if prior_min_weight is None:
            prior_floor = self.DEFAULT_PRIOR_MIN_WEIGHT if schedule_prior else dur_floor
        else:
            prior_floor = float(prior_min_weight)
        if schedule_prior and prior_floor <= 0.0:
            raise ValueError(
                "prior_min_weight must be > 0 when schedule_prior=true "
                "(prior weight 0 lets encoder drift and val prior_loss spikes)"
            )

        self.dur_track = _LossTrack(
            name="dur",
            monitor=dur_monitor,
            patience=int(dur_patience if dur_patience is not None else patience),
            decay_epochs=max(1, int(dur_decay_epochs if dur_decay_epochs is not None else decay_epochs)),
            decay_val_stride=max(1, int(dur_decay_val_stride if dur_decay_val_stride is not None else decay_val_stride)),
            min_weight=dur_floor,
        )
        self.prior_track = _LossTrack(
            name="prior",
            monitor=prior_monitor,
            patience=int(prior_patience if prior_patience is not None else patience),
            decay_epochs=max(1, int(prior_decay_epochs if prior_decay_epochs is not None else decay_epochs)),
            decay_val_stride=max(1, int(prior_decay_val_stride if prior_decay_val_stride is not None else decay_val_stride)),
            min_weight=float(prior_floor),
        )

    @staticmethod
    def _metric_value(metrics: dict, key: str) -> Optional[float]:
        if key not in metrics:
            return None
        value = metrics[key]
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().item()
        return float(value)

    def _set_module_weights(self, pl_module: LightningModule) -> None:
        pl_module.dur_loss_weight = float(self.dur_track.current_weight)
        if self.schedule_prior and getattr(pl_module, "prior_loss", True):
            pl_module.prior_loss_weight = float(self.prior_track.current_weight)
        else:
            pl_module.prior_loss_weight = 1.0

    @rank_zero_only
    def _log_status(self, trainer: Trainer) -> None:
        metrics = {
            "aux_schedule/dur_weight": self.dur_track.current_weight,
            "aux_schedule/dur_stage_id": {"full": 0.0, "decay": 1.0, "done": 2.0}[self.dur_track.stage],
            "aux_schedule/dur_epochs_no_improve": float(self.dur_track.epochs_no_improve),
        }
        if self.dur_track.best_metric is not None:
            metrics["aux_schedule/dur_best_metric"] = self.dur_track.best_metric
        if self.schedule_prior:
            metrics.update({
                "aux_schedule/prior_weight": self.prior_track.current_weight,
                "aux_schedule/prior_stage_id": {"full": 0.0, "decay": 1.0, "done": 2.0}[self.prior_track.stage],
                "aux_schedule/prior_epochs_no_improve": float(self.prior_track.epochs_no_improve),
            })
            if self.prior_track.best_metric is not None:
                metrics["aux_schedule/prior_best_metric"] = self.prior_track.best_metric
        trainer.logger.log_metrics(metrics, step=trainer.global_step)

    def _sync_tracks(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.world_size <= 1:
            return
        payload = torch.tensor(
            self.dur_track.to_payload() + self.prior_track.to_payload(),
            device=pl_module.device,
        )
        torch.distributed.broadcast(payload, src=0)
        self.dur_track.load_payload(payload[:6].tolist())
        self.prior_track.load_payload(payload[6:].tolist())

    def _apply_pending_first_decay(self, trainer: Trainer) -> None:
        """Resume: ckpt may have stage=decay but decay_steps=0 (old code or fresh trigger)."""
        if not trainer.is_global_zero:
            return
        for track in (self.dur_track, self.prior_track):
            if track.stage == "decay" and track.decay_steps == 0:
                track._apply_decay_step()
                track.val_epochs_in_decay = 0

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self.enabled:
            return
        self._apply_pending_first_decay(trainer)
        self._sync_tracks(trainer, pl_module)
        self._set_module_weights(pl_module)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self.enabled:
            return

        dur_metric = self._metric_value(trainer.callback_metrics, self.dur_track.monitor)
        if dur_metric is None:
            log.warning(
                f"Aux loss schedule: missing {self.dur_track.monitor}; skipping update"
            )
            return

        if trainer.is_global_zero:
            self.dur_track.advance_after_min_steps(trainer, dur_metric, self.min_steps)
            if self.schedule_prior and getattr(pl_module, "prior_loss", True):
                prior_metric = self._metric_value(trainer.callback_metrics, self.prior_track.monitor)
                if prior_metric is None:
                    log.warning(
                        f"Aux loss schedule: missing {self.prior_track.monitor}; "
                        "keeping prior weight unchanged"
                    )
                else:
                    self.prior_track.advance_after_min_steps(
                        trainer, prior_metric, self.min_steps,
                    )

        self._sync_tracks(trainer, pl_module)
        self._set_module_weights(pl_module)
        if trainer.is_global_zero:
            self._log_status(trainer)

    def state_dict(self) -> dict[str, Any]:
        return {
            "dur_track": {
                "stage": self.dur_track.stage,
                "best_metric": self.dur_track.best_metric,
                "epochs_no_improve": self.dur_track.epochs_no_improve,
                "decay_steps": self.dur_track.decay_steps,
                "val_epochs_in_decay": self.dur_track.val_epochs_in_decay,
                "current_weight": self.dur_track.current_weight,
            },
            "prior_track": {
                "stage": self.prior_track.stage,
                "best_metric": self.prior_track.best_metric,
                "epochs_no_improve": self.prior_track.epochs_no_improve,
                "decay_steps": self.prior_track.decay_steps,
                "val_epochs_in_decay": self.prior_track.val_epochs_in_decay,
                "current_weight": self.prior_track.current_weight,
            },
            # legacy single-track checkpoint
            "stage": self.dur_track.stage,
            "best_metric": self.dur_track.best_metric,
            "epochs_no_improve": self.dur_track.epochs_no_improve,
            "decay_epoch": self.dur_track.decay_steps,
            "current_weight": self.dur_track.current_weight,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if "dur_track" in state_dict:
            for key, track in (("dur_track", self.dur_track), ("prior_track", self.prior_track)):
                blob = state_dict[key]
                track.stage = blob.get("stage", "full")
                track.best_metric = blob.get("best_metric")
                track.epochs_no_improve = int(blob.get("epochs_no_improve", 0))
                track.decay_steps = int(blob.get("decay_steps", blob.get("decay_epoch", 0)))
                track.val_epochs_in_decay = int(blob.get("val_epochs_in_decay", 0))
                track.current_weight = float(blob.get("current_weight", 1.0))
            return
        # legacy: one shared weight → apply to dur only; prior restarts at 1.0
        self.dur_track.stage = state_dict.get("stage", "full")
        self.dur_track.best_metric = state_dict.get("best_metric")
        self.dur_track.epochs_no_improve = int(state_dict.get("epochs_no_improve", 0))
        self.dur_track.decay_steps = int(state_dict.get("decay_epoch", 0))
        self.dur_track.val_epochs_in_decay = 0
        self.dur_track.current_weight = float(state_dict.get("current_weight", 1.0))
