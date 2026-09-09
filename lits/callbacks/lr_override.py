"""Override per-param-group learning rates at train start.

Motivation: when resuming from a Lightning checkpoint, ``optimizer.load_state_dict``
restores the old ``lr`` of every param group, silently discarding any new values
set in the model config (e.g. ``prior_encoder_lr``). This callback re-applies the
desired learning rates *after* the checkpoint has been restored, keyed by the
param-group ``name`` assigned in ``LITS._build_optimizer_param_groups``
("prior_encoder", "duration_predictor", "spk_emb", "decoder").

Typical use (downshift the encoder once its val prior loss turns):

    callbacks.lr_override.enabled=true
    callbacks.lr_override.overrides.prior_encoder=1e-5
"""

from __future__ import annotations

from typing import Optional

from lightning import Callback, LightningModule, Trainer

from lits.utils import pylogger

log = pylogger.get_pylogger(__name__)


class ParamGroupLROverride(Callback):
    def __init__(
        self,
        enabled: bool = False,
        overrides: Optional[dict[str, float]] = None,
    ) -> None:
        super().__init__()
        self.enabled = enabled
        self.overrides = {k: float(v) for k, v in (overrides or {}).items()}

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self.enabled or not self.overrides:
            return
        applied: dict[str, float] = {}
        known_groups: list[str] = []
        for optimizer in trainer.optimizers:
            for group in optimizer.param_groups:
                name = group.get("name")
                if name is None:
                    continue
                known_groups.append(name)
                if name in self.overrides:
                    group["lr"] = self.overrides[name]
                    # Keep initial_lr consistent in case a scheduler is added later.
                    if "initial_lr" in group:
                        group["initial_lr"] = self.overrides[name]
                    applied[name] = self.overrides[name]
        missing = set(self.overrides) - set(applied)
        if missing:
            log.warning(
                "ParamGroupLROverride: groups %s not found (available: %s)",
                sorted(missing),
                sorted(set(known_groups)),
            )
        if applied:
            log.info("ParamGroupLROverride: set lr %s", applied)
