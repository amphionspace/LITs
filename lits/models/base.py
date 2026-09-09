"""
This is a base lightning module that can be used to train a model.
The benefit of this abstraction is that all the logic outside of model definition can be reused for different models.
"""
import inspect
from abc import ABC
from typing import Any, Dict, Optional

import torch
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm

from lits import utils
from lits.utils.utils import plot_tensor

# Get module-level logger
log = utils.get_pylogger(__name__)


class BaseLits(LightningModule, ABC):
    def __init__(self):
        super().__init__()
        self.aux_loss_weight: Optional[float] = None  # legacy; prefer dur/prior_loss_weight
        self.dur_loss_weight: Optional[float] = None
        self.prior_loss_weight: Optional[float] = None

    def update_data_statistics(self, data_statistics: Optional[dict] = None) -> None:
        """
        Register mel_mean and mel_std as buffers for normalization.
        If data_statistics is None, use default values.
        """
        if data_statistics is None:
            data_statistics = {
                "mel_mean": 0.0,
                "mel_std": 1.0,
            }
        self.register_buffer("mel_mean", torch.tensor(data_statistics["mel_mean"]))
        self.register_buffer("mel_std", torch.tensor(data_statistics["mel_std"]))

    def _build_optimizer_param_groups(self) -> Optional[list[dict[str, Any]]]:
        """Return optimizer param groups, or None to optimize all parameters with one lr."""
        return None

    def configure_optimizers(self) -> Any:
        """
        Configure optimizer and learning rate scheduler for training.
        Ensures compatibility with checkpoint resume and Lightning's requirements.
        """
        param_groups = self._build_optimizer_param_groups()
        optimizer = self.hparams.optimizer(
            params=self.parameters() if param_groups is None else param_groups
        )
        if self.hparams.scheduler not in (None, {}):
            # 从scheduler中提取lightning_args，避免传递给scheduler构造函数
            lightning_args = getattr(self.hparams.scheduler, 'lightning_args', {})
            
            # 创建scheduler时排除lightning_args
            scheduler_kwargs = {}
            if hasattr(self.hparams.scheduler, 'func'):
                # 处理functools.partial对象
                scheduler_func = self.hparams.scheduler.func
                scheduler_kwargs = dict(self.hparams.scheduler.keywords)
                # 移除lightning_args
                scheduler_kwargs.pop('lightning_args', None)
            else:
                # 直接使用scheduler
                scheduler_func = self.hparams.scheduler
            
            # 添加optimizer参数
            scheduler_kwargs['optimizer'] = optimizer
            
            # 处理last_epoch参数
            if 'last_epoch' in scheduler_kwargs:
                if hasattr(self, "ckpt_loaded_epoch"):
                    scheduler_kwargs['last_epoch'] = self.ckpt_loaded_epoch - 1
                else:
                    scheduler_kwargs['last_epoch'] = -1
            
            # 创建scheduler实例
            scheduler = scheduler_func(**scheduler_kwargs)
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": lightning_args.get("interval", "epoch"),
                    "frequency": lightning_args.get("frequency", 1),
                    "name": "learning_rate",
                },
            }
        
        return {"optimizer": optimizer}

    def _dur_loss_weight(self) -> float:
        if getattr(self, "encoder_frozen", False):
            return 0.0
        if self.dur_loss_weight is not None:
            return float(self.dur_loss_weight)
        if self.aux_loss_weight is not None:
            return float(self.aux_loss_weight)
        return self._step_based_aux_weight()

    def _prior_loss_weight(self) -> float:
        if getattr(self, "encoder_frozen", False):
            return 0.0
        if self.prior_loss_weight is not None:
            return float(self.prior_loss_weight)
        if self.aux_loss_weight is not None:
            return float(self.aux_loss_weight)
        return self._step_based_aux_weight()

    def _step_based_aux_weight(self) -> float:
        """Optional step-based fallback (applies equally when callback is disabled)."""
        start = int(getattr(self.hparams, "aux_loss_decay_start_step", -1))
        end = int(getattr(self.hparams, "aux_loss_decay_end_step", -1))
        if start < 0 or end < 0 or end <= start:
            return 1.0
        step = int(self.global_step)
        if step <= start:
            return 1.0
        if step >= end:
            return 0.0
        return 1.0 - (step - start) / (end - start)

    @staticmethod
    def _alignment_duration_stats(batch: dict, attn: torch.Tensor) -> dict[str, torch.Tensor]:
        """Summarize token durations induced by the current alignment path."""
        if attn is None:
            return {}
        if attn.dim() == 4:
            attn = attn.squeeze(1)
        if attn.dim() != 3:
            return {}

        durations = attn.sum(dim=-1).float()
        device = durations.device
        x_lengths = batch["x_lengths"].to(device)
        y_lengths = batch["y_lengths"].to(device)
        x = batch["x"].to(device)

        max_text_len = durations.size(1)
        positions = torch.arange(max_text_len, device=device).unsqueeze(0)
        valid_mask = positions < x_lengths.unsqueeze(1)
        x = x[:, :max_text_len]
        blank_id = 0
        separator_id = 3  # All current mixed-language inventories reserve "_" at id 3.
        blank_mask = valid_mask & (x == blank_id)
        separator_mask = valid_mask & ((x == blank_id) | (x == separator_id))
        word_separator_mask = valid_mask & (x == separator_id)
        speech_mask = valid_mask & ~separator_mask

        def _ratio(event_mask: torch.Tensor, base_mask: torch.Tensor) -> torch.Tensor:
            base = base_mask.float().sum().clamp_min(1.0)
            return (event_mask & base_mask).float().sum() / base

        def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            denom = mask.float().sum().clamp_min(1.0)
            return (values * mask.float()).sum() / denom

        def _masked_min(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            masked = values.masked_fill(~mask, float("inf"))
            min_value = masked.amin()
            return torch.where(torch.isfinite(min_value), min_value, torch.zeros_like(min_value))

        zero_mask = durations <= 0
        one_mask = durations == 1
        le2_mask = durations <= 2
        ge3_mask = durations >= 3
        ge5_mask = durations >= 5
        mel_per_token = y_lengths.float() / x_lengths.float().clamp_min(1.0)
        speech_counts = speech_mask.float().sum(dim=1).clamp_min(1.0)
        mel_per_speech_token = y_lengths.float() / speech_counts

        return {
            "zero_ratio": _ratio(zero_mask, valid_mask),
            "one_ratio": _ratio(one_mask, valid_mask),
            "le2_ratio": _ratio(le2_mask, valid_mask),
            "speech_zero_ratio": _ratio(zero_mask, speech_mask),
            "speech_one_ratio": _ratio(one_mask, speech_mask),
            "speech_le2_ratio": _ratio(le2_mask, speech_mask),
            "separator_zero_ratio": _ratio(zero_mask, separator_mask),
            "separator_one_ratio": _ratio(one_mask, separator_mask),
            "separator_ge3_ratio": _ratio(ge3_mask, separator_mask),
            "separator_ge5_ratio": _ratio(ge5_mask, separator_mask),
            "word_separator_ge3_ratio": _ratio(ge3_mask, word_separator_mask),
            "word_separator_ge5_ratio": _ratio(ge5_mask, word_separator_mask),
            "blank_zero_ratio": _ratio(zero_mask, blank_mask),
            "blank_one_ratio": _ratio(one_mask, blank_mask),
            "mean": _masked_mean(durations, valid_mask),
            "speech_mean": _masked_mean(durations, speech_mask),
            "separator_mean": _masked_mean(durations, separator_mask),
            "word_separator_mean": _masked_mean(durations, word_separator_mask),
            "min": _masked_min(durations, valid_mask),
            "speech_min": _masked_min(durations, speech_mask),
            "separator_max": durations.masked_fill(~separator_mask, 0.0).amax(),
            "word_separator_max": durations.masked_fill(~word_separator_mask, 0.0).amax(),
            "mel_per_token_mean": mel_per_token.mean(),
            "mel_per_token_min": mel_per_token.amin(),
            "mel_per_speech_token_mean": mel_per_speech_token.mean(),
            "mel_per_speech_token_min": mel_per_speech_token.amin(),
            "sample_mel_per_token_lt1_ratio": (mel_per_token < 1.0).float().mean(),
            "sample_mel_per_token_lt1_5_ratio": (mel_per_token < 1.5).float().mean(),
        }

    def _log_alignment_duration_stats(
        self,
        stats: dict[str, torch.Tensor],
        stage: str,
        *,
        on_step: bool,
        on_epoch: bool,
    ) -> None:
        for name, value in stats.items():
            self.log(
                f"mas_duration/{stage}_{name}",
                value,
                on_step=on_step,
                on_epoch=on_epoch,
                logger=True,
                sync_dist=True,
            )

    def _log_duration_mask_stats(
        self,
        stats: dict[str, torch.Tensor],
        stage: str,
        *,
        on_step: bool,
        on_epoch: bool,
    ) -> None:
        for name, value in stats.items():
            self.log(
                f"duration_mask/{stage}_{name}",
                value,
                on_step=on_step,
                on_epoch=on_epoch,
                logger=True,
                sync_dist=True,
            )

    def get_losses(self, batch: dict) -> dict:
        """
        Compute all loss components for a batch.
        Returns a dict with dur_loss, prior_loss, diff_loss.
        """
        x, x_lengths = batch["x"], batch["x_lengths"]
        y, y_lengths = batch["y"], batch["y_lengths"]
        spks = batch["spks"]
        outputs = self(
            x=x,
            x_lengths=x_lengths,
            y=y,
            y_lengths=y_lengths,
            spks=spks,
            out_size=self.out_size,
            durations=batch["durations"],
            x_tones=batch.get("x_tones"),
        )
        dur_loss, prior_loss, diff_loss, attn = outputs[:4]
        duration_mask_stats = outputs[4] if len(outputs) > 4 else {}
        return {
            "dur_loss": dur_loss,
            "prior_loss": prior_loss,
            "diff_loss": diff_loss,
            "alignment_duration_stats": self._alignment_duration_stats(batch, attn),
            "duration_mask_stats": duration_mask_stats,
        }

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        Restore epoch information from checkpoint for scheduler compatibility.
        """
        self.ckpt_loaded_epoch = checkpoint["epoch"]  # pylint: disable=attribute-defined-outside-init

    def training_step(self, batch: Any, batch_idx: int) -> dict:
        """
        Perform a training step, log all loss components, and return total loss.
        """
        loss_dict = self.get_losses(batch)
        self.log(
            "step",
            float(self.global_step),
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "sub_loss/train_dur_loss",
            loss_dict["dur_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "sub_loss/train_prior_loss",
            loss_dict["prior_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "sub_loss/train_diff_loss",
            loss_dict["diff_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self._log_alignment_duration_stats(
            loss_dict["alignment_duration_stats"],
            "train",
            on_step=True,
            on_epoch=True,
        )
        self._log_duration_mask_stats(
            loss_dict["duration_mask_stats"],
            "train",
            on_step=True,
            on_epoch=True,
        )
        dur_w = self._dur_loss_weight()
        prior_w = self._prior_loss_weight()
        self.log("train/dur_loss_weight", dur_w, on_step=True, on_epoch=False, logger=True, sync_dist=True)
        self.log("train/prior_loss_weight", prior_w, on_step=True, on_epoch=False, logger=True, sync_dist=True)
        if getattr(self, "encoder_frozen", False):
            self.log(
                "freeze_encoder/frozen",
                1.0,
                on_step=True,
                on_epoch=False,
                logger=True,
                sync_dist=True,
            )
        total_loss = (
            dur_w * loss_dict["dur_loss"] +
            prior_w * loss_dict["prior_loss"] +
            loss_dict["diff_loss"]
        )
        self.log(
            "loss/train",
            total_loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            sync_dist=True,
        )

        if self.trainer is not None and hasattr(self.trainer, "optimizers") and self.trainer.optimizers:
            optimizer = self.trainer.optimizers[0]
            for idx, group in enumerate(optimizer.param_groups):
                group_name = group.get("name", f"group_{idx}")
                self.log(
                    f"learning_rate/{group_name}",
                    group["lr"],
                    on_step=True,
                    on_epoch=True,
                    prog_bar=(idx == 0),
                    logger=True,
                    sync_dist=True,
                )
            # legacy alias: first param group
            self.log(
                "learning_rate/lr",
                optimizer.param_groups[0]["lr"],
                on_step=True,
                on_epoch=True,
                logger=True,
                sync_dist=True,
            )
        else:
            log.warning("No optimizer found in trainer")

        return {
            "loss": total_loss,
            "log": {
                "dur_loss": loss_dict["dur_loss"],
                "prior_loss": loss_dict["prior_loss"],
                "diff_loss": loss_dict["diff_loss"],
            },
        }

    def validation_step(self, batch: Any, batch_idx: int) -> float:
        """
        Perform a validation step, log all loss components, and return total loss.
        """
        loss_dict = self.get_losses(batch)
        self.log(
            "sub_loss/val_dur_loss",
            loss_dict["dur_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "sub_loss/val_prior_loss",
            loss_dict["prior_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "sub_loss/val_diff_loss",
            loss_dict["diff_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self._log_alignment_duration_stats(
            loss_dict["alignment_duration_stats"],
            "val",
            on_step=True,
            on_epoch=True,
        )
        self._log_duration_mask_stats(
            loss_dict["duration_mask_stats"],
            "val",
            on_step=True,
            on_epoch=True,
        )
        total_loss = (
            loss_dict["dur_loss"] +
            loss_dict["prior_loss"] +
            loss_dict["diff_loss"]
        )
        self.log(
            "loss/val",
            total_loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            sync_dist=True,
        )
        return total_loss

    def on_validation_end(self) -> None:
        """
        Visualize original and generated samples at validation end.
        Only runs on global rank zero.
        """
        if self.trainer.is_global_zero:
            one_batch = next(iter(self.trainer.val_dataloaders))
            n_vis = min(2, one_batch["y"].shape[0])
            if self.current_epoch == 0:
                log.debug("Plotting original samples")
                for i in range(n_vis):
                    y = one_batch["y"][i].unsqueeze(0).to(self.device)
                    self.logger.experiment.add_image(
                        f"original/{i}",
                        plot_tensor(y.squeeze().cpu()),
                        self.current_epoch,
                        dataformats="HWC",
                    )
            log.debug("Synthesising...")
            for i in range(n_vis):
                x = one_batch["x"][i].unsqueeze(0).to(self.device)
                x_lengths = one_batch["x_lengths"][i].unsqueeze(0).to(self.device)
                spks = one_batch["spks"][i].unsqueeze(0).to(self.device) if one_batch["spks"] is not None else None
                x_tones = (
                    one_batch["x_tones"][i].unsqueeze(0).to(self.device)[:, :x_lengths]
                    if one_batch.get("x_tones") is not None
                    else None
                )
                output = self.synthesise(x[:, :x_lengths], x_lengths, n_timesteps=10, spks=spks, x_tones=x_tones)
                y_enc, y_dec = output["encoder_outputs"], output["decoder_outputs"]
                attn = output["attn"]
                self.logger.experiment.add_image(
                    f"generated_enc/{i}",
                    plot_tensor(y_enc.squeeze().cpu()),
                    self.current_epoch,
                    dataformats="HWC",
                )
                self.logger.experiment.add_image(
                    f"generated_dec/{i}",
                    plot_tensor(y_dec.squeeze().cpu()),
                    self.current_epoch,
                    dataformats="HWC",
                )
                self.logger.experiment.add_image(
                    f"alignment/{i}",
                    plot_tensor(attn.squeeze().cpu()),
                    self.current_epoch,
                    dataformats="HWC",
                )

    def on_before_optimizer_step(self, optimizer: Any) -> None:
        """
        Log gradient norm for all parameters before optimizer step.
        """
        self.log_dict({f"grad_norm/{k}": v for k, v in grad_norm(self, norm_type=2).items()})
