import math
import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks.progress.tqdm_progress import convert_inf

matplotlib.use("Agg")


def save_figure_to_numpy(fig: plt.Figure) -> np.ndarray:
    """
    Save a matplotlib figure to a numpy array.

    Args:
        fig (Figure): Matplotlib figure object.

    Returns:
        ndarray: Numpy array representing the figure.
    """
    data = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    return data


def plot_spectrogram_to_numpy(spectrogram: np.ndarray) -> np.ndarray:
    """
    Plot a spectrogram and convert it to a numpy array.

    Args:
        spectrogram (ndarray): Spectrogram data.

    Returns:
        ndarray: Numpy array representing the plotted spectrogram.
    """
    spectrogram = spectrogram.astype(np.float32)
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


class GradNormCallback(Callback):
    """
    Callback to log the gradient norm.
    """

    def on_after_backward(self, trainer, model):
        model.log("grad_norm", gradient_norm(model))


class EpochProgressBar(TQDMProgressBar):
    """Progress bar with Epoch current/total in the description; bar shows in-epoch steps."""

    def __init__(self, refresh_rate: int = 1):
        super().__init__(refresh_rate=refresh_rate)
        self._max_epochs: int | str = "?"

    def on_train_start(self, trainer, pl_module) -> None:
        super().on_train_start(trainer, pl_module)
        train_dataloader = trainer.train_dataloader
        if train_dataloader is None:
            return

        steps_per_epoch = len(train_dataloader)
        if trainer.max_epochs and trainer.max_epochs > 0:
            self._max_epochs = trainer.max_epochs
        elif trainer.max_steps and trainer.max_steps > 0:
            num_optimizers = len(trainer.optimizers) if trainer.optimizers else 2
            max_batches = trainer.max_steps // max(num_optimizers, 1)
            self._max_epochs = max(1, math.ceil(max_batches / steps_per_epoch))
        else:
            self._max_epochs = "?"

    def _epoch_desc(self, trainer) -> str:
        return f"Epoch {trainer.current_epoch + 1}/{self._max_epochs}"

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        total_batches = self.total_batches_current_epoch
        self.main_progress_bar.reset(convert_inf(total_batches))
        self.main_progress_bar.initial = 0
        self.main_progress_bar.set_description(self._epoch_desc(trainer))

    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs) -> None:
        super().on_train_batch_end(trainer, pl_module, *args, **kwargs)
        if self.is_enabled and self._main_progress_bar is not None:
            self.main_progress_bar.set_description(self._epoch_desc(trainer))


def gradient_norm(model: torch.nn.Module, norm_type: float = 2.0) -> torch.Tensor:
    """
    Compute the gradient norm.

    Args:
        model (Module): PyTorch model.
        norm_type (float, optional): Type of the norm. Defaults to 2.0.

    Returns:
        Tensor: Gradient norm.
    """
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    total_norm = torch.norm(torch.stack([torch.norm(g.detach(), norm_type) for g in grads]), norm_type)
    return total_norm
