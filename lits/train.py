import logging
logging.getLogger("phonemizer").setLevel(logging.ERROR)
logging.getLogger("jieba").setLevel(logging.ERROR)
import os
import re
import sys
import warnings
warnings.filterwarnings("ignore", module="phonemizer")

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from lits import utils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #


log = utils.get_pylogger(__name__)
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


class _ConsoleTee:
    """Mirror plain console writes to a file while leaving terminal behavior intact."""

    def __init__(self, stream, log_file):
        self._stream = stream
        self._log_file = log_file

    def write(self, data):
        written = self._stream.write(data)
        if data and "\r" not in data:
            clean_data = _ANSI_ESCAPE_RE.sub("", data)
            if clean_data:
                self._log_file.write(clean_data)
        return written

    def flush(self):
        self._stream.flush()
        self._log_file.flush()

    def isatty(self):
        return self._stream.isatty()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _is_rank_zero_process() -> bool:
    rank = os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or os.environ.get("SLURM_PROCID")
    return rank in (None, "", "0")


@contextmanager
def _tee_console_output(output_dir: str):
    if not _is_rank_zero_process():
        yield None
        return

    log_path = Path(output_dir) / "console_output.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8", buffering=1) as log_file:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = _ConsoleTee(old_stdout, log_file)
        sys.stderr = _ConsoleTee(old_stderr, log_file)
        try:
            yield log_path
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout, sys.stderr = old_stdout, old_stderr


@utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")  # pylint: disable=protected-access
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")  # pylint: disable=protected-access
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    if cfg.get("init_ckpt_path"):
        if cfg.get("ckpt_path"):
            raise ValueError(
                "Both init_ckpt_path and ckpt_path are set. "
                "Use init_ckpt_path for warm-start (weights only), "
                "or ckpt_path for full resume."
            )
        init_ckpt_path = cfg.get("init_ckpt_path")
        log.info(f"Warm-starting model weights from: {init_ckpt_path}")
        checkpoint = torch.load(init_ckpt_path, map_location="cpu", weights_only=False)
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["state_dict"], strict=False)
        if missing_keys:
            log.warning(f"Missing keys when loading warm-start checkpoint: {missing_keys}")
        if unexpected_keys:
            log.warning(f"Unexpected keys when loading warm-start checkpoint: {unexpected_keys}")

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))


    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")  # pylint: disable=protected-access
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        log.info("Before trainer.fit()")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
        log.info("After trainer.fit()")

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    with _tee_console_output(cfg.paths.output_dir) as console_log_path:
        if console_log_path is not None:
            log.info(f"Mirroring console output to: {console_log_path}")

        # apply extra utilities
        # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
        utils.extras(cfg)

        # train the model
        metric_dict, _ = train(cfg)

        # safely retrieve metric value for hydra-based hyperparameter optimization
        metric_value = utils.get_metric_value(metric_dict=metric_dict, metric_name=cfg.get("optimized_metric"))

        # return optimized metric
        return metric_value


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
