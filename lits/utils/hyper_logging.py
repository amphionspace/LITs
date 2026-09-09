from typing import Dict, Any
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import OmegaConf
from lits.utils import pylogger

_logger = pylogger.get_pylogger(__name__)

def _count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    return total, trainable, non_trainable

def _collect_hparams(cfg, model):
    hparams = {}
    hparams["model"] = cfg["model"]
    total, trainable, non_trainable = _count_params(model)
    hparams["model/params/total"] = total
    hparams["model/params/trainable"] = trainable
    hparams["model/params/non_trainable"] = non_trainable
    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]
    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")
    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")
    return hparams

@rank_zero_only
def log_hyperparameters(trainer: Any = None, model: Any = None, cfg: Any = None, object_dict: Dict[str, Any] = None) -> None:
    if object_dict is not None:
        cfg = OmegaConf.to_container(object_dict["cfg"])
        model = object_dict["model"]
        trainer = object_dict["trainer"]
    if not trainer or not getattr(trainer, "logger", None):
        _logger.warning("Logger not found! Skipping hyperparameter logging...")
        return
    hparams = _collect_hparams(cfg, model)
    for logger in getattr(trainer, "loggers", []):
        logger.log_hyperparams(hparams) 