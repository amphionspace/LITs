import os
import sys
import warnings
from importlib.util import find_spec
from math import ceil
from pathlib import Path
from typing import Any, Callable, Dict, Tuple, Optional, List

import gdown
import matplotlib.pyplot as plt
import numpy as np
import torch
import wget
from omegaconf import DictConfig

from lits.utils import pylogger, rich_utils

_logger = pylogger.get_pylogger(__name__)

def _log_and_return(msg, level="info"):
    getattr(_logger, level)(msg)
    return None

def _is_tensor(obj):
    return isinstance(obj, torch.Tensor)

def _is_numpy(obj):
    return isinstance(obj, np.ndarray)

def _is_list(obj):
    return isinstance(obj, list)

def get_metric_value(metric_dict: Dict[str, Any], metric_name: str) -> Optional[float]:
    if not metric_name:
        return _log_and_return("Metric name is None! Skipping metric value retrieval...", "info")
    if metric_name not in metric_dict:
        raise ValueError(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Check if the metric name in LightningModule and config is correct!"
        )
    value = metric_dict[metric_name].item()
    _logger.info(f"Retrieved metric value! <{metric_name}={value}>")
    return value

def intersperse(lst: List[Any], item: Any) -> List[Any]:
    return [item if i % 2 == 0 else lst[i // 2] for i in range(len(lst) * 2 + 1)]

def save_figure_to_numpy(fig) -> np.ndarray:
    arr = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    arr = arr.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return arr

def plot_tensor(tensor: np.ndarray) -> np.ndarray:
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.canvas.draw()
    arr = save_figure_to_numpy(fig)
    plt.close()
    return arr

def save_plot(tensor: np.ndarray, save_path: str) -> None:
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.canvas.draw()
    plt.savefig(save_path)
    plt.close()

def to_numpy(tensor: Any) -> np.ndarray:
    if _is_numpy(tensor):
        return tensor
    if _is_tensor(tensor):
        return tensor.detach().cpu().numpy()
    if _is_list(tensor):
        return np.array(tensor)
    raise TypeError("Unsupported type for conversion to numpy array")

def get_user_data_dir(app_name: str = "lits_tts") -> Path:
    env_home = os.environ.get("lits_HOME")
    if env_home:
        ans = Path(env_home).expanduser().resolve(strict=False)
    elif sys.platform == "win32":
        import winreg
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\Shell Folders",
        )
        dir_, _ = winreg.QueryValueEx(key, "Local AppData")
        ans = Path(dir_).resolve(strict=False)
    elif sys.platform == "darwin":
        ans = Path("~/Library/Application Support/").expanduser()
    else:
        ans = Path.home().joinpath(".local/share")
    final_path = ans.joinpath(app_name)
    final_path.mkdir(parents=True, exist_ok=True)
    return final_path

def assert_model_downloaded(checkpoint_path: str, url: str, use_wget: bool = True) -> None:
    p = Path(checkpoint_path)
    if p.exists():
        _logger.debug(f"[+] Model already present at {checkpoint_path}!")
        print(f"[+] Model already present at {checkpoint_path}!")
        return
    _logger.info(f"[-] Model not found at {checkpoint_path}! Will download it")
    print(f"[-] Model not found at {checkpoint_path}! Will download it")
    if not use_wget:
        gdown.download(url=url, output=str(p), quiet=False, fuzzy=True)
    else:
        wget.download(url=url, out=str(p))

def get_phoneme_durations(durations: List[int], phones: List[str]) -> List[Dict[str, Dict[str, int]]]:
    prev = durations[0]
    merged = []
    for i in range(1, len(durations), 2):
        if i == len(durations) - 2:
            next_half = durations[i + 1]
        else:
            next_half = ceil(durations[i + 1] / 2)
        curr = prev + durations[i] + next_half
        prev = durations[i + 1] - next_half
        merged.append(curr)
    assert len(phones) == len(merged)
    assert len(merged) == (len(durations) - 1) // 2
    merged_tensor = torch.cumsum(torch.tensor(merged), 0, dtype=torch.long)
    start = torch.tensor(0)
    duration_json = []
    for i, duration in enumerate(merged_tensor):
        duration_json.append({
            phones[i]: {
                "starttime": start.item(),
                "endtime": duration.item(),
                "duration": duration.item() - start.item(),
            }
        })
        start = duration
    assert list(duration_json[-1].values())[0]["endtime"] == sum(durations), \
        f"{list(duration_json[-1].values())[0]['endtime'], sum(durations)}"
    return duration_json

def extras(cfg: DictConfig) -> None:
    extras_cfg = cfg.get("extras")
    if not extras_cfg:
        _logger.warning("Extras config not found! <cfg.extras=null>")
        return
    if extras_cfg.get("ignore_warnings"):
        _logger.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")
    if extras_cfg.get("enforce_tags"):
        _logger.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)
    if extras_cfg.get("print_config"):
        _logger.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)

def task_wrapper(task_func: Callable) -> Callable:
    def wrap(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        try:
            metric_dict, object_dict = task_func(cfg=cfg)
        except Exception as ex:
            _logger.exception("Exception occurred during task execution.")
            raise ex
        finally:
            _logger.info(f"Output dir: {cfg.paths.output_dir}")
            if find_spec("wandb"):
                import wandb
                if wandb.run:
                    _logger.info("Closing wandb!")
                    wandb.finish()
        return metric_dict, object_dict
    return wrap
