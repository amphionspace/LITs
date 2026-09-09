from pathlib import Path
from typing import Sequence

import rich
import rich.syntax
import rich.tree
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf, open_dict
from rich.prompt import Prompt

from lits.utils import pylogger

_logger = pylogger.get_pylogger(__name__)


def _add_branch(tree, field, value, style, resolve):
    branch = tree.add(field, style=style, guide_style=style)
    if isinstance(value, DictConfig):
        content = OmegaConf.to_yaml(value, resolve=resolve)
    else:
        content = str(value)
    branch.add(rich.syntax.Syntax(content, "yaml"))


def _get_print_queue(cfg, order):
    queue = [f for f in order if f in cfg]
    for f in order:
        if f not in cfg:
            _logger.warning(f"Field '{f}' not found in config. Skipping '{f}' config printing...")
    queue.extend(f for f in cfg if f not in queue)
    return queue


def _save_tree_to_file(tree, output_dir):
    with open(Path(output_dir, "config_tree.log"), "w", encoding="utf-8") as f:
        rich.print(tree, file=f)


def _save_tags_to_file(tags, output_dir):
    with open(Path(output_dir, "tags.log"), "w", encoding="utf-8") as f:
        rich.print(tags, file=f)


@rank_zero_only
def print_config_tree(cfg: DictConfig, resolve: bool = False, save_to_file: bool = False, print_order: Sequence[str] = None) -> None:
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)
    order = print_order or ("data", "model", "callbacks", "logger", "trainer", "paths", "extras")
    queue = _get_print_queue(cfg, order)
    for field in queue:
        _add_branch(tree, field, cfg[field], style, resolve)
    rich.print(tree)
    if save_to_file:
        _save_tree_to_file(tree, cfg.paths.output_dir)


@rank_zero_only
def enforce_tags(cfg: DictConfig, save_to_file: bool = False) -> None:
    """Prompts user to input tags from command line if no tags are provided in config.

    :param cfg: A DictConfig composed by Hydra.
    :param save_to_file: Whether to export tags to the hydra output folder. Default is ``False``.
    """
    if not cfg.get("tags"):
        if "id" in HydraConfig().cfg.hydra.job:
            raise ValueError("Specify tags before launching a multirun!")

        _logger.warning("No tags provided in config. Prompting user to input tags...")
        tags = Prompt.ask("Enter a list of comma separated tags", default="dev")
        tags = [t.strip() for t in tags.split(",") if t != ""]

        with open_dict(cfg):
            cfg.tags = tags

        _logger.info(f"Tags: {cfg.tags}")

    if save_to_file:
        with open(Path(cfg.paths.output_dir, "tags.log"), "w", encoding="utf-8") as file:
            rich.print(cfg.tags, file=file)
