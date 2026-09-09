import argparse
import json
import os
import sys
from pathlib import Path

import rootutils
import torch
from hydra import compose, initialize
from omegaconf import open_dict
from tqdm.auto import tqdm

from lits.data.datamodule import TextMelDataModule
from lits.utils.hyper_logging import pylogger

_logger = pylogger.get_pylogger(__name__)

def _stat_accumulate(mels, mel_lengths):
    return torch.sum(mels), torch.sum(torch.pow(mels, 2)), torch.sum(mel_lengths)

def _finalize_stats(total_sum, total_sq_sum, total_len, out_channels):
    mean = total_sum / (total_len * out_channels)
    std = torch.sqrt((total_sq_sum / (total_len * out_channels)) - torch.pow(mean, 2))
    return {"mel_mean": mean.item(), "mel_std": std.item()}

def compute_data_statistics(loader: torch.utils.data.DataLoader, out_channels: int):
    total_sum, total_sq_sum, total_len = 0, 0, 0
    for batch in tqdm(loader, leave=False):
        s, sq, l = _stat_accumulate(batch["y"], batch["y_lengths"])
        total_sum += s
        total_sq_sum += sq
        total_len += l
    return _finalize_stats(total_sum, total_sq_sum, total_len, out_channels)

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-config", type=str, default="vctk.yaml", help="The name of the yaml config file under configs/data")
    parser.add_argument("-b", "--batch-size", type=int, default="256", help="Can have increased batch size for faster computation")
    parser.add_argument("-f", "--force", action="store_true", default=False, required=False, help="force overwrite the file")
    return parser.parse_args()

def _prepare_cfg(cfg, args, root_path):
    with open_dict(cfg):
        for k in ["hydra", "_target_"]:
            if k in cfg:
                del cfg[k]
        cfg["data_statistics"] = None
        cfg["seed"] = 1234
        cfg["batch_size"] = args.batch_size
        cfg["train_filelist_path"] = str(os.path.join(root_path, cfg["train_filelist_path"]))
        cfg["valid_filelist_path"] = str(os.path.join(root_path, cfg["valid_filelist_path"]))
        cfg["load_durations"] = False
    return cfg

def main_entry():
    args = _parse_args()
    output_file = Path(args.input_config).with_suffix(".json")
    if os.path.exists(output_file) and not args.force:
        print("File already exists. Use -f to force overwrite")
        sys.exit(1)
    with initialize(version_base="1.3", config_path="../../configs/data"):
        cfg = compose(config_name=args.input_config, return_hydra_config=True, overrides=[])
    root_path = rootutils.find_root(search_from=__file__, indicator=".project-root")
    cfg = _prepare_cfg(cfg, args, root_path)
    datamodule = TextMelDataModule(**cfg)
    datamodule.setup()
    loader = datamodule.train_dataloader()
    _logger.info("Dataloader loaded! Now computing stats...")
    stats = compute_data_statistics(loader, cfg["n_feats"])
    print(stats)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(stats, f)

if __name__ == "__main__":
    main_entry()
