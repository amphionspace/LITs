"""Vocos mel-to-waveform vocoder loader for LITS inference."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import torch
import yaml
from torch import nn

class AttrDict(dict):
    """Dict with attribute access (replaces removed hifigan.env.AttrDict)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_VOCOS_ROOT = REPO_ROOT
DEFAULT_VOCOS_CHECKPOINT = REPO_ROOT / "generator.ckpt"

class VocosMelVocoder(nn.Module):
    """HiFiGAN-compatible interface: mel (B, C, T) -> waveform (B, 1, T)."""

    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.backbone(mel)
        audio = self.head(x)
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        return audio


def _ensure_vocos_importable(vocos_root: Path) -> None:
    vocos_root = vocos_root.resolve()
    if not vocos_root.is_dir():
        raise FileNotFoundError(f"Vocos package root not found: {vocos_root}")
    root_str = str(vocos_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _prepare_vocos_imports(vocos_root: Path) -> None:
    """Import Vocos submodules without pulling optional HF/Encodec deps from vocos/__init__.py."""
    _ensure_vocos_importable(vocos_root)

    if "encodec" not in sys.modules:
        encodec_mod = types.ModuleType("encodec")

        class _DummyEncodecModel:
            @staticmethod
            def encodec_model_24khz(*_args, **_kwargs):
                raise NotImplementedError("Encodec is not used by the mel vocoder path")

            @staticmethod
            def encodec_model_48khz(*_args, **_kwargs):
                raise NotImplementedError("Encodec is not used by the mel vocoder path")

        encodec_mod.EncodecModel = _DummyEncodecModel
        sys.modules["encodec"] = encodec_mod

    vocos_pkg_dir = vocos_root / "vocos"
    if "vocos" not in sys.modules:
        pkg = types.ModuleType("vocos")
        pkg.__path__ = [str(vocos_pkg_dir)]
        sys.modules["vocos"] = pkg


def _read_vocos_config(ckpt_path: Path) -> tuple[dict, dict, dict, int]:
    config_candidates = (
        ckpt_path.parent / "config.yaml",
        ckpt_path.parent.parent / "config.yaml",
    )
    config_path = next((p for p in config_candidates if p.exists()), None)

    if config_path is not None:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        fe_args = cfg["model"]["init_args"]["feature_extractor"]["init_args"]
        bb_args = cfg["model"]["init_args"]["backbone"]["init_args"]
        hd_args = cfg["model"]["init_args"]["head"]["init_args"]
        sample_rate = int(cfg["model"]["init_args"]["sample_rate"])
    else:
        fe_args = dict(
            sample_rate=24000,
            n_fft=2048,
            hop_length=384,
            win_length=1536,
            n_mels=100,
            fmin=0.0,
            fmax=12000.0,
        )
        bb_args = dict(input_channels=100, dim=512, intermediate_dim=1024, num_layers=8)
        hd_args = dict(dim=512, n_fft=2048, hop_length=384, win_length=1536, padding="same")
        sample_rate = 24000

    return fe_args, bb_args, hd_args, sample_rate


def _load_generator_modules(ckpt_path: Path, device: torch.device):
    from vocos.heads import ISTFTHead
    from vocos.models import VocosBackbone

    fe_args, bb_args, hd_args, sample_rate = _read_vocos_config(ckpt_path)
    backbone = VocosBackbone(**bb_args)
    head = ISTFTHead(**hd_args)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    bb_state = {k.removeprefix("backbone."): v for k, v in state.items() if k.startswith("backbone.")}
    hd_state = {k.removeprefix("head."): v for k, v in state.items() if k.startswith("head.")}
    if not bb_state or not hd_state:
        raise ValueError(
            f"No generator weights found in {ckpt_path}; expected backbone.* and head.* keys"
        )

    backbone.load_state_dict(bb_state, strict=True)
    head.load_state_dict(hd_state, strict=True)
    sample_rate = int(ckpt.get("hyper_parameters", {}).get("sample_rate", sample_rate))
    return backbone, head, fe_args, sample_rate


def load_vocos_vocoder(
    checkpoint_path: str | None = None,
    device: torch.device | None = None,
    vocos_root: str | Path | None = None,
):
    """Load a Vocos checkpoint for mel-conditioned synthesis (backbone + ISTFT head)."""
    vocos_root = Path(vocos_root or DEFAULT_VOCOS_ROOT)
    _prepare_vocos_imports(vocos_root)

    ckpt_path = Path(checkpoint_path or DEFAULT_VOCOS_CHECKPOINT).resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Vocos checkpoint not found: {ckpt_path}")

    if device is None:
        device = torch.device("cpu")

    backbone, head, fe_args, sample_rate = _load_generator_modules(ckpt_path, device)
    vocoder = VocosMelVocoder(backbone, head).to(device).eval()

    h = AttrDict(
        {
            "num_mels": int(fe_args["n_mels"]),
            "sampling_rate": sample_rate,
            "hop_size": int(fe_args["hop_length"]),
        }
    )
    return vocoder, h
