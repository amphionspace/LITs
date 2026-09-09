"""English G2P backend selection: Transsion HTTP frontend vs local CMUdict."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_PKG_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_DIR.parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

from transsion_frontend import (  # noqa: E402
    DEFAULT_FRONTEND_URL,
    convert_mixed_text_to_arpa,
    is_frontend_available,
)

G2PBackendName = Literal["auto", "cmudict", "transsion"]


def default_g2p_frontend_url() -> str:
    """Transsion G2P frontend URL: env override, else built-in corp default."""
    return (
        os.environ.get("G2P_FRONTEND_URL")
        or os.environ.get("EN_G2P_FRONTEND_URL")
        or DEFAULT_FRONTEND_URL
    )


def default_g2p_backend_name() -> str:
    """Default English G2P mode for CLI entry points.

    ``G2P_BACKEND`` env overrides the built-in default (``cmudict``).
    If you set ``G2P_BACKEND=auto`` in shell profile, Transsion frontend
    will be used on corp networks even when script defaults say cmudict.
    """
    return os.environ.get("G2P_BACKEND", "cmudict").lower()


@dataclass
class G2PBackendConfig:
    backend: G2PBackendName = "auto"
    frontend_url: str = ""
    frontend_lang: str = "en"
    probe_timeout: float = 3.0
    request_timeout: float = 10.0
    transsion_available: bool | None = None
    resolved_backend: str | None = None
    logged: bool = False

    def __post_init__(self) -> None:
        if not self.frontend_url:
            self.frontend_url = default_g2p_frontend_url()


_config = G2PBackendConfig(
    backend="cmudict",  # type: ignore[arg-type]
)


def configure_g2p_backend(
    *,
    backend: G2PBackendName | str | None = None,
    frontend_url: str | None = None,
    frontend_lang: str | None = None,
    probe_timeout: float | None = None,
    request_timeout: float | None = None,
) -> None:
    """Set G2P routing for the current process (call once at inference startup)."""
    global _config
    if backend is not None:
        _config.backend = backend.lower()  # type: ignore[assignment]
    if frontend_url is not None:
        _config.frontend_url = frontend_url
    if frontend_lang is not None:
        _config.frontend_lang = frontend_lang
    if probe_timeout is not None:
        _config.probe_timeout = probe_timeout
    if request_timeout is not None:
        _config.request_timeout = request_timeout
    _config.transsion_available = None
    _config.resolved_backend = None
    _config.logged = False


def _probe_transsion() -> bool:
    if _config.transsion_available is None:
        _config.transsion_available = is_frontend_available(
            _config.frontend_url,
            timeout=_config.probe_timeout,
        )
    return _config.transsion_available


def get_resolved_backend(*, force_probe: bool = False) -> str:
    """Return ``transsion`` or ``cmudict`` for this process."""
    if force_probe:
        _config.transsion_available = None
        _config.resolved_backend = None

    if _config.resolved_backend is not None:
        return _config.resolved_backend

    mode = _config.backend
    if mode == "cmudict":
        _config.resolved_backend = "cmudict"
    elif mode == "transsion":
        _config.resolved_backend = "transsion" if _probe_transsion() else "cmudict"
    else:
        _config.resolved_backend = "transsion" if _probe_transsion() else "cmudict"

    return _config.resolved_backend


def log_g2p_backend_status() -> None:
    if _config.logged:
        return
    resolved = get_resolved_backend()
    env_backend = os.environ.get("G2P_BACKEND")
    config_line = f"[g2p] requested={_config.backend!r} resolved={resolved!r}"
    if env_backend is not None:
        config_line += f" (env G2P_BACKEND={env_backend!r})"
    print(config_line)
    if resolved == "transsion":
        print(
            f"[g2p] Using Transsion frontend: {_config.frontend_url} "
            f"(lang={_config.frontend_lang})"
        )
    else:
        if _config.backend == "transsion":
            print(
                f"[g2p] Transsion frontend unreachable ({_config.frontend_url}); "
                "falling back to local CMUdict",
                file=sys.stderr,
            )
        elif _config.backend == "auto":
            print(
                f"[g2p] Transsion frontend not reachable ({_config.frontend_url}); "
                "using local CMUdict",
            )
        else:
            print("[g2p] Using local CMUdict + supplement lexicon")
    _config.logged = True


def preprocess_via_transsion(text: str) -> str:
    from english_frontend import has_raw_english_words, is_arpabet_input, normalize_arpabet_input

    text = text.strip()
    if is_arpabet_input(text):
        return normalize_arpabet_input(text)
    if not has_raw_english_words(text):
        return text
    return convert_mixed_text_to_arpa(
        text,
        frontend_url=_config.frontend_url,
        lang=_config.frontend_lang,
        timeout=_config.request_timeout,
    )


def preprocess_via_cmudict(text: str, g2p=None) -> str:
    from english_frontend import (
        _normalize_english_glue_tokens,
        convert_mixed_text_to_slash_arpa,
        get_default_g2p,
        has_raw_english_words,
        is_arpabet_input,
        normalize_arpabet_input,
    )

    text = text.strip()
    text = _normalize_english_glue_tokens(text)
    engine = g2p or get_default_g2p()
    if is_arpabet_input(text):
        return normalize_arpabet_input(text)
    if has_raw_english_words(text):
        return convert_mixed_text_to_slash_arpa(text, engine)
    return text


def preprocess_english_with_backend(text: str, g2p=None) -> str:
    """Route English G2P through Transsion frontend or local CMUdict."""
    if not text or not text.strip():
        return text
    if get_resolved_backend() == "transsion":
        return preprocess_via_transsion(text)
    return preprocess_via_cmudict(text, g2p)
