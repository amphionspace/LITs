"""Runtime adapter for the vendored Chinese-English ``TextNormalizer``.

Production deployments should provide ``libtts_normalizer`` and use its stable C
API.  Developer checkouts can use the unified ``tts_cli`` built by
``install_e2e_tn.sh``.  Both backends load the same ``data/<profile>/config.json``
and therefore have identical normalization behavior.
"""

from __future__ import annotations

import atexit
import ctypes
import os
import sys
import threading
from pathlib import Path

from lits.text.cpp_process_wrapper import CPPProcessWrapper


REPO_ROOT = Path(__file__).resolve().parents[2]
TN_ROOT = REPO_ROOT / "frontend"


class TextNormalizerUnavailable(RuntimeError):
    """Raised when no usable unified normalizer runtime can be found."""


class _CapiNormalizer:
    def __init__(self, library_path: Path, resource_dir: Path):
        self.library_path = library_path
        self.resource_dir = resource_dir
        self._lock = threading.Lock()
        self._lib = ctypes.CDLL(str(library_path))
        self._lib.tts_init.argtypes = [ctypes.c_char_p]
        self._lib.tts_init.restype = ctypes.c_void_p
        self._lib.tts_normalize.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self._lib.tts_normalize.restype = ctypes.c_char_p
        self._lib.tts_last_error.argtypes = []
        self._lib.tts_last_error.restype = ctypes.c_char_p
        self._lib.tts_free.argtypes = [ctypes.c_void_p]
        self._lib.tts_free.restype = None

        self._handle = self._lib.tts_init(os.fsencode(resource_dir))
        if not self._handle:
            raw_error = self._lib.tts_last_error()
            error = raw_error.decode("utf-8", errors="replace") if raw_error else "unknown error"
            raise TextNormalizerUnavailable(
                f"tts_init failed for {resource_dir}: {error}"
            )

    def normalize(self, text: str) -> str:
        if not text:
            return ""
        with self._lock:
            if not self._handle:
                raise TextNormalizerUnavailable("normalizer handle is closed")
            raw = self._lib.tts_normalize(self._handle, text.encode("utf-8"))
            if raw is None:
                raise RuntimeError(f"tts_normalize returned NULL for {self.resource_dir}")
            # The C API owns this buffer only until the next call.  Decode/copy it now.
            return raw.decode("utf-8")

    def close(self) -> None:
        with self._lock:
            if self._handle:
                self._lib.tts_free(self._handle)
                self._handle = None


class _CliNormalizer:
    def __init__(self, cli_path: Path, resource_dir: Path):
        self.cli_path = cli_path
        self.resource_dir = resource_dir
        self._proc = CPPProcessWrapper(
            [str(cli_path), "--data", str(resource_dir)],
            env=_tn_subprocess_env(),
            strict_startup=True,
            strict_runtime=True,
            process_name=f"TextNormalizer[{resource_dir.name}]",
        )

    def normalize(self, text: str) -> str:
        return self._proc.communicate(text)

    def close(self) -> None:
        self._proc.close()


_normalizers: dict[tuple[int, str], object] = {}
_normalizers_lock = threading.Lock()
_warned_unavailable: set[str] = set()


def _reset_after_fork() -> None:
    """Drop inherited handles without terminating the parent-owned CLI process."""
    global _normalizers_lock
    _normalizers.clear()
    _normalizers_lock = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)


def _resource_dir(profile: str) -> Path:
    data_root = Path(os.environ.get("TTS_NORMALIZER_DATA_ROOT", TN_ROOT / "data"))
    resource_dir = data_root / profile
    if not (resource_dir / "config.json").is_file():
        raise TextNormalizerUnavailable(
            f"TextNormalizer resource profile not found: {resource_dir}"
        )
    return resource_dir


def _library_candidates() -> list[Path]:
    candidates: list[Path] = []
    explicit = os.environ.get("TTS_NORMALIZER_LIB")
    if explicit:
        candidates.append(Path(explicit))
    extensions = ("dylib", "so") if sys.platform == "darwin" else ("so", "dylib")
    for ext in extensions:
        candidates.extend(
            [
                REPO_ROOT / "e2e_infer" / "lib" / f"libtts_normalizer.{ext}",
                TN_ROOT / "build" / f"libtts_normalizer.{ext}",
            ]
        )
    return candidates


def _cli_candidates() -> list[Path]:
    candidates: list[Path] = []
    explicit = os.environ.get("TTS_NORMALIZER_CLI")
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend(
        [
            REPO_ROOT / "e2e_infer" / "bin" / "tts_cli",
            TN_ROOT / "build" / "tts_cli_so",
            TN_ROOT / "test" / "bin" / "tts_cli",
        ]
    )
    return candidates


def _tn_subprocess_env() -> dict[str, str] | None:
    """ICU-only LD_LIBRARY_PATH for tts_cli children; keep parent clean for CUDA."""
    icu_lib = os.environ.get("E2E_ICU_LIB_DIR", "").strip()
    if not icu_lib:
        icu_root = os.environ.get("ICU_ROOT", "").strip()
        if icu_root:
            icu_lib = str(Path(icu_root) / "lib")
    if not icu_lib or not Path(icu_lib).is_dir():
        return None
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = icu_lib
    return env


def _create_normalizer(profile: str):
    resource_dir = _resource_dir(profile)
    backend = os.environ.get("TTS_NORMALIZER_BACKEND", "auto").strip().lower()
    if backend not in {"auto", "capi", "cli"}:
        raise TextNormalizerUnavailable(
            "TTS_NORMALIZER_BACKEND must be one of: auto, capi, cli"
        )

    errors: list[str] = []
    if backend in {"auto", "capi"}:
        for library_path in _library_candidates():
            if not library_path.is_file():
                continue
            try:
                return _CapiNormalizer(library_path.resolve(), resource_dir.resolve())
            except (OSError, RuntimeError) as exc:
                errors.append(f"C API {library_path}: {exc}")

    if backend in {"auto", "cli"}:
        for cli_path in _cli_candidates():
            if not cli_path.is_file() or not os.access(cli_path, os.X_OK):
                continue
            try:
                return _CliNormalizer(cli_path.resolve(), resource_dir.resolve())
            except (OSError, RuntimeError) as exc:
                errors.append(f"CLI {cli_path}: {exc}")

    detail = "; ".join(errors) if errors else "no library or CLI artifact was found"
    raise TextNormalizerUnavailable(
        f"unified TextNormalizer is unavailable for profile={profile!r}: {detail}. "
        "Run `bash install_e2e_tn.sh`, or set TTS_NORMALIZER_LIB/TTS_NORMALIZER_CLI."
    )


def get_text_normalizer(profile: str, *, required: bool = False):
    """Return a process-local normalizer for ``data/<profile>``.

    A handle/process is cached per profile and serialized internally, so callers
    can safely share it between Python threads.  Forked workers get fresh handles.
    """

    key = (os.getpid(), profile)
    with _normalizers_lock:
        cached = _normalizers.get(key)
        if isinstance(cached, TextNormalizerUnavailable):
            if required:
                raise cached
            return None
        if cached is not None:
            return cached
        try:
            normalizer = _create_normalizer(profile)
        except TextNormalizerUnavailable as exc:
            _normalizers[key] = exc
            if required:
                raise
            if profile not in _warned_unavailable:
                print(f"[TextNormalizer] {exc}; optional caller fallback remains active", file=sys.stderr)
                _warned_unavailable.add(profile)
            return None
        _normalizers[key] = normalizer
        return normalizer


def normalize_text(profile: str, text: str, *, required: bool = False) -> str | None:
    """Normalize one line, returning ``None`` only for an optional unavailable runtime."""

    normalizer = get_text_normalizer(profile, required=required)
    if normalizer is None:
        return None
    return normalizer.normalize(text)


def close_text_normalizers() -> None:
    with _normalizers_lock:
        normalizers = list(_normalizers.values())
        _normalizers.clear()
    for normalizer in normalizers:
        if isinstance(normalizer, TextNormalizerUnavailable):
            continue
        try:
            normalizer.close()
        except Exception:
            pass


atexit.register(close_text_normalizers)
