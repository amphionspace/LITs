"""Resolve mu / teacher-decoder / student-decoder streaming flags for distillation."""

from __future__ import annotations

import argparse
import warnings


def add_streaming_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Legacy: enable streaming for mu encoder, teacher decoder, and student decoder.",
    )
    parser.add_argument(
        "--mu-streaming",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Mu encoder streaming during distillation (default false = bidirectional, "
        "matches bidirectional_mu inference and teacher offline quality).",
    )
    parser.add_argument(
        "--teacher-decoder-streaming",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Legacy flag for full-sequence teacher ODE when --no-kv-cache-distill. "
        "When kv_cache_distill is enabled, teacher targets always use the chunked "
        "KV-cache path (decoder_left_frames, chunk_size) matching deployment.",
    )
    parser.add_argument(
        "--decoder-streaming",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Student decoder causal mask during distillation "
        "(default true; required for kv_cache_distill).",
    )


def resolve_streaming_flags(args) -> None:
    """Populate mu/teacher_decoder/decoder streaming flags on *args* in place."""
    if getattr(args, "streaming", False):
        if args.mu_streaming is None:
            args.mu_streaming = True
        if args.teacher_decoder_streaming is None:
            args.teacher_decoder_streaming = True
        if args.decoder_streaming is None:
            args.decoder_streaming = True

    if args.mu_streaming is None:
        args.mu_streaming = False
    if args.decoder_streaming is None:
        args.decoder_streaming = True
    if args.teacher_decoder_streaming is None:
        args.teacher_decoder_streaming = bool(getattr(args, "kv_cache_distill", True))


def warn_streaming_config(args) -> None:
    if args.mu_streaming and not getattr(args, "streaming", False):
        warnings.warn(
            "mu_streaming=true: mu encoder uses causal chunks during distillation, "
            "but bidirectional_mu inference uses streaming=False for mu. "
            "Prefer mu_streaming=false unless you also change inference.",
            stacklevel=2,
        )
    if getattr(args, "kv_cache_distill", True) and args.decoder_streaming:
        if args.teacher_decoder_streaming is False:
            warnings.warn(
                "teacher_decoder_streaming=false is ignored when kv_cache_distill=true: "
                "teacher targets use the same chunked KV-cache path as inference.",
                stacklevel=2,
            )
    elif not getattr(args, "kv_cache_distill", True):
        warnings.warn(
            "kv_cache_distill=false: teacher/student ODE paths may not match "
            "causal_kv deployment inference. Prefer kv_cache_distill=true.",
            stacklevel=2,
        )


def streaming_config_summary(args) -> str:
    kv = getattr(args, "kv_cache_distill", True)
    teacher_mode = "kv_cache" if kv and args.decoder_streaming else (
        f"full_seq(streaming={args.teacher_decoder_streaming})"
    )
    return (
        f"mu_streaming={args.mu_streaming} "
        f"teacher_decoder={teacher_mode} "
        f"decoder_streaming={args.decoder_streaming} "
        f"kv_cache_distill={kv}"
    )
