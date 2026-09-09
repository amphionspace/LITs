"""Inspect the deployment path: optional C++ TN -> C++ G2P -> JSON Text2Id."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from infer_e2e import (  # noqa: E402
    DEFAULT_TN_BIN_DIR,
    DEFAULT_TN_DATA_ROOT,
    G2P_PROFILE,
    SUPPORTED_MODEL_LANGS,
    TtsCliEngine,
    _tn_lang_for_line,
)
from lits.runtime.text2id import Text2Id  # noqa: E402


def run_text2id_debug(
    text: str,
    lang: str,
    *,
    with_tn: bool = False,
    tn_primary: str | None = None,
    add_blank: bool = False,
    tn_bin_dir: Path | None = None,
    tn_data_root: Path | None = None,
    text2id_config: Path | None = None,
) -> dict:
    if lang not in SUPPORTED_MODEL_LANGS:
        raise ValueError(f"Unsupported lang={lang!r}; choose from {sorted(SUPPORTED_MODEL_LANGS)}")

    bin_dir = tn_bin_dir or DEFAULT_TN_BIN_DIR
    data_root = tn_data_root or DEFAULT_TN_DATA_ROOT
    mapper = Text2Id(text2id_config or data_root / G2P_PROFILE / "model_tokens.json")
    engine = TtsCliEngine(bin_dir, data_root=data_root)
    try:
        tn_lang = _tn_lang_for_line(text, tn_primary) if with_tn else None
        tn_text = engine.tn_line(text, tn_lang) if tn_lang is not None else text.strip()
        phonemes = engine.g2p_line(tn_text)
        encoded = mapper.encode(phonemes, add_blank=add_blank)
    finally:
        engine.close()

    return {
        "lang": lang,
        "input": text,
        "tn_lang": tn_lang,
        "tn_text": tn_text,
        "phonemes": encoded.phonemes,
        "tokens": encoded.phonemes.split(),
        "token_ids": encoded.token_ids,
        "tone_ids": encoded.tone_ids,
        "add_blank": add_blank,
        "text2id_config": str(mapper.config_path),
    }


def _print_result(result: dict) -> None:
    print(f"Lang:       {result['lang']}")
    print(f"Input:      {result['input']}")
    print(f"TN lang:    {result['tn_lang'] or 'skipped'}")
    print(f"TN text:    {result['tn_text']}")
    print(f"Phonemes:   {result['phonemes']}")
    print(f"Token IDs:  {result['token_ids']}")
    print(f"Tone IDs:   {result['tone_ids']}")
    print(f"Length:     {len(result['token_ids'])}")
    print(f"Contract:   {result['text2id_config']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="C++ TN/G2P -> JSON Text2Id deployment-path debugger"
    )
    parser.add_argument("text")
    parser.add_argument("--lang", default="en-zh-dict", choices=sorted(SUPPORTED_MODEL_LANGS))
    parser.add_argument("--with-tn", action="store_true")
    parser.add_argument("--tn-primary", choices=["zh", "en"])
    parser.add_argument("--add-blank", action="store_true")
    parser.add_argument("--tn-bin-dir", type=Path, default=None)
    parser.add_argument("--tn-data-root", type=Path, default=None)
    parser.add_argument("--text2id-config", type=Path, default=None)
    args = parser.parse_args(argv)

    _print_result(
        run_text2id_debug(
            args.text,
            args.lang,
            with_tn=args.with_tn,
            tn_primary=args.tn_primary,
            add_blank=args.add_blank,
            tn_bin_dir=args.tn_bin_dir,
            tn_data_root=args.tn_data_root,
            text2id_config=args.text2id_config,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
