"""Debug per-token predicted durations (same path as inference_stream / LITS.synthesise)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from lits.models.lits import LITS
from lits.text import text_to_sequence_with_tones
from lits.text.char_symbols.symbol_inventories import lang2inventory
from lits.utils.infer_duration_floor import apply_infer_duration_patches

LANG2CLEANER = {
    "en-zh-dict": "en_zh_dict_mixed_rhyme_body_tone_cleaners",
    "en-zh": "pinyin_direct_mixed_rhyme_body_tone_cleaners",
    "en-zh-dict-rhyme-body-tone": "en_zh_dict_mixed_rhyme_body_tone_cleaners",
    "en-zh-rhyme-body-tone": "pinyin_direct_mixed_rhyme_body_tone_cleaners",
}

# Vocos default hop in this repo (24 kHz).
DEFAULT_MEL_HOP = 384
DEFAULT_SAMPLE_RATE = 24000


def _symbols_for_vocab(n_vocab: int) -> list[str]:
    for payload in lang2inventory.values():
        symbols = payload.get("symbols")
        if symbols and len(symbols) == n_vocab:
            return list(symbols)
    return []


def predict_token_durations(
    text: str,
    *,
    cleaner: str,
    checkpoint: str | Path,
    spk_id: int = 0,
    length_scale: float = 1.0,
    device: str | torch.device = "cpu",
    apply_patches: bool = True,
    lang: str | None = None,
) -> dict:
    token_ids, tone_ids, cleaned = text_to_sequence_with_tones(text, [cleaner])
    if not token_ids:
        raise ValueError("Empty token sequence after cleaning.")

    model = LITS.load_from_checkpoint(str(checkpoint), map_location=device)
    model.eval()
    model_has_tone_emb = getattr(model, "n_tones", 0) > 0
    if model_has_tone_emb != (tone_ids is not None):
        mode_hint = lang or cleaner
        raise ValueError(
            f"Checkpoint/cleaner mismatch: model n_tones={getattr(model, 'n_tones', 0)} "
            f"but lang/cleaner={mode_hint!r} "
            f"{'requires' if model_has_tone_emb else 'does not use'} tone ids. "
            "Rhyme-body-tone checkpoints need en-zh / en-zh-dict modes."
        )
    symbols = _symbols_for_vocab(model.n_vocab)

    x = torch.tensor([token_ids], dtype=torch.long, device=device)
    x_lengths = torch.tensor([len(token_ids)], dtype=torch.long, device=device)
    x_tones = None
    if tone_ids is not None:
        x_tones = torch.tensor([tone_ids], dtype=torch.long, device=device)
    spk_emb = None
    if model.n_spks > 1:
        spks = torch.tensor([spk_id], dtype=torch.long, device=device)
        spk_emb = model.spk_emb(spks)

    with torch.no_grad():
        _mu_x, logw, x_mask = model.encoder(x, x_lengths, spk_emb, x_tones=x_tones)
        w = torch.exp(logw) * x_mask
        w_raw = torch.ceil(w) * length_scale
        w_final = (
            apply_infer_duration_patches(w_raw.clone(), x, x_mask, n_vocab=model.n_vocab)
            if apply_patches
            else w_raw
        )

    raw = w_raw.squeeze().tolist()
    final = w_final.squeeze().tolist()
    logws = logw.squeeze().tolist()
    weights = w.squeeze().tolist()

    rows: list[dict] = []
    frame = 0.0
    tone_id_list = tone_ids if tone_ids is not None else [0] * len(token_ids)
    for idx, token_id in enumerate(token_ids):
        dur = float(final[idx])
        rows.append(
            {
                "index": idx,
                "token": symbols[token_id] if token_id < len(symbols) else f"<id:{token_id}>",
                "token_id": token_id,
                "tone_id": int(tone_id_list[idx]),
                "logw": float(logws[idx]),
                "w": float(weights[idx]),
                "raw_frames": float(raw[idx]),
                "frames": dur,
                "frame_start": frame,
                "frame_end": frame + dur,
            }
        )
        frame += dur

    return {
        "text": text,
        "lang": lang,
        "cleaner": cleaner,
        "cleaned": cleaned,
        "token_ids": token_ids,
        "tone_ids": tone_ids,
        "rows": rows,
        "total_frames": frame,
        "approx_sec": frame * DEFAULT_MEL_HOP / DEFAULT_SAMPLE_RATE,
        "apply_patches": apply_patches,
        "length_scale": length_scale,
        "checkpoint": str(checkpoint),
        "spk_id": spk_id,
    }


def format_duration_report(result: dict, *, show_raw: bool = True) -> str:
    lines = [
        f"Input:      {result['text']}",
    ]
    if result.get("lang"):
        lines.append(f"Lang:       {result['lang']}")
    lines.extend(
        [
        f"Cleaner:    {result['cleaner']}",
        f"Checkpoint: {result['checkpoint']}",
        f"Cleaned:    {result['cleaned']}",
        f"Patches:    {result['apply_patches']}",
        f"LengthScale:{result['length_scale']}",
        "",
        ]
    )
    show_tones = result.get("tone_ids") is not None
    if show_raw:
        header = (
            f"{'idx':>3}  {'token':^6}  {'id':>3}  "
            + (f"{'tone':>4}  " if show_tones else "")
            + f"{'logw':>7}  {'w':>8}  {'raw':>5}  {'final':>5}  {'start':>6}  {'end':>6}"
        )
        lines.append(header)
        lines.append("-" * (68 if show_tones else 62))
        for row in result["rows"]:
            tone_col = f"{row['tone_id']:4d}  " if show_tones else ""
            lines.append(
                f"{row['index']:3d}  {row['token']:^6}  {row['token_id']:3d}  "
                f"{tone_col}"
                f"{row['logw']:7.3f}  {row['w']:8.4f}  "
                f"{row['raw_frames']:5.0f}  {row['frames']:5.0f}  "
                f"{row['frame_start']:6.0f}  {row['frame_end']:6.0f}"
            )
    else:
        header = (
            f"{'idx':>3}  {'token':^6}  "
            + (f"{'tone':>4}  " if show_tones else "")
            + f"{'frames':>5}  {'start':>6}  {'end':>6}"
        )
        lines.append(header)
        lines.append("-" * (42 if show_tones else 36))
        for row in result["rows"]:
            tone_col = f"{row['tone_id']:4d}  " if show_tones else ""
            lines.append(
                f"{row['index']:3d}  {row['token']:^6}  "
                f"{tone_col}"
                f"{row['frames']:5.0f}  "
                f"{row['frame_start']:6.0f}  {row['frame_end']:6.0f}"
            )

    lines.extend(
        [
            "",
            f"Total mel frames: {result['total_frames']:.0f}",
            f"Approx duration:  {result['approx_sec']:.3f}s "
            f"(hop={DEFAULT_MEL_HOP}, sr={DEFAULT_SAMPLE_RATE})",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print per-token duration (mel frames) predicted by the LITS duration model.",
    )
    parser.add_argument("text", help="Input text (same as inference / text2id_debug.sh)")
    parser.add_argument(
        "--lang",
        default="en-zh-dict",
        choices=sorted(LANG2CLEANER),
        help="Model language / cleaner mapping (default: en-zh-dict)",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="External LITS checkpoint",
    )
    parser.add_argument("--spk-id", type=int, default=0, help="Speaker id (default: 0)")
    parser.add_argument(
        "--length-scale",
        type=float,
        default=1.0,
        help="Same as inference_stream --length_scale (default: 1.0)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (default: cuda if available else cpu)",
    )
    parser.add_argument(
        "--no-patches",
        action="store_true",
        help="Skip infer_duration_floor patches (show raw ceil durations only)",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Hide logw / raw columns",
    )
    args = parser.parse_args(argv)

    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        parser.error(f"Checkpoint not found: {ckpt}")

    cleaner = LANG2CLEANER[args.lang]
    result = predict_token_durations(
        args.text,
        cleaner=cleaner,
        checkpoint=ckpt,
        spk_id=args.spk_id,
        length_scale=args.length_scale,
        device=args.device,
        apply_patches=not args.no_patches,
        lang=args.lang,
    )
    print(format_duration_report(result, show_raw=not args.compact))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
