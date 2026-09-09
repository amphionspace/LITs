#!/usr/bin/env python3
"""Generate JSON with original_line / updated_line for abbrev_detected.txt."""

from __future__ import annotations

import json
import re
from pathlib import Path

INPUT = Path(__file__).with_name("abbrev_detected.txt")
OUTPUT = Path(__file__).with_name("abbrev_detected_updated.json")

_CYR = r"[а-яёА-ЯЁ]"
_SENTENCE_END_PUNCT = ".?!…;:"


def ensure_sentence_ending_punctuation(text: str) -> str:
    """Append a period when the text does not already end with punctuation."""
    text = text.strip()
    if not text:
        return text
    if text[-1] in _SENTENCE_END_PUNCT:
        return text
    return f"{text} ."


def _is_cyr_letter(tok: str) -> bool:
    return len(tok) == 1 and bool(re.fullmatch(_CYR, tok))


def deslash_text(slash_text: str) -> str:
    """Rebuild plain text from slash tokens, joining letter-dot abbreviation chains."""
    tokens = [t.strip() for t in slash_text.split("/") if t.strip()]
    out: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if _is_cyr_letter(tok) and i + 1 < len(tokens) and tokens[i + 1] == ".":
            chunk = tok
            i += 1
            while i < len(tokens) and tokens[i] == ".":
                chunk += "."
                i += 1
                if (
                    i < len(tokens)
                    and _is_cyr_letter(tokens[i])
                    and i + 1 < len(tokens)
                    and tokens[i + 1] == "."
                ):
                    chunk += tokens[i]
                    i += 1
                else:
                    break
            out.append(chunk)
            continue
        out.append(tok)
        i += 1
    return " ".join(out)


def retokenize_with_slashes(text: str) -> str:
    tokens = text.split()
    return " / ".join(tokens)


def capitalize_if_needed(replacement: str, was_capitalized: bool) -> str:
    if not was_capitalized or not replacement:
        return replacement
    return replacement[0].upper() + replacement[1:]


def expand_abbreviations(text: str) -> str:
    """Apply readings from ru_abbrev_reading_table.md."""

    def sub_fixed(pattern: str, repl: str, s: str, *, flags: int = re.IGNORECASE) -> str:
        def _repl(m: re.Match[str]) -> str:
            matched = m.group(0)
            first_alpha = re.search(r"[а-яёА-ЯЁa-zA-Z]", matched)
            cap = bool(first_alpha and first_alpha.group(0).isupper())
            out = repl
            if cap and repl:
                out = repl[0].upper() + repl[1:]
            return out

        return re.sub(pattern, _repl, s, flags=flags)

    # Longer fixed phrases first.
    text = sub_fixed(r"(?<![а-яёА-ЯЁ])до\s+н\.э\.?", "до нашей эры", text)
    text = sub_fixed(r"(?<![а-яёА-ЯЁ])и\s+т\.д\.?", "и так далее", text)
    text = sub_fixed(r"(?<![а-яёА-ЯЁ])и\s+т\.п\.?", "и тому подобное", text)
    text = sub_fixed(r"(?<![а-яёА-ЯЁ])т\.е\.?", "то есть", text)
    text = sub_fixed(r"(?<![а-яёА-ЯЁ])н\.э\.?", "нашей эры", text)

    # Known full-name expansions (with declension where needed).
    text = re.sub(
        r"(?<![а-яёА-ЯЁ])А\.С\.\s+Пушкина",
        "Александра Сергеевича Пушкина",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(?<![а-яёА-ЯЁ])А\.П\.\s+Керн",
        "Анна Петровна Керн",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(?<![а-яёА-ЯЁ])В\.И\.\s+Ленин",
        "Владимир Ильич Ленин",
        text,
        flags=re.IGNORECASE,
    )

    # Letter-name readings when full name is unknown.
    text = re.sub(
        r"(?<![а-яёА-ЯЁ])Д\.Я\.Р\.?",
        "дэ я эр",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(?<![а-яёА-ЯЁ])Г\.П\.?",
        "гэ пэ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(?<![а-яёА-ЯЁ])М\.(?=\s)",
        "эм",
        text,
        flags=re.IGNORECASE,
    )

    return re.sub(r" +", " ", text).strip()


def split_filelist_line(line: str) -> tuple[str, str, str]:
    parts = line.rstrip("\n").rsplit("|", 2)
    if len(parts) != 3:
        raise ValueError(f"Bad filelist line: {line!r}")
    return parts[0], parts[1], parts[2]


def build_updated_line(wav: str, spk_id: str, slash_text: str) -> str:
    plain = deslash_text(slash_text)
    expanded = expand_abbreviations(plain)
    expanded = ensure_sentence_ending_punctuation(expanded)
    retokenized = retokenize_with_slashes(expanded)
    return f"{wav}|{spk_id}|{retokenized}"


def main() -> None:
    rows: list[dict[str, str]] = []
    for raw in INPUT.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        wav, spk_id, text = split_filelist_line(raw)
        rows.append(
            {
                "original_line": raw,
                "updated_line": build_updated_line(wav, spk_id, text),
            }
        )

    OUTPUT.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} entries -> {OUTPUT}")


if __name__ == "__main__":
    main()
