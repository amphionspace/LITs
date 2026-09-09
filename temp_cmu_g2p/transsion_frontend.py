"""Transsion internal HTTP G2P frontend (reachable only on corp network).

Falls back to local CMUdict when unavailable; see ``g2p_backend.py``.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DEFAULT_FRONTEND_URL = "http://10.150.224.54:22222"

MONOPHTHONGS = ["AO", "AA", "IY", "UW", "EH", "IH", "UH", "AH", "AX", "AE"]
DIPHTHONGS = ["EY", "AY", "OW", "AW", "OY"]
RCOLORED = ["ER", "AXR", "EHR", "UHR", "AOR", "AAR", "IHR", "IYR", "AWR"]
VOWELS = MONOPHTHONGS + DIPHTHONGS + RCOLORED

_ARPA_LANGS_DIR = Path(__file__).resolve().parents[1] / "lits" / "text" / "char_symbols" / "langs"
if str(_ARPA_LANGS_DIR) not in sys.path:
    sys.path.insert(0, str(_ARPA_LANGS_DIR))
from ARPA import ARPA_TOKENS  # noqa: E402

ARPA_PHONE_SET = set(ARPA_TOKENS)
ARPA_BOUNDARY_TOKENS = {"/"}
ARPA_PUNCT_TOKENS = {",", ".", "!", "?", ";", ":", "'", '"', "-", "(", ")"}

_ENGLISH_WORD_RE = re.compile(r"[A-Za-z]+(?:['''-][A-Za-z]+)*")
_NON_ENGLISH_SCRIPT_RE = re.compile(
    r"[\u4e00-\u9fff\u0600-\u06ff\u0980-\u09ff\u0400-\u04ff\u0300-\u0301]"
)


def request_frontend(
    frontend_url: str,
    text: str,
    *,
    lang: str = "en-US-LjNeutral",
    timeout: float = 10.0,
) -> list[dict]:
    payload = urllib.parse.urlencode({"name": "1", "text": text, "lang": lang}).encode("utf-8")
    req = urllib.request.Request(frontend_url, data=payload, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
    data = json.loads(raw)
    return data["tobiJson"]


def is_frontend_available(
    frontend_url: str = DEFAULT_FRONTEND_URL,
    *,
    timeout: float = 3.0,
) -> bool:
    """Return True if the Transsion G2P frontend responds to a probe request."""
    try:
        request_frontend(frontend_url, ".", lang="en-US-LjNeutral", timeout=timeout)
        return True
    except Exception:
        return False


def has_raw_english_words(text: str) -> bool:
    return bool(_ENGLISH_WORD_RE.search(text))


def iter_english_word_segments(text: str):
    i = 0
    n = len(text)
    while i < n:
        if re.match(r"[A-Za-z]", text[i]):
            match = _ENGLISH_WORD_RE.match(text, i)
            if match:
                yield match.group(), True
                i = match.end()
                continue
        j = i + 1
        while j < n and not re.match(r"[A-Za-z]", text[j]):
            j += 1
        yield text[i:j], False
        i = j


def get_arpa_seq(
    text: str,
    *,
    frontend_url: str = DEFAULT_FRONTEND_URL,
    lang: str = "en",
    fragment: bool = False,
    timeout: float = 10.0,
) -> str:
    """Convert text to ARPAbet via Transsion frontend HTTP API."""
    res = request_frontend(frontend_url, text, lang=lang, timeout=timeout)
    seq: list[str] = []
    for sent in res:
        for item in sent["unit_list"]:
            if item["unit_type"] == "speech":
                syllables = item["chunk_info"]["unit_info"]["syllable_list"]
                for syl in syllables:
                    stress = syl["stress"]
                    try:
                        int_stress = int(stress)
                        if int_stress not in (0, 1, 2):
                            continue
                    except TypeError:
                        continue
                    phns = []
                    for phn in syl["phone_list"]:
                        if phn["text"].upper() in VOWELS:
                            phns.append(f"{phn['text']}{stress}")
                        else:
                            phns.append(f"{phn['text']}")
                    seq.extend(phns)
                seq.append("/")
            elif item["unit_type"] == "punctuation":
                if fragment:
                    continue
                punc = item["chunk_info"]["orth"]
                seq.extend([punc, "/"])
    if fragment:
        while seq and seq[-1] == "/":
            seq.pop()
        return " ".join(seq)
    return " ".join(seq[:-1]) if seq else ""


def convert_mixed_text_to_arpa(
    text: str,
    *,
    frontend_url: str = DEFAULT_FRONTEND_URL,
    lang: str = "en",
    timeout: float = 10.0,
) -> str:
    """Convert English words to ARPAbet; leave zh/ar/bn/ru segments unchanged."""
    if not has_raw_english_words(text):
        return text

    if not _NON_ENGLISH_SCRIPT_RE.search(text):
        return get_arpa_seq(text, frontend_url=frontend_url, lang=lang, timeout=timeout)

    parts: list[str] = []
    for segment, is_english in iter_english_word_segments(text):
        if is_english:
            arpa = get_arpa_seq(
                segment,
                frontend_url=frontend_url,
                lang=lang,
                fragment=True,
                timeout=timeout,
            )
            parts.append(f" / {arpa} / ")
        else:
            parts.append(segment)
    return "".join(parts).strip()
