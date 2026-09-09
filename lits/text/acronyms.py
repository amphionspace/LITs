import re
import string
from pathlib import Path

import zhon


_ACRONYM_PATH = Path(__file__).resolve().parent / "sources" / "common_acronyms.txt"

_UPPERCASE_LETTER_PHONEMES_STRESSED = {
    "A": "ˈeɪ",
    "B": "b|ˈiː",
    "C": "s|ˈiː",
    "D": "d|ˈiː",
    "E": "ˈiː",
    "F": "ˈɛ|f",
    "G": "dʒ|ˈiː",
    "H": "ˈeɪ|tʃ",
    "I": "ˈaɪ",
    "J": "dʒ|ˈeɪ",
    "K": "k|ˈeɪ",
    "L": "ˈɛ|l",
    "M": "ˈɛ|m",
    "N": "ˈɛ|n",
    "O": "ˈoʊ",
    "P": "p|ˈiː",
    "Q": "k|j|ˈuː",
    "R": "ˈɑ|ɹ",
    "S": "ˈɛ|s",
    "T": "t|ˈiː",
    "U": "j|ˈuː",
    "V": "v|ˈiː",
    "W": "d|ˈʌ|b|ə|l|j|ˈuː",
    "X": "ˈɛ|k|s",
    "Y": "w|ˈaɪ",
    "Z": "z|ˈiː",
}

_NUMBERS_0to9_PHONEMES_STRESSED = {
    "0": "z|ˈiə|ɹ|oʊ",
    "1": "w|ˈʌ|n",
    "2": "t|ˈuː",
    "3": "θ|ɹ|ˈiː",
    "4": "f|ˈoːɹ",
    "5": "f|ˈaɪ|v",
    "6": "s|ˈɪ|k|s",
    "7": "s|ˈɛ|v|ə|n",
    "8": "ˈeɪ|t",
    "9": "n|ˈaɪ|n",
}

def strip_edge_punctuation(text):
    if not isinstance(text, str):
        return ""
    return text.strip(zhon.hanzi.punctuation + string.punctuation)


def normalize_acronym_key(text):
    return re.sub(r"\s+", "", strip_edge_punctuation(text)).upper()


def _load_common_acronyms():
    acronyms = set()
    try:
        with open(_ACRONYM_PATH, "r", encoding="utf-8") as readf:
            for line in readf:
                token = line.split("#", 1)[0].strip()
                if not token:
                    continue
                acronyms.add(normalize_acronym_key(token))
    except Exception:
        pass
    return acronyms


_common_acronyms = _load_common_acronyms()


def load_common_acronyms():
    return _common_acronyms


def should_read_as_acronym(text):
    if not isinstance(text, str):
        return False

    key = normalize_acronym_key(text)
    if not key:
        return False

    try:
        from lits.text.frontend_rules.ops.acronym_case import get_case_pronunciation_mode

        case_mode = get_case_pronunciation_mode(text)
        if case_mode == "spell":
            return True
        if case_mode == "word":
            return False
    except Exception:
        pass

    return any(c.isupper() for c in text) and key in _common_acronyms


def spell_acronym_token(text):
    parts = []
    idx = 0
    while idx < len(text):
        if text[idx].isdigit():
            end = idx + 1
            while end < len(text) and text[end].isdigit():
                end += 1
            parts.append(text[idx:end])
            idx = end
            continue
        if not text[idx].isspace():
            parts.append(text[idx])
        idx += 1
    return " ".join(parts)


def preprocess_common_acronyms(text):
    if not isinstance(text, str):
        return text

    words = text.split(" ")
    new_text = ""
    for word in words:
        if should_read_as_acronym(word):
            new_text += spell_acronym_token(word) + " "
        else:
            new_text += word + " "
    return new_text.strip(" ")


def get_letter_phonemes():
    return _UPPERCASE_LETTER_PHONEMES_STRESSED


def get_number_phonemes():
    return _NUMBERS_0to9_PHONEMES_STRESSED


def direct_acronym_phonemes(
    text,
    text_tokenizer=None,
):
    if not isinstance(text, str):
        return None

    stripped = text.strip()
    if not stripped:
        return None

    match = re.fullmatch(r"([A-Z0-9](?:\s+[A-Z0-9]){1,}|[A-Z0-9]*[A-Z][A-Z0-9]*)([,\.\?!;:\'…]?)", stripped)
    if not match:
        return None

    acronym = re.sub(r"\s+", "", match.group(1))
    if not should_read_as_acronym(acronym):
        return None

    punctuation = match.group(2)
    letter_phonemes = get_letter_phonemes()
    number_phonemes = get_number_phonemes()
    phoneme_parts = []
    idx = 0
    while idx < len(acronym):
        char = acronym[idx]
        if char in letter_phonemes:
            phoneme_parts.extend(letter_phonemes[char].split("|"))
            idx += 1
        elif char in number_phonemes:
            end = idx + 1
            while end < len(acronym) and acronym[end].isdigit():
                end += 1
            number = acronym[idx:end]
            if len(number) > 1 and text_tokenizer is not None:
                phoneme_parts.extend([part for part in text_tokenizer(number).split("|") if part and part != "_"])
            else:
                for digit in number:
                    phoneme_parts.extend(number_phonemes[digit].split("|"))
            idx = end
        else:
            return None

    if punctuation:
        phoneme_parts.append(punctuation)
    return "|".join(phoneme_parts)


def split_leading_acronym(text):
    if not isinstance(text, str):
        return None, None

    stripped = text.strip()
    if not stripped:
        return None, None

    match = re.fullmatch(r"([A-Z0-9](?:\s+[A-Z0-9]){1,}|[A-Z0-9]*[A-Z][A-Z0-9]*)(?:\s+(.*))?", stripped)
    if not match:
        return None, None

    acronym = re.sub(r"\s+", "", match.group(1))
    if not should_read_as_acronym(acronym):
        return None, None

    remainder = (match.group(2) or "").strip()
    return acronym, remainder


def segment_english_text(text):
    if not isinstance(text, str):
        return []

    stripped = text.strip()
    if not stripped:
        return []

    pattern = re.compile(r"(?<![A-Za-z0-9])([A-Z0-9](?:\s+[A-Z0-9]){1,}|[A-Z0-9]*[A-Z][A-Z0-9]*)([,\.\?!;:\'…]?)(?![A-Za-z0-9])")
    segments = []
    cursor = 0

    for match in pattern.finditer(stripped):
        candidate = (match.group(1) + (match.group(2) or "")).strip()
        if not should_read_as_acronym(candidate):
            continue

        start, end = match.span()
        if start > cursor:
            plain = stripped[cursor:start].strip()
            if plain:
                segments.append(("plain", plain))

        segments.append(("acronym", candidate))
        cursor = end

    if cursor < len(stripped):
        plain = stripped[cursor:].strip()
        if plain:
            segments.append(("plain", plain))

    if not segments:
        return [("plain", stripped)]
    return segments
