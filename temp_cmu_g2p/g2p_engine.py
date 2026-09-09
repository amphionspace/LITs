"""CMUdict + optional supplement lexicon G2P engine."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from cmudict_loader import CMUDict, DEFAULT_CMUDICT_PATH

_PKG_ROOT = Path(__file__).resolve().parent
DEFAULT_SUPPLEMENT_PATH = _PKG_ROOT / "data" / "supplement_lexicon.json"

# Latin letters incl. common name diacritics (é, ö, ñ, ç, …).
_LATIN_WORD_CHAR = r"A-Za-z\u00C0-\u024F"
_ENGLISH_TOKEN_RE = re.compile(
    rf"[{_LATIN_WORD_CHAR}]+(?:['''-][{_LATIN_WORD_CHAR}]+)*|"
    r"\.\.\.|"
    r"[,.\?!;:\"'()—\-]"
)
_ENGLISH_WORD_SURFACE_RE = re.compile(
    rf"[{_LATIN_WORD_CHAR}]+(?:[''-][{_LATIN_WORD_CHAR}]+)*"
)

# PascalCase segments: ShadowWarrior -> Shadow, Warrior
_PASCAL_CASE_PART_RE = re.compile(r"[A-Z][a-z]+")


class SupplementLexicon:
    def __init__(self, entries: dict[str, list[str]]) -> None:
        self._entries = {k.upper(): v for k, v in entries.items()}

    @staticmethod
    def _ascii_fold_key(word: str) -> str:
        folded = unicodedata.normalize("NFKD", word.strip())
        folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
        return folded.upper()

    @classmethod
    def from_json(cls, path: str | Path) -> SupplementLexicon:
        path = Path(path)
        if not path.is_file():
            return cls({})
        data = json.loads(path.read_text(encoding="utf-8"))
        raw_entries = data.get("entries", data)
        parsed: dict[str, list[str]] = {}
        for key, value in raw_entries.items():
            if isinstance(value, dict) and "phones" in value:
                parsed[key] = list(value["phones"])
            elif isinstance(value, list):
                parsed[key] = list(value)
            elif isinstance(value, str):
                parsed[key] = value.split()
        return cls(parsed)

    def lookup(self, word: str) -> list[str] | None:
        key = word.strip().upper()
        phones = self._entries.get(key)
        if phones is not None:
            return list(phones)
        folded = self._ascii_fold_key(word)
        if folded != key:
            phones = self._entries.get(folded)
            if phones is not None:
                return list(phones)
        return None

    def __len__(self) -> int:
        return len(self._entries)


class CMUDictG2P:
    def __init__(
        self,
        cmudict: CMUDict,
        supplement: SupplementLexicon | None = None,
    ) -> None:
        self.cmudict = cmudict
        self.supplement = supplement or SupplementLexicon({})

    @classmethod
    def from_paths(
        cls,
        cmudict_path: str | Path = DEFAULT_CMUDICT_PATH,
        supplement_path: str | Path = DEFAULT_SUPPLEMENT_PATH,
    ) -> CMUDictG2P:
        return cls(
            CMUDict.from_file(cmudict_path),
            SupplementLexicon.from_json(supplement_path),
        )

    def lookup_word_phonemes_with_fallback(
        self,
        word: str,
        *,
        prev_word: str | None = None,
        next_word: str | None = None,
        allow_spell: bool = True,
    ) -> tuple[list[str], str]:
        from acronym_case import select_acronym_case_action
        from homograph import select_homograph_variant

        variants = self.cmudict.lookup_variants(word)
        if variants:
            case_action = select_acronym_case_action(word, len(variants))
            if case_action == "spell":
                spelled = self._spell_word(word)
                if spelled:
                    return spelled, "spell"
            elif case_action is not None:
                return list(variants[case_action]), "cmudict"

            pick = select_homograph_variant(
                word, len(variants), prev=prev_word, next=next_word
            )
            idx = pick if pick is not None else 0
            return list(variants[idx]), "cmudict"

        phones = self.supplement.lookup(word)
        if phones is not None:
            return phones, "supplement"

        if allow_spell:
            spelled = self._spell_word(word)
            if spelled:
                return spelled, "spell"
        return [], "missing"

    def lookup_isolated_letter_phonemes(
        self,
        letter: str,
        *,
        prefer_letter_name: bool = False,
    ) -> tuple[list[str], str]:
        """Letter-name or article reading for an isolated A–Z token.

        CMUdict's primary ``A`` is the indefinite article (AH0); ``A(1)`` is the
        letter name (EY1). Use ``prefer_letter_name=True`` only for mid-sentence
        uppercase ``A``; lowercase ``a`` and sentence-edge ``A`` keep AH0.
        """
        if len(letter) != 1 or not letter.isalpha():
            return [], "missing"

        variants = self.cmudict.lookup_variants(letter)
        if variants:
            if letter.upper() == "A" and prefer_letter_name and len(variants) > 1:
                return list(variants[1]), "cmudict"
            return list(variants[0]), "cmudict"

        phones = self.supplement.lookup(letter)
        if phones:
            return list(phones), "supplement"
        return [], "missing"

    def lookup_isolated_letter_phonemes(
        self,
        letter: str,
        *,
        prefer_letter_name: bool = False,
    ) -> tuple[list[str], str]:
        """Letter-name or article reading for an isolated A–Z token.

        CMUdict's primary ``A`` is the indefinite article (AH0); ``A(1)`` is the
        letter name (EY1). Use ``prefer_letter_name=True`` for plate-style
        isolated letters in mixed zh-en text; English ``A dog`` keeps AH0.
        """
        if len(letter) != 1 or not letter.isalpha():
            return [], "missing"

        variants = self.cmudict.lookup_variants(letter)
        if variants:
            if letter.upper() == "A" and prefer_letter_name and len(variants) > 1:
                return list(variants[1]), "cmudict"
            return list(variants[0]), "cmudict"

        phones = self.supplement.lookup(letter)
        if phones:
            return list(phones), "supplement"
        return [], "missing"

    def _spell_word(self, word: str) -> list[str]:
        letters = [ch for ch in word.upper() if ch.isalpha()]
        if not letters:
            return []
        phones: list[str] = []
        for letter in letters:
            letter_phones, _ = self.lookup_isolated_letter_phonemes(
                letter,
                prefer_letter_name=True,
            )
            if not letter_phones:
                return []
            phones.extend(letter_phones)
        return phones

    def _lookup_delimited(self, word: str, delimiter: str) -> list[str] | None:
        if delimiter not in word:
            return None
        parts = [part for part in word.split(delimiter) if part]
        if len(parts) < 2:
            return None
        combined: list[str] = []
        for part in parts:
            phones, source = self.lookup_word_phonemes_with_fallback(part)
            if source == "missing":
                return None
            combined.extend(phones)
        return combined

    def _lookup_hyphenated(self, word: str) -> list[str] | None:
        return self._lookup_delimited(word, "-")

    def _lookup_underscored(self, word: str) -> list[str] | None:
        return self._lookup_delimited(word, "_")

    @staticmethod
    def _split_underscore_glue(word: str) -> list[str] | None:
        """Split legacy TN glue such as UNDERSCORECOLUMN -> UNDERSCORE + COLUMN."""
        if len(word) <= len("underscore"):
            return None
        if not word.isalpha():
            return None
        upper = word.upper()
        if not upper.startswith("UNDERSCORE"):
            return None
        rest = word[len("underscore") :]
        if not rest or not rest.isalpha():
            return None
        return ["UNDERSCORE", rest]

    def _lookup_underscore_glue(self, word: str) -> list[str] | None:
        parts = self._split_underscore_glue(word)
        if parts is None:
            return None
        combined: list[str] = []
        for part in parts:
            phones, source = self.lookup_word_phonemes_with_fallback(part)
            if source == "missing":
                return None
            combined.extend(phones)
        return combined

    @staticmethod
    def _split_pascal_case(word: str) -> list[str] | None:
        """Split PascalCase compounds, e.g. ShadowWarrior -> Shadow + Warrior."""
        parts = _PASCAL_CASE_PART_RE.findall(word)
        if len(parts) < 2:
            return None
        if "".join(parts) != word:
            return None
        return parts

    def _lookup_pascal_compound(self, word: str) -> list[str] | None:
        parts = self._split_pascal_case(word)
        if parts is None:
            return None
        combined: list[str] = []
        for part in parts:
            phones, source = self.lookup_word_phonemes_with_fallback(part)
            if source == "missing":
                return None
            combined.extend(phones)
        return combined

    def lookup_phrase_phonemes(
        self,
        word: str,
        *,
        prev_word: str | None = None,
        next_word: str | None = None,
    ) -> tuple[list[str], str]:
        if "_" in word:
            underscored = self._lookup_underscored(word)
            if underscored:
                return underscored, "underscore"

        direct, source = self.lookup_word_phonemes_with_fallback(
            word,
            prev_word=prev_word,
            next_word=next_word,
            allow_spell=False,
        )
        if direct:
            if (
                source == "supplement"
                and "-" in word
                and len(direct) < max(3, len(word.replace("-", "")) // 3)
            ):
                hyphenated = self._lookup_hyphenated(word)
                if hyphenated:
                    return hyphenated, "hyphen"
            return direct, source

        if "-" in word:
            hyphenated = self._lookup_hyphenated(word)
            if hyphenated:
                return hyphenated, "hyphen"

        pascal = self._lookup_pascal_compound(word)
        if pascal:
            return pascal, "pascal_compound"

        underscore_glue = self._lookup_underscore_glue(word)
        if underscore_glue:
            return underscore_glue, "underscore_glue"

        direct, source = self.lookup_word_phonemes_with_fallback(
            word,
            prev_word=prev_word,
            next_word=next_word,
            allow_spell=True,
        )
        if direct:
            return direct, source

        return [], "missing"


def _collapse_dash_runs(tokens: list[tuple[str, str]]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    i = 0
    while i < len(tokens):
        kind, surface = tokens[i]
        if (
            kind == "punct"
            and surface == "-"
            and i + 1 < len(tokens)
            and tokens[i + 1] == ("punct", "-")
            and out
            and out[-1][0] == "word"
            and i + 2 < len(tokens)
            and tokens[i + 2][0] == "word"
        ):
            out.append(("punct", "—"))
            i += 2
            continue
        out.append((kind, surface))
        i += 1
    return out


def tokenize_english_text(text: str) -> list[tuple[str, str]]:
    """Tokenize English text into ('word'|'punct', surface) pairs."""
    from abbreviation import collapse_letter_dot_abbreviations

    text = collapse_letter_dot_abbreviations(text)
    tokens: list[tuple[str, str]] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i].isspace():
            i += 1
            continue
        match = _ENGLISH_TOKEN_RE.match(text, i)
        if not match:
            i += 1
            continue
        surface = match.group()
        i = match.end()
        if _ENGLISH_WORD_SURFACE_RE.fullmatch(surface):
            tokens.append(("word", surface))
        else:
            tokens.append(("punct", surface))
    return _collapse_dash_runs(tokens)
