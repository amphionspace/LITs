"""Pipeline runners for frontend rule documents."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from .loader import load_rules, resolve_char_set, resolve_constant
from .ops.punctuation import apply_punctuation_op
from .ops.sandhi import MandarinSandhiOps
from .tn_frontend import (
    apply_tn_frontend_full,
    get_char_set as get_tn_frontend_char_set,
)


class PunctuationEngine:
    """Apply punctuation rules from ``frontend_rules/rules/punctuation/*.json``."""

    def __init__(self, locale: str = "common") -> None:
        self.locale = locale
        self.doc = load_rules("punctuation", locale)

    def apply(self, text: str) -> str:
        for stage in self.doc["pipeline"]["stages"]:
            text = apply_punctuation_op(stage["op"], text, self.doc, stage.get("params"))
        return text

    def get_char_set(self, name: str) -> frozenset[str]:
        return resolve_char_set(self.doc, name)

    def get_constant(self, name: str) -> str:
        return resolve_constant(self.doc, name)


@lru_cache(maxsize=None)
def get_punctuation_engine(locale: str) -> PunctuationEngine:
    return PunctuationEngine(locale)


class TnFrontendEngine:
    """Applies ``rules_v2/<locale>.full.json`` → ``frontend`` bookend stages."""

    def apply(self, text: str, lang: str = "en") -> str:
        return apply_tn_frontend_full(text, lang)

    def get_char_set(self, name: str, lang: str = "en") -> frozenset[str]:
        return get_tn_frontend_char_set(name, lang)


@lru_cache(maxsize=None)
def get_common_punctuation() -> TnFrontendEngine:
    return TnFrontendEngine()


@lru_cache(maxsize=None)
def get_zh_punctuation() -> PunctuationEngine:
    return PunctuationEngine("zh")


class G2PSandhiEngine:
    """Apply Mandarin tone sandhi from ``frontend_rules/rules/g2p_sandhi/zh.json``."""

    def __init__(self, locale: str = "zh") -> None:
        self.locale = locale
        self.doc = load_rules("g2p_sandhi", locale)
        self._ops = MandarinSandhiOps(self.doc)

    @property
    def ops(self) -> MandarinSandhiOps:
        return self._ops

    def merge_words(self, words: list[str]) -> list[str]:
        return self._ops.merge_words(words)

    def apply_word_sandhi_bopomofo(self, word: str, bopomofos: list[str]) -> list[str]:
        return self._ops.apply_word_sandhi_bopomofo(word, bopomofos)

    def apply_third_tone_sandhi_pinyin(self, syllables: list[str]) -> list[str]:
        return self._ops.apply_third_tone_sandhi_pinyin(syllables)

    def apply_third_tone_sandhi_to_pinyin_tokens(self, tokens: list[str]) -> list[str]:
        return self._ops.apply_third_tone_sandhi_to_pinyin_tokens(tokens)

    def apply_interjection_tones(
        self,
        word: str,
        syllables: list[str],
        *,
        full_text: str | None,
        char_offset: int,
        word_char_positions: list[int] | None = None,
        prev_word: str | None = None,
    ) -> list[str]:
        return self._ops.apply_interjection_tones(
            word,
            syllables,
            full_text=full_text,
            char_offset=char_offset,
            word_char_positions=word_char_positions,
            prev_word=prev_word,
        )

    def apply_a_interjection_tones(
        self,
        word: str,
        syllables: list[str],
        *,
        full_text: str | None,
        char_offset: int,
        word_char_positions: list[int] | None = None,
    ) -> list[str]:
        return self.apply_interjection_tones(
            word,
            syllables,
            full_text=full_text,
            char_offset=char_offset,
            word_char_positions=word_char_positions,
        )

    def apply_bu_question_neutral_tones(
        self,
        word: str,
        syllables: list[str],
        *,
        full_text: str | None,
        char_offset: int,
        word_char_positions: list[int] | None = None,
        prev_word: str | None = None,
    ) -> list[str]:
        return self._ops.apply_bu_question_neutral_tones(
            word,
            syllables,
            full_text=full_text,
            char_offset=char_offset,
            word_char_positions=word_char_positions,
            prev_word=prev_word,
        )

    def get_sandhi_pause_tokens(self) -> frozenset[str]:
        return self._ops._sandhi_pause_tokens

    def get_doc(self) -> dict[str, Any]:
        return self.doc


@lru_cache(maxsize=None)
def get_g2p_sandhi_engine(locale: str = "zh") -> G2PSandhiEngine:
    return G2PSandhiEngine(locale)
