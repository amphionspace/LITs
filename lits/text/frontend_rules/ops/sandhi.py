"""Chinese G2P tone sandhi operations (skeleton driven by g2p_sandhi/zh.json)."""

from __future__ import annotations

import re
from typing import Any

from ..loader import load_rules, resolve_char_set, resolve_constant, resolve_word_list

_PINYIN_SYLLABLE_RE = re.compile(r"^[a-z]+[0-6]$")
_ARPABET_TOKEN_RE = re.compile(r"^[A-Z]{1,3}[012]?$")
_BPMF_TONE_MARKS = frozenset("˙ˊˇˋˉ")


class MandarinSandhiOps:
    """Applies Mandarin tone sandhi rules from JSON config."""

    def __init__(self, doc: dict[str, Any] | None = None) -> None:
        self.doc = doc or load_rules("g2p_sandhi", "zh")
        resources = self.doc["resources"]
        self._pinyin_tone_chars = frozenset(resources["char_sets"]["pinyin_tone_chars"])
        self._pinyin_tone_dict = {"0": "˙", "5": "˙", "6": "ˊ", "1": "ˉ", "2": "ˊ", "3": "ˇ", "4": "ˋ"}
        self._bopomofo_tone_marks = resources["char_sets"]["bopomofo_tone_marks"]
        self._sandhi_pause_punct = resolve_char_set(self.doc, "sandhi_pause_punct")
        extra = resources["char_sets"]["sandhi_pause_tokens_extra"]
        self._ellipsis_char = resolve_constant(self.doc, "ellipsis_char")
        self._sandhi_pause_tokens = self._sandhi_pause_punct | frozenset(extra) | frozenset({self._ellipsis_char})
        self._yi_punc = resources["char_sets"]["yi_punctuation_chars"]
        self._yi_number_digits = frozenset(resources["char_sets"]["yi_number_digits"])
        self._must_not_er_words = resolve_word_list(self.doc, "must_not_er_words")
        self._bu_exceptions = resolve_word_list(self.doc, "bu_exceptions")
        self._bu_question_keep_tone_words = frozenset(
            resolve_word_list(self.doc, "bu_question_keep_tone_words")
        )
        self._yi_exceptions = resolve_word_list(self.doc, "yi_exceptions")
        self._yi_ordinal_prefix = resources["word_lists"]["yi_ordinal_prefix"]
        self._yi_date_prefixes = tuple(resources["word_lists"]["yi_date_prefixes"])
        self._interjection_tone_configs = self._load_interjection_tone_configs(resources)
        self._interjection_chars = frozenset(
            cfg["char"] for cfg in self._interjection_tone_configs if cfg.get("char")
        )
        self._bu_question_neutral_rule = resources["tone_rules"]["bu_sandhi"].get("question_neutral")
        self._char_citation_tone_cache: dict[str, str | None] = {}

    @staticmethod
    def _load_interjection_tone_configs(resources: dict[str, Any]) -> list[dict[str, Any]]:
        tone_rules = resources.get("tone_rules", {})
        configs = tone_rules.get("interjection_tones")
        if configs:
            return list(configs)
        legacy = tone_rules.get("a_interjection")
        if legacy:
            return [legacy]
        return []

    @staticmethod
    def _bopomofo_tone_mark(bopomofo: str) -> str | None:
        if bopomofo and bopomofo[-1] in _BPMF_TONE_MARKS:
            return bopomofo[-1]
        return None

    def _char_citation_bopomofo_tone_mark(self, char: str) -> str | None:
        """Citation (dictionary) tone for sandhi; neutral surface forms use this."""
        if char in self._char_citation_tone_cache:
            return self._char_citation_tone_cache[char]
        if len(char) != 1:
            return None
        mark: str | None = None
        try:
            from pypinyin import Style, pinyin

            token = pinyin(
                char,
                style=Style.TONE3,
                neutral_tone_with_five=False,
                heteronym=False,
                errors="ignore",
            )[0][0]
            if token and token[-1] in self._pinyin_tone_chars:
                mark = self._pinyin_tone_dict.get(token[-1])
        except ImportError:
            pass
        self._char_citation_tone_cache[char] = mark
        return mark

    def _following_tone_mark_for_yi_sandhi(self, word: str, index: int, bopomofos: list[str]) -> str | None:
        if index + 1 >= len(bopomofos) or not bopomofos[index + 1]:
            return None
        mark = self._bopomofo_tone_mark(bopomofos[index + 1])
        if mark == "˙" and index + 1 < len(word):
            citation = self._char_citation_bopomofo_tone_mark(word[index + 1])
            if citation is not None:
                return citation
        return mark

    @staticmethod
    def change_bopomofo_tone(bopomofo: str, tone: str) -> str:
        if bopomofo[-1] not in "˙ˊˇˋ":
            return bopomofo + tone
        return bopomofo[:-1] + tone

    def _pinyin_syllable_tone(self, syllable: str) -> str | None:
        if len(syllable) >= 2 and syllable[-1] in self._pinyin_tone_chars:
            return syllable[-1]
        return None

    def _set_pinyin_syllable_tone(self, syllable: str, tone: str) -> str:
        if self._pinyin_syllable_tone(syllable) is not None:
            return syllable[:-1] + tone
        return syllable + tone

    def apply_third_tone_sandhi_pinyin(self, syllables: list[str]) -> list[str]:
        """三三变调 on numbered pinyin syllables."""
        if len(syllables) < 2:
            return syllables
        result = list(syllables)
        rule = self.doc["resources"]["tone_rules"]["third_tone_sandhi"]
        from_tone, to_tone = rule["from_tone"], rule["to_tone"]
        for i in range(len(result) - 1):
            if (
                self._pinyin_syllable_tone(result[i]) == from_tone
                and self._pinyin_syllable_tone(result[i + 1]) == rule["lookahead_tone"]
            ):
                result[i] = self._set_pinyin_syllable_tone(result[i], to_tone)
        return result

    def _is_punctuation_token(self, token: str) -> bool:
        halfwidth = frozenset(",.!?;:\"()[]<>-")
        if token == self._ellipsis_char:
            return True
        return bool(token) and all(ch in halfwidth for ch in token)

    def apply_third_tone_sandhi_to_pinyin_tokens(self, tokens: list[str]) -> list[str]:
        if len(tokens) < 2:
            return tokens
        out = list(tokens)
        pinyin_indices: list[int] = []
        for i, tok in enumerate(tokens):
            if _PINYIN_SYLLABLE_RE.match(tok):
                pinyin_indices.append(i)
                continue
            if tok in self._sandhi_pause_tokens or self._is_punctuation_token(tok) or _ARPABET_TOKEN_RE.match(tok):
                if len(pinyin_indices) >= 2:
                    adjusted = self.apply_third_tone_sandhi_pinyin([out[j] for j in pinyin_indices])
                    for idx, syllable in zip(pinyin_indices, adjusted):
                        out[idx] = syllable
                pinyin_indices = []
        if len(pinyin_indices) >= 2:
            adjusted = self.apply_third_tone_sandhi_pinyin([out[j] for j in pinyin_indices])
            for idx, syllable in zip(pinyin_indices, adjusted):
                out[idx] = syllable
        return out

    def apply_triple_third_tone_bopomofo(self, word: str, bopomofos: list[str]) -> list[str]:
        rule = self.doc["resources"]["tone_rules"]["triple_third_tone_bopomofo"]
        mark = rule["tone_mark"]
        result_mark = rule["result_mark"]
        if (
            len(word) == rule["word_length"]
            and len(bopomofos) == rule["word_length"]
            and all(b[-1] == mark for b in bopomofos)
        ):
            bopomofos = list(bopomofos)
            bopomofos[0] = bopomofos[0][:-1] + result_mark
            bopomofos[1] = bopomofos[1][:-1] + result_mark
            return bopomofos
        return bopomofos

    def apply_third_tone_sandhi_bopomofo(self, word: str, bopomofos: list[str]) -> list[str]:
        if len(word) == 2 and len(bopomofos) == 2 and bopomofos[0][-1] == "ˇ" and bopomofos[-1][-1] == "ˇ":
            bopomofos = list(bopomofos)
            bopomofos[0] = bopomofos[0][:-1] + "ˊ"
        return bopomofos

    def bu_sandhi(self, word: str, bopomofos: list[str]) -> list[str]:
        bopomofos = list(bopomofos)
        valid_char = set(word)
        if len(valid_char) == 1 and "不" in valid_char:
            return bopomofos
        if word in self._bu_exceptions:
            return bopomofos
        if len(word) == 3 and word[1] == "不" and len(bopomofos) > 1 and bopomofos[1][:-1] == "ㄅㄨ":
            bopomofos[1] = bopomofos[1][:-1] + "˙"
            return bopomofos
        for i, char in enumerate(word):
            if (
                i + 1 < len(bopomofos)
                and char == "不"
                and i + 1 < len(word)
                and len(bopomofos[i + 1]) > 0
                and bopomofos[i + 1][-1] == "ˋ"
            ):
                bopomofos[i] = bopomofos[i][:-1] + "ˊ"
        return bopomofos

    def yi_sandhi(self, word: str, bopomofos: list[str]) -> list[str]:
        bopomofos = list(bopomofos)
        if word in self._yi_exceptions:
            return bopomofos
        if word.find("一") != -1 and any(item.isnumeric() for item in word if item != "一"):
            for i in range(len(word)):
                if i == 0 and word[0] == "一" and len(word) > 1 and word[1] not in self._yi_number_digits:
                    if len(bopomofos) > 1 and len(bopomofos[0]) > 0 and bopomofos[1][-1] in ["ˋ", "˙"]:
                        bopomofos[0] = self.change_bopomofo_tone(bopomofos[0], "ˊ")
                    else:
                        bopomofos[0] = self.change_bopomofo_tone(bopomofos[0], "ˋ")
                elif word[i] == "一":
                    bopomofos[i] = self.change_bopomofo_tone(bopomofos[i], "")
            return bopomofos
        if len(word) == 3 and word[1] == "一" and word[0] == word[-1]:
            bopomofos[1] = self.change_bopomofo_tone(bopomofos[1], "˙")
        elif word.startswith(self._yi_ordinal_prefix):
            bopomofos[1] = self.change_bopomofo_tone(bopomofos[1], "")
        elif any(word.startswith(p) for p in self._yi_date_prefixes):
            bopomofos[0] = self.change_bopomofo_tone(bopomofos[0], "")
        else:
            for i, char in enumerate(word):
                if char == "一" and i + 1 < len(word):
                    following_mark = self._following_tone_mark_for_yi_sandhi(word, i, bopomofos)
                    if following_mark == "ˋ":
                        bopomofos[i] = self.change_bopomofo_tone(bopomofos[i], "ˊ")
                    elif word[i + 1] not in self._yi_punc:
                        bopomofos[i] = self.change_bopomofo_tone(bopomofos[i], "ˋ")
        return bopomofos

    def er_sandhi(self, word: str, bopomofos: list[str]) -> list[str]:
        bopomofos = list(bopomofos)
        if len(word) > 1 and word[-1] == "儿" and word not in self._must_not_er_words:
            bopomofos[-1] = self.change_bopomofo_tone(bopomofos[-1], "˙")
        return bopomofos

    def merge_bu(self, seg: list[str]) -> list[str]:
        new_seg: list[str] = []
        last_word = ""
        for word in seg:
            if word != "不":
                if last_word == "不":
                    word = last_word + word
                new_seg.append(word)
            last_word = word
        if last_word == "不":
            new_seg.append("不")
        return new_seg

    def merge_er(self, seg: list[str]) -> list[str]:
        new_seg: list[str] = []
        for i, word in enumerate(seg):
            if i - 1 >= 0 and word == "儿":
                new_seg[-1] = new_seg[-1] + seg[i]
            else:
                new_seg.append(word)
        return new_seg

    def merge_yi(self, seg: list[str]) -> list[str]:
        new_seg: list[str] = []
        for i, word in enumerate(seg):
            if i - 1 >= 0 and word == "一" and i + 1 < len(seg) and seg[i - 1] == seg[i + 1]:
                if i - 1 < len(new_seg):
                    new_seg[i - 1] = new_seg[i - 1] + "一" + new_seg[i - 1]
                else:
                    new_seg.append(word)
                    new_seg.append(seg[i + 1])
            else:
                if i - 2 >= 0 and seg[i - 1] == "一" and seg[i - 2] == word:
                    continue
                new_seg.append(word)

        seg = new_seg
        new_seg = []
        isnumeric_flag = False
        for word in seg:
            if all(item.isnumeric() for item in word) and not isnumeric_flag:
                isnumeric_flag = True
                new_seg.append(word)
            else:
                new_seg.append(word)

        seg = new_seg
        new_seg = []
        for word in seg:
            if new_seg and new_seg[-1] == "一":
                new_seg[-1] = new_seg[-1] + word
            else:
                new_seg.append(word)
        return new_seg

    def apply_word_sandhi_bopomofo(self, word: str, bopomofos: list[str]) -> list[str]:
        """Run per-word Bopomofo sandhi pipeline from JSON."""
        bopomofos = self.apply_triple_third_tone_bopomofo(word, bopomofos)
        bopomofos = self.apply_third_tone_sandhi_bopomofo(word, bopomofos)
        bopomofos = self.bu_sandhi(word, bopomofos)
        bopomofos = self.yi_sandhi(word, bopomofos)
        bopomofos = self.er_sandhi(word, bopomofos)
        return bopomofos

    def merge_words(self, words: list[str]) -> list[str]:
        words = self.merge_yi(words)
        words = self.merge_bu(words)
        words = self.merge_er(words)
        return words

    @staticmethod
    def _next_non_space_char(text: str, idx: int) -> str | None:
        j = idx + 1
        while j < len(text) and text[j].isspace():
            j += 1
        return text[j] if j < len(text) else None

    def adjust_interjection_tone_from_following_punct(
        self,
        syllable: str,
        following_char: str | None,
        config: dict[str, Any],
    ) -> str:
        for rule in config.get("rules", []):
            if following_char in rule.get("following_punct", []):
                return self._set_pinyin_syllable_tone(syllable, rule["tone"])
        default_tone = config.get("default_tone")
        if default_tone is not None:
            return self._set_pinyin_syllable_tone(syllable, default_tone)
        return syllable

    def adjust_a_tone_from_following_punct(self, syllable: str, following_char: str | None) -> str:
        for config in self._interjection_tone_configs:
            if config.get("char") == "啊" or config.get("id") == "a_interjection":
                return self.adjust_interjection_tone_from_following_punct(syllable, following_char, config)
        return syllable

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
        if full_text is None or len(word) != len(syllables):
            return syllables
        if not any(ch in self._interjection_chars for ch in word):
            return syllables
        adjusted = list(syllables)
        for config in self._interjection_tone_configs:
            interjection_char = config.get("char")
            if not interjection_char or interjection_char not in word:
                continue
            if config.get("standalone_only") and (
                word != interjection_char or prev_word is not None
            ):
                continue
            for local_idx, ch in enumerate(word):
                if ch != interjection_char:
                    continue
                if word_char_positions is not None and local_idx < len(word_char_positions):
                    global_idx = word_char_positions[local_idx]
                else:
                    global_idx = char_offset + local_idx
                following = self._next_non_space_char(full_text, global_idx)
                adjusted[local_idx] = self.adjust_interjection_tone_from_following_punct(
                    adjusted[local_idx],
                    following,
                    config,
                )
        return adjusted

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

    def _resolve_bu_question_neutral_target(
        self,
        word: str,
        prev_word: str | None,
    ) -> tuple[str, int] | None:
        """Return (matched phrase, index of 「不」 within *word*) when this token participates."""
        rule = self._bu_question_neutral_rule
        if rule is None:
            return None
        bu_char = rule.get("char", "不")
        if not word.endswith(bu_char):
            return None
        bu_index = word.rfind(bu_char)
        phrase = f"{prev_word}{bu_char}" if word == bu_char and prev_word else word
        if phrase in self._bu_question_keep_tone_words:
            return None
        return phrase, bu_index

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
        """词尾「不」+ 问号 → 默认读轻声（如 好不？、来不？）；例外见 bu_question_keep_tone_words。"""
        if full_text is None or word in self._bu_exceptions or len(word) != len(syllables):
            return syllables

        target = self._resolve_bu_question_neutral_target(word, prev_word)
        if target is None:
            return syllables

        _, bu_index = target
        if bu_index >= len(word) or word[bu_index] != self._bu_question_neutral_rule.get("char", "不"):
            return syllables

        if word_char_positions is not None and bu_index < len(word_char_positions):
            global_idx = word_char_positions[bu_index]
        else:
            global_idx = char_offset + bu_index
        following = self._next_non_space_char(full_text, global_idx)
        rule = self._bu_question_neutral_rule
        if rule is None or following not in rule.get("following_punct", []):
            return syllables

        tone = str(rule.get("tone", "5"))
        adjusted = list(syllables)
        adjusted[bu_index] = self._set_pinyin_syllable_tone(adjusted[bu_index], tone)
        return adjusted
