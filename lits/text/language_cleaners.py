"""Text cleaners for Chinese-English pipelines.

Public cleaners:
  - pinyin_direct_mixed_cleaners: auto-routes pinyin Chinese vs ARPAbet English
  - pinyin_direct_mixed_rhyme_body_tone_cleaners: rhyme-body + inline tone marks
  - english_direct_phoneme_cleaners: normalizes pre-computed ARPAbet tokens
  - en_zh_dict_mixed_cleaners: hanzi lexicon + English CMUdict G2P + ARPAbet pass-through
    via chinese_lexicon.txt (longest-match), then through the same pinyin → Bopomofo
    path used by training.
  - en_zh_dict_mixed_rhyme_body_tone_cleaners: rhyme-body + inline tone marks
"""

import os
import re
from pathlib import Path

from lits.text.frontend_rules.pipeline import (
    get_g2p_sandhi_engine,
    get_punctuation_engine,
    get_zh_punctuation,
)
from lits.text.unified_text_normalizer import get_text_normalizer, normalize_text

REPO_ROOT = Path(__file__).resolve().parents[2]

def _frontend_char_set(name: str) -> frozenset[str]:
    from lits.text.frontend_rules.tn_frontend import get_char_set

    return get_char_set(name)

_WHITESPACE_PATTERN = re.compile(r"\s+")


def _strip_structural_quotes(text: str) -> str:
    """Remove paired quote marks (all scripts); keep English apostrophes."""
    from lits.text.frontend_rules.ops.punctuation import strip_structural_quotes

    return strip_structural_quotes(
        text,
        single_quote_chars=_frontend_char_set("structural_single_quote_chars"),
        jp_quote_chars=_frontend_char_set("jp_single_quote_chars"),
        double_quote_chars=_frontend_char_set("structural_double_quote_chars"),
    )


def _strip_structural_single_quotes(text: str) -> str:
    return _strip_structural_quotes(text)


def _collapse_empty_slash_segments(text: str) -> str:
    from lits.text.frontend_rules.ops.punctuation import collapse_empty_slash_segments

    return collapse_empty_slash_segments(text)


def _remove_dashes(text: str) -> str:
    from lits.text.frontend_rules.ops.punctuation import remove_dashes

    return remove_dashes(text, dash_chars=_frontend_char_set("dash_chars"))


def _dedupe_trailing_punctuation(text: str) -> str:
    from lits.text.frontend_rules.ops.punctuation import dedupe_trailing_punctuation

    return dedupe_trailing_punctuation(
        text,
        exempt_chars=_frontend_char_set("dedupe_punct_exempt"),
    )


def _ensure_trailing_sentence_punct(text: str) -> str:
    """Append '.' when text lacks sentence-final punctuation (smoother TTS ending)."""
    from lits.text.frontend_rules.ops.punctuation import ensure_trailing_sentence_punct

    return ensure_trailing_sentence_punct(
        text,
        acceptable_trailing=_frontend_char_set("acceptable_trailing_punct"),
    )


def _replace_vertical_line_punct(text: str) -> str:
    """Delegate to ``frontend.post_stages`` (``fe_006a``); idempotent at G2P time."""
    from lits.text.frontend_rules.ops.punctuation import replace_vertical_line_punct

    return replace_vertical_line_punct(
        text,
        source_chars=_frontend_char_set("vertical_line_punct_chars"),
        replacement="，",
    )


def normalize_clause_break_punct(text: str) -> str:
    """Insert clause punctuation at line breaks and bare spaces.

    Rules live in ``Transsion_Multilingual_Text_Normalization_for_TTS/rules_v2/en.full.json``
    under ``frontend``.
    """
    from lits.text.frontend_rules.ops.punctuation import normalize_clause_break_punct as _op

    return _op(text, acceptable_trailing=_frontend_char_set("acceptable_trailing_punct"))


def strip_line_breaks(text: str) -> str:
    """Backward-compatible alias for :func:`normalize_clause_break_punct`."""
    return normalize_clause_break_punct(text)


def preprocess_text(text: str) -> str:
    """Legacy hook for training paths that skip TN; no-op when ``|`` already handled."""
    if not text:
        return ""
    if not any(ch in text for ch in _frontend_char_set("vertical_line_punct_chars")):
        return text
    return _replace_vertical_line_punct(text)

# ==============================================================================
# English ARPAbet cleaner
# ==============================================================================

def zh_en_phoneme_passthrough_cleaners(text):
    """Preserve precomputed zh-en model phoneme tokens from an offline filelist."""
    return re.sub(r"\s+", " ", text.strip())


def english_direct_phoneme_cleaners(text):
    """Normalize precomputed English phoneme text into token sequence.

    Input:  "DH EH1 R / IH1 Z / M AH0 S Y ER1"
    Output: "DH EH1 R _ IH1 Z _ M AH0 S Y ER1"
    """
    if not text:
        return ""

    normalized = re.sub(r"\s+", " ", text.strip())
    normalized = re.sub(r"\s*[|/]\s*", " _ ", normalized)
    normalized = re.sub(r"(?: _ )+", " _ ", normalized).strip()
    return normalized


# ==============================================================================
# Pinyin → Bopomofo conversion
# ==============================================================================

from lits.text.bopomofo_utils import (
    PINYIN_TONE_TO_BPMF as _pinyin_tone_dict,
    bpmf_syllable_to_tokens as _bpmf_syllable_to_tokens,
    load_pinyin_2_bpmf as _load_pinyin_2_bpmf_from_utils,
)

_pinyin_2_bpmf_cache = None
_pinyin_tone_chars = frozenset(_pinyin_tone_dict)
_pinyin_syllable_re = re.compile(r"^[a-z]+[0-6]$")
_pinyin_syllable_scan_re = re.compile(r"[a-z]+[0-6]")
_arpabet_token_re = re.compile(r"^[A-Z]{1,3}[012]?$")
_arpabet_token_scan_re = re.compile(r"[A-Z]{1,3}[012]?")


def _zh_punct_resources():
    return get_zh_punctuation()


def _ellipsis_char() -> str:
    return _zh_punct_resources().get_constant("ellipsis_char")


def _halfwidth_punct_chars() -> frozenset[str]:
    return _zh_punct_resources().get_char_set("halfwidth_punct_chars")


def _zh_structural_punct_chars() -> frozenset[str]:
    return _zh_punct_resources().get_char_set("structural_punct_chars")


def _normalize_zh_punct_to_halfwidth(text: str) -> str:
    """Map CJK punctuation to halfwidth; rules in ``frontend_rules/rules/punctuation/zh.json``."""
    if not text:
        return text
    return _zh_punct_resources().apply(text)


def _canonical_punctuation_token(token: str) -> str | None:
    """Return one output punctuation character/token, or None if not punctuation."""
    if not token:
        return None
    ellipsis = _ellipsis_char()
    aliases = tuple(_zh_punct_resources().doc["resources"]["char_sets"]["ellipsis_aliases"])
    if token == ellipsis or token in aliases:
        return ellipsis
    halfwidth = _halfwidth_punct_chars()
    if all(ch in halfwidth for ch in token):
        return token
    return None


def _is_punctuation_token(token: str) -> bool:
    return _canonical_punctuation_token(token) is not None


_INTERNAL_LATIN_CLAUSE_PUNCT_RE = re.compile(r"(?<=[A-Za-z])[;,](?=[A-Za-z])")
_LATIN_GLUE_RUN_RE = re.compile(
    r"[A-Za-z]+(?:['\u2019\-][A-Za-z]+)*(?:[,;][A-Za-z]+(?:['\u2019\-][A-Za-z]+)*)*"
)


def split_latin_token_on_internal_commas(token: str) -> list[str]:
    """Split TN-glued Latin tokens like ``Ziqi,GEM`` into separate word/punct pieces."""
    if not token or not _INTERNAL_LATIN_CLAUSE_PUNCT_RE.search(token):
        return [token]
    parts: list[str] = []
    start = 0
    for match in _INTERNAL_LATIN_CLAUSE_PUNCT_RE.finditer(token):
        parts.append(token[start : match.start()])
        parts.append(match.group(0))
        start = match.end()
    parts.append(token[start:])
    return [piece for piece in parts if piece]


def _is_zh_punct_char(ch: str) -> bool:
    """Single-character punctuation after halfwidth normalization."""
    return ch == _ellipsis_char() or ch in _halfwidth_punct_chars()


def _is_zh_structural_punct_char(ch: str) -> bool:
    """Punctuation kept in output but ignored when grouping hanzi for lexicon lookup."""
    return ch in _zh_structural_punct_chars()


def _is_zh_lexicon_break_char(ch: str) -> bool:
    """Punctuation that splits lexicon lookup (clause / sentence pauses)."""
    return _is_zh_punct_char(ch) and not _is_zh_structural_punct_char(ch)


def _strip_zh_structural_punct(text: str) -> str:
    """Remove unspoken structural punctuation (quotes, brackets, parens)."""
    from lits.text.frontend_rules.ops.punctuation import strip_zh_structural_punct

    return strip_zh_structural_punct(
        text,
        structural_chars=_zh_structural_punct_chars(),
        acceptable_trailing=_frontend_char_set("acceptable_trailing_punct"),
    )


_strip_structural_punct = _strip_zh_structural_punct


def _hanzi_char_positions_in_segment(segment: str, segment_start: int) -> list[int]:
    """Map each hanzi char in *segment* to its index in the original full text."""
    positions: list[int] = []
    for local_i, ch in enumerate(segment):
        if _is_zh_structural_punct_char(ch):
            continue
        if _hanzi_char_re.match(ch):
            positions.append(segment_start + local_i)
    return positions


def _tokenize_mixed_pinyin_text(text: str) -> list[str]:
    """Split space-separated or glued pinyin/ARPAbet/punctuation into tokens.

    Fullwidth punctuation is normalized to halfwidth first.  Pinyin syllables
    glued to punctuation (e.g. ``ni3,hao3`` or ``shi4 jie4.``) are split apart.
    """
    text = _normalize_zh_punct_to_halfwidth(text)
    tokens: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if ch in ("_", "/", "|"):
            tokens.append(ch)
            i += 1
            continue
        if _is_zh_punct_char(ch):
            tokens.append(ch)
            i += 1
            continue
        latin_m = _LATIN_GLUE_RUN_RE.match(text, i)
        latin_run = latin_m.group() if latin_m else ""
        arpa_m = _arpabet_token_scan_re.match(text, i) if "A" <= ch <= "Z" else None
        if (
            latin_m
            and ("," in latin_run or ";" in latin_run)
            and (not arpa_m or latin_m.end() > arpa_m.end())
        ):
            for piece in split_latin_token_on_internal_commas(latin_run):
                tokens.append(piece)
            i = latin_m.end()
            continue
        if arpa_m and arpa_m.start() == i:
            tokens.append(arpa_m.group())
            i = arpa_m.end()
            continue
        if "a" <= ch <= "z":
            m = _pinyin_syllable_scan_re.match(text, i)
            if m and m.start() == i:
                tokens.append(m.group())
                i = m.end()
                continue
        tokens.append(ch)
        i += 1
    return tokens


def _append_punctuation_tokens(result_tokens: list[str], token: str) -> None:
    """Emit punctuation into the phoneme stream (with ``_`` boundaries)."""
    punct = _canonical_punctuation_token(token)
    if punct is None:
        return
    if len(punct) == 1:
        chars = [punct]
    else:
        chars = [ch for ch in punct if ch in _halfwidth_punct_chars()]
    for ch in chars:
        if result_tokens and result_tokens[-1] != "_":
            result_tokens.append("_")
        result_tokens.append(ch)
        result_tokens.append("_")


def _apply_third_tone_sandhi(syllables):
    """Legacy fallback used when the unified ``data/zh_g2p`` runtime is absent."""
    return get_g2p_sandhi_engine().apply_third_tone_sandhi_pinyin(syllables)


def _use_unified_zh_g2p() -> bool:
    return get_text_normalizer("zh_g2p") is not None


def _apply_third_tone_sandhi_to_tokens(tokens: list[str]) -> list[str]:
    """Apply the unified post-lexicon G2P rules to one complete token stream.

    The C++ profile covers yi/bu/third-tone/neutral/interjection rules and keeps
    ARPAbet plus punctuation untouched.  The Python rule engine remains only as
    a development fallback when no unified runtime artifact has been installed.
    """
    if not tokens:
        return []
    normalized = normalize_text("zh_g2p", " ".join(tokens))
    if normalized is not None:
        return _tokenize_mixed_pinyin_text(normalized)
    return get_g2p_sandhi_engine().apply_third_tone_sandhi_to_pinyin_tokens(tokens)


def _load_pinyin_2_bpmf():
    global _pinyin_2_bpmf_cache
    if _pinyin_2_bpmf_cache is not None:
        return _pinyin_2_bpmf_cache
    _pinyin_2_bpmf_cache = _load_pinyin_2_bpmf_from_utils()
    return _pinyin_2_bpmf_cache


def _ensure_boundary_after_arpabet(result_tokens: list[str]) -> None:
    """Insert ``_`` when an ARPAbet phoneme is immediately followed by pinyin."""
    if result_tokens and _is_arpabet_token(result_tokens[-1]):
        result_tokens.append("_")


def _ensure_boundary_before_arpabet(result_tokens: list[str]) -> None:
    """Insert ``_`` when pinyin/Bopomofo is immediately followed by an ARPAbet phoneme."""
    if not result_tokens or result_tokens[-1] == "_":
        return
    if _is_arpabet_token(result_tokens[-1]):
        return
    result_tokens.append("_")


def _append_bpmf_syllable_tokens(
    result_tokens: list[str],
    bpmf: str,
    tone_mark: str,
    *,
    rhyme_body_tone: bool = False,
    append_syllable_boundary: bool = True,
) -> None:
    result_tokens.extend(
        _bpmf_syllable_to_tokens(
            bpmf,
            tone_mark,
            rhyme_body_tone=rhyme_body_tone,
        )
    )
    if append_syllable_boundary:
        result_tokens.append("_")


def _apply_sentence_tone_sandhi(tokens: list[str]) -> list[str]:
    """One unified zh_g2p pass over a complete pinyin token stream (may include ``|``)."""
    if not tokens:
        return []
    sandhi = _apply_third_tone_sandhi_to_tokens(tokens)
    return [token for token in sandhi if token != "|"]


def _pinyin_to_bopomofo_tokens(
    text,
    *,
    rhyme_body_tone: bool = False,
    apply_entry_sandhi: bool = True,
):
    """Convert pinyin+ARPAbet mixed string to space-delimited tokens.

    Pinyin syllables are converted to Bopomofo char tokens with ``_``
    boundaries; ARPAbet phonemes pass through unchanged. A ``_`` is inserted
    between the last ARPAbet phoneme of an English unit and the next pinyin
    syllable when the input line omits an explicit boundary.
    Halfwidth punctuation tokens (e.g. ``,``, ``.``) are kept in the output.

    Input:  "ni3 hao3 _ DH IH1 S"
    Output: "ㄋ ㄧ ˇ _ ㄏ ㄠ ˇ _ DH IH1 S"
    Input:  "EY1 yi1"
    Output: "EY1 _ ㄧ ˉ"
    Input:  "ni3 hao3 , shi4 jie4 ."
    Output: "ㄋ ㄧ ˇ _ ㄏ ㄠ ˇ _ , _ ㄕ ˋ _ ㄐ ㄧ ㄝ ˋ _ ."
    Input:  "ni3,hao3 shi4 jie4."
    Output: same as spaced punctuation form above.

    With rhyme_body_tone=True (no ``_`` between zh syllables; zh/en boundary kept):
    Output: "ㄋ ㄧ ˊ ㄏ ㄠ ˇ _ , _ ㄕ ˋ ㄐ ㄧㄝ ˋ _ ."
    Mixed: "ni3 hao3 DH IH1 S" -> "ㄋ ㄧ ˊ ㄏ ㄠ ˇ _ DH IH1 S"
    """
    append_syllable_boundary = not rhyme_body_tone
    token_list = _tokenize_mixed_pinyin_text(text)
    tokens = (
        _apply_third_tone_sandhi_to_tokens(token_list)
        if apply_entry_sandhi
        else list(token_list)
    )
    pinyin_2_bpmf = _load_pinyin_2_bpmf()

    result_tokens = []
    for syllable in tokens:
        if not syllable:
            continue
        if syllable == "|":
            continue
        if syllable in ("_", "/"):
            if not result_tokens or result_tokens[-1] != "_":
                result_tokens.append("_")
            continue
        if _is_punctuation_token(syllable):
            _append_punctuation_tokens(result_tokens, syllable)
            continue
        if _is_arpabet_token(syllable):
            _ensure_boundary_before_arpabet(result_tokens)
            result_tokens.append(syllable)
            continue
        if len(syllable) >= 2 and syllable[-1] in _pinyin_tone_dict:
            _ensure_boundary_after_arpabet(result_tokens)
            tone = syllable[-1]
            base = syllable[:-1]
            if base in pinyin_2_bpmf:
                _append_bpmf_syllable_tokens(
                    result_tokens,
                    pinyin_2_bpmf[base],
                    _pinyin_tone_dict[tone],
                    rhyme_body_tone=rhyme_body_tone,
                    append_syllable_boundary=append_syllable_boundary,
                )
                continue
            if len(base) > 1 and base.endswith("r") and base[:-1] in pinyin_2_bpmf:
                _ensure_boundary_after_arpabet(result_tokens)
                _append_bpmf_syllable_tokens(
                    result_tokens,
                    pinyin_2_bpmf[base[:-1]],
                    _pinyin_tone_dict[tone],
                    rhyme_body_tone=rhyme_body_tone,
                    append_syllable_boundary=append_syllable_boundary,
                )
                result_tokens.append("ㄦ")
                result_tokens.append("˙")
                if append_syllable_boundary:
                    result_tokens.append("_")
                continue
        # Non-pinyin token (ARPAbet phoneme, etc.): pass through as-is
        result_tokens.append(syllable)

    while result_tokens and result_tokens[-1] == "_":
        result_tokens.pop()
    return " ".join(result_tokens)


# ==============================================================================
# Main hybrid cleaner
# ==============================================================================


def _is_arpabet_token(token: str) -> bool:
    """Training English uses uppercase ARPAbet (e.g. ER0, DH); pinyin is lowercase (ni3)."""
    return bool(_arpabet_token_re.match(token))


def pinyin_direct_mixed_rhyme_body_tone_cleaners(text):
    """Rhyme-body + inline tone mark paradigm: ``initial rhyme_body tone_mark``
    per syllable; no ``_`` between zh syllables (zh/en boundaries unchanged)."""
    if not text:
        return ""
    return _pinyin_to_bopomofo_tokens(text, rhyme_body_tone=True)


def pinyin_direct_mixed_cleaners(text):
    """Hybrid cleaner for zh-en training with pinyin Chinese + ARPAbet English.

    Handles pure pinyin, pure ARPAbet, and mixed pinyin+ARPAbet text.
    Pinyin tokens are converted to Bopomofo; ARPAbet tokens pass through.
    Chinese punctuation in the pinyin line (e.g. from add_punct_to_pinyin) is
    normalized to halfwidth and emitted as punctuation tokens in the output.

    Chinese input (pinyin):  "ni3 hao3 shi4 jie4"
    With punctuation:        "ni3 hao3 , shi4 jie4 ."
    English input (ARPAbet): "DH AE1 T / W ER1 L D"
    Mixed input:             "ni3 hao3 _ DH IH1 S"
    """
    if not text:
        return ""
    return _pinyin_to_bopomofo_tokens(text)


# ==============================================================================
# Chinese hanzi (character) cleaner — raw 汉字 input
# ==============================================================================

_chinese_frontend = None
_en_tokenizer = None
_chinese_lexicon_cache = None


def _get_chinese_frontend():
    global _chinese_frontend
    if _chinese_frontend is not None:
        return _chinese_frontend
    from lits.text.g2p.mandarin import Frontend_chinese
    resource_path = str(REPO_ROOT / "lits" / "text")
    _chinese_frontend = Frontend_chinese(resource_path, BLANK_LEVEL=2)
    return _chinese_frontend


def _get_en_tokenizer():
    global _en_tokenizer
    if _en_tokenizer is not None:
        return _en_tokenizer
    from lits.text.g2p.text_tokenizers import TextTokenizer
    _en_tokenizer = TextTokenizer(language="en-us")
    return _en_tokenizer


def _load_chinese_lexicon():
    """Load chinese_lexicon.txt (+ optional user_dict.txt) for hanzi G2P."""
    global _chinese_lexicon_cache
    if _chinese_lexicon_cache is not None:
        return _chinese_lexicon_cache

    word_pinyin_dict: dict[str, str] = {}
    lexicon_file = Path(
        os.environ.get(
            "LITS_ZH_LEXICON",
            REPO_ROOT / "lits" / "text" / "sources" / "chinese_lexicon.txt",
        )
    )
    with open(lexicon_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) == 2:
                word_pinyin_dict[parts[0]] = parts[1]

    user_dict_file = Path(
        os.environ.get(
            "LITS_ZH_USER_DICT",
            REPO_ROOT / "lits" / "text" / "sources" / "user_dict.txt",
        )
    )
    if user_dict_file.exists():
        with open(user_dict_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    word_pinyin_dict[parts[0]] = parts[-1]

    _chinese_lexicon_cache = {
        "word_pinyin_dict": word_pinyin_dict,
        "max_word_len": max((len(w) for w in word_pinyin_dict), default=1),
    }
    return _chinese_lexicon_cache


def _normalize_lexicon_pinyin(pinyin: str) -> list[str]:
    syllables = []
    for py in pinyin.strip().split():
        py = py.replace("ü", "v").replace("Ü", "v")
        if _pinyin_syllable_re.match(py):
            syllables.append(py)
    return syllables


def _candidate_conflicts_with_contextual_polyphone(cand: str, pinyin: str) -> bool:
    """Reject lexicon matches whose first polyphone contradicts contextual G2P.

    The lexicon is used for greedy longest-match segmentation, but it contains
    rare entries such as ``的情 -> di2 qing2``.  In normal phrases like
    ``...的情况`` that match incorrectly swallows the structural particle ``的``.
    Use pypinyin as a lightweight contextual check for this high-impact
    polyphonic prefix instead of adding phrase-specific overrides.
    """
    if len(cand) <= 1 or not cand.startswith("的"):
        return False

    lexicon_syllables = _normalize_lexicon_pinyin(pinyin)
    if not lexicon_syllables:
        return False

    contextual = _pypinyin_syllables_for_text(cand)
    if not contextual:
        return False

    return lexicon_syllables[0] != contextual[0]


def _candidate_splits_following_lexicon_word(
    cand: str,
    text: str,
    i: int,
    lexicon: dict,
) -> bool:
    """Reject a greedy match that eats the first char of a following dictionary word.

    Example: ``或一加`` must not match lexicon ``或一`` when ``一加`` is also listed.
    """
    if len(cand) < 2:
        return False
    word_pinyin_dict = lexicon["word_pinyin_dict"]
    max_word_len = lexicon["max_word_len"]
    overlap_start = i + len(cand) - 1
    n = len(text)
    prefix = text[i:i + len(cand) - 1]
    for length in range(min(max_word_len, n - overlap_start), 1, -1):
        if length < 2:
            continue
        word = text[overlap_start:overlap_start + length]
        if word not in word_pinyin_dict or word == cand:
            continue
        if word[0] != cand[-1]:
            continue
        if prefix not in word_pinyin_dict:
            continue
        if not text[i + len(cand):].startswith(word[1:]):
            continue
        return True
    return False


def _segment_hanzi_with_lexicon(text: str, lexicon: dict) -> list[str]:
    """Greedy longest-match segmentation using chinese_lexicon.txt."""
    word_pinyin_dict = lexicon["word_pinyin_dict"]
    max_word_len = lexicon["max_word_len"]
    words: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        matched = None
        for length in range(min(max_word_len, n - i), 1, -1):
            cand = text[i:i + length]
            if cand in word_pinyin_dict:
                if _candidate_conflicts_with_contextual_polyphone(cand, word_pinyin_dict[cand]):
                    continue
                if cand == "得不" and i > 0 and text[i - 1] == "觉":
                    continue
                if cand == "个能" and i > 0 and text[i - 1] == "一":
                    continue
                if _candidate_splits_following_lexicon_word(cand, text, i, lexicon):
                    continue
                matched = cand
                break
        if matched is not None:
            words.append(matched)
            i += len(matched)
        else:
            words.append(text[i])
            i += 1
    return words


def _pypinyin_syllables_for_text(hanzi_text: str) -> list[str]:
    from pypinyin import Style, pinyin

    syllables = []
    for item in pinyin(
        hanzi_text.strip(),
        style=Style.TONE3,
        neutral_tone_with_five=True,
        errors=lambda chars: list(chars),
    ):
        token = item[0].strip()
        if not token:
            continue
        token = token.replace("ü", "v").replace("Ü", "v")
        if _pinyin_syllable_re.match(token):
            syllables.append(token)
    return syllables


def _lexicon_pinyin_for_word(word: str, lexicon: dict) -> list[str] | None:
    """Resolve one segmented hanzi word to numbered pinyin syllables."""
    word_pinyin_dict = lexicon["word_pinyin_dict"]

    if word in word_pinyin_dict:
        syllables = _normalize_lexicon_pinyin(word_pinyin_dict[word])
        if syllables:
            return syllables

    # merge_bu joins leading 不 onto the next word for tone sandhi, but the
    # combined form is often missing from the lexicon. Compose 不 + tail so
    # entries like 马虎 (ma3 hu5) keep their listed reading.
    if len(word) > 1 and word[0] == "不":
        tail_syllables = _lexicon_pinyin_for_word(word[1:], lexicon)
        if tail_syllables is not None:
            bu_syllables = _lexicon_pinyin_for_word("不", lexicon)
            if bu_syllables is not None:
                return bu_syllables + tail_syllables

    syllables: list[str] = []
    for ch in word:
        if ch in word_pinyin_dict:
            part = _normalize_lexicon_pinyin(word_pinyin_dict[ch])
            if not part:
                return None
            syllables.extend(part)
        else:
            return None
    return syllables


def _hanzi_prefix_before_segment(
    full_text: str,
    segment_start: int,
    max_len: int,
) -> str:
    """Trailing hanzi before *segment_start*, skipping clause/structural punctuation."""
    prefix_chars: list[str] = []
    i = segment_start - 1
    while i >= 0 and len(prefix_chars) < max_len:
        ch = full_text[i]
        if ch.isspace() or _is_zh_lexicon_break_char(ch) or _is_zh_structural_punct_char(ch):
            i -= 1
            continue
        if not _hanzi_char_re.match(ch):
            break
        prefix_chars.append(ch)
        i -= 1
    prefix_chars.reverse()
    return "".join(prefix_chars)


def _lexicon_pinyin_for_cross_clause_suffix(
    segment_hanzi: str,
    full_text: str | None,
    segment_start: int | None,
    lexicon: dict,
) -> list[str] | None:
    """Resolve a post-comma chunk using a longer lexicon phrase from the prior clause.

    Pause commas split lookup (``兄弟,相称``) but user/lexicon entries such as
  ``兄弟相称`` still need to apply to the suffix chunk so ``称`` reads chēng.
    """
    if not full_text or segment_start is None or not segment_hanzi:
        return None

    word_pinyin_dict = lexicon["word_pinyin_dict"]
    max_word_len = lexicon["max_word_len"]
    prefix = _hanzi_prefix_before_segment(
        full_text,
        segment_start,
        max_word_len - len(segment_hanzi),
    )
    if not prefix:
        return None

    combined = prefix + segment_hanzi
    suffix_len = len(segment_hanzi)
    for length in range(min(max_word_len, len(combined)), suffix_len, -1):
        cand = combined[-length:]
        if cand == segment_hanzi or not cand.endswith(segment_hanzi):
            continue
        pinyin = word_pinyin_dict.get(cand)
        if not pinyin:
            continue
        if _candidate_conflicts_with_contextual_polyphone(cand, pinyin):
            continue
        cand_syllables = _normalize_lexicon_pinyin(pinyin)
        if len(cand_syllables) < suffix_len:
            continue
        return cand_syllables[-suffix_len:]
    return None


_bpmf_base_to_pinyin_cache = None
_BPMF_TONE_TO_NUMBER = {"ˉ": "1", "ˊ": "2", "ˇ": "3", "ˋ": "4", "˙": "5"}


def _get_bpmf_base_to_pinyin() -> dict[str, str]:
    global _bpmf_base_to_pinyin_cache
    if _bpmf_base_to_pinyin_cache is not None:
        return _bpmf_base_to_pinyin_cache
    mapping = {bpmf: py for py, bpmf in _load_pinyin_2_bpmf().items()}
    _bpmf_base_to_pinyin_cache = mapping
    return mapping


def _bopomofo_syllable_to_numbered_pinyin(bopomofo: str) -> str | None:
    """Convert one numbered-pinyin-style bopomofo syllable back to e.g. yi4."""
    if not bopomofo:
        return None
    tone = "1"
    body = bopomofo
    if body[-1] in _BPMF_TONE_TO_NUMBER:
        tone = _BPMF_TONE_TO_NUMBER[body[-1]]
        body = body[:-1]
    py_base = _get_bpmf_base_to_pinyin().get(body)
    if py_base is None:
        return None
    return f"{py_base}{tone}"


def _apply_mandarin_tone_sandhi(hanzi_text: str, syllables: list[str]) -> list[str]:
    """Apply the legacy per-word rules only when unified zh_g2p is unavailable."""
    if _use_unified_zh_g2p():
        return syllables
    if not hanzi_text or not syllables or len(hanzi_text) != len(syllables):
        return syllables

    frontend = _get_chinese_frontend()
    bopomofos: list[str] = []
    erhua_tails: list[list[str]] = []
    for py in syllables:
        parts = frontend._pinyin_to_bopomofos(py)
        if parts is None:
            return syllables
        bopomofos.append(parts[0])
        erhua_tails.append(parts[1:])

    bopomofos = get_g2p_sandhi_engine().ops.apply_word_sandhi_bopomofo(hanzi_text, bopomofos)

    adjusted: list[str] = []
    for bpmf, tail, orig_py in zip(bopomofos, erhua_tails, syllables):
        converted = _bopomofo_syllable_to_numbered_pinyin(bpmf)
        if converted is None:
            adjusted.append(orig_py)
            continue
        if tail:
            base, tone = converted[:-1], converted[-1]
            converted = f"{base}r{tone}"
        adjusted.append(converted)
    return adjusted


def _adjust_a_tone_from_following_punct(syllable: str, following_char: str | None) -> str:
    if _use_unified_zh_g2p():
        return syllable
    return get_g2p_sandhi_engine().ops.adjust_a_tone_from_following_punct(syllable, following_char)


# Optional G2P overrides for the syllabic-nasal interjection 嗯.
# _SYLLABIC_NG_ACTIVE: en* -> ng* (ㄫ token) for tone-embedding inventories.
# _SKIP_SYLLABIC_NG_FOR_INTERJECTION: keep en* (ㄣ) and sandhi, but skip ng/ㄫ.
_SYLLABIC_NG_ACTIVE = False
_SKIP_SYLLABIC_NG_FOR_INTERJECTION = False
_SYLLABIC_NG_CHARS = frozenset("嗯")
_syllabic_ng_re = re.compile(r"^e?n(\d)$")
_en_interjection_degraded_re = re.compile(r"^n(\d)$")


def _apply_syllabic_ng(hanzi_word: str, syllables: list[str]) -> list[str]:
    if not _SYLLABIC_NG_ACTIVE:
        return syllables
    if len(hanzi_word) != len(syllables) or not (_SYLLABIC_NG_CHARS & set(hanzi_word)):
        return syllables
    out = []
    for ch, py in zip(hanzi_word, syllables):
        if ch in _SYLLABIC_NG_CHARS:
            m = _syllabic_ng_re.match(py)
            if m:
                py = "ng" + m.group(1)
        out.append(py)
    return out


def _coalesce_en_interjection_pinyin(hanzi_word: str, syllables: list[str]) -> list[str]:
    """Map degraded n* back to en* for 嗯 when skipping the ng/ㄫ rewrite."""
    if not _SKIP_SYLLABIC_NG_FOR_INTERJECTION:
        return syllables
    if len(hanzi_word) != len(syllables):
        return syllables
    out = list(syllables)
    for idx, ch in enumerate(hanzi_word):
        if ch not in _SYLLABIC_NG_CHARS:
            continue
        m = _en_interjection_degraded_re.match(out[idx])
        if m:
            out[idx] = "en" + m.group(1)
    return out


def _finalize_word_pinyin_syllables(
    word: str,
    lex_syllables: list[str],
    *,
    full_text: str | None = None,
    char_offset: int = 0,
    word_char_positions: list[int] | None = None,
    prev_word: str | None = None,
) -> list[str]:
    """Apply Mandarin sandhi then optional 嗯 overrides."""
    syllables = list(lex_syllables)
    syllables = _apply_mandarin_tone_sandhi(word, syllables)
    syllables = _apply_interjection_tones(
        word,
        syllables,
        full_text=full_text,
        char_offset=char_offset,
        word_char_positions=word_char_positions,
        prev_word=prev_word,
    )
    syllables = _apply_bu_question_neutral_tones(
        word,
        syllables,
        full_text=full_text,
        char_offset=char_offset,
        word_char_positions=word_char_positions,
        prev_word=prev_word,
    )
    syllables = _apply_syllabic_ng(word, syllables)
    return _coalesce_en_interjection_pinyin(word, syllables)


def _apply_interjection_tones(
    word: str,
    syllables: list[str],
    *,
    full_text: str | None,
    char_offset: int,
    word_char_positions: list[int] | None = None,
    prev_word: str | None = None,
) -> list[str]:
    if _use_unified_zh_g2p():
        return syllables
    return get_g2p_sandhi_engine().apply_interjection_tones(
        word,
        syllables,
        full_text=full_text,
        char_offset=char_offset,
        word_char_positions=word_char_positions,
        prev_word=prev_word,
    )


def _apply_bu_question_neutral_tones(
    word: str,
    syllables: list[str],
    *,
    full_text: str | None,
    char_offset: int,
    word_char_positions: list[int] | None = None,
    prev_word: str | None = None,
) -> list[str]:
    if _use_unified_zh_g2p():
        return syllables
    return get_g2p_sandhi_engine().apply_bu_question_neutral_tones(
        word,
        syllables,
        full_text=full_text,
        char_offset=char_offset,
        word_char_positions=word_char_positions,
        prev_word=prev_word,
    )


def _apply_a_interjection_tones(
    word: str,
    syllables: list[str],
    *,
    full_text: str | None,
    char_offset: int,
    word_char_positions: list[int] | None = None,
) -> list[str]:
    return _apply_interjection_tones(
        word,
        syllables,
        full_text=full_text,
        char_offset=char_offset,
        word_char_positions=word_char_positions,
    )


def _hanzi_chunk_to_pinyin_syllables(
    hanzi_text: str,
    *,
    full_text: str | None = None,
    chunk_offset: int = 0,
    char_positions: list[int] | None = None,
) -> list[str]:
    """Convert a hanzi-only run to numbered pinyin syllables via chinese_lexicon.txt."""
    hanzi_text = hanzi_text.strip()
    if not hanzi_text:
        return []

    lexicon = _load_chinese_lexicon()
    segment_start = char_positions[0] if char_positions else None
    cross_clause_syllables = _lexicon_pinyin_for_cross_clause_suffix(
        hanzi_text,
        full_text,
        segment_start,
        lexicon,
    )
    if cross_clause_syllables is not None:
        lex_syllables = list(cross_clause_syllables)
        return _finalize_word_pinyin_syllables(
            hanzi_text,
            lex_syllables,
            full_text=full_text,
            char_offset=chunk_offset,
            word_char_positions=char_positions,
        )

    words = _segment_hanzi_with_lexicon(hanzi_text, lexicon)
    words = get_g2p_sandhi_engine().merge_words(words)

    syllables: list[str] = []
    char_offset = 0
    for word_idx, word in enumerate(words):
        prev_word = words[word_idx - 1] if word_idx > 0 else None
        lex_syllables = _lexicon_pinyin_for_word(word, lexicon)
        if lex_syllables is None:
            lex_syllables = _pypinyin_syllables_for_text(word)
        word_syllables = _finalize_word_pinyin_syllables(
            word,
            lex_syllables,
            full_text=full_text,
            char_offset=chunk_offset + char_offset,
            word_char_positions=(
                char_positions[char_offset:char_offset + len(word)]
                if char_positions is not None
                else None
            ),
            prev_word=prev_word,
        )
        if word_syllables:
            if syllables:
                syllables.append("|")
            syllables.extend(word_syllables)
        char_offset += len(word)
    return syllables


def _debug_hanzi_chunk_g2p_stages(
    hanzi_text: str,
    *,
    full_text: str | None = None,
    chunk_offset: int = 0,
    char_positions: list[int] | None = None,
) -> dict:
    """Return lexicon segmentation / merge / per-word pinyin stages (debug tooling)."""
    hanzi_text = hanzi_text.strip()
    if not hanzi_text:
        return {"segmented": [], "merged": [], "words": [], "pinyin": []}

    lexicon = _load_chinese_lexicon()
    segment_start = char_positions[0] if char_positions else None
    cross_clause_syllables = _lexicon_pinyin_for_cross_clause_suffix(
        hanzi_text,
        full_text,
        segment_start,
        lexicon,
    )
    if cross_clause_syllables is not None:
        lex_syllables = list(cross_clause_syllables)
        final_syllables = _finalize_word_pinyin_syllables(
            hanzi_text,
            lex_syllables,
            full_text=full_text,
            char_offset=chunk_offset,
            word_char_positions=char_positions,
        )
        lex_only = final_syllables
        if _use_unified_zh_g2p():
            final_syllables = _apply_sentence_tone_sandhi(final_syllables)
        return {
            "segmented": [hanzi_text],
            "merged": [hanzi_text],
            "words": [
                {
                    "word": hanzi_text,
                    "lexicon": lex_syllables,
                    "sandhi": final_syllables,
                    "pypinyin_fallback": False,
                    "cross_clause": True,
                }
            ],
            "pinyin": final_syllables,
            "lexicon_pinyin": lex_only,
        }

    segmented = _segment_hanzi_with_lexicon(hanzi_text, lexicon)
    merged = get_g2p_sandhi_engine().merge_words(list(segmented))

    word_stages: list[dict] = []
    syllables: list[str] = []
    char_offset = 0
    for word_idx, word in enumerate(merged):
        prev_word = merged[word_idx - 1] if word_idx > 0 else None
        lex_syllables = _lexicon_pinyin_for_word(word, lexicon)
        pypinyin_fallback = False
        if lex_syllables is None:
            lex_syllables = _pypinyin_syllables_for_text(word)
            pypinyin_fallback = True
        final_syllables = _finalize_word_pinyin_syllables(
            word,
            lex_syllables,
            full_text=full_text,
            char_offset=chunk_offset + char_offset,
            word_char_positions=(
                char_positions[char_offset:char_offset + len(word)]
                if char_positions is not None
                else None
            ),
            prev_word=prev_word,
        )
        word_stages.append(
            {
                "word": word,
                "lexicon": lex_syllables,
                "sandhi": final_syllables,
                "pypinyin_fallback": pypinyin_fallback,
            }
        )
        if final_syllables:
            if syllables:
                syllables.append("|")
            syllables.extend(final_syllables)
        char_offset += len(word)

    lex_only = list(syllables)
    if _use_unified_zh_g2p() and syllables:
        syllables = _apply_sentence_tone_sandhi(syllables)

    return {
        "segmented": segmented,
        "merged": merged,
        "words": word_stages,
        "pinyin": syllables,
        "lexicon_pinyin": lex_only,
    }


def format_zh_lexicon_g2p_debug(text: str, *, cleaner: str | None = None) -> str | None:
    """Format a multi-line G2P trace for ``text2id_debug.sh`` (hanzi lexicon path only)."""
    if not text or not _hanzi_char_re.search(text):
        return None

    global _SKIP_SYLLABIC_NG_FOR_INTERJECTION
    skip_ng_prev = _SKIP_SYLLABIC_NG_FOR_INTERJECTION
    try:
        return _format_zh_lexicon_g2p_debug_body(text, cleaner=cleaner)
    finally:
        _SKIP_SYLLABIC_NG_FOR_INTERJECTION = skip_ng_prev


def _format_zh_lexicon_g2p_debug_body(text: str, *, cleaner: str | None = None) -> str | None:
    lines: list[str] = []
    working = text.strip()

    if cleaner in {
        "en_zh_dict_mixed_cleaners",
        "en_zh_dict_mixed_rhyme_body_tone_cleaners",
    }:
        try:
            from temp_cmu_g2p.mixed_cleaners import (
                _ARPA_BEFORE_HANZI_RE,
                _HANZI_BEFORE_ARPA_RE,
                preprocess_english_input,
            )

            pre = preprocess_english_input(working)
            pre = _HANZI_BEFORE_ARPA_RE.sub(r"\1 ", pre)
            pre = _ARPA_BEFORE_HANZI_RE.sub(r"\1 ", pre)
            if pre.strip() != working:
                lines.append(f"Preprocess: {pre.strip()}")
            working = pre
        except Exception:
            pass

    working = _normalize_zh_punct_to_halfwidth(working.strip())
    chunk_idx = 0
    i = 0
    n = len(working)
    while i < n:
        ch = working[i]
        if ch.isspace():
            i += 1
            continue
        if _is_zh_lexicon_break_char(ch):
            i += 1
            continue
        j = i
        while j < n and not working[j].isspace() and not _is_zh_lexicon_break_char(working[j]):
            j += 1
        segment_start = i
        segment = working[i:j]
        i = j
        if not _hanzi_char_re.search(segment):
            continue
        chunk_idx += 1
        if chunk_idx > 1:
            lines.append("")
        if chunk_idx > 1 or segment != working.strip():
            lines.append(f"Hanzi chunk: {segment}")
        clean = _strip_zh_structural_punct(segment) or segment
        hanzi_positions = _hanzi_char_positions_in_segment(segment, segment_start)
        hanzi_pos_cursor = 0
        for sub_chunk, is_hanzi in _split_hanzi_and_non_hanzi_runs(clean):
            if not is_hanzi:
                if sub_chunk.strip():
                    lines.append(f"Non-hanzi: {sub_chunk.strip()}")
                continue
            sub_len = len(sub_chunk)
            positions = hanzi_positions[hanzi_pos_cursor:hanzi_pos_cursor + sub_len]
            hanzi_pos_cursor += sub_len
            stages = _debug_hanzi_chunk_g2p_stages(
                sub_chunk,
                full_text=working,
                char_positions=positions,
            )
            lines.append(f"Segment: {' | '.join(stages['segmented'])}")
            if stages["merged"] != stages["segmented"]:
                lines.append(f"Merged:  {' | '.join(stages['merged'])}")
            for item in stages["words"]:
                lex = " ".join(item["lexicon"])
                tag = " [pypinyin]" if item["pypinyin_fallback"] else ""
                if item.get("cross_clause"):
                    tag += " [cross-clause]"
                lines.append(f"  {item['word']}: {lex}{tag}")
            lex_pinyin = stages.get("lexicon_pinyin")
            final_pinyin = stages["pinyin"]
            if lex_pinyin and " ".join(lex_pinyin) != " ".join(final_pinyin):
                lines.append(f"Sandhi:  {' '.join(final_pinyin)}")
            else:
                lines.append(f"Pinyin:  {' '.join(final_pinyin)}")

    return "\n".join(lines) if lines else None


def _split_hanzi_and_non_hanzi_runs(text: str) -> list[tuple[str, bool]]:
    """Split a no-space chunk into hanzi and non-hanzi runs.

    The mixed English frontend may emit ARPAbet directly next to hanzi, e.g.
    ``成S AH1`` or ``把D EH1``. Keeping that as one hanzi chunk drops the
    adjacent ARPAbet token when pypinyin handles the chunk.
    """
    runs: list[tuple[str, bool]] = []
    start = 0
    current_is_hanzi = bool(_hanzi_char_re.match(text[0])) if text else False
    for idx, ch in enumerate(text[1:], 1):
        is_hanzi = bool(_hanzi_char_re.match(ch))
        if is_hanzi != current_is_hanzi:
            runs.append((text[start:idx], current_is_hanzi))
            start = idx
            current_is_hanzi = is_hanzi
    if text:
        runs.append((text[start:], current_is_hanzi))
    return runs


def _process_zh_clause_segment(
    segment: str,
    *,
    full_text: str,
    segment_start: int,
) -> list[str]:
    """Expand a pause-bounded clause into pinyin syllables (+ pause punct if any).

    Structural punctuation is stripped before lexicon lookup; parenthetical asides
    get a ``，`` pause (``现行犯（如…）`` -> ``现行犯，如…``). Forms like
    ``（兄弟）相称`` still allow ``兄弟相称`` to match in the post-comma chunk.
    """
    tokens: list[str] = []
    clean = _strip_zh_structural_punct(segment)
    if not clean:
        return tokens

    hanzi_positions = _hanzi_char_positions_in_segment(segment, segment_start)
    hanzi_pos_cursor = 0

    if _hanzi_char_re.search(clean):
        for sub_chunk, is_hanzi in _split_hanzi_and_non_hanzi_runs(clean):
            if is_hanzi:
                n = len(sub_chunk)
                positions = hanzi_positions[hanzi_pos_cursor:hanzi_pos_cursor + n]
                hanzi_pos_cursor += n
                tokens.extend(
                    _hanzi_chunk_to_pinyin_syllables(
                        sub_chunk,
                        full_text=full_text,
                        char_positions=positions,
                    )
                )
            else:
                tokens.extend(_tokenize_mixed_pinyin_text(sub_chunk))
    else:
        tokens.extend(_tokenize_mixed_pinyin_text(clean))

    return tokens


def _hanzi_to_bopomofo_tokens(
    text,
    *,
    rhyme_body_tone: bool = False,
):
    """Convert raw Chinese characters through numbered pinyin to Bopomofo tokens.

    Hanzi segments are segmented with chinese_lexicon.txt (longest match), then
    resolved to numbered pinyin from the same lexicon; unknown chars fall back to
    pypinyin.  Output follows the same pinyin-to-Bopomofo path as direct input.
    Punctuation (fullwidth or halfwidth) is normalized to halfwidth and preserved.
    Parentheses / quotes / brackets are removed (not spoken); comma-like marks
    still split clauses and are kept in the output.
    """
    if not text:
        return ""

    text = _normalize_zh_punct_to_halfwidth(text.strip())
    token_parts: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if _is_zh_lexicon_break_char(ch):
            token_parts.append(ch)
            i += 1
            continue
        j = i
        while j < n and not text[j].isspace() and not _is_zh_lexicon_break_char(text[j]):
            j += 1
        segment = text[i:j]
        if _hanzi_char_re.search(segment):
            token_parts.extend(
                _process_zh_clause_segment(segment, full_text=text, segment_start=i)
            )
        else:
            token_parts.extend(_tokenize_mixed_pinyin_text(segment))
        i = j

    unified_g2p = _use_unified_zh_g2p()
    token_stream = list(token_parts)
    if unified_g2p and token_stream:
        token_stream = _apply_sentence_tone_sandhi(token_stream)
    return _pinyin_to_bopomofo_tokens(
        " ".join(token_stream),
        rhyme_body_tone=rhyme_body_tone,
        apply_entry_sandhi=not unified_g2p,
    )


_hanzi_char_re = re.compile(r"[\u4e00-\u9fff]")
_english_word_re = re.compile(r"[A-Za-z]+(?:['''-][A-Za-z]+)*")


def _segment_zh_en(text):
    """Split mixed text into (segment, lang) pairs: 'zh' or 'en'."""
    segments = []
    i = 0
    n = len(text)
    while i < n:
        if re.match(r"[A-Za-z]", text[i]):
            m = _english_word_re.match(text, i)
            if m:
                segments.append((m.group(), "en"))
                i = m.end()
            else:
                segments.append((text[i], "other"))
                i += 1
        else:
            j = i
            while j < n and not re.match(r"[A-Za-z]", text[j]):
                j += 1
            segments.append((text[i:j], "zh"))
            i = j
    return segments


def en_zh_dict_mixed_cleaners(text):
    """Hybrid cleaner: hanzi lexicon lookup + English CMUdict G2P + ARPAbet pass-through.

    Implementation lives in temp_cmu_g2p.
    """
    from temp_cmu_g2p.mixed_cleaners import en_zh_dict_mixed_cleaners as _impl
    return _impl(text)


def en_zh_dict_mixed_rhyme_body_tone_cleaners(text):
    """Like en_zh_dict_mixed_cleaners with rhyme-body + inline tone marks."""
    from temp_cmu_g2p.mixed_cleaners import en_zh_dict_mixed_rhyme_body_tone_cleaners as _impl
    return _impl(text)


def chinese_cleaners(text, phonemize=True):
    """Normalize Chinese text (number/symbol expansion). Used by mixed pipeline."""
    try:
        from lits.text.g2p.cn2an_transform import chinese_to_num
        text = chinese_to_num(text)
    except ImportError:
        pass
    text = re.sub(
        rf"[{re.escape(_zh_punct_resources().doc['resources']['char_sets']['chinese_cleaner_strip_chars'])}]",
        "",
        text,
    )
    text = re.sub(r"\s+", " ", text).strip()
    return text


def english_cleaners(text, phonemize=True):
    """Normalize English text (casing, whitespace). Used by mixed pipeline."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text
    return english_direct_phoneme_cleaners(text)
