"""Punctuation rule operations (skeleton implementations driven by JSON)."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from ..loader import resolve_char_map, resolve_char_set, resolve_constant

_WHITESPACE_PATTERN = re.compile(r"\s+")
_TRAILING_PUNCT_RUN_PATTERN = re.compile(
    r"(\s*[^\w\s_]\s*)+$",
    re.UNICODE,
)
_LITERAL_LINE_BREAK_ESCAPE_RE = re.compile(r"\\r\\n|\\n|\\r")
_REAL_LINE_BREAK_RE = re.compile(r"\r\n|\r|\n")
_PARAGRAPH_BREAK_RE = re.compile(r"[ \t]{2,}")
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U00002600-\U000026FF"
    "\U00002300-\U000023FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "]+",
    flags=re.UNICODE,
)
# Tag chars used in subregion flags (e.g. England 🏴󠁧󠁢󠁥󠁮󠁧󠁿); not always removed by pictograph ranges alone.
_TAG_CHAR_RE = re.compile(r"[\U000E0020-\U000E007F]", flags=re.UNICODE)
# Orphan VS16 / combining keycap after pictograph strip (e.g. 1️⃣ -> 1 + leftovers).
_EMOJI_TAIL_RE = re.compile(r"\uFE0F?\u20E3", flags=re.UNICODE)
_CONTROL_CHARS_RE = re.compile(
    r"[\x00-\x1f\x7f-\x9f"
    r"\u200b-\u200f"
    r"\u202a-\u202e"
    r"\u2066-\u2069"
    r"\ufeff\ufff9-\ufffb]"
)


def _is_ascii_alnum_char(ch: str) -> bool:
    return ch.isascii() and ch.isalnum()


def _is_cjk_char(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


def _newline_clause_break_char(segment: str) -> str:
    """Append fullwidth 。 only after CJK; otherwise use halfwidth period."""
    stripped = segment.rstrip()
    if stripped and _is_cjk_char(stripped[-1]):
        return "。"
    return "."


def _needs_clause_break_punct(text_before: str, acceptable: frozenset[str]) -> bool:
    segment = text_before.rstrip()
    if not segment:
        return False
    if segment.endswith("..."):
        return False
    return segment[-1] not in acceptable


def _is_english_word_internal_space(left: str, right: str) -> bool:
    if not left or not right:
        return False
    lch = left.rstrip()[-1]
    rch = right.lstrip()[0]
    return _is_ascii_alnum_char(lch) and _is_ascii_alnum_char(rch) and lch.isalpha() and rch.isalpha()


def _should_insert_comma_before_space(left: str, right: str, acceptable: frozenset[str]) -> bool:
    if not _needs_clause_break_punct(left, acceptable) or not left or not right:
        return False
    if _is_english_word_internal_space(left, right):
        return False
    lch = left.rstrip()[-1]
    rch = right.lstrip()[0]
    if _is_ascii_alnum_char(lch) or _is_ascii_alnum_char(rch):
        return False
    return _is_cjk_char(lch) and _is_cjk_char(rch)


def _next_nonempty_part(parts: list[str], start_idx: int) -> str:
    for part in parts[start_idx:]:
        if part.strip():
            return part
    return ""


def _next_part_is_markdown_list_item(part: str) -> bool:
    """True when the following chunk is another markdown list item (``, Word``)."""
    return bool(re.match(r"^,\s*[A-Za-z]", part.lstrip()))


def _merge_break_separated_segments(
    text: str,
    *,
    split_re: re.Pattern[str],
    acceptable_trailing: frozenset[str],
) -> str:
    if not split_re.search(text):
        return text
    parts = split_re.split(text)
    merged: list[str] = []
    for idx, part in enumerate(parts):
        segment = part.rstrip()
        next_part = _next_nonempty_part(parts, idx + 1)
        if (
            idx < len(parts) - 1
            and segment
            and _needs_clause_break_punct(segment, acceptable_trailing)
            and not _next_part_is_markdown_list_item(next_part)
        ):
            segment += _newline_clause_break_char(segment)
        if segment:
            merged.append(segment)
    return " ".join(merged)


def strip_literal_line_break_escapes(text: str) -> str:
    """Remove leftover literal \\n / \\r escape sequences as spaces."""
    if not text:
        return text
    return _LITERAL_LINE_BREAK_ESCAPE_RE.sub(" ", text)


def strip_trailing_backslash(text: str) -> str:
    """Drop a stray trailing backslash before post adds sentence punctuation."""
    if not text:
        return text
    stripped = text.rstrip()
    if stripped.endswith("\\"):
        return stripped[:-1] + text[len(stripped) :]
    return text


def expand_literal_line_break_escapes(text: str) -> str:
    """Turn literal \\n / \\r escape sequences into real line breaks."""
    if not text:
        return text
    return _LITERAL_LINE_BREAK_ESCAPE_RE.sub("\n", text)


def normalize_clause_break_punct(text: str, *, acceptable_trailing: frozenset[str]) -> str:
    if not text:
        return text

    text = _merge_break_separated_segments(
        text, split_re=_REAL_LINE_BREAK_RE, acceptable_trailing=acceptable_trailing
    )
    text = _merge_break_separated_segments(
        text, split_re=_PARAGRAPH_BREAK_RE, acceptable_trailing=acceptable_trailing
    )
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch != " ":
            out.append(ch)
            i += 1
            continue
        j = i
        while j < n and text[j] == " ":
            j += 1
        left = "".join(out)
        right = text[j:]
        if _should_insert_comma_before_space(left, right, acceptable_trailing):
            out.append("，")
        out.append(" ")
        i = j

    return "".join(out)


def strip_structural_quotes(
    text: str,
    *,
    single_quote_chars: frozenset[str],
    jp_quote_chars: frozenset[str],
    double_quote_chars: frozenset[str],
) -> str:
    """Remove structural quote marks; keep English apostrophes (``I'm``, ``don't``)."""
    if not text:
        return text
    out: list[str] = []
    n = len(text)
    for i, ch in enumerate(text):
        if ch in jp_quote_chars or ch in double_quote_chars:
            continue
        if ch in single_quote_chars:
            prev_is_letter = i > 0 and text[i - 1].isalpha()
            next_is_letter = i + 1 < n and text[i + 1].isalpha()
            if prev_is_letter and next_is_letter:
                out.append(ch)
            continue
        out.append(ch)
    return "".join(out)


def strip_structural_single_quotes(
    text: str,
    *,
    quote_chars: frozenset[str],
    jp_quote_chars: frozenset[str],
) -> str:
    return strip_structural_quotes(
        text,
        single_quote_chars=quote_chars,
        jp_quote_chars=jp_quote_chars,
        double_quote_chars=frozenset(),
    )


def collapse_empty_slash_segments(text: str) -> str:
    text = re.sub(r"(?:\s*/\s*){2,}", " / ", text)
    return re.sub(r"\s*/\s*$", "", text)


_LETTER_DOT_ABBREV_DEFAULT = r"\b(?:[A-Z][a-z]?\.)+(?:[A-Z][a-z]?)?\."


def collapse_letter_dot_abbrevs(text: str, *, pattern: str) -> str:
    """``U.S.`` / ``Ph.D.`` → ``US`` / ``PhD`` (drop internal periods)."""
    if not text:
        return text
    rx = re.compile(pattern)

    def repl(match: re.Match[str]) -> str:
        return "".join(ch for ch in match.group(0) if ch.isalpha())

    return rx.sub(repl, text)


_NUMERIC_DASH_RANGE_RE = re.compile(
    r"(?<!\d)(?<!:)(\d{1,2})\s*[-–—]\s*(\d{1,2})(?!\d)(?!\s*[/\.])(?!\s*°)"
)


def normalize_numeric_dash_ranges(text: str) -> str:
    """Turn prose ranges like ``6-8`` into ``6 to 8`` before dash stripping."""
    if not text:
        return text
    return _NUMERIC_DASH_RANGE_RE.sub(r"\1 to \2", text)


def replace_vertical_line_punct(
    text: str,
    *,
    source_chars: frozenset[str],
    replacement: str,
) -> str:
    """Turn news/UI clause separators (| / ｜ / 丨) into a comma-like pause."""
    if not text or not source_chars:
        return text
    for ch in source_chars:
        text = text.replace(ch, replacement)
    if replacement:
        text = re.sub(rf"{re.escape(replacement)}{{2,}}", replacement, text)
    return text


def remove_dashes(text: str, *, dash_chars: frozenset[str]) -> str:
    return text.translate(str.maketrans({ch: " " for ch in dash_chars}))


def collapse_whitespace(text: str) -> str:
    return _WHITESPACE_PATTERN.sub(" ", text).strip()


def dedupe_trailing_punctuation(text: str, *, exempt_chars: frozenset[str]) -> str:
    match = _TRAILING_PUNCT_RUN_PATTERN.search(text)
    if not match:
        return text
    trailing = text[match.start():]
    punct_chars = [ch for ch in trailing if not ch.isspace() and ch not in exempt_chars]
    if len(punct_chars) <= 1:
        return text
    if len(set(punct_chars)) == 1:
        return text[: match.start()] + punct_chars[0]
    return text


def ensure_trailing_sentence_punct(text: str, *, acceptable_trailing: frozenset[str]) -> str:
    if not text:
        return text
    stripped = text.rstrip()
    if not stripped:
        return text
    if stripped.endswith("..."):
        return text
    if stripped[-1] in acceptable_trailing:
        return text
    if stripped[-1].isalnum():
        return stripped + " ."
    return stripped + "."


def normalize_ellipsis(text: str, *, aliases: tuple[str, ...], target: str) -> str:
    for alias in aliases:
        text = text.replace(alias, target)
    return text


def nfkc_normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def strip_emojis(text: str) -> str:
    """Remove emoji sequences (flags, keycaps, tag-subregion flags, etc.)."""
    if not text:
        return text
    try:
        import emoji

        text = emoji.replace_emoji(text, replace="")
    except ImportError:
        text = _EMOJI_RE.sub("", text)
    text = _TAG_CHAR_RE.sub("", text)
    text = _EMOJI_TAIL_RE.sub("", text)
    return text


_ASTERISK_BOUNDARY_PUNCT = frozenset(",.;:!?。，；：！？""''…")


def _is_markdown_list_asterisk(text: str, i: int) -> bool:
    """True for bullet/list ``*`` markers (``* Love``, ``Love * Learn``), not ``**`` or ``*italic*``."""
    if i < 0 or i >= len(text) or text[i] != "*":
        return False
    if (i > 0 and text[i - 1] == "*") or (i + 1 < len(text) and text[i + 1] == "*"):
        return False
    if i > 0 and not text[i - 1].isspace() and text[i - 1] not in _ASTERISK_BOUNDARY_PUNCT:
        return False
    j = i + 1
    if j >= len(text) or not text[j].isspace():
        return False
    while j < len(text) and text[j].isspace():
        j += 1
    return j < len(text) and text[j].isalpha()


def _strip_spaced_markdown_asterisks(text: str) -> str:
    """Remove list/bullet ``*`` markers; insert ``, `` after the previous item."""
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        if not _is_markdown_list_asterisk(text, i):
            out.append(text[i])
            i += 1
            continue

        left = "".join(out).rstrip()
        while out and out[-1].isspace():
            out.pop()
        left = "".join(out)

        j = i + 1
        while j < n and text[j].isspace():
            j += 1

        if left and left[-1].isdigit() and j < n and text[j].isdigit():
            i = j
            continue

        if left and left[-1] in _ASTERISK_BOUNDARY_PUNCT:
            out.append(" ")
        elif left:
            out.append(", ")

        i = j

    return "".join(out)


def strip_markdown_asterisks(text: str) -> str:
    """Drop markdown bullets/bold/italic markers; keep the wrapped words."""
    if not text:
        return text
    text = _strip_spaced_markdown_asterisks(text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    # SQL empty-arg wildcard: COUNT(*) / COUNT( * ) -> COUNT()
    text = re.sub(r"\(\s*\*\s*\)", "()", text)
    # Exam optional markers: И)* -> И)
    text = re.sub(r"([А-ЯЁа-яё])\)\s*\*(?=\s)", r"\1)", text)
    # Other code wildcards when * directly follows alnum (not after '(')
    text = re.sub(r"(?<=[A-Za-z0-9])\*(?=[),.\s;]|$)", "", text)
    return text


def strip_markdown_code_ticks(text: str) -> str:
    """Remove markdown backtick code fences while keeping inner text."""
    if not text:
        return text
    text = text.replace("```", " ")
    return text.replace("`", " ")


def collapse_decorative_underscore_runs(text: str) -> str:
    """Drop form-fill / decorative underscore runs; keep single ``_`` in identifiers."""
    if not text:
        return text
    return re.sub(r"_{2,}", " ", text)


def expand_identifier_underscores(text: str) -> str:
    """Expand identifier underscores: date_column -> date underscore column."""
    if not text:
        return text
    return re.sub(r"(?<=[A-Za-z0-9])_(?=[A-Za-z0-9])", " underscore ", text)


# Common programming tokens: always expand to spoken English before TN/G2P.
_PROG_TOKEN_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"C\+\+"), "C plus plus"),
    (re.compile(r"C#"), "C sharp"),
)

# SQL/Oracle glued identifiers (no space in source) -> spaced tokens for CMUdict lookup.
_SQL_GLUED_SPLITS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bTRUNCDATE\b", re.I), "TRUNC DATE"),
    (re.compile(r"\bTRUNCMONTH\b", re.I), "TRUNC MONTH"),
    (re.compile(r"\bTRUNCYEAR\b", re.I), "TRUNC YEAR"),
    (re.compile(r"\bTRUNCDAY\b", re.I), "TRUNC DAY"),
    (re.compile(r"\bTRUNCHOUR\b", re.I), "TRUNC HOUR"),
    (re.compile(r"\bTRUNCMINUTE\b", re.I), "TRUNC MINUTE"),
    (re.compile(r"\bTRUNCSECOND\b", re.I), "TRUNC SECOND"),
    (re.compile(r"\bCOUNTDISTINCT\b", re.I), "COUNT DISTINCT"),
    (re.compile(r"\bGROUPBY\b", re.I), "GROUP BY"),
    (re.compile(r"\bORDERBY\b", re.I), "ORDER BY"),
    (re.compile(r"\bINNERJOIN\b", re.I), "INNER JOIN"),
    (re.compile(r"\bLEFTJOIN\b", re.I), "LEFT JOIN"),
    (re.compile(r"\bRIGHTJOIN\b", re.I), "RIGHT JOIN"),
    (re.compile(r"\bOUTERJOIN\b", re.I), "OUTER JOIN"),
    (re.compile(r"\bCROSSJOIN\b", re.I), "CROSS JOIN"),
    (re.compile(r"\bFULLJOIN\b", re.I), "FULL JOIN"),
    (re.compile(r"\bLENSTRING\b", re.I), "LEN STRING"),
    (re.compile(r"\bTOCHAR\b", re.I), "TO CHAR"),
    (re.compile(r"\bTODATE\b", re.I), "TO DATE"),
    (re.compile(r"\bTONUMBER\b", re.I), "TO NUMBER"),
    (re.compile(r"\bNVL2\b", re.I), "NVL 2"),
)


def normalize_programming_tokens_en(text: str) -> str:
    """Expand C++/C# etc. to spoken English (language-agnostic preprocess)."""
    if not text:
        return text
    for pattern, replacement in _PROG_TOKEN_REPLACEMENTS:
        text = pattern.sub(replacement, text)
    return text


def split_sql_glued_identifiers(text: str) -> str:
    """Split common glued SQL identifiers for per-word G2P / dict lookup."""
    if not text:
        return text
    for pattern, replacement in _SQL_GLUED_SPLITS:
        text = pattern.sub(replacement, text)
    return text


def strip_control_chars(text: str) -> str:
    if not text:
        return text
    return _CONTROL_CHARS_RE.sub("", text)


def translate_chars(text: str, char_map: dict[str, str | None]) -> str:
    return text.translate(str.maketrans(char_map))


def strip_chars(text: str, *, chars: frozenset[str]) -> str:
    return "".join(ch for ch in text if ch not in chars)


def collapse_double_hyphen(text: str) -> str:
    while "--" in text:
        text = text.replace("--", "-")
    return text


def dedupe_adjacent_sentence_punct(text: str, *, chars: frozenset[str]) -> str:
    if not chars:
        return text
    pattern = re.compile(r"([{}])\1+".format(re.escape("".join(sorted(chars)))))
    return pattern.sub(r"\1", text)


def _is_english_apostrophe(text: str, index: int) -> bool:
    """True when ``'`` is a contraction apostrophe (``don't``), not a quote mark."""
    if index < 0 or index >= len(text):
        return False
    ch = text[index]
    if ch not in ("'", "\u2018", "\u2019"):
        return False
    return (
        index > 0
        and text[index - 1].isalpha()
        and index + 1 < len(text)
        and text[index + 1].isalpha()
    )


def _should_insert_comma_after_paren_close(
    text: str,
    close_index: int,
    *,
    structural_chars: frozenset[str],
    acceptable_trailing: frozenset[str],
) -> bool:
    """True when non-empty text follows ``)`` without leading clause punctuation."""
    n = len(text)
    j = close_index + 1
    while j < n and text[j].isspace():
        j += 1
    if j >= n:
        return False
    next_ch = text[j]
    if next_ch in acceptable_trailing or next_ch in structural_chars:
        return False
    return True


def strip_zh_structural_punct(
    text: str,
    *,
    structural_chars: frozenset[str],
    acceptable_trailing: frozenset[str],
) -> str:
    if not text:
        return text

    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "(":
            close = text.find(")", i + 1)
            if close != -1:
                inner = text[i + 1:close]
                has_content = any(
                    c not in structural_chars and not c.isspace()
                    for c in inner
                )
                if has_content and out and _needs_clause_break_punct("".join(out), acceptable_trailing):
                    out.append("，")
            i += 1
            continue
        if ch == ")":
            if (
                out
                and _needs_clause_break_punct("".join(out), acceptable_trailing)
                and _should_insert_comma_after_paren_close(
                    text,
                    i,
                    structural_chars=structural_chars,
                    acceptable_trailing=acceptable_trailing,
                )
            ):
                out.append("，")
            i += 1
            continue
        if ch in structural_chars:
            if _is_english_apostrophe(text, i):
                out.append(ch)
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def apply_punctuation_op(
    op: str,
    text: str,
    doc: dict[str, Any],
    params: dict[str, Any] | None = None,
) -> str:
    params = params or {}
    resources = doc.get("resources", {})

    if op == "strip_literal_line_break_escapes":
        return strip_literal_line_break_escapes(text)

    if op == "strip_trailing_backslash":
        return strip_trailing_backslash(text)

    if op == "expand_literal_line_break_escapes":
        return expand_literal_line_break_escapes(text)

    if op == "normalize_clause_break_punct":
        acceptable = resolve_char_set(doc, params.get("acceptable_trailing", "acceptable_trailing_punct"))
        return normalize_clause_break_punct(text, acceptable_trailing=acceptable)

    if op == "strip_structural_quotes":
        single_quote_chars = resolve_char_set(
            doc, params.get("single_quote_chars", "structural_single_quote_chars")
        )
        jp_quote_chars = resolve_char_set(doc, params.get("jp_quote_chars", "jp_single_quote_chars"))
        double_quote_chars = resolve_char_set(
            doc, params.get("double_quote_chars", "structural_double_quote_chars")
        )
        return strip_structural_quotes(
            text,
            single_quote_chars=single_quote_chars,
            jp_quote_chars=jp_quote_chars,
            double_quote_chars=double_quote_chars,
        )

    if op == "strip_structural_single_quotes":
        quote_chars = resolve_char_set(doc, params.get("quote_chars", "structural_single_quote_chars"))
        jp_quote_chars = resolve_char_set(doc, params.get("jp_quote_chars", "jp_single_quote_chars"))
        return strip_structural_single_quotes(text, quote_chars=quote_chars, jp_quote_chars=jp_quote_chars)

    if op == "collapse_empty_slash_segments":
        return collapse_empty_slash_segments(text)

    if op == "normalize_numeric_dash_ranges":
        return normalize_numeric_dash_ranges(text)

    if op == "collapse_letter_dot_abbrevs":
        pattern_name = params.get("pattern", "letter_dot_abbrev")
        pattern_defs = resources.get("pattern_defs", {})
        pattern = pattern_defs.get(pattern_name, _LETTER_DOT_ABBREV_DEFAULT)
        return collapse_letter_dot_abbrevs(text, pattern=pattern)

    if op == "replace_vertical_line_punct":
        source_chars = resolve_char_set(doc, params["char_set"])
        replacement = params.get("replacement", "，")
        return replace_vertical_line_punct(
            text, source_chars=source_chars, replacement=replacement
        )

    if op == "remove_dashes":
        dash_chars = resolve_char_set(doc, params["char_set"])
        return remove_dashes(text, dash_chars=dash_chars)

    if op == "collapse_whitespace":
        return collapse_whitespace(text)

    if op == "dedupe_trailing_punctuation":
        exempt = resolve_char_set(doc, params.get("exempt_chars", "dedupe_punct_exempt"))
        return dedupe_trailing_punctuation(text, exempt_chars=exempt)

    if op == "ensure_trailing_sentence_punct":
        acceptable = resolve_char_set(doc, params.get("acceptable_trailing", "acceptable_trailing_punct"))
        return ensure_trailing_sentence_punct(text, acceptable_trailing=acceptable)

    if op == "normalize_ellipsis":
        aliases = tuple(resources["char_sets"][params["aliases"]])
        target = resolve_constant(doc, params["target"])
        return normalize_ellipsis(text, aliases=aliases, target=target)

    if op == "nfkc_normalize":
        return nfkc_normalize(text)

    if op == "strip_emojis":
        return strip_emojis(text)

    if op == "strip_markdown_asterisks":
        return strip_markdown_asterisks(text)

    if op == "strip_markdown_code_ticks":
        return strip_markdown_code_ticks(text)

    if op == "collapse_decorative_underscore_runs":
        return collapse_decorative_underscore_runs(text)

    if op == "expand_identifier_underscores":
        return expand_identifier_underscores(text)

    if op == "normalize_programming_tokens_en":
        return normalize_programming_tokens_en(text)

    if op == "split_sql_glued_identifiers":
        return split_sql_glued_identifiers(text)

    if op == "strip_control_chars":
        return strip_control_chars(text)

    if op == "translate_chars":
        char_map = resolve_char_map(doc, params["char_map"])
        return translate_chars(text, char_map=char_map)

    if op == "strip_chars":
        chars = resolve_char_set(doc, params["char_set"])
        return strip_chars(text, chars=chars)

    if op == "collapse_double_hyphen":
        return collapse_double_hyphen(text)

    if op == "dedupe_adjacent_sentence_punct":
        chars = resolve_char_set(doc, params["char_set"])
        return dedupe_adjacent_sentence_punct(text, chars=chars)

    if op == "strip_zh_structural_punct":
        structural = resolve_char_set(doc, params.get("structural_chars", "structural_punct_chars"))
        acceptable = resolve_char_set(doc, params.get("acceptable_trailing", "acceptable_trailing_punct"))
        return strip_zh_structural_punct(
            text,
            structural_chars=structural,
            acceptable_trailing=acceptable,
        )

    raise ValueError(f"unknown punctuation op: {op}")
