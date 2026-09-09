"""MFA mandarin_china_mfa (IPA) <-> 354-model Bopomofo token mapping.

354 zh tokens = 21 initials + 225 rhyme+tone tokens (+ ㄦ* handled inside rhyme_tone set).
Mapping is built from:
  1. Fixed consonant table for initials.
  2. MFA dictionary lookups on representative single-character readings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from lits.text.bopomofo_utils import (
    BOPOMOFO_INITIALS,
    BOPOMOFO_TONES,
    PINYIN_TONE_TO_BPMF,
    bpmf_syllable_to_tokens,
    load_pinyin_2_bpmf,
    split_bpmf_body,
)
from lits.text.char_symbols.langs import zh_en_rhyme_tone_tokens as zh_en_rt

PIPELINE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_ROOT.parent
DEFAULT_MFA_DICT = Path("/chenmingjie/xingwen/tools/MFA/pretrained_models/dictionary/mandarin_china_mfa.dict")
DEFAULT_MAPPING_TSV = PIPELINE_ROOT / "mapping" / "mfa_zh354_token_mapping.tsv"

# Bopomofo initial -> MFA IPA consonant (mandarin_china_mfa phone set).
BPMF_INITIAL_TO_MFA: dict[str, str] = {
    "ㄅ": "p",
    "ㄆ": "pʰ",
    "ㄇ": "m",
    "ㄈ": "f",
    "ㄉ": "t",
    "ㄊ": "tʰ",
    "ㄋ": "n",
    "ㄌ": "l",
    "ㄍ": "k",
    "ㄎ": "kʰ",
    "ㄏ": "x",
    "ㄐ": "tɕ",
    "ㄑ": "tɕʰ",
    "ㄒ": "ɕ",
    "ㄓ": "ʈʂ",
    "ㄔ": "ʈʂʰ",
    "ㄕ": "ʂ",
    "ㄖ": "ʐ",
    "ㄗ": "ts",
    "ㄘ": "tsʰ",
    "ㄙ": "s",
}

# MFA Chao tone marks stripped from nucleus phones.
_MFA_TONE_SUFFIXES = ("˥˩", "˨˩˦", "˧˥", "˥", "˧", "˩")

# Rare syllables missing single-char MFA dict entries (hm, yai).
_MANUAL_RHYME_TONE_MFA: dict[str, tuple[str, ...]] = {
    "ㄏㄇˉ": ("m̩˥",),
    "ㄏㄇˊ": ("m̩˧˥",),
    "ㄏㄇˇ": ("m̩˨˩˦",),
    "ㄏㄇˋ": ("m̩˥˩",),
    "ㄏㄇ˙": ("m̩˩",),
    "ㄧㄞˉ": ("aj˥",),
    "ㄧㄞˊ": ("aj˧˥",),
    "ㄧㄞˇ": ("aj˨˩˦",),
    "ㄧㄞˋ": ("aj˥˩",),
    "ㄧㄞ˙": ("aj˩",),
}

# Whole-syllable zhuyin nuclei (no separate initial in 354 tokenization).
_WHOLE_SYLLABLE_RHYMES = frozenset({"ㄓ", "ㄔ", "ㄕ", "ㄖ", "ㄗ", "ㄘ", "ㄙ", "ㄏㄇ", "ㄧㄞ"})

# Rhyme-body overrides applied after representative-char dict lookup.
# ㄨㄛ: 我's top-probability dict variant is the reduced "ʔ o" which drops the
# [w] glide, silently losing its duration for every -uo syllable (说 = ʂ w o).
# Expected phones list all possible onsets; ʔ/w are optional-if-present at
# match time (see run.py), so all three dict variants (o / ʔ o / w o) match.
_RHYME_BODY_MFA_OVERRIDES: dict[str, tuple[str, ...]] = {
    "ㄨㄛ": ("ʔ", "w", "o"),
}


@dataclass(frozen=True)
class TokenMapping:
    token_354: str
    token_type: str  # initial | rhyme_tone | erhua
    mfa_phones: tuple[str, ...]
    example_hanzi: str
    example_pinyin: str
    notes: str = ""

    @property
    def mfa_phones_str(self) -> str:
        return " ".join(self.mfa_phones)


def _strip_mfa_tone(phone: str) -> str:
    for suf in _MFA_TONE_SUFFIXES:
        if phone.endswith(suf):
            return phone[: -len(suf)]
    return phone


def mfa_phone_base(phone: str) -> str:
    """Tone-stripped MFA phone with common allophone collapse for alignment."""
    base = _strip_mfa_tone(phone)
    collapse = {
        "pʷ": "pʰ",
        "xʷ": "x",
        "tʲ": "t",
        "kʷ": "kʰ",
        "kʰʷ": "kʰ",
    }
    return collapse.get(base, base)


def mfa_phones_compatible(mfa_label: str, expected: str) -> bool:
    """Relaxed match for MFA alignments (tone sandhi, ʔ omission, labialization)."""
    if mfa_label == expected:
        return True
    if expected == "ʔ":
        return mfa_label == "ʔ"
    mb = mfa_phone_base(mfa_label)
    eb = mfa_phone_base(expected)
    if mb == eb:
        return True
    if eb in {"x", "ɕ"} and mb.startswith(eb):
        return True
    if eb == "w" and mb.startswith("w"):
        return True
    if eb == "j" and mb in {"j", "tɕ", "ɕ"}:
        return True
    if eb == "pʰ" and mb in {"pʰ", "pʷ", "p"}:
        return True
    if eb == "kʰ" and mb in {"kʰ", "kʷ", "k"}:
        return True
    if eb == "t" and mb in {"t", "tʲ"}:
        return True
  # j/q/x + üe: MFA uses labialized onset (ɕʷ/tɕʷ) for the initial token.
    if eb == "ɕ" and mb in {"ɕ", "ɕʷ"}:
        return True
    if eb == "tɕ" and mb in {"tɕ", "tɕʷ"}:
        return True
    if eb == "ə" and mb == "ə":
        return True
    return False


# j/q/x + üe in MFA: ɕʷ/tɕʷ e maps to 354 tokens ㄒ/ㄐ/ㄑ + ㄩㄝ*
JQX_INITIALS = frozenset("ㄐㄑㄒ")


def contextual_mfa_phones(
    token: str,
    prev_token: str | None,
    mapping: dict[str, TokenMapping],
) -> tuple[str, ...]:
    """Return MFA phones expected for *token* given the previous 354 token."""
    phones = mapping[token].mfa_phones
    if prev_token in JQX_INITIALS:
        if token.startswith("ㄩㄝ"):
            # ü absorbed into labialized onset; rhyme nucleus is e only.
            if phones and mfa_phone_base(phones[0]) == "ɥ":
                return phones[1:]
            return ("e",) if not phones else phones
        if token.startswith("ㄩㄢ"):
            # j/q/x + üan: Cʷ + e + n (e.g. ɕʷ e n → ㄒ ㄩㄢˇ)
            out = tuple(p for p in phones if mfa_phone_base(p) != "ɥ")
            return out if out else phones
        if token.startswith("ㄩㄥ"):
            # j/q/x + iong: glide absorbed into onset (熊 = ɕ u ŋ, 穷 = tɕʰ u ŋ),
            # so drop the leading j expected from zero-initial yong (用 = j u ŋ).
            out = tuple(p for p in phones if mfa_phone_base(p) != "j")
            return out if out else phones
    if token == "ㄜ˙" and prev_token in BPMF_INITIAL_TO_MFA:
        # 的/了 de/le: t/l + ə, not ʔo (鹅 e2 uses ㄜˊ).
        return ("ə",)
    return phones

def _is_mfa_consonant(phone: str) -> bool:
    base = _strip_mfa_tone(phone)
    return base in set(BPMF_INITIAL_TO_MFA.values()) or base in {
        "ɕ",
        "ʂ",
        "ʐ",
        "ʈʂ",
        "ʈʂʰ",
        "tɕ",
        "tɕʰ",
        "ts",
        "tsʰ",
        "f",
        "x",
        "j",
        "w",
        "l",
        "m",
        "n",
        "ŋ",
        "p",
        "pʰ",
        "t",
        "tʰ",
        "k",
        "kʰ",
        "ɻ",
        "ʔ",
    }


def _is_mfa_nucleus(phone: str) -> bool:
    if phone == "ɻ":
        return True
    return not _is_mfa_consonant(phone)


def load_mfa_dict_entries(dict_path: Path | None = None) -> dict[str, list[tuple[float, tuple[str, ...]]]]:
    """Parse mandarin_china_mfa.dict -> {word: [(prob, phones), ...]}."""
    dict_path = dict_path or DEFAULT_MFA_DICT
    entries: dict[str, list[tuple[float, tuple[str, ...]]]] = {}
    with dict_path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            word = parts[0]
            prob = float(parts[1])
            phones = tuple(parts[5].split())
            entries.setdefault(word, []).append((prob, phones))
    for word in entries:
        entries[word].sort(key=lambda x: -x[0])
    return entries


def _best_mfa_phones(entries: dict[str, list[tuple[float, tuple[str, ...]]]], word: str) -> tuple[str, ...] | None:
    opts = entries.get(word)
    if not opts:
        return None
    return opts[0][1]


def _pinyin_for_bpmf_syllable(bpmf_with_tone: str) -> str | None:
    """Find one numbered pinyin whose bopomofo matches *bpmf_with_tone*."""
    if not bpmf_with_tone:
        return None
    tone_mark = bpmf_with_tone[-1]
    if tone_mark not in BOPOMOFO_TONES:
        return None
    body = bpmf_with_tone[:-1]
    inv_tone = {v: k for k, v in PINYIN_TONE_TO_BPMF.items() if k in "12345"}
    tone_num = inv_tone.get(tone_mark)
    if tone_num is None:
        return None
    p2b = load_pinyin_2_bpmf()
    for py, bpmf in p2b.items():
        if bpmf + tone_mark == bpmf_with_tone or bpmf == body:
            if py[-1] in "12345":
                return py
            return f"{py}{tone_num}"
    return None


def _find_example_char(
    entries: dict[str, list[tuple[float, tuple[str, ...]]]],
    target_phones: tuple[str, ...],
) -> str:
    for word, opts in entries.items():
        if len(word) != 1:
            continue
        if opts[0][1] == target_phones:
            return word
    return ""


def _char_for_pinyin_syllable(py: str) -> str:
    """Pick a single hanzi whose MFA reading matches the syllable pinyin."""
    p2b = load_pinyin_2_bpmf()
    base = py[:-1] if py[-1] in "12345" else py
    if base not in p2b:
        return ""
    # Common single-char fallbacks for syllable lookup.
    fallbacks = {
        "ba": "八", "ma": "妈", "fa": "发", "da": "大", "ta": "他", "na": "那", "la": "拉",
        "ga": "嘎", "ka": "卡", "ha": "哈", "ji": "机", "qi": "七", "xi": "西",
        "zhi": "之", "chi": "吃", "shi": "是", "ri": "日", "zi": "子", "ci": "词", "si": "四",
        "a": "啊", "o": "哦", "e": "鹅", "ai": "爱", "ei": "诶", "ao": "奥", "ou": "欧",
        "an": "安", "en": "恩", "ang": "昂", "eng": "嗯", "er": "二",
        "yi": "一", "ya": "呀", "yo": "哟", "ye": "也", "yao": "要", "you": "有",
        "yan": "烟", "yin": "音", "yang": "阳", "ying": "英", "yong": "用",
        "wu": "五", "wa": "哇", "wo": "我", "wai": "外", "wei": "为", "wan": "万", "wen": "文", "wang": "王",
        "yu": "鱼", "yue": "月", "yuan": "元", "yun": "云",
        "bi": "比", "pie": "撇", "mie": "灭", "die": "爹", "tie": "铁", "nie": "捏", "lie": "列",
        "jia": "家", "qia": "恰", "xia": "下", "jian": "间", "qian": "千", "xian": "先",
        "jiang": "江", "jiong": "炯", "xiong": "胸",
        "bin": "宾", "pin": "拼", "min": "民", "lin": "林",
        "bing": "冰", "ping": "平", "ming": "明", "ding": "顶", "ting": "听", "ning": "宁", "ling": "灵",
        "diu": "丢", "liu": "六", "niu": "牛", "jiu": "九", "qiu": "秋", "xiu": "秀",
        "duo": "多", "tuo": "拖", "nuo": "诺", "luo": "罗", "guo": "国", "kuo": "阔", "huo": "火",
        "zuo": "做", "cuo": "错", "suo": "所", "zhuo": "桌", "chuo": "戳", "shuo": "说", "ruo": "若",
        "dui": "对", "tui": "腿", "gui": "贵", "kui": "亏", "hui": "会", "zui": "最", "cui": "翠", "sui": "岁",
        "zhui": "追", "chui": "吹", "shui": "水", "rui": "瑞",
        "duan": "段", "tuan": "团", "guan": "关", "kuan": "宽", "huan": "欢",
        "zuan": "钻", "cuan": "窜", "suan": "酸", "zhuan": "转", "chuan": "穿", "shuan": "栓", "ruan": "软",
        "zun": "尊", "cun": "村", "sun": "孙", "zhun": "准", "chun": "春", "shun": "顺", "run": "润",
        "zong": "宗", "cong": "从", "song": "松", "zhong": "中", "chong": "冲", "shong": "冲", "rong": "容",
        "hua": "花", "huai": "怀", "huang": "黄", "hun": "婚", "huo": "火",
        "lv": "绿", "lve": "略", "nü": "女", "nüe": "虐",
        "hm": "噷", "hng": "哼", "m": "呣", "n": "嗯", "ng": "嗯",
    }
    return fallbacks.get(base, "")


def _phones_for_pinyin_syllable(
    py: str,
    entries: dict[str, list[tuple[float, tuple[str, ...]]]],
) -> tuple[str, tuple[str, ...], str]:
    ch = _char_for_pinyin_syllable(py)
    if ch:
        phones = _best_mfa_phones(entries, ch)
        if phones:
            return ch, phones, py
    return "", (), py


def build_initial_mappings() -> list[TokenMapping]:
    rows: list[TokenMapping] = []
    for bpmf, mfa in BPMF_INITIAL_TO_MFA.items():
        rows.append(
            TokenMapping(
                token_354=bpmf,
                token_type="initial",
                mfa_phones=(mfa,),
                example_hanzi="",
                example_pinyin="",
                notes="fixed consonant table",
            )
        )
    return rows


def build_rhyme_tone_mappings(
    entries: dict[str, list[tuple[float, tuple[str, ...]]]] | None = None,
) -> list[TokenMapping]:
    entries = entries or load_mfa_dict_entries()
    p2b = load_pinyin_2_bpmf()
    rows: list[TokenMapping] = []
    missing: list[str] = []

    for token in sorted(t for t in zh_en_rt.symbols if t not in BOPOMOFO_INITIALS and any(x in t for x in BOPOMOFO_TONES)):
        tone = token[-1]
        rhyme_body = token[:-1]
        token_type = "erhua" if rhyme_body == "ㄦ" and tone == "˙" else "rhyme_tone"

        example_py = ""
        example_ch = ""
        mfa_phones: tuple[str, ...] = ()
        note_extra = ""

        best_score = -1
        for py_base, bpmf in p2b.items():
            for tone_num, tone_mark in PINYIN_TONE_TO_BPMF.items():
                if tone_num not in "12345":
                    continue
                if tone_mark != tone:
                    continue
                toks = bpmf_syllable_to_tokens(bpmf, tone_mark, combine_rhyme_tone=True)
                if token not in toks:
                    continue
                # Prefer exact whole-syllable tokenization (yi1 -> [ㄧˉ], not bi1 -> [ㄅ,ㄧˉ]).
                if toks == [token]:
                    score = 100
                elif toks[-1] == token and len(toks) == 2:
                    score = 50
                elif toks[-1] == token:
                    score = 10
                else:
                    score = 1
                if score <= best_score:
                    continue
                numbered_py = f"{py_base}{tone_num}"
                ch, phones, _ = _phones_for_pinyin_syllable(numbered_py, entries)
                if not phones:
                    continue
                best_score = score
                example_py = numbered_py
                example_ch = ch
                mfa_phones = phones
            if best_score >= 100:
                break

        if token == "ㄦ˙":
            # Corpus prep spaces out hanzi, so MFA aligns erhua 儿 as a
            # standalone word: "o˧˥ ɻ" or just "o˧˥" (never a bare ɻ).
            # ʔ is optional-if-present; trailing ɻ is an optional coda (run.py).
            mfa_phones = ("ʔ", "o", "ɻ")
            example_ch = "儿"
            example_py = "er5"
            note_extra = "erhua tail (o + optional ɻ)"
        elif not mfa_phones and token in _MANUAL_RHYME_TONE_MFA:
            mfa_phones = _MANUAL_RHYME_TONE_MFA[token]
            example_py = {"ㄏㄇ": "hm4", "ㄧㄞ": "yai1"}.get(rhyme_body, "")
            note_extra = "manual MFA override"
        elif not mfa_phones and rhyme_body == "ㄦ" and tone == "˙":
            mfa_phones = ("ʔ", "o", "ɻ")
            example_ch = "儿"
            example_py = "er5"
            note_extra = "erhua tail (o + optional ɻ)"
        elif not mfa_phones:
            missing.append(token)
            continue

        if rhyme_body in _RHYME_BODY_MFA_OVERRIDES:
            mfa_phones = _RHYME_BODY_MFA_OVERRIDES[rhyme_body]
            note_extra = "rhyme-body override (optional onsets)"

        note_extra = note_extra if note_extra else ""
        initial_in_354 = ""
        if py_base in p2b:
            initial_in_354, _ = split_bpmf_body(p2b[py_base])
        if initial_in_354 and rhyme_body not in _WHOLE_SYLLABLE_RHYMES:
            # Syllable uses initial + rhyme_tone in 354: exclude only the leading
            # onset already represented by the initial token, keeping coda phones.
            if mfa_phones and _is_mfa_consonant(mfa_phones[0]):
                rhyme_phones = mfa_phones[1:]
            else:
                rhyme_phones = mfa_phones
            mapped_phones = rhyme_phones or mfa_phones
            note = "rhyme phones only (initial mapped separately)"
        else:
            # Whole-syllable token (ㄓˉ, ㄙˉ, ㄧˉ, …): all MFA phones for the syllable.
            mapped_phones = mfa_phones
            note = note_extra or "whole syllable (all MFA phones)"

        rows.append(
            TokenMapping(
                token_354=token,
                token_type=token_type,
                mfa_phones=mapped_phones,
                example_hanzi=example_ch,
                example_pinyin=example_py,
                notes=note,
            )
        )

    if missing:
        raise RuntimeError(f"Failed to map {len(missing)} rhyme_tone tokens: {missing[:20]} ...")

    return rows


def build_all_mappings(dict_path: Path | None = None) -> list[TokenMapping]:
    entries = load_mfa_dict_entries(dict_path)
    return build_initial_mappings() + build_rhyme_tone_mappings(entries)


def validate_354_coverage(mappings: list[TokenMapping]) -> dict[str, list[str]]:
    """Ensure every Chinese token in 354 vocab has a mapping."""
    zh_symbols = {
        s
        for s in zh_en_rt.symbols
        if s in BOPOMOFO_INITIALS or (any(t in s for t in BOPOMOFO_TONES) and s[0] in "ㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎㄏㄐㄑㄒㄓㄔㄕㄖㄗㄘㄙㄚㄛㄜㄝㄞㄟㄠㄡㄢㄣㄤㄥㄦㄧㄨㄩ")
    }
    mapped = {m.token_354 for m in mappings}
    missing = sorted(zh_symbols - mapped)
    extra = sorted(mapped - zh_symbols)
    return {"missing": missing, "extra": extra, "zh_count": [str(len(zh_symbols))], "mapped_count": [str(len(mapped))]}


def write_mapping_tsv(path: Path, mappings: list[TokenMapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("token_354\ttoken_type\tmfa_phones\texample_hanzi\texample_pinyin\tnotes\n")
        for m in mappings:
            f.write(
                f"{m.token_354}\t{m.token_type}\t{m.mfa_phones_str}\t{m.example_hanzi}\t{m.example_pinyin}\t{m.notes}\n"
            )


def load_mapping_tsv(path: Path | None = None) -> dict[str, TokenMapping]:
    path = path or DEFAULT_MAPPING_TSV
    out: dict[str, TokenMapping] = {}
    with path.open(encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            token, typ, phones, hanzi, py, notes = parts[:6]
            out[token] = TokenMapping(token, typ, tuple(phones.split()), hanzi, py, notes)
    return out


if __name__ == "__main__":
    mappings = build_all_mappings()
    report = validate_354_coverage(mappings)
    print(f"354 zh symbols: {report['zh_count'][0]}, mapped: {report['mapped_count'][0]}")
    if report["missing"]:
        raise SystemExit(f"Missing mappings: {report['missing']}")
    write_mapping_tsv(DEFAULT_MAPPING_TSV, mappings)
    print(f"Wrote {DEFAULT_MAPPING_TSV} ({len(mappings)} rows)")
    print("\n=== Initials (21) ===")
    for m in mappings:
        if m.token_type == "initial":
            print(f"  {m.token_354} -> {m.mfa_phones_str}")
    print("\n=== Sample rhyme_tone ===")
    for m in mappings:
        if m.token_type == "rhyme_tone" and m.token_354 in {"ㄧˉ", "ㄨㄚˉ", "ㄓˉ", "ㄣˋ", "ㄨㄢˊ"}:
            print(f"  {m.token_354} -> {m.mfa_phones_str}  ({m.example_hanzi}/{m.example_pinyin}) [{m.notes}]")
    print("\n=== Erhua ===")
    for m in mappings:
        if m.token_type == "erhua":
            print(f"  {m.token_354} -> {m.mfa_phones_str}  ({m.example_hanzi})")
