"""zh-en symbols for the rhyme-body + inline tone mark paradigm.

Cleaners emit ``initial rhyme_body tone_mark`` per syllable with no ``_`` between
zh syllables (zh/en word boundaries unchanged); MAS uses parallel x_tones
backfilled from inline tone marks (see lits.text).
"""

from lits.text.bopomofo_utils import BOPOMOFO_TONES, collect_rhyme_bodies

_arpabet_tokens = [
    "AA0", "AA1", "AA2",
    "AE0", "AE1", "AE2",
    "AH0", "AH1", "AH2",
    "AO0", "AO1", "AO2",
    "AW0", "AW1", "AW2",
    "AX0", "AX1", "AX2",
    "AY0", "AY1", "AY2",
    "B", "CH", "D", "DH",
    "EH0", "EH1", "EH2",
    "ER0", "ER1", "ER2",
    "EY0", "EY1", "EY2",
    "F", "G", "HH",
    "IH0", "IH1", "IH2",
    "IY0", "IY1", "IY2",
    "JH", "K", "L", "M", "N", "NG",
    "OW0", "OW1", "OW2",
    "OY0", "OY1", "OY2",
    "P", "R", "S", "SH", "T", "TH",
    "UH0", "UH1", "UH2",
    "UW0", "UW1", "UW2",
    "V", "W", "Y", "Z", "ZH",
]

_bopomofo_initials = [
    "ㄅ", "ㄆ", "ㄇ", "ㄈ", "ㄉ", "ㄊ", "ㄋ", "ㄌ", "ㄍ", "ㄎ", "ㄏ",
    "ㄐ", "ㄑ", "ㄒ", "ㄓ", "ㄔ", "ㄕ", "ㄖ", "ㄗ", "ㄘ", "ㄙ",
]

_punctuation = [
    ";", ":", ",", ".", "!", "?", "¡", "¿", "—", "…", "'", "\"", "«", "»", "“", "”", " ",
    "/", "-", "٪", "×", "÷", "+", "=", "*", "%", "^", "°", "’", "–",
    "(", ")"
]

_base_symbols: list[str] = []
for _token in (
    "<blank>", "<sil>", "<unk>", "_",
    *_bopomofo_initials,
    *collect_rhyme_bodies(),
    *BOPOMOFO_TONES,
    *_punctuation,
):
    if _token not in _base_symbols:
        _base_symbols.append(_token)

symbols = list(_base_symbols)
for token in _arpabet_tokens:
    if token not in _base_symbols:
        symbols.append(token)

PAD_ID = symbols.index("<blank>")
SIL_ID = symbols.index("<sil>")
SPACE_ID = symbols.index("_")
UNK_ID = symbols.index("<unk>")


if __name__ == "__main__":
    rhymes = collect_rhyme_bodies()
    print(f"无调韵母 token 数: {len(rhymes)}")
    print(f"声调 mark 数: {len(BOPOMOFO_TONES)}")
    print(f"EN-ZH rhyme-body-tone 符号集总数: {len(symbols)}")
