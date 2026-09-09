"""Unified zh-en symbols: static Bopomofo vocab + ARPAbet extension."""

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

_base_symbols = [
    "<blank>", "<sil>", "<unk>", "_",
    "ㄅ", "ㄆ", "ㄇ", "ㄈ", "ㄉ", "ㄊ", "ㄋ", "ㄌ", "ㄍ", "ㄎ", "ㄏ",
    "ㄐ", "ㄑ", "ㄒ", "ㄓ", "ㄔ", "ㄕ", "ㄖ", "ㄗ", "ㄘ", "ㄙ",
    "ㄚ", "ㄛ", "ㄜ", "ㄝ", "ㄞ", "ㄟ", "ㄠ", "ㄡ", "ㄢ", "ㄣ", "ㄤ", "ㄥ", "ㄦ", "ㄧ", "ㄨ", "ㄩ",
    "ˉ", "ˊ", "ˇ", "ˋ", "˙",
    ";", ":", ",", ".", "!", "?", "¡", "¿", "—", "…", "'", "\"", "«", "»", "“", "”", " ",
    "/", "-", "٪", "×", "÷", "+", "=", "*", "%", "^", "°", "’", "–",
    "(", ")"
]

symbols = list(_base_symbols)
for token in _arpabet_tokens:
    if token not in _base_symbols:
        symbols.append(token)

PAD_ID = symbols.index("<blank>")
SIL_ID = symbols.index("<sil>")
SPACE_ID = symbols.index("_")
UNK_ID = symbols.index("<unk>")


if __name__ == "__main__":
    id2symbol = {idx: symb for idx, symb in enumerate(symbols)}
    symbol2id = {symb: idx for idx, symb in enumerate(symbols)}
    print(f"EN-ZH 混合模型符号集总数: {len(symbols)}")
    print(symbol2id)