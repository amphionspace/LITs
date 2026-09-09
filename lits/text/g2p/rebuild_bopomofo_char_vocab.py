#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

# 1. 基础与特殊控制符号
_pad = "<blank>" # add_blank
_sil = "<sil>"
_unknown = "<unk>"
_word_boundary = "_" # add between words, e.g., how_are_you

_special_symbols = [_pad, _sil, _unknown, _word_boundary]

# 2. 核心字母表
_english_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


# 3. 国际音标 (IPA) - 用于处理英语发音映射
_letters_ipa = (
    "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘ"
    "ɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
)

# 4. 中文：注音符号与音调
_bopomofo_tokens = [
    "ㄅ", "ㄆ", "ㄇ", "ㄈ",
    "ㄉ", "ㄊ", "ㄋ", "ㄌ",
    "ㄍ", "ㄎ", "ㄏ",
    "ㄐ", "ㄑ", "ㄒ",
    "ㄓ", "ㄔ", "ㄕ", "ㄖ",
    "ㄗ", "ㄘ", "ㄙ",
    "ㄚ", "ㄛ", "ㄜ", "ㄝ",
    "ㄞ", "ㄟ", "ㄠ", "ㄡ",
    "ㄢ", "ㄣ", "ㄤ", "ㄥ",
    "ㄦ", "ㄧ", "ㄨ", "ㄩ",
    "ˉ", "ˊ", "ˇ", "ˋ", "˙"
]

# 5. 标点符号 (整合阿语、英语及通用标点)
_punctuation = ';:,.!?¡¿—…"«»“” '
_additional_symbols = [
    '/', '-', '٪', '×', '÷', '+', '=', '*', '%', '^', '°', "'", '’', '–'
]

# --- 构建最终符号表 ---
symbols = (
    [_pad, _sil] + 
    list(_special_symbols) +
    list(_english_letters) +
    list(_letters_ipa) +
    list(_bopomofo_tokens) +
    list(_punctuation) +
    _additional_symbols
)

# 核心步骤：去重并保持顺序 (Ordered Set)
symbols = list(dict.fromkeys(symbols))

# ID 映射
PAD_ID = symbols.index(_pad)
SIL_ID = symbols.index(_sil)
UNK_ID = symbols.index("<unk>")
WORD_SEP_ID = symbols.index("_")

assert [PAD_ID, SIL_ID, UNK_ID, WORD_SEP_ID] == [0, 1, 2, 3], f"Expected [PAD_ID, SIL_ID, UNK_ID, WORD_SEP_ID] == [0, 1, 2, 3] for all char-based models. Please check your symbol mapping!"

def rebuild_vocab(symbols, output_path: Path) -> None:
    new_vocab = {token: idx for idx, token in enumerate(symbols)}
    output = {"vocab": new_vocab}
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a clean vocab.json with English phoneme tokens plus Mandarin Bopomofo char tokens."
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).with_name("vocab.json")),
        help="Path to write the rebuilt vocab.json",
    )
    args = parser.parse_args()

    rebuild_vocab(symbols, Path(args.output))


if __name__ == "__main__":
    main()
