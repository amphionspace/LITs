import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# from lits.text.char_symbols.langs.en_g2p_tokens import symbols as en_g2p_symbols
from lits.text.char_symbols.langs.zh_en_direct_tokens import symbols as zh_en_direct_symbols
from lits.text.char_symbols.langs.zh_en_rhyme_body_tone_tokens import symbols as zh_en_rhyme_body_tone_symbols


lang2symbols = {
    # "en-g2p": en_g2p_symbols,
    "zh-en-direct": zh_en_direct_symbols,
    "zh-en-rhyme-body-tone": zh_en_rhyme_body_tone_symbols,
}


def _build_inventory(symbols):
    symbol_to_id = {symbol: idx for idx, symbol in enumerate(symbols)}
    id_to_symbol = {idx: symbol for idx, symbol in enumerate(symbols)}
    return {
        "symbols": list(symbols),
        "symbol_to_id": symbol_to_id,
        "id_to_symbol": id_to_symbol,
    }


lang2inventory = {
    model_key: _build_inventory(symbols)
    for model_key, symbols in lang2symbols.items()
}
