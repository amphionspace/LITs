import re

from . import language_cleaners as cleaners
from .bopomofo_utils import TONE_MARK_TO_ID
from .char_symbols.symbol_inventories import lang2inventory


class UnknownCleanerException(Exception):
    pass


_SYMBOLS_DB = lang2inventory

_CLEANER_TO_MODEL_KEY = {
    "pinyin_direct_mixed_cleaners": "zh-en-direct",
    "pinyin_direct_mixed_rhyme_body_tone_cleaners": "zh-en-rhyme-body-tone",
    "en_zh_dict_mixed_cleaners": "zh-en-direct",
    "en_zh_dict_mixed_rhyme_body_tone_cleaners": "zh-en-rhyme-body-tone",
    "english_direct_phoneme_cleaners": "en-g2p",
    "zh_en_phoneme_passthrough_cleaners": "zh-en-rhyme-body-tone",
}

_ACTIVE_MODEL_KEY = None
symbols = []
_symbol_to_id = {}
_id_to_symbol = {}
_BLANK_ID = 0
_TOKEN_SEQUENCE_CLEANERS = {
    "pinyin_direct_mixed_cleaners",
    "pinyin_direct_mixed_rhyme_body_tone_cleaners",
    "en_zh_dict_mixed_cleaners",
    "en_zh_dict_mixed_rhyme_body_tone_cleaners",
    "english_direct_phoneme_cleaners",
    "zh_en_phoneme_passthrough_cleaners",
}
_TOKEN_MODEL_KEYS = {
    "en-g2p",
    "zh-en-direct",
    "zh-en-rhyme-body-tone",
}
_RHYME_BODY_TONE_BACKFILL_KEYS = {"zh-en-rhyme-body-tone"}
_TONE_PARALLEL_MODEL_KEYS = _RHYME_BODY_TONE_BACKFILL_KEYS


def _activate_model_symbols(model_key: str):
    global _ACTIVE_MODEL_KEY, symbols, _symbol_to_id, _id_to_symbol

    payload = _SYMBOLS_DB.get(model_key)
    if payload is None:
        raise ValueError(f"model_key '{model_key}' not found in symbol inventories")

    symbols_local = payload.get("symbols", [])
    if not symbols_local:
        raise ValueError(f"'symbols' for model_key '{model_key}' is empty")

    symbol_to_id = payload.get("symbol_to_id")
    if not symbol_to_id:
        symbol_to_id = {s: i for i, s in enumerate(symbols_local)}

    id_to_symbol = payload.get("id_to_symbol")
    if id_to_symbol:
        id_to_symbol = {int(k): v for k, v in id_to_symbol.items()}
    else:
        id_to_symbol = {i: s for s, i in symbol_to_id.items()}

    _ACTIVE_MODEL_KEY = model_key
    symbols = symbols_local
    _symbol_to_id = symbol_to_id
    _id_to_symbol = id_to_symbol


def _resolve_model_key(cleaner_names):
    for name in cleaner_names:
        if name in _CLEANER_TO_MODEL_KEY:
            return _CLEANER_TO_MODEL_KEY[name]
    return next(iter(_SYMBOLS_DB.keys()))


def _ensure_active_symbols():
    if not _symbol_to_id:
        _activate_model_symbols(next(iter(_SYMBOLS_DB.keys())))


def _token_to_id(token, unk_id):
    token_id = _symbol_to_id.get(token)
    if token_id is not None:
        return token_id
    return unk_id


def _backfill_rhyme_body_tones(tokens, unk_id):
    """Map inline tone marks to parallel tone ids for MAS 2D lookup."""
    sequence = []
    tones = []
    for token in tokens:
        if token in TONE_MARK_TO_ID:
            tone_id = TONE_MARK_TO_ID[token]
            if sequence:
                tones[-1] = tone_id
            sequence.append(_symbol_to_id.get(token, unk_id))
            tones.append(0)
            continue
        token_id, tone_id = _token_to_id(token, unk_id), 0
        sequence.append(token_id)
        tones.append(tone_id)
    return sequence, tones


def phonemes_to_sequence_with_tones(cleaned_text: str, cleaner_names):
    """Map a pre-cleaned phoneme token line to ids (skips cleaner execution)."""
    model_key = _resolve_model_key(cleaner_names)
    if model_key != _ACTIVE_MODEL_KEY:
        _activate_model_symbols(model_key)

    clean_text = cleaned_text.strip()
    backfill_tones = model_key in _RHYME_BODY_TONE_BACKFILL_KEYS
    emit_tones = model_key in _TONE_PARALLEL_MODEL_KEYS
    unk_id = _symbol_to_id.get("<unk>", _symbol_to_id.get(" ", 0))

    token_list = clean_text.split()
    if backfill_tones:
        sequence, tones = _backfill_rhyme_body_tones(token_list, unk_id)
    else:
        sequence = [_token_to_id(token, unk_id) for token in token_list]
        tones = [0] * len(sequence)
    return sequence, tones if emit_tones else None, clean_text


def phonemes_to_sequence(cleaned_text: str, cleaner_names):
    sequence, _, clean_text = phonemes_to_sequence_with_tones(cleaned_text, cleaner_names)
    return sequence, clean_text


def text_to_sequence(text, cleaner_names):
    sequence, _, clean_text = text_to_sequence_with_tones(text, cleaner_names)
    return sequence, clean_text


def text_to_sequence_with_tones(text, cleaner_names, prepend_sil=True):
    """Convert text to IDs and optionally prepend sentence-initial ``<sil>``.

    The leading silence token gives MAS a dedicated place for recording onset
    silence instead of assigning those frames to the first spoken token. For
    zh-en-rhyme-body-tone inventories, inline tone marks are also backfilled
    into parallel tone IDs.
    """
    model_key = _resolve_model_key(cleaner_names)
    if model_key != _ACTIVE_MODEL_KEY:
        _activate_model_symbols(model_key)

    clean_text = _clean_text(text, cleaner_names)
    backfill_tones = model_key in _RHYME_BODY_TONE_BACKFILL_KEYS
    emit_tones = model_key in _TONE_PARALLEL_MODEL_KEYS

    sequence = []
    tones = []
    unk_id = _symbol_to_id.get("<unk>", _symbol_to_id.get(" ", 0))
    if any(name in _TOKEN_SEQUENCE_CLEANERS for name in cleaner_names):
        token_list = clean_text.split()
        if backfill_tones:
            sequence, tones = _backfill_rhyme_body_tones(token_list, unk_id)
        else:
            for token in token_list:
                sequence.append(_token_to_id(token, unk_id))
                tones.append(0)
    else:
        for symbol in clean_text:
            sequence.append(_symbol_to_id.get(symbol, unk_id))
            tones.append(0)
    if prepend_sil:
        sil_id = _symbol_to_id.get("<sil>")
        if sil_id is None:
            raise ValueError(f"model inventory {model_key!r} has no <sil> token")
        if not sequence or sequence[0] != sil_id:
            sequence.insert(0, sil_id)
            tones.insert(0, 0)
    return sequence, tones if emit_tones else None, clean_text


def _clean_text(text, cleaner_names):
    text = cleaners.preprocess_text(text)
    for name in cleaner_names:
        cleaner = getattr(cleaners, name, None)
        if not cleaner:
            raise UnknownCleanerException(f"Unknown cleaner: '{name}'")
        text = cleaner(text)
    return text


def sequence_to_text(sequence):
    _ensure_active_symbols()
    result = ""
    for symbol_id in sequence:
        if symbol_id != _BLANK_ID and symbol_id in _id_to_symbol:
            result += _id_to_symbol[symbol_id]
    return result


def ids_to_tokens(sequence):
    return sequence_to_text(sequence)


def cleaned_text_to_sequence(cleaned_text):
    _ensure_active_symbols()
    unk_id = _symbol_to_id.get("<unk>", 0)
    sequence = []
    for token in cleaned_text.split():
        sequence.append(_symbol_to_id.get(token, unk_id))
    return sequence
