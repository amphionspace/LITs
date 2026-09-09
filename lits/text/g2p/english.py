import re
from unidecode import unidecode
import os
from lits.text.frontend_config import resource_path
import importlib.util
from lits.text.acronyms import (
    direct_acronym_phonemes,
    load_common_acronyms,
    preprocess_common_acronyms,
    segment_english_text,
    split_leading_acronym,
)

_special_map = [
    ('t|ɹ', 'tɹ'),
    ('d|ɹ', 'dɹ'),
    ('t|s', 'ts'),
    ('d|z', 'dz'),
    ('ɪ|ɹ', 'ɪɹ'),
    ('ɐ', 'ɚ'),
    ('ᵻ', 'ɪ'),
    ('əl', 'l'),
    ('x', 'k'),
    ('ɬ', 'l'),
    ('ʔ', 't'),
    ('n̩', 'n'),
    ('oː|ɹ', 'oːɹ')
]

# 加载缩写词音素字典

def _load_contraction_phonemes():
    """内部函数：加载缩写词音素字典，仅模块初始化时调用"""
    try:
        spec = importlib.util.spec_from_file_location(
            "contraction_phonemes",
            os.path.join(resource_path, "sources", "contraction_phonemes.py")
        )
        if spec is None or spec.loader is None:
            return {}
        contraction_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(contraction_module)
        return contraction_module.CONTRACTION_PHONEMES
    except Exception:
        return {}

# 模块加载时即读取，后续直接用
_contraction_phonemes = _load_contraction_phonemes()

def load_contraction_phonemes():
    """兼容旧接口，直接返回已缓存的缩写词音素字典"""
    return _contraction_phonemes


def apply_contraction_corrections(phonemes, original_text):
    """
    只在原始文本中出现缩写词时，才对音素序列做替换，避免误伤。
    phonemes: str 或 List[str]，tokenizer 输出
    original_text: str 或 List[str]，原始文本
    返回修正后的 phonemes
    """
    contraction_dict = load_contraction_phonemes()
    if isinstance(phonemes, str) and isinstance(original_text, str):
        return _replace_contractions_in_string(phonemes, original_text, contraction_dict)
    elif isinstance(phonemes, list) and isinstance(original_text, list):
        return [
            _replace_contractions_in_string(p, t, contraction_dict)
            for p, t in zip(phonemes, original_text)
        ]
    else:
        return phonemes

def _replace_contractions_in_string(phoneme_str, text, contraction_dict):
    # 以空格分词，逐词查找缩写
    words = text.split()
    phs = phoneme_str.split("|")
    new_phs = []
    idx = 0
    for word in words:
        # 跳过空词
        if not word:
            continue
        # 处理大小写
        key = word.lower()
        if key in contraction_dict:
            # 用字典音素替换
            src_ph = contraction_dict[key]["src"].split("|")
            dst_ph = contraction_dict[key]["dst"].split("|")
            # 在当前音素序列中查找 src_ph
            match_found = False
            for i in range(idx, len(phs) - len(src_ph) + 1):
                if phs[i:i+len(src_ph)] == src_ph:
                    new_phs.extend(dst_ph)
                    idx = i + len(src_ph)
                    match_found = True
                    break
            if not match_found:
                # 没找到就原样保留
                next_sep = idx
                while next_sep < len(phs) and phs[next_sep] not in [',','.','?','!',';','\'','…',':','_']:
                    next_sep += 1
                new_phs.extend(phs[idx:next_sep])
                idx = next_sep
        else:
            # 原样保留当前词的音素
            next_sep = idx
            while next_sep < len(phs) and phs[next_sep] not in [',','.','?','!',';','\'','…',':','_']:
                next_sep += 1
            new_phs.extend(phs[idx:next_sep])
            idx = next_sep
        # 跳过分隔符
        while idx < len(phs) and phs[idx] in [',','.','?','!',';','\'','…',':','_']:
            new_phs.append(phs[idx])
            idx += 1
    # 补充剩余音素
    if idx < len(phs):
        new_phs.extend(phs[idx:])
    return "|".join([p for p in new_phs if p != ""]) if new_phs else phoneme_str


#添加英文的特殊处理流程

def read_as_word(text):
    return text == preprocess_common_acronyms(text)

def special_process(text):
    return preprocess_common_acronyms(text)

def _normalize_english_text(text):
    text = re.sub(r"\s+", " ", text).strip()
    return text

# special map
def special_map(text):
    for regex, replacement in _special_map:
        regex = regex.replace("|", "\|")
        while re.search(r'(^|[_|]){}([_|]|$)'.format(regex), text):
            text = re.sub(r'(^|[_|]){}([_|]|$)'.format(regex), r'\1{}\2'.format(replacement), text)
    # text = re.sub(r'([,.!?])', r'|\1', text)
    return text

# Add some special operation
def english_to_ipa(text, text_tokenizer):
    #在这对文本进行归一化
    tn_text = ""
    if type(text) == str:
        # print("EN_INPUT_RAW", repr(text), type(text), flush=True)
        normalized_text = _normalize_english_text(text)
        # print("EN_INPUT_NORM", repr(normalized_text), flush=True)
        segments = segment_english_text(normalized_text)
        if len(segments) > 1 and any(kind == "acronym" for kind, _ in segments):
            combined_parts = []
            for kind, segment_text in segments:
                if kind == "acronym":
                    segment_phonemes = direct_acronym_phonemes(segment_text, text_tokenizer)
                else:
                    processed_text = special_process(segment_text)
                    segment_phonemes = text_tokenizer(processed_text)
                    segment_phonemes = apply_contraction_corrections(segment_phonemes, processed_text)

                segment_parts = [part for part in segment_phonemes.split("|") if part]
                if not segment_parts:
                    continue
                if combined_parts:
                    combined_parts.append("_")
                combined_parts.extend(segment_parts)

            if combined_parts:
                tn_text = normalized_text
                phonemes = "|".join(combined_parts)
                if phonemes[-1] in ",.?!_iːɪɜɚoɹɔɑuʊʌɛæeapbtdkɡfvθðszʃʒhmnŋjwləɾr̃çɐʲ⁼ʰx`→↑↓ɥɯçɸɰᵝɴgʑqɕɒɫyøœʁɲ:;'…ɣʈʐʂɤ̆ăʷ52436":
                    phonemes += "|_"
                return special_map(phonemes), tn_text

        direct_phonemes = direct_acronym_phonemes(normalized_text, text_tokenizer)
        # print("EN_DIRECT_CHECK", repr(normalized_text), repr(direct_phonemes), flush=True)
        if direct_phonemes is not None:
            # print("DIRECT_ACRONYM", normalized_text, direct_phonemes, flush=True)

            tn_text = normalized_text
            phonemes = direct_phonemes
            if phonemes[-1] in ",.?!_iːɪɜɚoɹɔɑuʊʌɛæeapbtdkɡfvθðszʃʒhmnŋjwləɾr̃çɐʲ⁼ʰx`→↑↓ɥɯçɸɰᵝɴgʑqɕɒɫyøœʁɲ:;'…ɣʈʐʂɤ̆ăʷ52436":
                phonemes += "|_"
            return special_map(phonemes), tn_text

        acronym, remainder = split_leading_acronym(normalized_text)
         #print("EN_LEADING_ACRONYM_CHECK", repr(normalized_text), repr(acronym), repr(remainder), flush=True)
        if acronym is not None and remainder:
            prefix_phonemes = direct_acronym_phonemes(acronym, text_tokenizer)
            remainder_text = special_process(remainder)
            remainder_phonemes = text_tokenizer(remainder_text)
            remainder_phonemes = apply_contraction_corrections(remainder_phonemes, remainder_text)
            combined_parts = [part for part in prefix_phonemes.split("|") if part]
            combined_parts.append("_")
            combined_parts.extend([part for part in remainder_phonemes.split("|") if part])
            phonemes = "|".join(combined_parts)
            tn_text = normalized_text
            if phonemes[-1] in ",.?!_iːɪɜɚoɹɔɑuʊʌɛæeapbtdkɡfvθðszʃʒhmnŋjwləɾr̃çɐʲ⁼ʰx`→↑↓ɥɯçɸɰᵝɴgʑqɕɒɫyøœʁɲ:;'…ɣʈʐʂɤ̆ăʷ52436":
                phonemes += "|_"
            return special_map(phonemes), tn_text
        text = special_process(normalized_text)
        tn_text = text
    else:
        text = [special_process(_normalize_english_text(t)) for t in text]
        tn_text = "".join(text)
    phonemes = text_tokenizer(text)
    # 缩写词音素修正
    phonemes = apply_contraction_corrections(phonemes, text)
    #将所有的ipa音素放到字符串里面，如果以ipa音素结尾，则添加blank，如果是非ipa音素结尾(可能是标点符号)，则不添加blank
    # if phonemes[-1] in "p⁼ʰmftnlkxʃs`ɹaoəɛɪeɑʊŋiuɥwæjːhdɡc":
    #对不完整的音素集进行补充
    if phonemes[-1] in ",.?!_iːɪɜɚoɹɔɑuʊʌɛæeapbtdkɡfvθðszʃʒhmnŋjwləɾr̃çɐʲ⁼ʰx`→↑↓ɥɯçɸɰᵝɴgʑqɕɒɫyøœʁɲ:;'…ɣʈʐʂɤ̆ăʷ52436":
        phonemes += "|_"
    if type(text) == str:
        return special_map(phonemes), tn_text
    else:
        result_ph = []
        for phone in phonemes:
            result_ph.append(special_map(phone))
        return result_ph, tn_text
