import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from lits.text.frontend import frontend as NewFrotend
from lits.text.unified_text_normalizer import get_text_normalizer
from typing import List, Tuple
import os
import time
import re
import json
import regex
import unicodedata
try:
    import ttsfrd
except Exception:
    ttsfrd = None


chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
english_pattern = re.compile(r'[a-zA-Z.\']')
number_pattern = re.compile(r'\d')

control_chars_pattern = re.compile(r'[\x00-\x1f\x7f-\x9f]')
whitespace_pattern = re.compile(r'\s+')
_tn_module_enabled = False
_frontend_instance = None
_frontend_pid = None
_cosy_frd = None
_cosy_frd_pid = None


class _FrontendProxy:
    """Backward-compatible proxy for callers that import `frontend` directly."""

    def __getattr__(self, name):
        return getattr(_get_frontend(), name)


frontend = _FrontendProxy()


def set_tn_module_enabled(enabled: bool):
    global _tn_module_enabled
    _tn_module_enabled = bool(enabled)


def _get_frontend():
    global _frontend_instance, _frontend_pid

    pid = os.getpid()
    if _frontend_instance is None or _frontend_pid != pid:
        _frontend_instance = NewFrotend()
        _frontend_pid = pid
    return _frontend_instance


def _get_cosy_frd():
    global _cosy_frd, _cosy_frd_pid

    pid = os.getpid()
    if _cosy_frd_pid == pid:
        return _cosy_frd

    _cosy_frd = None
    _cosy_frd_pid = pid
    if ttsfrd is None:
        return None

    try:
        engine = ttsfrd.TtsFrontendEngine()
        engine.initialize('pretrained_models/CosyVoice-ttsfrd/resource')
        engine.set_lang_type('pinyinvg')
        _cosy_frd = engine
    except Exception:
        _cosy_frd = None
    return _cosy_frd


def _get_tn_normalizer(lang: str):
    if lang not in {"zh", "en"}:
        return None
    return get_text_normalizer(lang)


def tn_normalize_mixed_text(text: str) -> str:
    """按 EN-ZH 前端同款分流，先做统一 TextNormalizer 再交给 mix_g2p。"""
    if not text:
        return text
    if not _tn_module_enabled:
        return text
    # 使用与 en-zh-Lits 一致的分流逻辑：数字/符号会并入上下文，孤立 other 最终归到 zh
    segments = _get_frontend().get_segment(text)
    if not segments:
        return text

    out = []
    for seg, lang in segments:
        tn_lang = "zh" if lang == "zh" else "en"
        norm = _get_tn_normalizer(tn_lang)
        if norm is None:
            out.append(seg)
            continue
        normalized = norm.normalize(seg)
        out.append(normalized if normalized else seg)
    return "".join(out)

def normalize_unicode(text: str) -> str:
    """Unicode标准化"""
    return unicodedata.normalize('NFKC', text)

def remove_control_chars(text: str) -> str:
    """移除控制字符"""
    return control_chars_pattern.sub('', text)

def normalize_whitespace(text: str) -> str:
    """标准化空白字符"""
    return whitespace_pattern.sub(' ', text).strip()

def english_lower(text: str) -> str:
    return text.lower()


def detect_language_segments(text: str) -> List[Tuple[str, str]]:
        """
        检测文本中的语言片段
        返回: [(text_segment, language), ...]
        language: 'chinese', 'english', 'mixed', 'number', 'punctuation'
        """
        segments = []
        current_segment = ""
        current_language = "unknown"
        
        for char in text:
            if chinese_pattern.match(char):
                char_language = "chinese"
            elif english_pattern.match(char):
                char_language = "english"
            else:
                char_language = "unknown"
            
            # 如果语言类型改变，保存当前片段
            if current_language != char_language and current_language != 'unknown' and char_language != 'unknown':
                # 确定片段的语言类型
                segment_language = current_language
                segments.append((current_segment, segment_language))
                current_segment = ""
            
            current_segment += char
            current_language = char_language
        
        # 处理最后一个片段
        if current_segment:
            segment_language = current_language
            segments.append((current_segment, segment_language))
        
        return segments

def is_only_punctuation(text):
    # Regular expression: Match strings that consist only of punctuation marks or are empty.
    punctuation_pattern = r'^[\p{P}\p{S}]*$'
    return bool(regex.fullmatch(punctuation_pattern, text))

cosy_frd = None

def clean_text(text: str) -> str:
        """
        清理代码切换文本
        
        Args:
            text: 输入文本
            level: 清理级别 ('basic', 'standard', 'advanced')
        
        Returns:
            清理后的文本
        """
        if not text:
            return text
            
        # 基础清理
        text = normalize_unicode(text)
        text = remove_control_chars(text)
        text = normalize_whitespace(text)
            
        # 标准清理
        # 1. 检测语言片段
        # segments = detect_language_segments(text)
            
        # 2. 处理每个片段
        # processed_segments = []
        # for segment, language in segments:
        #     if language == 'english':
        #         segment = english_lower(segment)
            # processed_segment = do_cosy_tn(segment)
                
            # processed_segments.append(processed_segment)
            
        # 3. 合并处理后的片段
        # result = "".join(processed_segments)
            
        # 4. 标准化空白字符
        result = normalize_whitespace(text)
        
        if result and len(result) > 0:
            text = result

        # Reuse existing C++ normalizers (zh/en) before mix_g2p.
        text = tn_normalize_mixed_text(text)

        phonems, tokens, tn_text = _get_frontend().mix_g2p(text, 'zh-en')
        return phonems, tokens, tn_text

def do_cosy_tn(text):
    cosy_frd = _get_cosy_frd()
    if cosy_frd is None:
        return text
    texts = [i["text"] for i in json.loads(cosy_frd.do_voicegen_frd(text))["sentences"]]
    text = ''.join(texts)
    texts = [i for i in texts if not is_only_punctuation(i)]
    return text


if __name__ == '__main__':
    for i in range(10):
        start = time.time()
        phonems, tokens, tn_text = clean_text("Mr. Black，给我$10")
        # phonems, tokens, tn_text = clean_text("And they're really good.")
        # phonems, tn_text = text_norm_process("风扇扇起来好凉快11千克, it cost me $10.我买了11千克苹果")
        # tn_text = text_norm_process(" 台州今天多云转晴，18度～28度,升温3度，北风5-6级，当前空气质量指数57，空气还可以 ")
        print("phonems", phonems, "tn_text:", tn_text, "time used:", time.time() - start)
