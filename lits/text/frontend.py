from lits.text.g2p import PhonemeBpeTokenizer
from typing import List
import tqdm
import os
import time
import re


class frontend():
    def __init__(self):
        self.text_tokenizer = PhonemeBpeTokenizer()
        self._tone_mark_chars = "āáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜńňǹḿ"
        self._hanzi_tone_pinyin_pattern = re.compile(
            r"[\u4e00-\u9fff][（(][A-Za-züÜ" + self._tone_mark_chars + r"'\s]+[）)]"
        )

    def g2p(self, text, sentence, language):
        return self.text_tokenizer.tokenize(text=text, sentence=sentence, language=language)

    #判断是否为中文汉字
    def is_chinese(self, char):
        if char >= '\u4e00' and char <= '\u9fa5':
            return True
        else:
            return False

    #判断是否为字母
    def is_alphabet(self, char):
        if (char >= '\u0041' and char <= '\u005a') or (char >= '\u0061' and
                                                        char <= '\u007a'):
            return True
        else:
            return False

    def is_all_english_text(self, text: str) -> bool:
        has_english = False
        for ch in text:
            if self.is_alphabet(ch):
                has_english = True
            elif self.is_chinese(ch):
                return False
        return has_english

    #既不是汉字也不是字母，可能是阿拉伯数字
    def is_other(self, char):
        if not (self.is_chinese(char) or self.is_alphabet(char)):
            return True
        else:
            return False

    #分类文本中的中英文
    def get_segment(self, text: str) -> List[str]:
        # sentence --> [ch_part, en_part, ch_part, ...]
        segments = []
        types = []
        flag = 0
        temp_seg = ""
        temp_lang = ""

        # 标注拼音（带声调）场景：汉字（chǔ）
        # 将括号内字符视作 other，避免被切成英文段导致重复读。
        force_other = [False] * len(text)
        for m in self._hanzi_tone_pinyin_pattern.finditer(text):
            span_txt = m.group(0)
            if any(ch in self._tone_mark_chars for ch in span_txt):
                for idx in range(m.start() + 1, m.end()):
                    force_other[idx] = True

        # Determine the type of each character. type: blank, chinese, alphabet, number, unk and point.
        for i, ch in enumerate(text):
            if force_other[i]:
                types.append("other")
                continue
            if self.is_chinese(ch):
                types.append("zh")
            elif self.is_alphabet(ch):
                types.append("en")
            else:
                types.append("other")

        assert len(types) == len(text)

        for i in range(len(types)):
            # find the first char of the seg
            if flag == 0:
                temp_seg += text[i]
                temp_lang = types[i]
                flag = 1
            else:
                if temp_lang == "other":
                    #当前片段语种与上个片段语种为"other"
                    if types[i] == temp_lang:
                        temp_seg += text[i]
                    else:
                        temp_seg += text[i]
                        temp_lang = types[i]
                else:
                    #当前片段语种与上个片段语种相同，则进行合并
                    if types[i] == temp_lang:
                        temp_seg += text[i]
                    #当前语种不属于中文或英文，则可能是阿拉伯数字
                    elif types[i] == "other":
                        temp_seg += text[i]
                    else:
                        segments.append([temp_seg, temp_lang])
                        temp_seg = text[i]
                        temp_lang = types[i]
                        flag = 1
        #将最后一个片段的结果写入segments
        segments.append([temp_seg, temp_lang])
        #中英混的情况可能会有单纯的数字或一些符号被标注为other
        for i in range(len(segments)):
            if segments[i][-1] == "other":
                segments[i][-1] = "zh"
        return segments
    
    #多语种的G2P调用
    def mix_g2p(self, text:str, language:str):
        #对文本前后的空格进行去除
        text = text.strip(" ")
        if text.endswith("."):
            text = text[:-1]
        if language == "" or language in ["zh-en", "zh", "en"]:
            if language == "zh-en" and self.is_all_english_text(text):
                return self.text_tokenizer.tokenize_english_with_cleaner(text)

            #对文本中文本片段语种进行打标,暂时仅支持中英文区分
            segments = self.get_segment(text)
            all_phoneme = ""
            all_tokens = []
            all_tn_text = []
            #中文结尾需要加blank，英文结尾不需要加blank,对英文结尾进行处理
            for index in range(len(segments)):
                seg = segments[index]
                # print("SEG", index, repr(seg[0]), seg[1], flush=True)
                phoneme, token, tn_text = self.g2p(seg[0], text, seg[1])
                all_phoneme += phoneme + "|"
                all_tokens += token
                all_tn_text.append(tn_text)
            return all_phoneme, all_tokens, "".join(all_tn_text)
        else:
            print(f'输入的语种{language}不支持，请输入支持的语种："en", "zh" "zh-en"')
            return ""

if __name__ == "__main__":
    #初始化前端
    frontend_g2p = frontend()
    
    #单条生层前端生成音素
    #输入需要转换的文本
    # text = "朝鲜后宫嫔御等级......这是level一个好时间"
    # text = "参考文献,参考文献,NEWSS"
    # text = "4. 数字人口型合成：通过输入变音后的 vocal的声音驱动数字人进行唱歌。(标准的数字人视频合成接口)"
    # text = "中在河南话中是一个很有意思的字"
    # text = "据英国天空新闻网28日报道，消息人士告诉路透社，伊朗最高领袖哈梅内伊已被转移到安全地点。"
    # text = "你是缺觉，不是缺元素，调整作息，保证充足的睡眠。"
    text = "风扇扇起来好凉快11千克, it cost me $10.我买了11千克苹果"
    # text = "Get the trust fund to the bank early."
    # text = "Biotechnology, or Biotech, is a very popular degree choice"
    start_time = time.time()
    phoneme, token, tn_text = frontend_g2p.mix_g2p(text = text, language = "zh-en")
    end_time = time.time()
    print(phoneme)
    print(token)
    print(tn_text)
    print("花费时间:{}".format(end_time-start_time))
    #计算音素与token长度时，注意看切分之后的有效长度
    print("音素长度:{}, token长度:{}".format(len(phoneme.split("|")[:-1]), len(token)))
    exit()
