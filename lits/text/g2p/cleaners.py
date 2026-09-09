import re
from lits.text.g2p.mandarin import Frontend_chinese
from lits.text.g2p.english import english_to_ipa
from lits.text.frontend_config import BLANK_LEVEL, resource_path


class Frontend_Choose():
    def __init__(self):
        #初始化中文前端
        self.Frontend_chinese = Frontend_chinese(resource_path, BLANK_LEVEL)
    
    def cjekfd_cleaners(self, text, sentence, language, text_tokenizers):

        if language == 'zh':
            return self.Frontend_chinese.chinese_to_ipa(text, sentence, text_tokenizers['zh'])
        #英语
        elif language == 'en':
            return english_to_ipa(text, text_tokenizers['en'])
        else:
            raise Exception('Unknown language: %s' % language)
            return None