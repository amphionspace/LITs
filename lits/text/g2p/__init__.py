from lits.text.g2p import cleaners
from tokenizers import Tokenizer
from lits.text.g2p.text_tokenizers import TextTokenizer
import json
import re
from lits.text.g2p.cleaners import Frontend_Choose
from lits.text.language_cleaners import english_cleaners as shared_english_cleaner
from .LangSegment import LangSegment

class PhonemeBpeTokenizer:
  # vocab_path 存放音素转id的映射文件
  def __init__(self, vocab_path="./lits/text/g2p/vocab.json"):
    self.lang2backend = {
        'zh': "cmn",
        "en": "en-us",
    }
    self.text_tokenizers = {}
    self.int_text_tokenizers()

    with open(vocab_path, 'r') as f:
      json_data = f.read()
    data = json.loads(json_data)
    self.vocab = data['vocab']
    LangSegment.setfilters(["en", "zh", "ja", "ko", "fr", "de", "es", "ru"])
    self.Frontend_Choose = Frontend_Choose()
    self._warned_oov_phonemes = set()

  def int_text_tokenizers(self):
    for key, value in self.lang2backend.items():
      self.text_tokenizers[key] = TextTokenizer(language=value)

  def tokenize(self, text, sentence, language):

    # 1. convert text to phoneme
    phonemes = []
    if language == 'auto':
      #自动切分语种片段
      seglist = LangSegment.getTexts(text)
      tmp_ph = []
      tn_text = ""
      for seg in seglist:
        clean_result = self._clean_text(seg['text'], sentence, seg['lang'], ['cjekfd_cleaners'])
        tmp_ph.append(clean_result[0])
        tn_text += clean_result[1]
      phonemes = "|_|".join(tmp_ph)
    else:
      phonemes, tn_text = self._clean_text(text, sentence, language, ['cjekfd_cleaners'])

    # 2. tokenize phonemes 在这将音素转换为对应的id
    phoneme_tokens = self.phoneme2token(phonemes)
    # print('encode: ', phoneme_tokens)

    # # 3. decode tokens [optional]
    # decoded_text = self.tokenizer.decode(phoneme_tokens)
    # print('decoded: ', decoded_text)

    return phonemes, phoneme_tokens, tn_text

  def tokenize_zh_en_mixed(self, text, sentence):
    phonemes, tn_text = self.Frontend_Choose.Frontend_chinese.chinese_english_to_ipa(
      text, sentence, self.text_tokenizers['en']
    )
    phoneme_tokens = self.phoneme2token(phonemes)
    return phonemes, phoneme_tokens, tn_text

  def tokenize_english_with_cleaner(self, text):
    normalized = shared_english_cleaner(text, phonemize=False).replace("_", " ")
    phonemes, tn_text = cleaners.english_to_ipa(normalized, self.text_tokenizers['en'])
    phoneme_tokens = self.phoneme2token(phonemes)
    return phonemes, phoneme_tokens, tn_text

  def _clean_text(self, text, sentence, language, cleaner_names):
    #在这跳转到对应的语种处理函数
    for name in cleaner_names:
      cleaner = getattr(self.Frontend_Choose, name)
      if not cleaner:
        raise Exception('Unknown cleaner: %s' % name)
    #获取对应语种的函数来处理
    text, tn_text = cleaner(text, sentence, language, self.text_tokenizers)
    return text, tn_text

  def _normalize_token_for_vocab(self, token: str):
    if token in self.vocab:
      return token

    # Strip common IPA modifier letters that often appear in espeak output
    # but are not always present in our fixed vocab.
    stripped = re.sub(r"[ʲʷ˞ˠˤ]", "", token)
    if stripped in self.vocab:
      return stripped

    # Fallback for explicit length marker.
    if token.endswith("ː") and token[:-1] in self.vocab:
      return token[:-1]

    return None

  def _token_to_id_with_fallback(self, token: str):
    normalized = self._normalize_token_for_vocab(token)
    if normalized is None:
      if token not in self._warned_oov_phonemes:
        # print(f"[G2P][WARN] OOV phoneme dropped: {token}")
        self._warned_oov_phonemes.add(token)
      return None

    if normalized != token and token not in self._warned_oov_phonemes:
      # print(f"[G2P][WARN] OOV phoneme mapped: {token} -> {normalized}")
      self._warned_oov_phonemes.add(token)

    return self.vocab[normalized]

  def _token_to_ids_charwise(self, token: str):
    ids = []
    for ch in token:
      token_id = self._token_to_id_with_fallback(ch)
      if token_id is not None:
        ids.append(token_id)
    return ids

  def phoneme2token(self, phonemes):
    tokens = []
    if isinstance(phonemes, list):
      for phone in phonemes:
        #由于可能在ipa音标后面添加了对应的常见音标，这里进行修改
        phone = phone.split("\t")[0]
        phonemes_split = phone.split("|")
        line_tokens = []
        for p in phonemes_split:
          # Char-level mapping for stress_en-zh: do not force compound IPA
          # tokens (e.g. eɪ/aɪ/ˈɪ) to stay bundled.
          line_tokens.extend(self._token_to_ids_charwise(p))
        tokens.append(line_tokens)
    else:
      #由于可能在ipa音标后面添加了对应的常见音标，这里进行修改
      phonemes = phonemes.split("\t")[0]
      phonemes_split = phonemes.split("|")
      tokens = []
      for p in phonemes_split:
        # Char-level mapping for stress_en-zh: do not force compound IPA
        # tokens (e.g. eɪ/aɪ/ˈɪ) to stay bundled.
        tokens.extend(self._token_to_ids_charwise(p))
    return tokens
