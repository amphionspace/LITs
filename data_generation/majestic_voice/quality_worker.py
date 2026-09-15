"""Shared transcript scoring for collection and checkpoint evaluation."""
import unicodedata
from functools import lru_cache


def normalize(text):
    return ''.join(c for c in unicodedata.normalize('NFKC', text).lower()
                   if unicodedata.category(c)[0] in ('L', 'N'))


@lru_cache(maxsize=100_000)
def tonal_sequence(text):
    """Tone-number pinyin for fully Chinese text; unknown readings fail closed."""
    import re
    from pypinyin import pinyin, Style
    text = normalize(text)
    if not text or not all('\u4e00' <= c <= '\u9fff' for c in text):
        return None
    sequence = tuple(item[0] for item in pinyin(text, style=Style.TONE3,
                                               heteronym=False, neutral_tone_with_five=True))
    if not all(re.fullmatch(r'[a-zvü]+[1-5]', token) for token in sequence):
        return None
    return sequence


def tonal_match(reference, hypothesis):
    expected = tonal_sequence(reference)
    return expected is not None and expected == tonal_sequence(hypothesis)


def cer(reference, hypothesis):
    reference, hypothesis = normalize(reference), normalize(hypothesis)
    return error_rate(reference, hypothesis)


def wer(reference, hypothesis):
    import re
    def words(text):
        text = unicodedata.normalize('NFKC', text).lower().replace('’', "'")
        return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)*", text)
    return error_rate(words(reference), words(hypothesis))


def error_rate(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, a in enumerate(reference, 1):
        current = [i]
        for j, b in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (a != b)))
        previous = current
    return previous[-1] / max(1, len(reference))
