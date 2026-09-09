"""LITs-side punctuation helpers and legacy G2P-sandhi fallback.

The production runtime uses the unified Transsion ``data/<locale>`` profiles for
TN and ``data/zh_g2p`` after lexicon lookup.  These Python runners remain for
model-side token cleanup, reference tests, and environments without a built
TextNormalizer runtime.
"""

from .loader import load_rules
from .pipeline import G2PSandhiEngine, PunctuationEngine

__all__ = [
    "G2PSandhiEngine",
    "PunctuationEngine",
    "load_rules",
]
