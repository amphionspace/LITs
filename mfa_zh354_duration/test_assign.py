"""Synthetic checks for assign_token_durations after the mapping/matching fixes."""

from __future__ import annotations

import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE_ROOT.parent))
sys.path.insert(0, str(PIPELINE_ROOT))

from map import load_mapping_tsv
from run import MEL_HOP, SAMPLE_RATE, PhoneInterval, assign_token_durations

SEC_PER_FRAME = MEL_HOP / SAMPLE_RATE


def seq(*labels_frames):
    t = 0.0
    out = []
    for label, frames in labels_frames:
        d = frames * SEC_PER_FRAME
        out.append(PhoneInterval(label, t, t + d))
        t += d
    return out


def frames(assigns):
    return [(tok, round(dur, 1)) for tok, dur in assigns]


mapping = load_mapping_tsv()

# 1. 我 wo3: all three dict variants must match, glide/glottal counted when present.
for variant, expect in [
    (seq(("o˨˩˦", 5.0)), 5.0),
    (seq(("ʔ", 2.0), ("o˨˩˦", 5.0)), 7.0),
    (seq(("w", 4.0), ("o˨˩˦", 5.0)), 9.0),
]:
    got = assign_token_durations(variant, ["ㄨㄛˇ"], mapping)
    assert frames(got) == [("ㄨㄛˇ", expect)], (frames(got), expect)
print("1. 我 o / ʔo / wo variants ok")

# 2. 说 shuo1 = ʂ w o: the w belongs to the rhyme, not dropped anymore.
got = assign_token_durations(
    seq(("ʂ", 6.0), ("w", 4.0), ("o˥", 5.0)), ["ㄕ", "ㄨㄛˉ"], mapping
)
assert frames(got) == [("ㄕ", 6.0), ("ㄨㄛˉ", 9.0)], frames(got)
print("2. 说 = ʂ + (w o) ok, rhyme gets 9 frames (was 5)")

# 3. 熊 xiong2 = ɕ u ŋ: jqx context drops the expected j; token no longer fails.
got = assign_token_durations(
    seq(("ɕ", 7.0), ("u˧˥", 6.0), ("ŋ", 5.0)), ["ㄒ", "ㄩㄥˊ"], mapping
)
assert frames(got) == [("ㄒ", 7.0), ("ㄩㄥˊ", 11.0)], frames(got)
print("3. 熊 = ɕ + (u ŋ) ok (previously dropped)")

# 4. 用 yong4 = j u ŋ (zero-initial): unchanged, j still counted.
got = assign_token_durations(
    seq(("j", 3.0), ("u˥˩", 6.0), ("ŋ", 5.0)), ["ㄩㄥˋ"], mapping
)
assert frames(got) == [("ㄩㄥˋ", 14.0)], frames(got)
print("4. 用 = (j u ŋ) ok, unchanged")

# 5. Nucleus barrier: 这 spoken as zhei (ʈʂ ej) followed by 我 (ʔ o).
# The expected o of ㄜˋ must NOT steal 我's o across the ej nucleus.
got = assign_token_durations(
    seq(("ʈʂ", 4.0), ("ej˥˩", 6.0), ("ʔ", 2.0), ("o˨˩˦", 5.0)),
    ["ㄓ", "ㄜˋ", "ㄨㄛˇ"],
    mapping,
)
assert ("ㄜˋ", 5.0) not in frames(got) and ("ㄜˋ", 7.0) not in frames(got), frames(got)
assert frames(got)[0] == ("ㄓ", 4.0), frames(got)
print(f"5. zhei variant: ㄜˋ dropped instead of stealing 我's nucleus -> {frames(got)}")

# 6. Regular sentence-like span still fully matches: 王 wang2 = w a ŋ.
got = assign_token_durations(
    seq(("w", 3.0), ("a˥˩", 7.0), ("ŋ", 4.0)), ["ㄨㄤˊ"], mapping
)
assert frames(got) == [("ㄨㄤˊ", 14.0)], frames(got)
print("6. 王 = (w a ŋ) ok, unchanged")

# 7. Erhua: 门儿 = m ə n + 儿-as-word (o ɻ); erhua tail gets o + ɻ.
got = assign_token_durations(
    seq(("m", 3.0), ("ə˧˥", 5.0), ("n", 3.0), ("o˧˥", 4.0), ("ɻ", 3.0)),
    ["ㄇ", "ㄣˊ", "ㄦ˙"],
    mapping,
)
assert frames(got) == [("ㄇ", 3.0), ("ㄣˊ", 8.0), ("ㄦ˙", 7.0)], frames(got)
print("7. 门儿: erhua tail = o + ɻ = 7 frames (was unmatched)")

# 8. Erhua variant without trailing ɻ (弯儿 aligned as w a n o):
got = assign_token_durations(
    seq(("w", 3.0), ("a˥", 6.0), ("n", 3.0), ("o˧˥", 4.0), ("t", 3.0), ("ow˥", 5.0)),
    ["ㄨㄢˉ", "ㄦ˙", "ㄉ", "ㄡˉ"],
    mapping,
)
assert frames(got) == [("ㄨㄢˉ", 12.0), ("ㄦ˙", 4.0), ("ㄉ", 3.0), ("ㄡˉ", 5.0)], frames(got)
print("8. 弯儿 (no ɻ variant): erhua tail = o only, following tokens unaffected")

# 9. Full er syllable 二 = o ɻ still matches; ɻ-less variant also accepted.
got = assign_token_durations(seq(("o˥˩", 6.0), ("ɻ", 4.0)), ["ㄦˋ"], mapping)
assert frames(got) == [("ㄦˋ", 10.0)], frames(got)
got = assign_token_durations(seq(("o˥˩", 6.0),), ["ㄦˋ"], mapping)
assert frames(got) == [("ㄦˋ", 6.0)], frames(got)
print("9. 二 = (o ɻ) and (o) both ok")

print("all checks passed")
