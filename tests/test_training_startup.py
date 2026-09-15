"""Startup regressions for environment-driven config and duration lookups."""
import csv
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from lits.text.bopomofo_utils import split_rhyme_tone_token, zh354_duration_tokens


def test_numeric_environment_values_are_decoded(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    for key, value in {'PROJECT_ROOT': str(root), 'TRAIN_FILELIST': '/tmp/train.txt',
                       'VALID_FILELIST': '/tmp/val.txt', 'N_SPKS': '2',
                       'MEL_MEAN': '-5.4', 'MEL_STD': '2.3'}.items():
        monkeypatch.setenv(key, value)
    with initialize_config_dir(config_dir=str(root / 'configs'), version_base='1.3'):
        cfg = compose(config_name='train', overrides=['experiment=en-zh'])
    assert cfg.model.n_spks == 2
    assert cfg.model.data_statistics.mel_mean == -5.4
    assert cfg.model.data_statistics.mel_std == 2.3


def test_duration_helpers_cover_local_stats():
    root = Path(__file__).resolve().parents[1]
    stats = root / 'mfa_zh354_duration/runs/chuanyin_biaobei/zh354_duration_stats.tsv'
    if not stats.is_file():
        pytest.skip('Optional local duration statistics are not part of the repository')
    with stats.open() as stream:
        keys = {row['token_354'] for row in csv.DictReader(stream, delimiter='\t')}
    assert keys <= set(zh354_duration_tokens())


def test_duration_helpers_split_tones():
    assert split_rhyme_tone_token('ㄣˊ') == ('ㄣ', 2)
    assert split_rhyme_tone_token('ㄦ˙') == ('ㄦ', 5)
    assert split_rhyme_tone_token('ㄅ') == ('ㄅ', 0)


def test_bfloat16_alignment_matches_float32_and_preserves_dtype():
    import torch
    from lits.utils.monotonic_align import maximum_path, maximum_path_constrained
    scores = torch.tensor([[[4, 3, -2, -3], [-3, -2, 3, 4]]], dtype=torch.bfloat16)
    mask = torch.ones_like(scores)
    floors = torch.ones(1, 2, dtype=torch.bfloat16)
    ceilings = torch.full((1, 2), 3, dtype=torch.bfloat16)
    simple = maximum_path(scores, mask)
    assert simple.dtype == torch.bfloat16
    assert torch.equal(simple.float(), maximum_path(scores.float(), mask.float()))
    constrained, _ = maximum_path_constrained(scores, mask, floors, ceilings)
    expected, _ = maximum_path_constrained(scores.float(), mask.float(), floors.float(), ceilings.float())
    assert constrained.dtype == torch.bfloat16
    assert torch.equal(constrained.float(), expected)
    assert torch.equal(constrained.sum(1), torch.ones(1, 4, dtype=torch.bfloat16))
