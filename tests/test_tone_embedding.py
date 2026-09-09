"""Tests for optional tone-embedding encoder and rhyme-body-tone inventories."""

import types

import pytest
import torch

import lits.text as lits_text
from lits.text import text_to_sequence, text_to_sequence_with_tones
from lits.text.bopomofo_utils import BOPOMOFO_TONES
from lits.text.char_symbols.symbol_inventories import lang2inventory
from lits.models.components.text_encoder import TextEncoder
from lits.models.lits import LITS

RHYME_BODY_TONE_CLEANERS = ["en_zh_dict_mixed_rhyme_body_tone_cleaners"]
RHYME_BODY_TONE_INV = lang2inventory["zh-en-rhyme-body-tone"]["symbol_to_id"]


@pytest.fixture()
def passthrough_cleaner(monkeypatch):
    """Bypass the heavy hanzi/G2P frontend: cleaned text goes in as-is."""
    monkeypatch.setattr(lits_text, "_clean_text", lambda text, names: text)


def test_rhyme_body_tone_inventory_size():
    symbols = lang2inventory["zh-en-rhyme-body-tone"]["symbols"]
    assert len(symbols) == 173


def test_rhyme_body_tone_sequence_backfills_tones(passthrough_cleaner):
    cleaned = "ㄋ ㄧ ˇ ㄏ ㄠ ˇ _ ."
    ids, tones, _ = text_to_sequence_with_tones(cleaned, RHYME_BODY_TONE_CLEANERS)
    assert tones is not None
    assert len(ids) == len(tones)
    expected = [
        ("ㄋ", 0), ("ㄧ", 3), ("ˇ", 0), ("ㄏ", 0), ("ㄠ", 3), ("ˇ", 0),
        ("_", 0), (".", 0),
    ]
    assert ids == [RHYME_BODY_TONE_INV[token] for token, _ in expected]
    assert tones == [tone for _, tone in expected]


def test_training_tokenization_prepends_sil_once(passthrough_cleaner):
    sil_id = RHYME_BODY_TONE_INV["<sil>"]
    ids, tones, _ = text_to_sequence_with_tones("ㄋ ㄧ ˇ", RHYME_BODY_TONE_CLEANERS)
    assert ids[0] == sil_id
    assert tones[0] == 0

    ids, tones, _ = text_to_sequence_with_tones(
        "<sil> ㄋ ㄧ ˇ", RHYME_BODY_TONE_CLEANERS
    )
    assert ids.count(sil_id) == 1
    assert tones[0] == 0


def test_training_tokenization_can_disable_sil(passthrough_cleaner):
    ids, _, _ = text_to_sequence_with_tones(
        "ㄋ ㄧ ˇ", RHYME_BODY_TONE_CLEANERS, prepend_sil=False
    )
    assert ids[0] == RHYME_BODY_TONE_INV["ㄋ"]


def test_plain_text_to_sequence_matches_ids(passthrough_cleaner):
    cleaned = "ㄋ ㄧ ˇ _ ."
    ids_only, _ = text_to_sequence(cleaned, RHYME_BODY_TONE_CLEANERS)
    ids, _, _ = text_to_sequence_with_tones(cleaned, RHYME_BODY_TONE_CLEANERS)
    assert ids_only == ids


def _make_text_encoder(n_tones):
    encoder_params = types.SimpleNamespace(
        n_feats=8,
        n_channels=16,
        filter_channels=32,
        n_heads=2,
        n_layers=1,
        kernel_size=3,
        p_dropout=0.0,
        prenet=False,
        n_spks=1,
    )
    duration_predictor_params = types.SimpleNamespace(
        filter_channels_dp=16,
        kernel_size=3,
        p_dropout=0.0,
    )
    return TextEncoder(
        "transformer", encoder_params, duration_predictor_params,
        n_vocab=30, n_spks=1, spk_emb_dim=8, n_tones=n_tones,
    )


def test_text_encoder_tone_zero_row_is_identity():
    torch.manual_seed(0)
    enc = _make_text_encoder(n_tones=5).eval()
    x = torch.randint(1, 30, (2, 7))
    x_lengths = torch.tensor([7, 5])
    with torch.no_grad():
        mu_no_tones, logw_no_tones, _ = enc(x, x_lengths)
        mu_zero_tones, logw_zero_tones, _ = enc(x, x_lengths, x_tones=torch.zeros_like(x))
        mu_toned, _, _ = enc(x, x_lengths, x_tones=torch.randint(1, 6, x.shape))
    assert torch.allclose(mu_no_tones, mu_zero_tones)
    assert torch.allclose(logw_no_tones, logw_zero_tones)
    assert not torch.allclose(mu_no_tones, mu_toned)


def test_text_encoder_disabled_has_no_tone_table():
    enc = _make_text_encoder(n_tones=0)
    assert enc.tone_emb is None
    state_keys = set(enc.state_dict().keys())
    assert not any("tone_emb" in k for k in state_keys)


def test_floor_lookup_does_not_mutate_tone_ids_saved_for_backward():
    """Regression: the floor lookup ran clamp_ in place on x_tones, which the
    tone embedding had saved for backward -> autograd version mismatch."""
    torch.manual_seed(0)
    enc = _make_text_encoder(n_tones=5)
    x = torch.randint(1, 30, (2, 7))
    x_tones = torch.randint(0, 6, (2, 7))
    x_lengths = torch.tensor([7, 7])

    mu, logw, _ = enc(x, x_lengths, x_tones=x_tones)

    stub = types.SimpleNamespace()
    table = torch.rand(30, 6)
    _ = LITS._zh_frames_lookup(stub, torch.zeros(30), table, x, x_tones)

    (mu.sum() + logw.sum()).backward()
    assert x_tones.max() <= 5
