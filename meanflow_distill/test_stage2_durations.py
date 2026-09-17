"""Regression coverage for independent per-utterance duration limits."""
import unittest
from types import SimpleNamespace

import torch

from lits.models.lits import LITS
from lits.utils.infer_duration_floor import _cap_tensor_per_position


class DurationBatchTests(unittest.TestCase):
    def test_tone_clamp_does_not_mix_utterances(self):
        model = SimpleNamespace(tone_floor_frames=1, tone_ceiling_frames=3,
                                _tone_mark_mask=lambda x: x == 7)
        tokens = torch.tensor([[7, 2, 3], [1, 7, 3]])
        durations = torch.tensor([[[9., 8., 7.]], [[6., 9., 5.]]])
        actual = LITS._clamp_inference_tone_durations(model, durations, tokens,
                                                    torch.ones_like(durations))
        expected = torch.tensor([[[3., 8., 7.]], [[6., 3., 5.]]])
        torch.testing.assert_close(actual, expected)

    def test_position_caps_do_not_mix_utterances(self):
        durations = torch.tensor([[[9., 8., 7.]], [[6., 9., 5.]]])
        caps = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
        need = torch.tensor([[True, False, True], [False, True, False]])
        actual = _cap_tensor_per_position(durations, durations.squeeze(1), need, caps)
        expected = torch.tensor([[[1., 8., 3.]], [[6., 5., 5.]]])
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
