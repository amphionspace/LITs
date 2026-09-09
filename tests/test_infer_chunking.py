#!/usr/bin/env python3
"""Tests for post-TN inference chunking."""

from __future__ import annotations

import unittest
from pathlib import Path

from lits.text import text_to_sequence
from lits.runtime.infer_chunking import chunk_tn_text, plan_infer_chunks, plan_infer_chunks_for_line

CLEANER = "en_zh_dict_mixed_rhyme_body_tone_cleaners"
REPO_ROOT = Path(__file__).resolve().parents[1]


def _word_count(text: str) -> int:
    return len(text.split())


def _cleaner_count(text: str) -> int:
    ids, _ = text_to_sequence(text, [CLEANER])
    return len(ids)


class InferChunkingTest(unittest.TestCase):
    def test_short_text_unchanged(self):
        text = "Hello world."
        result = chunk_tn_text(text, 256, _word_count)
        self.assertEqual(result.chunks, (text,))

    def test_strong_punct_split_within_budget(self):
        text = "First sentence. Second sentence. Third sentence."
        result = chunk_tn_text(text, 2, _word_count)
        self.assertGreater(len(result.chunks), 1)
        self.assertTrue(all(_word_count(c) <= 2 for c in result.chunks))

    def test_weak_punct_when_no_strong(self):
        text = "alpha, beta, gamma, delta"
        result = chunk_tn_text(text, 2, _word_count)
        self.assertGreater(len(result.chunks), 1)
        self.assertTrue(all(_word_count(c) <= 2 for c in result.chunks))

    def test_hard_split_when_no_punct(self):
        text = "one two three four five six seven eight"
        result = chunk_tn_text(text, 3, _word_count)
        self.assertGreater(len(result.chunks), 1)
        self.assertTrue(all(_word_count(c) <= 3 for c in result.chunks))
        self.assertIn("hard", {b.kind for b in result.boundaries})

    def test_plan_preserves_orig_line_mapping(self):
        lines = ["Short.", "First part. Second part. Third part."]
        plan = plan_infer_chunks(lines, 2, _word_count)
        self.assertEqual(plan[0].n_chunks, 1)
        orig2 = [e for e in plan if e.orig_line == 2]
        self.assertGreater(len(orig2), 1)

    def test_plan_for_line_uses_local_synth_line(self):
        text = "First part. Second part."
        plan = plan_infer_chunks_for_line(text, 3, 2, _word_count)
        self.assertEqual([e.synth_line for e in plan], [1, 2])
        self.assertTrue(all(e.orig_line == 3 for e in plan))



if __name__ == "__main__":
    unittest.main()
