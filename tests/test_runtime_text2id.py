import json
from pathlib import Path
import tempfile
import unittest

from lits.runtime.text2id import EncodedPhonemes, Text2Id, read_jsonl, write_jsonl


REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE = REPO_ROOT / "tests" / "fixtures" / "model_tokens.json"


class Text2IdTests(unittest.TestCase):
    def setUp(self):
        self.mapper = Text2Id(PROFILE)

    def test_inventory_matches_checkpoint_contract(self):
        self.assertEqual(self.mapper.model_key, "zh-en-rhyme-body-tone")
        self.assertEqual(self.mapper.n_vocab, 173)

    def test_inline_tone_backfill_matches_legacy_contract(self):
        encoded = self.mapper.encode("ㄋ ㄧ ˇ ㄏ ㄠ ˇ _ HH AY1")
        expected_tokens = [
            self.mapper.token_to_id[token]
            for token in "ㄋ ㄧ ˇ ㄏ ㄠ ˇ _ HH AY1".split()
        ]
        self.assertEqual(encoded.token_ids, expected_tokens)
        self.assertEqual(encoded.tone_ids, [0, 3, 0, 0, 3, 0, 0, 0, 0])

    def test_prepend_sil_is_optional_and_idempotent(self):
        encoded = self.mapper.encode("ㄋ ㄧ ˇ", prepend_sil=True)
        self.assertEqual(encoded.token_ids[0], self.mapper.silence_id)
        self.assertEqual(encoded.tone_ids[0], 0)
        self.assertTrue(encoded.phonemes.startswith("<sil> "))

        already_prefixed = self.mapper.encode("<sil> ㄋ ㄧ ˇ", prepend_sil=True)
        self.assertEqual(already_prefixed.token_ids.count(self.mapper.silence_id), 1)

    def test_prepend_sil_then_blank_preserves_order(self):
        encoded = self.mapper.encode("ㄋ", prepend_sil=True, add_blank=True)
        self.assertEqual(
            encoded.token_ids[:3],
            [self.mapper.blank_id, self.mapper.silence_id, self.mapper.blank_id],
        )
        self.assertEqual(encoded.tone_ids[:3], [0, 0, 0])

    def test_unknown_and_blank_insertion_are_data_driven(self):
        encoded = self.mapper.encode("NOT_IN_INVENTORY", add_blank=True)
        self.assertEqual(
            encoded.token_ids,
            [self.mapper.blank_id, self.mapper.unknown_id, self.mapper.blank_id],
        )
        self.assertEqual(encoded.tone_ids, [0, 0, 0])

    def test_jsonl_round_trip_validates_parallel_ids(self):
        row = self.mapper.encode("ㄨㄛ ˊ")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ids.jsonl"
            write_jsonl(path, [row])
            self.assertEqual(read_jsonl(path), [row])

            path.write_text(
                json.dumps({"phonemes": "x", "token_ids": [1, 2], "tone_ids": [0]}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "tone_ids"):
                read_jsonl(path)

    def test_record_requires_numeric_non_empty_ids(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            EncodedPhonemes.from_record(
                {"phonemes": "", "token_ids": [], "tone_ids": None}, line_no=1
            )


if __name__ == "__main__":
    unittest.main()
