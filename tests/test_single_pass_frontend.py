from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from infer_e2e import (
    _preferred_cut,
    build_frontend_artifacts,
    build_normalized_txt,
    run_dry_run,
    split_encoded_phonemes,
)
from lits.infer_wav_merge import load_chunk_manifest
from lits.runtime.text2id import Text2Id, read_jsonl


REPO_ROOT = Path(__file__).resolve().parents[1]
TOKEN_CONFIG = REPO_ROOT / "tests" / "fixtures" / "model_tokens.json"


class CountingG2P:
    def __init__(self, output: str):
        self.output = output
        self.calls: list[str] = []

    def g2p_line(self, text: str) -> str:
        self.calls.append(text)
        return self.output


class CountingTN:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def tn_line(self, text: str, lang: str) -> str:
        self.calls.append((text, lang))
        return text


class CountingText2Id:
    def __init__(self, mapper: Text2Id):
        self.mapper = mapper
        self.calls: list[tuple[str, bool, bool]] = []
        self.blank_id = mapper.blank_id
        self.n_vocab = mapper.n_vocab

    def encode(
        self,
        phonemes: str,
        *,
        add_blank: bool = False,
        prepend_sil: bool = False,
    ):
        self.calls.append((phonemes, add_blank, prepend_sil))
        return self.mapper.encode(
            phonemes,
            add_blank=add_blank,
            prepend_sil=prepend_sil,
        )


class SinglePassFrontendTests(unittest.TestCase):
    def setUp(self):
        self.mapper = Text2Id(TOKEN_CONFIG)
        self.phonemes = "ㄋ ㄧ ˇ _ HH AY1 _ ㄏ ㄠ ˇ _ W ER1 L D ."

    def test_encoded_split_preserves_all_frontend_tokens(self):
        encoded = self.mapper.encode(self.phonemes)
        chunks, used_hard_split = split_encoded_phonemes(
            encoded,
            5,
            add_blank=False,
            blank_id=self.mapper.blank_id,
        )

        self.assertGreater(len(chunks), 1)
        self.assertFalse(used_hard_split)
        self.assertTrue(all(len(chunk.token_ids) <= 5 for chunk in chunks))
        self.assertEqual(
            [token for chunk in chunks for token in chunk.phonemes.split()],
            encoded.phonemes.split(),
        )
        self.assertEqual(
            [token_id for chunk in chunks for token_id in chunk.token_ids],
            encoded.token_ids,
        )
        self.assertEqual(
            [tone_id for chunk in chunks for tone_id in (chunk.tone_ids or [])],
            encoded.tone_ids,
        )

    def test_add_blank_is_applied_after_split_without_reencoding(self):
        encoded = self.mapper.encode(self.phonemes)
        chunks, _ = split_encoded_phonemes(
            encoded,
            11,
            add_blank=True,
            blank_id=self.mapper.blank_id,
        )

        self.assertTrue(all(len(chunk.token_ids) <= 11 for chunk in chunks))
        self.assertEqual(
            [token_id for chunk in chunks for token_id in chunk.token_ids[1::2]],
            encoded.token_ids,
        )
        for chunk in chunks:
            self.assertTrue(all(
                token_id == self.mapper.blank_id
                for token_id in chunk.token_ids[0::2]
            ))
            self.assertEqual(len(chunk.token_ids), 2 * len(chunk.phonemes.split()) + 1)

    def test_unsafe_budget_does_not_split_a_chinese_tone_unit(self):
        encoded = self.mapper.encode("ㄋ ㄧ ˇ ㄏ ㄠ ˇ")
        with self.assertRaisesRegex(ValueError, "cannot split encoded phonemes safely"):
            split_encoded_phonemes(
                encoded,
                2,
                add_blank=False,
                blank_id=self.mapper.blank_id,
            )

    def test_punctuation_at_budget_edge_remains_a_valid_boundary(self):
        end, kind = _preferred_cut(
            ("HH", "AY1", ".", "_", "ㄋ", "ㄧ", "ˇ"),
            0,
            3,
        )
        self.assertEqual((end, kind), (3, "strong"))

    def test_each_input_line_calls_g2p_and_text2id_exactly_once(self):
        g2p = CountingG2P(self.phonemes)
        text2id = CountingText2Id(self.mapper)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "normalized.txt"
            input_path.write_text("same text\nsame text\n", encoding="utf-8")
            artifacts = build_frontend_artifacts(
                input_path,
                root / "phonemes.txt",
                root / "token_ids.jsonl",
                tts_cli=g2p,
                text2id=text2id,
                add_blank=True,
                max_tokens=11,
            )

            self.assertEqual(g2p.calls, ["same text", "same text"])
            self.assertEqual(len(text2id.calls), 2)
            self.assertTrue(all(call == (self.phonemes, False, True) for call in text2id.calls))
            self.assertGreater(len(artifacts.rows), 2)
            self.assertEqual(read_jsonl(artifacts.token_ids_path), artifacts.rows)
            self.assertEqual(
                len(artifacts.phonemes_path.read_text(encoding="utf-8").splitlines()),
                len(artifacts.rows),
            )

            self.assertIsNotNone(artifacts.chunk_manifest_path)
            manifest = load_chunk_manifest(artifacts.chunk_manifest_path)
            self.assertEqual(len(manifest), len(artifacts.rows))
            self.assertTrue(all(entry.source_text == "same text" for entry in manifest))

            before = (len(g2p.calls), len(text2id.calls))
            run_dry_run(artifacts, root / "dry.txt", root / "dry.tsv")
            self.assertEqual((len(g2p.calls), len(text2id.calls)), before)

    def test_tn_runs_once_per_raw_input_line(self):
        tn = CountingTN()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "raw.txt"
            input_path.write_text("你好\nhello\n", encoding="utf-8")
            stats = build_normalized_txt(
                input_path,
                root / "normalized.txt",
                tts_cli=tn,
                limit=0,
            )
        self.assertEqual(stats["written"], 2)
        self.assertEqual(tn.calls, [("你好", "zh"), ("hello", "en")])

    def test_no_token_budget_uses_normalized_text_without_manifest(self):
        g2p = CountingG2P("HH AY1 .")
        text2id = CountingText2Id(self.mapper)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "normalized.txt"
            input_path.write_text("hello\n", encoding="utf-8")
            artifacts = build_frontend_artifacts(
                input_path,
                root / "phonemes.txt",
                root / "token_ids.jsonl",
                tts_cli=g2p,
                text2id=text2id,
                max_tokens=0,
            )
        self.assertEqual(artifacts.synth_input_path, input_path)
        self.assertIsNone(artifacts.chunk_manifest_path)
        self.assertEqual(g2p.calls, ["hello"])
        self.assertEqual(len(text2id.calls), 1)

    def test_legacy_manifest_text_with_tab_remains_backward_compatible(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chunk_manifest.tsv"
            path.write_text(
                "synth_line\torig_line\tchunk_idx\tn_chunks\tn_tokens\t"
                "had_hard_split\ttext\n"
                "1\t1\t1\t1\t3\t0\thello\tworld\n",
                encoding="utf-8",
            )
            rows = load_chunk_manifest(path)
        self.assertEqual(rows[0].text, "hello\tworld")
        self.assertIsNone(rows[0].source_text)


if __name__ == "__main__":
    unittest.main()
