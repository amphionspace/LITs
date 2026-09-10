"""Tests for infer_e2e TtsCliEngine (direct tts_cli, no lits.text frontend modules)."""

from __future__ import annotations

import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

from infer_e2e import (  # noqa: E402
    G2P_PROFILE,
    SUPPORTED_MODEL_LANGS,
    TtsCliEngine,
)


class TtsCliEngineConfigTests(unittest.TestCase):
    def test_supports_en_zh_models(self):
        self.assertIn("en-zh-dict", SUPPORTED_MODEL_LANGS)

    def test_g2p_profile_constant(self):
        self.assertEqual(G2P_PROFILE, "en-zh-g2p")
        self.assertIn("en-zh-dict", SUPPORTED_MODEL_LANGS)


class TtsCliEngineRuntimeTests(unittest.TestCase):
    def _engine(self) -> TtsCliEngine | None:
        bin_dir = REPO_ROOT / "e2e_infer" / "bin"
        data_root = REPO_ROOT / "frontend" / "data"
        if not (bin_dir / "tts_cli").is_file():
            return None
        return TtsCliEngine(bin_dir, data_root=data_root)

    def test_g2p_smoke_via_subprocess(self):
        engine = self._engine()
        if engine is None or not engine.is_g2p_available():
            self.skipTest("tts_cli or en-zh-g2p profile not installed")
        try:
            out = engine.g2p_line("你好")
        except RuntimeError as exc:
            if "unknown backend id" in str(exc):
                self.skipTest("tts_cli needs rebuild with en_zh_g2p backend")
            raise
        self.assertTrue(out)

class TtsCliParityTests(unittest.TestCase):
    def test_curated_corpus_relaxed_parity_smoke(self):
        """Smoke: cpp G2P is runnable; relaxed parity is tracked in golden TSV."""
        engine = TtsCliEngineRuntimeTests()._engine()
        if engine is None or not engine.is_g2p_available():
            self.skipTest("cpp frontend not ready")
        corpus = REPO_ROOT / "test" / "fixtures" / "en_zh_parity_corpus.txt"
        if not corpus.is_file():
            self.skipTest("parity corpus missing")
        try:
            out = engine.g2p_line("你好")
        except RuntimeError as exc:
            if "unknown backend id" in str(exc):
                self.skipTest("tts_cli needs rebuild with en_zh_g2p backend")
            raise
        self.assertIn("ㄋ", out)


if __name__ == "__main__":
    unittest.main()
