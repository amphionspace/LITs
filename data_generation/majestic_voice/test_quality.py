"""Regression checks for admission gates and final manifest publication."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import select_candidates as selection
from quality_worker import cer, tonal_match, wer, quality_config


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / 'quality').mkdir()
        (self.root / 'logs').mkdir()
        self.base = {'id': 'train/0__c000', 'source_id': 'train/0', 'status': 'ok', 'text': '你好。'}
        self.asr = {**self.base, 'asr_text': '你好', 'cer': 0.0}
        self.metrics = {**self.base, 'wavlm_similarity': 0.8, 'camp_similarity': 0.8,
                        'dnsmos_ovrl': 3.5, 'dnsmos_sig': 3.6, 'dnsmos_bak': 4.0}
        self.patch = patch.object(selection, 'ROOT', self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def write(self, metrics=None, asr=None):
        for path, value in [('logs/synthesis.jsonl', self.base),
                            ('quality/asr.jsonl', self.asr if asr is None else asr),
                            ('quality/metrics.jsonl', self.metrics if metrics is None else metrics)]:
            (self.root / path).write_text(json.dumps(value) + '\n')

    def test_valid_candidate_is_admitted_despite_max_rounds_setting(self):
        self.write()
        _, judged, best = selection.assess()
        self.assertEqual(set(best), {'train/0'})
        self.assertEqual(judged[self.base['id']]['rejection_reasons'], [])

    def test_every_quality_gate_is_required(self):
        for key, value in [('wavlm_similarity', 0.59), ('camp_similarity', 0.59),
                           ('dnsmos_ovrl', 2.99), ('dnsmos_sig', 3.19), ('dnsmos_bak', 3.49)]:
            with self.subTest(key=key):
                self.write(metrics={**self.metrics, key: value})
                self.assertFalse(selection.assess()[2])
        self.write(asr={**self.asr, 'asr_text': '你号', 'cer': 0.5})
        self.assertFalse(selection.assess()[2])

    def test_missing_or_nonfinite_score_cannot_enter_dataset(self):
        incomplete = dict(self.metrics)
        del incomplete['camp_similarity']
        for metrics in (incomplete, {**self.metrics, 'dnsmos_ovrl': float('nan')}):
            self.write(metrics=metrics)
            _, judged, best = selection.assess()
            self.assertFalse(best)
            self.assertTrue(judged[self.base['id']]['rejection_reasons'])
        incomplete_asr = dict(self.asr)
        del incomplete_asr['cer']
        self.write(asr=incomplete_asr)
        self.assertFalse(selection.assess()[2])

    def test_unscored_candidate_waits(self):
        self.write()
        (self.root / 'quality/asr.jsonl').unlink()
        _, judged, best = selection.assess()
        self.assertFalse(judged)
        self.assertFalse(best)

    def test_text_normalization_keeps_real_errors(self):
        self.assertEqual(cer('你好，ＡＢＣ！', '你好 abc'), 0)
        self.assertGreater(cer('你好', '你号'), 0)

    def test_english_word_errors_are_checked_and_missing_wer_fails_closed(self):
        self.assertEqual(wer("It's fine!", "it’s fine"), 0)
        self.assertEqual(wer('This is fine.', 'This is mine.'), 1 / 3)
        config = {**quality_config(), 'max_wer': 0.05}
        with patch.object(selection, 'quality_config', return_value=config):
            self.write()
            self.assertFalse(selection.assess()[2])
            self.write(asr={**self.asr, 'wer': 0.1})
            self.assertFalse(selection.assess()[2])
            self.write(asr={**self.asr, 'wer': 0.0})
            self.assertTrue(selection.assess()[2])

    def test_same_tone_homophones_are_allowed_but_wrong_tones_are_not(self):
        self.assertTrue(tonal_match('假语村言', '贾雨村言'))
        self.assertFalse(tonal_match('你好', '你号'))
        self.assertFalse(tonal_match('你好A', '你好B'))
        self.base['text'] = '我再这里'
        self.write(asr={**self.asr, 'asr_text': '我在这里', 'cer': 0.25})
        _, judged, best = selection.assess()
        self.assertEqual(set(best), {'train/0'})
        self.assertEqual(judged[self.base['id']]['text_acceptance'], 'exact_tonal_pinyin')


if __name__ == '__main__':
    unittest.main()
