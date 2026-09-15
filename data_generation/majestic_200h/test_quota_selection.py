import unittest
from select_training_quota import select


class QuotaSelectionTests(unittest.TestCase):
    def test_exact_quota_preserves_stratum_proportions_and_is_reproducible(self):
        rows=[dict(audio=f'{src}{i}.wav',source=src,source_length_bin='short',language='Mixed',text='你好 world',duration=5.) for src,n in [('a',80),('b',20)] for i in range(n)]
        chosen,report=select(rows,250,42)
        self.assertEqual(len(chosen),50)
        self.assertEqual(sum(r['source']=='b' for r in chosen),10)
        self.assertEqual([r['audio'] for r in chosen],[r['audio'] for r in select(list(reversed(rows)),250,42)[0]])
        self.assertEqual(report['selected_hours'],250/3600)

    def test_shortage_fails_and_overshoot_is_bounded(self):
        rows=[dict(audio=f'{i}.wav',source='a',language='English',text='some text',duration=3+i) for i in range(10)]
        with self.assertRaises(ValueError):select(rows,1000,42)
        chosen,_=select(rows,31,42)
        self.assertGreaterEqual(sum(r['duration'] for r in chosen),31)
        self.assertLess(sum(r['duration'] for r in chosen),31+max(r['duration'] for r in rows))


if __name__=='__main__':unittest.main()
