import sqlite3
import unittest
from run_workers import synthesis_languages, active_candidate_count, synthesis_reference, quality_reference, candidate_limit


class LanguageSchedulingTests(unittest.TestCase):
    def test_extra_test_round_is_scoped_to_english_test(self):
        cfg=dict(max_candidates_per_text=4,max_candidates_per_text_by_language={'en':12,'mixed':8},max_candidates_per_text_by_split={'test':{'en':14}})
        self.assertEqual(candidate_limit(cfg,'en','test'),14)
        self.assertEqual(candidate_limit(cfg,'en','train'),12)
        self.assertEqual(candidate_limit(cfg,'en','val'),12)
        self.assertEqual(candidate_limit(cfg,'zh','test'),4)
        self.assertEqual(candidate_limit(cfg,'mixed','test'),8)

    def test_language_reference_does_not_change_quality_anchor_or_other_languages(self):
        cfg=dict(reference_audio='cn.wav',prompt_text='中文参考',synthesis_references={'en':dict(reference_audio='en4.wav',prompt_text='English reference.')})
        self.assertEqual(synthesis_reference(cfg,'en')['reference_audio'],'en4.wav')
        for lang in ['zh','mixed']:
            self.assertEqual(synthesis_reference(cfg,lang)['reference_audio'],'cn.wav')
        self.assertEqual(cfg['reference_audio'],'cn.wav')
        cfg['synthesis_references']['en'].pop('prompt_text')
        with self.assertRaises(ValueError):synthesis_reference(cfg,'en')

    def test_approved_english_quality_reference_and_priority(self):
        cfg=dict(reference_audio='cn.wav',quality_reference_audio_by_language={'en':'en4.wav'},
            active_languages=['zh','en','mixed'],synthesis_language_slots=['zh']*3+['mixed']*3+['en']*2)
        self.assertEqual(quality_reference(cfg,'en'),'en4.wav')
        self.assertEqual(quality_reference(cfg,'zh'),'cn.wav')
        self.assertEqual(quality_reference(cfg,'mixed'),'cn.wav')
        preferred=[synthesis_languages(cfg,i)[0] for i in range(8)]
        self.assertEqual(preferred.count('zh'),3)
        self.assertEqual(preferred.count('mixed'),3)
        self.assertEqual(preferred.count('en'),2)

    def test_inactive_english_never_used_for_fallback(self):
        cfg={'active_languages':['zh','mixed'],'synthesis_language_slots':['zh','zh','mixed','mixed']}
        orders=[synthesis_languages(cfg,i) for i in range(8)]
        self.assertEqual([x[0] for x in orders].count('zh'),4)
        self.assertEqual([x[0] for x in orders].count('mixed'),4)
        self.assertTrue(all(set(x)=={'zh','mixed'} for x in orders))
        self.assertEqual(synthesis_languages({'active_languages':[]},0),[])

    def test_paused_backlog_does_not_block_active_admission(self):
        db=sqlite3.connect(':memory:')
        db.executescript('create table texts(id integer,language text);create table candidates(text_id integer,state text);')
        db.executemany('insert into texts values(?,?)',[(1,'en'),(2,'zh'),(3,'mixed')])
        db.executemany('insert into candidates values(?,?)',[(1,'queued')]*2048+[(2,'queued'),(3,'asr_done'),(2,'passed')])
        self.assertEqual(active_candidate_count(db,['zh','mixed']),2)
        self.assertEqual(active_candidate_count(db,['en','zh','mixed']),2050)


if __name__=='__main__':unittest.main()
