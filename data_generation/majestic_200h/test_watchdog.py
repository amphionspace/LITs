import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import watchdog


class WatchdogTests(unittest.TestCase):
    def test_never_restarts_generation_during_or_after_handoff(self):
        for status in watchdog.HANDOFF:
            self.assertFalse(watchdog.recovery_allowed({},status))
        self.assertTrue(watchdog.recovery_allowed({},'stopped'))
        for control in ({'enabled':False},{'auto_recover':False},{'pause_generation':True}):
            self.assertFalse(watchdog.recovery_allowed(control,'stopped'))

    def test_recovery_limit_expires_and_is_per_component(self):
        history=[dict(component='pipeline',time=100,pid=1)]*3
        self.assertFalse(watchdog.budget_available(history,'pipeline',200))
        self.assertTrue(watchdog.budget_available(history,'tts_gpu0',200))
        self.assertTrue(watchdog.budget_available(history,'pipeline',3800))

    def test_process_identity_required(self):
        self.assertFalse(watchdog.alive(os.getpid(),'nonexistent-process-signature'))
        self.assertFalse(watchdog.alive(None,'python'))

    def test_report_counts_only_one_winner_per_text(self):
        db=sqlite3.connect(':memory:',isolation_level=None);db.row_factory=sqlite3.Row
        db.executescript('''
            create table texts(id integer,split text,language text,accepted_candidate integer,attempts integer);
            create table candidates(id integer,text_id integer,state text,duration real,reasons text,synthesis text,metrics text);
            create table pool(split text,language text,state text,estimated_seconds real,payload text);
            insert into texts values(1,'train','en',2,3);
            insert into candidates values(1,1,'passed',10,'[]','{"endpoint":"one"}',null);
            insert into candidates values(2,1,'passed',12,'[]','{"endpoint":"two"}',null);
            insert into candidates values(3,1,'rejected',8,'["asr"]','{"endpoint":"two"}',null);
        ''')
        cfg=dict(targets_train_hours={'zh':100,'en':50,'mixed':50},targets_heldout_hours_per_language={'val':1/6,'test':1/6},max_candidates_per_text=4)
        with tempfile.TemporaryDirectory() as d, patch.object(watchdog,'ROOT',Path(d)), patch.object(watchdog,'connection',return_value=db), patch.object(watchdog,'config',return_value=cfg):
            result=watchdog.report({},[])
            self.assertEqual(result['generated_candidates']['count'],3)
            self.assertAlmostEqual(result['accepted_unique_hours'],12/3600)
            en=next(q for q in result['quotas'] if q['language']=='en' and q['split']=='train')
            self.assertEqual(en['texts'],1)
            self.assertAlmostEqual(en['missing_hours'],50-12/3600)


if __name__=='__main__':unittest.main()
