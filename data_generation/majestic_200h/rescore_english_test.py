"""Bounded recheck of existing English test audio after the approved reference change."""
import json
import time
from common import ROOT, atomic_json, config, connection


def main():
    cfg=config();ref=cfg['quality_reference_audio_by_language']['en']
    assert ref.endswith('/72003.wav')
    db=connection()
    db.execute('''create table if not exists candidate_quality_history(
        id integer primary key,candidate_id integer,changed_at real,reason text,previous_record text)''')
    db.execute('begin immediate')
    try:
        candidates=db.execute('''select c.* from texts t join candidates c on c.text_id=t.id
            where t.language='en' and t.split='test' and t.accepted_candidate is null
            and c.state='rejected' and c.metrics is not null and c.asr is not null
            and c.duration between 3 and 20 and json_extract(c.metrics,'$.speaker_reference_audio') is null
            and not exists(select 1 from candidates x where x.text_id=t.id and x.state not in ('passed','rejected'))
            order by c.score desc,c.id''').fetchall()
        selected=[];seen=set()
        for r in candidates:
            if r['text_id'] in seen:continue
            seen.add(r['text_id']);selected.append(dict(r))
            if len(selected)>=512:break
        now=time.time()
        for r in selected:
            db.execute('insert into candidate_quality_history(candidate_id,changed_at,reason,previous_record) values(?,?,?,?)',
                (r['id'],now,'User approved English sample 4 as quality reference; bounded test-gap recheck',json.dumps(r)))
            db.execute("update candidates set state='asr_done',metrics=null,reasons=null,score=null,owner=null,lease_until=0,updated_at=? where id=?",(now,r['id']))
        db.execute('commit')
    except BaseException:db.execute('rollback');raise
    result=dict(status='queued_for_rescoring',reference=ref,queued_candidates=len(selected),distinct_texts=len(seen),
        candidate_ids=[r['id'] for r in selected],split='test',new_audio_generated=False,
        previous_scores_preserved_in='candidate_quality_history',created_at_unix=time.time())
    atomic_json(ROOT/'reports/english_test_reference4_rescore.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()
