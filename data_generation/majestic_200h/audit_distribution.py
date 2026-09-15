"""Compare accepted training text with completed attempts, excluding pending text."""
import json
import re
import time
from collections import Counter, defaultdict

from common import ROOT, atomic_json, config, connection
from run_workers import candidate_limit


def main():
    cfg=config();db=connection();db.execute('begin')
    groups=defaultdict(Counter);status=defaultdict(Counter)
    for row in db.execute('''select t.*,exists(select 1 from candidates c where c.text_id=t.id
        and c.state not in ('passed','rejected')) busy from texts t where t.split='train' '''):
        lang=row['language'];cap=candidate_limit(cfg,lang,row['split'])
        state='accepted' if row['accepted_candidate'] is not None else 'exhausted' if row['attempts']>=cap and not row['busy'] else 'pending'
        status[lang][state]+=1
        payload=json.loads(row['payload']);text=row['text']
        dimensions={'length_bin':row['length_bin'],'source':payload.get('source','unknown')}
        if lang=='mixed':
            words=len(re.findall(r"[A-Za-z]+(?:['’-][A-Za-z]+)*",text))
            dimensions['english_words']='1-2' if words<=2 else '3-5' if words<=5 else '6-10' if words<=10 else '11+'
            spans=re.findall(r'[A-Za-z]+|[\u4e00-\u9fff]+',text)
            langs=['en' if x[0].isascii() else 'zh' for x in spans]
            switches=sum(a!=b for a,b in zip(langs,langs[1:]))
            dimensions['language_switches']=str(switches) if switches<=2 else '3-4' if switches<=4 else '5+'
        for dimension,value in dimensions.items():groups[(lang,dimension,value)][state]+=1
    db.execute('commit');db.close()
    result=[]
    for (lang,dimension,value),counts in sorted(groups.items()):
        finished=counts['accepted']+counts['exhausted'];all_finished=status[lang]['accepted']+status[lang]['exhausted']
        before=finished/all_finished if all_finished else 0
        after=counts['accepted']/status[lang]['accepted'] if status[lang]['accepted'] else 0
        result.append(dict(language=lang,dimension=dimension,bucket=value,**dict(counts),completed_texts=finished,
            text_acceptance_rate=counts['accepted']/finished if finished else None,
            completed_distribution_share=before,accepted_distribution_share=after,share_change_percentage_points=100*(after-before)))
    report=dict(updated_at_unix=time.time(),scope='train only',baseline='accepted plus retry-exhausted texts; pending excluded from rate and distribution denominators',
        limitations=['Not a semantic/topic balance audit','Lengths are source text-rate bins, not generated audio duration','Per-text success differs from per-candidate pass rate','Uniformity is not itself the target; compare with intended use and source distribution'],
        counts={k:dict(v) for k,v in status.items()},distribution=result)
    atomic_json(ROOT/'reports/text_distribution_audit.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
