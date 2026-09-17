"""Detached collection watchdog: bounded recovery, progress/quality reports, handoff guard."""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

from common import ROOT, ASSETS, REPO, CODE, LANGS, atomic_json, config, connection, goals, totals

HANDOFF = {'prepare', 'preflight', 'training', 'training_complete', 'handoff_failed'}


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {} if default is None else default


def alive(pid, signature):
    try:
        proc = Path(f'/proc/{int(pid)}')
        return (proc/'stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z' and signature.encode() in (proc/'cmdline').read_bytes()
    except (FileNotFoundError, ProcessLookupError, ValueError, TypeError):
        return False


def group_alive(pgid):
    # A dead API parent can leave GPU engine children. Do not create a competing replica.
    for proc in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = proc.read_text().rsplit(')', 1)[1].split()
            if fields[0] != 'Z' and int(fields[2]) == pgid:
                return True
        except (FileNotFoundError, ProcessLookupError):
            pass
    return False


def healthy(endpoint):
    try:
        with urllib.request.urlopen(endpoint+'/health', timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


def recovery_allowed(control, status):
    return control.get('enabled', True) and control.get('auto_recover', True) and not control.get('pause_generation', False) and status not in HANDOFF


def budget_available(history, component, now, cap=3):
    return sum(x['component'] == component and now-x['time'] < 3600 for x in history) < cap


def spawn(command, logfile, env=None):
    with logfile.open('a') as log:
        return subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def report(health, actions):
    cfg=config(); db=connection()
    db.execute('begin')
    accepted=totals(db)
    quota=[]
    for (split, lang), seconds in goals(cfg).items():
        got=accepted.get((split, lang), {'seconds': 0, 'count': 0})
        quota.append(dict(split=split, language=lang, hours=got['seconds']/3600, texts=got['count'], target_hours=seconds/3600, missing_hours=max(0, seconds-got['seconds'])/3600))
    states=dict(db.execute('select state,count(*) from candidates group by state'))
    generated=dict(db.execute('select count(*) count,coalesce(sum(duration),0)/3600 hours from candidates where duration is not null').fetchone())
    pool=[dict(r) for r in db.execute('select split,language,state,count(*) texts,sum(estimated_seconds)/3600 estimated_text_hours from pool group by split,language,state')]
    sources=[dict(r) for r in db.execute("select json_extract(payload,'$.source') source,language,count(*) texts,sum(estimated_seconds)/3600 estimated_text_hours from pool group by source,language")]
    by_language={}
    for lang in LANGS:
        counts=dict(db.execute("select c.state,count(*) from candidates c join texts t on t.id=c.text_id where t.language=? group by c.state", (lang,)))
        final=counts.get('passed', 0)+counts.get('rejected', 0)
        reasons=Counter()
        for row in db.execute("select reasons from candidates c join texts t on t.id=c.text_id where t.language=? and c.state='rejected'", (lang,)):
            reasons.update(json.loads(row['reasons'] or '[]'))
        cap=cfg.get('max_candidates_per_text_by_language', {}).get(lang, cfg['max_candidates_per_text'])
        exhausted=db.execute('''select count(*) from texts t where language=? and accepted_candidate is null and attempts>=?
            and not exists(select 1 from candidates c where c.text_id=t.id and c.state not in ('passed','rejected'))''', (lang, cap)).fetchone()[0]
        by_language[lang]=dict(candidate_states=counts, scored_candidates=final, candidate_pass_rate=counts.get('passed', 0)/final if final else None, exhausted_texts=exhausted, max_candidates_per_text=cap, rejection_reasons=dict(reasons.most_common(10)))
    endpoint_counts=[dict(r) for r in db.execute("select json_extract(synthesis,'$.endpoint') endpoint,count(*) candidates from candidates where synthesis is not null group by endpoint")]
    reference_quality=[dict(r) for r in db.execute('''select t.language,
        coalesce(json_extract(c.metrics,'$.speaker_reference_audio'),?) quality_reference,
        coalesce(json_extract(c.synthesis,'$.reference_audio'),?) synthesis_reference,
        count(*) scored_candidates,sum(c.state='passed') passed_candidates,
        avg(json_extract(c.metrics,'$.wavlm_similarity')) mean_wavlm,
        avg(json_extract(c.metrics,'$.camp_similarity')) mean_camp,
        avg(json_extract(c.metrics,'$.dnsmos_ovrl')) mean_dnsmos
        from candidates c join texts t on t.id=c.text_id where c.metrics is not null
        group by t.language,quality_reference,synthesis_reference''',(cfg.get('reference_audio'),cfg.get('reference_audio')))]
    db.execute('commit');db.close()
    job=read(ROOT/'job.json'); now=time.time(); plans=[]
    for lang in LANGS:
        deficit=sum(q['missing_hours'] for q in quota if q['language']==lang)
        if deficit<=0:continue
        quality=by_language[lang];available=sum(x['texts'] for x in pool if x['language']==lang and x['state']=='available')
        if lang not in cfg.get('active_languages',LANGS):
            plans.append(f'{lang} 按用户当前安排暂缓新合成；已有音频继续质检。原配额尚缺 {deficit:.2f} 小时（含验证/测试），全部训练交接仍需满足原配额。')
            continue
        plans.append(f'{lang} 尚缺 {deficit:.2f} 小时（含验证/测试）；继续从剩余 {available:,} 条文本生成，每条最多 {quality["max_candidates_per_text"]} 个候选，保留通过全部门槛的最佳候选。')
        if quality['scored_candidates']>=100 and quality['candidate_pass_rate']<.1:
            top='、'.join(list(quality['rejection_reasons'])[:3])
            plans.append(f'{lang} 候选通过率偏低，主要失败项：{top}。先检查对应失败音频和参考音色；当前重试上限耗尽后可评估增加候选或补充独立文本。不得降低质检阈值或重复计入同一句时长。')
        for q in quota:
            if q['language']!=lang or q['missing_hours']<=0:continue
            remaining=sum(x['estimated_text_hours'] or 0 for x in pool if x['language']==lang and x['split']==q['split'] and x['state']=='available')
            if remaining<q['missing_hours']:
                plans.append(f'{lang}/{q["split"]} 剩余文本估算时长 {remaining:.2f}h 小于音频缺口 {q["missing_hours"]:.2f}h：需补充独立来源的完整文本，经过来源分组和历史去重后入池。文本估算不能算作合格音频。')
    previous=read(ROOT/'reports/supervisor_status.json')
    previous_count=previous.get('generated_candidates',{}).get('count',0)
    last_progress=now if generated['count']!=previous_count else previous.get('last_generation_progress_at_unix',now)
    if now-last_progress>900 and job.get('status')=='synthesizing_and_filtering':
        plans.append('连续十五分钟未新增已生成候选：检查合成服务、队列积压和文本池；不以此单独判定进程死亡。')
    if job.get('status')=='handoff_failed':
        plans.append(f'自动训练交接失败，阶段 {job.get("failed_stage")}：检查 logs 中对应 handoff 日志；保留数据及 checkpoint，修复后人工确认续训方式，不自动从零重训。')
    if job.get('status') in {'prepare','preflight','training'} and not alive(job.get('supervisor_pid'),str(CODE/'start_pipeline.py')):
        plans.append('训练交接主进程已退出：先检查训练子进程、日志和已有 checkpoint；禁止自动重启数据生成或从零启动训练。')
    result=dict(updated_at_unix=now,watchdog_pid=os.getpid(),job=job,health=health,quotas=quota,generated_candidates=generated,
        accepted_unique_hours=sum(q['hours'] for q in quota),candidate_states=states,quality_by_language=by_language,
        pool_inventory=pool,source_inventory=sources,endpoint_candidates=endpoint_counts,plans=plans,recent_actions=actions[-20:],
        quality_by_reference=reference_quality,
        last_generation_progress_at_unix=last_progress,automatic_training_gate='all nine quotas + final dataset audit + GPU preflight')
    atomic_json(ROOT/'reports/supervisor_status.json',result)
    lines=['# 数据生成与训练看守报告', '', time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime(now)), '',
        f'任务状态：{job.get("status")}；看守 PID：{os.getpid()}。', '',
        '| 分区 | 语言 | 合格唯一音频（小时） | 目标（小时） | 缺口（小时） |', '|---|---|---:|---:|---:|']
    lines += [f'| {q["split"]} | {q["language"]} | {q["hours"]:.3f} | {q["target_hours"]:.3f} | {q["missing_hours"]:.3f} |' for q in quota]
    lines += ['',f'已生成候选：{generated["count"]:,} 条 / {generated["hours"]:.2f} 小时（包含不合格和同句多候选，不能算训练数据）。', '', '## 下一步方案', '']+[f'- {p}' for p in plans]
    lines += ['', '## 服务与恢复', '', *[f'- {k}：{v}' for k,v in health.items()], '', *[f'- {a}' for a in actions[-10:]], '', '训练细节见 TRAINING_PLAN.md；所有数据和预检通过后自动交接。']
    lines += ['', '## 按合成和音色质检参考分别统计', '', '| 语言 | 合成参考 | 质检参考 | 通过 / 已评分候选 | 平均 WavLM | 平均 CAMPPlus | 平均 DNSMOS |', '|---|---|---|---:|---:|---:|---:|']
    lines += [f'| {r["language"]} | {Path(r["synthesis_reference"] or "unknown").name} | {Path(r["quality_reference"] or "unknown").name} | {r["passed_candidates"]}/{r["scored_candidates"]} | {r["mean_wavlm"] or 0:.3f} | {r["mean_camp"] or 0:.3f} | {r["mean_dnsmos"] or 0:.3f} |' for r in reference_quality]
    path=ROOT/'SUPERVISION.md';temp=path.with_suffix('.md.tmp');temp.write_text('\n'.join(lines)+'\n');temp.replace(path)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--once',action='store_true');args=parser.parse_args()
    lock=(ROOT/'watchdog.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=read(ROOT/'watchdog_state.json');history=state.get('restarts',[]);children=[];last_report=0;actions=[]
    while True:
        children[:]=[p for p in children if p.poll() is None]
        control=read(ROOT/'supervision_control.json',dict(enabled=True,auto_recover=True,pause_generation=False))
        if not control.get('enabled',True):break
        cfg=config();job=read(ROOT/'job.json');status=job.get('status');servers=read(ROOT/'servers.json')
        health={ep:healthy(ep) for ep in cfg['tts_endpoints']} if status not in HANDOFF else {'generation_services':'released for training handoff'}
        pid=job.get('supervisor_pid');main_alive=alive(pid,str(CODE/'start_pipeline.py'))
        health['pipeline_process_alive']=main_alive
        if control.get('pause_generation') and main_alive and status not in HANDOFF:
            os.kill(pid,signal.SIGTERM);actions.append('按 pause_generation 停止生成工作进程；保留 TTS 服务和数据。')
        if recovery_allowed(control,status) and not args.once:
            now=time.time();history=[x for x in history if now-x['time']<3600]
            def launch(component,command,log,env=None):
                if not budget_available(history,component,now):
                    actions.append(f'{component} 达到每小时三次自动恢复上限；检查日志后再处理。');return None
                child=spawn(command,log,env);children.append(child)
                history.append(dict(component=component,time=now,pid=child.pid));actions.append(f'自动启动 {component}，PID {child.pid}。')
                return child
            for name,record in servers.items():
                endpoint=f'http://127.0.0.1:{record["port"]}'
                if health.get(endpoint) or alive(record['pid'],str(ASSETS/'VoxCPM2')) or group_alive(record['pid']):continue
                gpu=int(name.removeprefix('tts_gpu'))
                child=launch(name,['bash',str(REPO/'data_generation/majestic_voice/serve.sh'),str(gpu),str(record['port'])],ROOT/'logs'/f'tts_server_gpu{gpu}.log',dict(os.environ,MAJESTIC_VOICE_DATA_ROOT=str(ROOT)))
                if child:
                    record['pid']=child.pid;atomic_json(ROOT/'servers.json',servers)
            previous_workers=read(ROOT/'workers.json')
            workers_alive=any(alive(r['pid'],str(CODE/'run_workers.py')) for r in previous_workers.values())
            # Check again immediately before recovery so a new handoff cannot be mistaken for a crash.
            fresh=read(ROOT/'job.json')
            if not main_alive and not workers_alive and all(health.get(ep) for ep in cfg['tts_endpoints']) and recovery_allowed(control,fresh.get('status')):
                # A previous start can still be loading before publishing job.json.
                loading=any(alive(x['pid'],str(CODE/'start_pipeline.py')) for x in history if x['component']=='pipeline')
                if not loading:launch('pipeline',[str(ASSETS/'.venv-voxcpm2/bin/python'),'-u',str(CODE/'start_pipeline.py')],ROOT/'logs/supervisor.log')
        atomic_json(ROOT/'watchdog_state.json',dict(pid=os.getpid(),updated_at_unix=time.time(),restarts=history,control=control))
        if args.once or time.time()-last_report>=300:
            try:
                report(health,actions)
            except Exception as exc:
                atomic_json(ROOT/'reports/watchdog_report_error.json',dict(error=repr(exc),time=time.time()))
                print(f'Report failed: {exc!r}',flush=True)
                if args.once:raise
            last_report=time.time();actions=actions[-20:]
        if args.once:break
        time.sleep(30)


if __name__=='__main__':main()
