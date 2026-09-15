"""Final training audit, fixed-set trend and paired Chinese probe report."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import html
import json
import math
import os
from pathlib import Path
import statistics


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main(root):
    import numpy as np
    import soundfile as sf
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = root/'diagnostics/checkpoint_progress_200000'
    state = json.loads((root/'training_state.json').read_text())
    assert state['status'] == 'complete' and state['global_step'] == 200000
    assert all(math.isfinite(v) for v in state['metrics'].values())
    validation = rows(root/'validation.jsonl')
    assert all(all(math.isfinite(v) for v in row['metrics'].values()) for row in validation)
    assert '`Trainer.fit` stopped: `max_steps=200000` reached.' in (root/'training.log').read_text()
    final_summary = json.loads((root/'eval/step_00200000/summary.json').read_text())
    assert final_summary['status'] == 'complete'
    assert all(g['evaluation_failures'] == 0 for g in final_summary['groups'].values())
    current = rows(root/'eval/step_00200000/details.jsonl')
    previous = {r['id']:r for r in rows(root/'eval/step_00145000/details.jsonl')}
    assert len(current) == len(previous) == len({r['id'] for r in current}) == 450
    for r in current:
        assert all(r[k] == previous[r['id']][k] for k in ['ref_text','token_ids','tone_ids'])
        wave, sr = sf.read(r['audio'])
        assert sr == 24000 and wave.ndim == 1 and np.isfinite(wave).all()
        assert abs(len(wave)/sr-r['duration']) < 1e-8
    history = []
    for p in sorted((root/'eval').glob('step_*/summary.json')):
        d = json.loads(p.read_text())
        step = int(p.parent.name.split('_')[1])
        if d.get('samples') != 450 or step < 145000:
            continue
        history.append(dict(step=step, groups=d['groups']))
    labels = dict(foundation_en='英文 WER', foundation_zh='中文 CER', foundation_mixed='中英混读 CER')
    best = {}
    for group in labels:
        key = 'wer_micro' if group.endswith('_en') else 'cer_micro'
        r = min(history, key=lambda h:h['groups'][group][key])
        best[group] = dict(step=r['step'], metric=key, value=r['groups'][group][key])
    probe = rows(out/'asr.jsonl')
    assert len(probe) == len({r['id'] for r in probe}) == 48
    groups = defaultdict(list)
    cases = defaultdict(dict)
    for r in probe:
        wave, sr = sf.read(r['audio'])
        assert sr == 24000 and wave.ndim == 1 and np.isfinite(wave).all()
        assert abs(len(wave)/sr-r['duration']) < 1e-8
        groups[('all_zh',r['mode'])].append(r)
        groups[(r['group'],r['mode'])].append(r)
        cases[r['sample']][r['mode']] = r
    scores = {}
    for (group,mode), items in groups.items():
        n = sum(r['reference_characters'] for r in items)
        e = sum(round(r['cer']*r['reference_characters']) for r in items)
        scores.setdefault(group,{})[mode] = dict(cer=e/n, edits=e, characters=n, samples=len(items))
    paired = {}
    for mode in ['pred','mas']:
        delta = [c[mode+'_200000']['cer']-c[mode+'_145000']['cer'] for c in cases.values()]
        paired[mode] = dict(improved=sum(v < -1e-9 for v in delta), unchanged=sum(abs(v) < 1e-9 for v in delta), worse=sum(v > 1e-9 for v in delta))
    completed = datetime.fromtimestamp(state['updated_at_unix'], timezone.utc).isoformat()
    result = dict(training_state=state, training_completed_utc=completed, fixed_eval_history=history,
        best_since_145000_by_metric=best, chinese_probe=scores, paired_probe=paired,
        verification=dict(fixed_eval_samples=450, same_eval_ids_text_tokens_tones=True, finite_final_eval_audio=True,
            normal_max_steps_exit=True, validation_records=len(validation), all_validation_metrics_finite=True, final_evaluation_failures=0,
            paired_probe_audio=48, finite_probe_audio=True, checkpoint=json.loads((out/'metadata.json').read_text())))
    (out/'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    fig, axes = plt.subplots(3,1,figsize=(11,9),sharex=True,layout='constrained')
    for ax,(group,label) in zip(axes,labels.items()):
        key = 'wer_micro' if group.endswith('_en') else 'cer_micro'
        ax.plot([h['step']/1000 for h in history],[h['groups'][group][key]*100 for h in history],marker='.',linewidth=1)
        ax.set(ylabel=('English WER %' if group.endswith('_en') else 'Chinese CER %' if group.endswith('_zh') else 'Mixed CER %'))
        ax.grid(alpha=.2)
    axes[-1].set_xlabel('Training step (thousands)')
    fig.savefig(out/'evaluation_trend.png',dpi=140)
    plt.close(fig)
    esc=html.escape
    page=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>200000 步训练检查</title>',
        '<style>body{max-width:1400px;margin:30px auto;font:16px/1.5 system-ui}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccc}audio{width:280px}img{max-width:100%}details{margin:12px 0}</style>',
        f'<h1>200000 步训练检查</h1><p>训练在 {completed} 正常达到 max_steps=200000。最终 450 条评估完成；文本、token 和声调序列与 145000 步逐项一致，所有音频通过有限值、采样率与时长检查。</p>',
        '<h2>固定 450 条评估</h2><p>英文 200 条、中文 200 条、中英混读 50 条。英文使用 WER，其他组使用 CER；均为按参考长度加权的 micro 指标。历史最低值仅用于定位候选 checkpoint，不代表独立测试集确认的最优模型。</p><table><tr><th>步数</th><th>英文 WER</th><th>中文 CER</th><th>混读 CER</th></tr>']
    for h in history:
        if h['step'] in [145000,179000,197000,200000]:
            page.append(f'<tr><td>{h["step"]}</td>'+''.join(f'<td>{h["groups"][g]["wer_micro" if g.endswith("_en") else "cer_micro"]:.2%}</td>' for g in labels)+'</tr>')
    page.append('</table><img src="evaluation_trend.png"><h2>12 条中文样本的配对复核</h2><p>复用上一轮训练/验证各 6 条样本。固定噪声、10 步生成和声码器；正常模式使用各自预测时长，MAS 模式固定录音总帧数。两份 checkpoint 的 48 条音频全部在本轮使用 batch size 1 重新做 ASR。小样本结果与固定 200 条中文评估是不同集合，不能直接混算。</p><table><tr><th>生成方式</th><th>145000 步 CER</th><th>200000 步 CER</th><th>逐句改善 / 持平 / 变差</th></tr>')
    for mode,label in [('pred','正常预测时长'),('mas','MAS 对齐时长')]:
        p=paired[mode]
        page.append(f'<tr><td>{label}</td><td>{scores["all_zh"][mode+"_145000"]["cer"]:.2%}</td><td>{scores["all_zh"][mode+"_200000"]["cer"]:.2%}</td><td>{p["improved"]} / {p["unchanged"]} / {p["worse"]}</td></tr>')
    page.append('</table>')
    for name,c in sorted(cases.items()):
        page.append(f'<details><summary>{name}：{esc(c["pred_200000"]["ref_text"])}</summary><table><tr><th>设置</th><th>音频</th><th>ASR</th><th>CER</th></tr>')
        for mode in ['pred_145000','pred_200000','mas_145000','mas_200000']:
            r=c[mode];link=os.path.relpath(r['audio'],out)
            page.append(f'<tr><td>{mode}</td><td><audio controls preload="none" src="{esc(link)}"></audio></td><td>{esc(r["asr_text"])}</td><td>{r["cer"]:.2%}</td></tr>')
        page.append('</table></details>')
    page.append('<h2>最终固定评估：逐句音频</h2>')
    for group,label in labels.items():
        items=[r for r in current if r['group']==group]
        metric='wer' if group.endswith('_en') else 'cer'
        items.sort(key=lambda r:r[metric],reverse=True)
        page.append(f'<details><summary>{label} · {len(items)} 条，按错误率从高到低</summary><table><tr><th>原文</th><th>音频</th><th>ASR</th><th>错误率</th></tr>')
        for r in items:
            link=os.path.relpath(r['audio'],out)
            page.append(f'<tr><td>{esc(r["ref_text"])}</td><td><audio controls preload="none" src="{esc(link)}"></audio></td><td>{esc(r["asr_text"])}</td><td>{r[metric]:.2%}</td></tr>')
        page.append('</table></details>')
    page.append('<p>自动 ASR 和 DNSMOS 不能替代人工听感评估；MAS 不是音素时长真值。本次检查未更改训练配置或启动下一阶段。</p></html>')
    (out/'index.html').write_text('\n'.join(page))
    print(json.dumps(dict(training_completed_utc=completed,best=best,chinese_probe=scores,paired=paired),ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    main(parser.parse_args().run)
