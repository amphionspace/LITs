"""Summarize paired model-health probes and render plots/audio for inspection."""
import argparse
from collections import defaultdict
import html
import json
from pathlib import Path
import re
import statistics
import unicodedata


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def main(out):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    source=rows(out/'synthesis.jsonl');asr={r['id']:r for r in rows(out/'asr.jsonl')}
    expected=json.loads((out/'metadata.json').read_text())['expected_audio_count']
    assert len(source)==len(asr)==expected
    samples={r['diagnostic_id']:r for r in json.loads((out/'selected_samples.json').read_text())}
    diagnostics={name:json.loads((out/name/'diagnostic.json').read_text()) for name in samples}
    groups=defaultdict(list)
    cases=defaultdict(list)
    for r in source:
        groups[(r['group'],r['mode'])].append(r);cases[r['sample']].append(r)
    scores={}
    for (group,mode),items in groups.items():
        is_en=group.endswith('_en');key='wer' if is_en else 'cer'
        numerator=denominator=0
        for r in items:
            ref=unicodedata.normalize('NFKC',r['ref_text']).lower()
            n=len(re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)*",ref.replace('’',"'"))) if is_en else sum(unicodedata.category(c)[0] in ('L','N') for c in ref)
            numerator+=asr[r['id']][key]*n;denominator+=n
        scores.setdefault(group,{})[mode]=dict(metric=key,micro=numerator/denominator,reference_units=denominator,
            samples=len(items),edit_count=round(numerator))
    paired={}
    for group in scores:
        names=[n for n,r in samples.items() if r['group']==group];metric='wer' if group.endswith('_en') else 'cer'
        items=[]
        for n in names:
            a=statistics.mean(asr[f'{n}/pred_10_s{s}'][metric] for s in [0,1])
            b=statistics.mean(asr[f'{n}/pred_30_s{s}'][metric] for s in [0,1])
            items.append(dict(sample=n,error_10=a,error_30=b))
        paired[group]=dict(items=items,improved=sum(r['error_30']<r['error_10']-1e-9 for r in items),
            unchanged=sum(abs(r['error_30']-r['error_10'])<=1e-9 for r in items),worse=sum(r['error_30']>r['error_10']+1e-9 for r in items))
    real=[r for r in diagnostics.values() if 'real_frames' in r]
    alignment={}
    for group in ['train_en','train_zh','val_en','val_zh']:
        selected=[r for r in real if r['group']==group]
        alignment[group]={k:dict(mean=statistics.mean(r[k] for r in selected),minimum=min(r[k] for r in selected),maximum=max(r[k] for r in selected))
                          for k in ['predicted_to_real_duration','duration_mae_frames','prior_mse','prior_zero_mean_mse']}
        alignment[group]['mas_stats_mean']={k:statistics.mean(r['mas_stats'][k] for r in selected) for k in selected[0]['mas_stats']}
    flow={}
    for t in [.05,.25,.5,.75,.95]:
        flow[str(t)]={c:statistics.mean(next(s['flow_mse'] for s in r['text_conditioning_loss'] if s['condition']==c and s['t']==t) for r in real)
                      for c in ['correct','reversed','zero']}
    optimizer=json.loads((out/'optimizer_audit.json').read_text())
    summary=dict(utterances=36,real_utterances=24,mixed_texts=12,audio_count=len(source),
        optimizer=dict(tensors=optimizer['parameter_tensors'],numel=optimizer['parameter_count'],exact_coverage=optimizer['optimizer_exact_coverage'],
            finite=optimizer['finite_checkpoint_tensors'],groups={g['group']:dict(tensors=len(g['parameters']),
                updated_tensors=sum(p['update_norm']>0 for p in g['parameters']),
                missing_optimizer_state=sum(not p['optimizer_state_present'] for p in g['parameters'])) for g in optimizer['groups']}),
        frontend_matches=sum(all(r['frontend'].values()) for r in diagnostics.values()),
        inference_entrypoint_max_error=next(r['inference_entrypoint_max_error'] for r in real if 'inference_entrypoint_max_error' in r),
        normalization_roundtrip_max_error=max(r['padding_roundtrip_error'] for r in real),
        alignment=alignment,text_conditioning=flow,asr=scores,paired_steps=paired)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(12,4.4))
    ts=[float(t) for t in flow]
    for c in ['correct','reversed','zero']:axes[0].plot(ts,[flow[str(t)][c] for t in ts],marker='o',label=c)
    axes[0].set(xlabel='Flow time (0 = noise, 1 = data)',ylabel='Velocity MSE',yscale='log',title='Text-conditioning intervention (24 real samples)');axes[0].legend();axes[0].grid(alpha=.2)
    gs=list(alignment)
    axes[1].boxplot([[r['predicted_to_real_duration'] for r in real if r['group']==g] for g in gs],tick_labels=gs)
    axes[1].axhline(1,color='gray',linestyle='--');axes[1].set(ylabel='Predicted / recorded duration',title='Duration comparison (6 samples per group)');axes[1].grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(out/'health_overview.png',dpi=160);plt.close(fig)
    for name,d in diagnostics.items():
        if 'real_frames' not in d:continue
        data=np.load(out/name/'alignment.npz');n=d['real_frames'];duration=n*.016
        fig,axes=plt.subplots(3,1,figsize=(10,7))
        axes[0].imshow(data['real_mel'][0],aspect='auto',origin='lower',extent=[0,duration,0,100]);axes[0].set(ylabel='Mel bin',title=name+' — recorded Mel')
        axes[1].imshow(data['alignment'][0,:,:n],aspect='auto',origin='lower',extent=[0,duration,0,len(d['mas_durations'])]);axes[1].set(xlabel='Recorded audio (seconds)',ylabel='Token index',title='Model-derived constrained MAS (not ground truth)')
        axes[2].plot(d['mas_durations'],label='MAS');axes[2].plot(d['predicted_durations'],label='Predicted',alpha=.8);axes[2].set(xlabel='Token index',ylabel='Frames per token');axes[2].legend()
        fig.tight_layout();fig.savefig(out/name/'alignment.png',dpi=120);plt.close(fig)
    labels={'original':'真实原音频','reconstruction':'真实 Mel 重建','mas_10_s0':'MAS 时长，10 步，噪声 0',
            'pred_matched_duration_10_s0':'预测时长整体调整至原录音长度，10 步，噪声 0',
            'pred_10_s0':'预测时长，10 步，噪声 0','pred_30_s0':'预测时长，30 步，噪声 0',
            'pred_10_s1':'预测时长，10 步，噪声 1','pred_30_s1':'预测时长，30 步，噪声 1'}
    body=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>模型健康诊断</title>',
          '<style>body{max-width:1150px;margin:25px auto;padding:0 18px;font:16px/1.6 system-ui;color:#222}table{border-collapse:collapse;width:100%}td,th{padding:7px;border:1px solid #ddd}audio,img{width:100%}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:12px}.card{border:1px solid #ddd;padding:12px;border-radius:8px}.case{margin:35px 0}small{color:#555}</style>',
          f'<h1>模型健康诊断：145k checkpoint</h1><p>24 条真实录音 + 12 条混读文本，{len(source)} 段配对音频。两份固定噪声分别比较 10/30 步推理；真实样本另有 Mel 重建与 MAS 时长对照。正式训练继续运行，未修改参数或数据。</p>',
          '<p><a href="FINDINGS.md">诊断结论及解释边界</a> · <a href="summary.json">汇总数据</a> · <a href="metadata.json">实验设置</a></p>',
          '<p>表中英文使用 WER，中文及混读使用 CER。每个真实组仅 6 条样本，混读 12 条；各噪声条件复用相同文本。MAS 是模型推导的对齐，不能视为真实音素时长。</p>',
          '<table><tr><th>条件</th><th>训练英文</th><th>验证英文</th><th>训练中文</th><th>验证中文</th><th>混读</th></tr>']
    for mode,label in labels.items():
        body.append('<tr><td>'+label+'</td>'+''.join('<td>'+ (f'{scores[g][mode]["micro"]:.2%}' if mode in scores[g] else '—')+'</td>' for g in ['train_en','val_en','train_zh','val_zh','mixed'])+'</tr>')
    body.extend(['</table><img src="health_overview.png" alt="文本条件与时长诊断">',
                 '<label>样本分组：<select id="group"><option value="all">全部</option>'+''.join(f'<option>{g}</option>' for g in scores)+'</select></label>'])
    for name,items in cases.items():
        body.append(f'<section class="case" data-group="{items[0]["group"]}"><h2>{html.escape(name)}</h2><p>{html.escape(items[0]["ref_text"])}</p>')
        if (out/name/'alignment.png').exists():body.append(f'<details><summary>查看对齐和时长</summary><img loading="lazy" src="{name}/alignment.png"></details>')
        body.append('<div class="grid">')
        for r in items:
            rel=Path(r['audio']).relative_to(out).as_posix();assert (out/rel).exists()
            body.append(f'<div class="card"><strong>{labels[r["mode"]]}</strong><audio controls preload="none" src="{html.escape(rel)}"></audio><p>ASR：{html.escape(asr[r["id"]]["asr_text"])}</p><small>{r["duration"]:.2f} 秒</small></div>')
        body.append('</div></section>')
    body.append('<script>document.getElementById("group").addEventListener("change",e=>document.querySelectorAll(".case").forEach(s=>s.hidden=e.target.value!=="all"&&s.dataset.group!==e.target.value));</script></html>')
    (out/'index.html').write_text('\n'.join(body))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
