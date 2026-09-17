"""Read existing completed evaluations; never launch or modify training/evaluation."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
DOCS = OUT.parents[1]
ROOT = Path('/119010446/tts-assets/training_runs')
RUNS = {
    'FM': ROOT / 'ljs_majestic_100h_backbone21k_20260914',
    'IMF': ROOT / 'ljs_majestic_100h_imf_h1_b48_from21k_500ep_20260915',
}
MAX_STEP = 110000  # Freeze the completed evaluation boundary at analysis start.
GROUPS = {'majestic_zh': 200, 'majestic_en': 200, 'majestic_mixed': 50, 'ljs_en': 200}
LABELS = {'majestic_zh': '目标中文', 'majestic_en': '目标英文', 'majestic_mixed': '目标混读', 'ljs_en': 'LJS 英文'}
METRICS = ['cer_micro', 'wer_micro', 'wavlm_similarity_mean', 'camp_similarity_mean',
           'dnsmos_ovrl_mean', 'dnsmos_sig_mean', 'dnsmos_bak_mean']

def read(p):
    return json.loads(p.read_text())

def lines(p):
    return [json.loads(x) for x in p.read_text().splitlines()]

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write_csv(name, rows):
    with (OUT / name).open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

protocol = read(RUNS['FM'] / 'data/eval_protocol.json')
assert protocol == read(RUNS['IMF'] / 'data/eval_protocol.json')
for p in RUNS.values():
    assert sha(p / 'data/eval_manifest.jsonl') == protocol['manifest_sha256']
available = []
for p in RUNS.values():
    available.append({int(x.parent.name.split('_')[1]) for x in (p/'eval').glob('step_*/summary.json')
                      if read(x)['status'] == 'complete' and read(x)['samples'] == 650})
steps = sorted(s for s in available[0] & available[1] if s <= MAX_STEP)
summaries, details, rows, provenance, duration_rows = {}, {}, [], [], []
for step in steps:
    for model, run in RUNS.items():
        p = run / f'eval/step_{step:08d}'
        summary, metadata = read(p/'summary.json'), read(p/'checkpoint_metadata.json')
        assert metadata['global_step'] == step and metadata['scope'] == 'full_650'
        d = lines(p/'details.jsonl')
        assert len(d) == 650 and len({r['id'] for r in d}) == 650
        for group, count in GROUPS.items():
            g = summary['groups'][group]
            assert g['samples'] == count and g['evaluation_failures'] == 0
            gd = [r for r in d if r['group'] == group]
            assert len(gd) == count
            for key in ['wavlm_similarity', 'camp_similarity', 'dnsmos_ovrl', 'dnsmos_sig', 'dnsmos_bak']:
                assert np.isfinite([r[key] for r in gd]).all()
                assert abs(np.mean([r[key] for r in gd]) - g[key+'_mean']) < 1e-10
            for metric, den in [('cer','reference_characters'),('wer','reference_words')]:
                total = sum(r.get(den,0) for r in gd)
                if total:
                    actual = sum(r[metric]*r.get(den,0) for r in gd)/total
                    assert abs(actual-g[metric+'_micro']) < 1e-10
            rows.append(dict(step=step,model=model,group=group,samples=count,**{k:g[k] for k in METRICS}))
        if model == 'IMF':
            assert all(r['flow_steps'] == 2 and r['flow_objective'] == 'imf' for r in d)
        summaries[step,model] = summary['groups']
        details[step,model] = {r['id']:r for r in d}
        provenance.append(dict(step=step,model=model,summary=str(p/'summary.json'),summary_sha256=sha(p/'summary.json'),
                               details_sha256=sha(p/'details.jsonl'),checkpoint_sha256=metadata['sha256']))
        ds = read(p/'duration_summary.json')
        assert ds['samples'] == 632
        for lang,g in ds['groups'].items():
            a=g['speech']['inference']
            duration_rows.append(dict(step=step,model=model,language=lang,**{k:a[k] for k in ['pearson','std_ratio','mae_frames','mean_ratio']}))
    a,b=details[step,'FM'],details[step,'IMF']
    assert a.keys()==b.keys()
    for key in a:
        for field in ['group','ref_text','speaker','token_ids','tone_ids','reference_characters','reference_words','dnsmos_protocol']:
            assert a[key][field] == b[key][field], (step,key,field)
write_csv('all_checkpoints.csv',rows)
write_csv('duration_diagnostics.csv',duration_rows)
(OUT/'provenance.json').write_text(json.dumps(dict(max_step=MAX_STEP,steps=steps,protocol=protocol,files=provenance),indent=2))

def target(step,model,key):
    g=summaries[step,model]
    return sum(g[k]['samples']*g[k][key] for k in GROUPS if k.startswith('majestic'))/450

selected=[2000,10000,20000,40000,60000,80000,100000,110000]
table=['| 训练 step | 英文 WER% ↓ | 中文 CER% ↓ | 混读 CER% ↓ | 目标音色 WavLM ↑ | 目标音质 DNSMOS ↑ |',
       '|---:|---:|---:|---:|---:|---:|']
for s in selected:
    values=[]
    for group,key in [('majestic_en','wer_micro'),('majestic_zh','cer_micro'),('majestic_mixed','cer_micro')]:
        values.append(' / '.join(f'{100*summaries[s,m][group][key]:.3f}' for m in RUNS))
    for key in ['wavlm_similarity_mean','dnsmos_ovrl_mean']:
        values.append(' / '.join(f'{target(s,m,key):.4f}' for m in RUNS))
    table.append(f'| {s//1000}k | '+' | '.join(values)+' |')

latest=steps[-1]
group_table=['| 组别 | ASR 错误率% ↓ | WavLM ↑ | CAMPPlus ↑ | DNSMOS OVRL ↑ | DNSMOS SIG ↑ | DNSMOS BAK ↑ |',
             '|---|---:|---:|---:|---:|---:|---:|']
for group in GROUPS:
    keys=['wer_micro' if group.endswith('_en') else 'cer_micro','wavlm_similarity_mean','camp_similarity_mean',
          'dnsmos_ovrl_mean','dnsmos_sig_mean','dnsmos_bak_mean']
    vals=[' / '.join(f'{summaries[latest,m][group][k]*(100 if i==0 else 1):.4f}' for m in RUNS) for i,k in enumerate(keys)]
    group_table.append('| '+LABELS[group]+' | '+' | '.join(vals)+' |')

# Paired sentence bootstrap at the latest checkpoint, fixed seed; no training-seed inference.
rng=np.random.default_rng(20260916)
bootstrap=[]
for group,count in GROUPS.items():
    ids=[k for k,r in details[latest,'FM'].items() if r['group']==group]
    ix=rng.integers(0,count,size=(4000,count))
    for metric in ['wer' if group.endswith('_en') else 'cer','wavlm_similarity','camp_similarity','dnsmos_ovrl']:
        a=np.array([details[latest,'FM'][k][metric] for k in ids])
        b=np.array([details[latest,'IMF'][k][metric] for k in ids])
        if metric in ['cer','wer']:
            den='reference_words' if metric=='wer' else 'reference_characters'
            weights=np.array([details[latest,'FM'][k][den] for k in ids])
            samples=100*((b-a)*weights)[ix].sum(1)/weights[ix].sum(1)
            delta=100*np.sum((b-a)*weights)/weights.sum()
        else:
            samples=(b-a)[ix].mean(1);delta=(b-a).mean()
        low,high=np.quantile(samples,[.025,.975])
        bootstrap.append(dict(step=latest,group=group,metric=metric,imf_minus_fm=float(delta),low95=float(low),high95=float(high)))
write_csv('latest_paired_bootstrap.csv',bootstrap)

fig,axes=plt.subplots(4,3,figsize=(14,13),sharex=True,constrained_layout=True)
for row,group in enumerate(GROUPS):
    err='wer_micro' if group.endswith('_en') else 'cer_micro'
    for col,key in enumerate([err,'wavlm_similarity_mean','dnsmos_ovrl_mean']):
        ax=axes[row,col]
        for model,color in [('FM','#2776b8'),('IMF','#e37925')]:
            y=[summaries[s,model][group][key]*(100 if col==0 else 1) for s in steps]
            ax.plot(np.array(steps)/1000,y,label=model+(' 10 steps' if model=='FM' else ' 2 steps'),color=color,marker='.',lw=1.6)
        ax.set_title(group+' | '+({'wer_micro':'WER (%)','cer_micro':'CER (%)'}.get(key,key.replace('_mean',''))))
        ax.grid(alpha=.2)
        if row==3:ax.set_xlabel('Training updates (k; excludes foundation 21k)')
        if row==0 and col==0:ax.legend()
fig.suptitle('Matched training checkpoints | same 650 fixed evaluation sentences',fontsize=15)
fig.savefig(OUT/'checkpoint_curves.png',dpi=160)
plt.close(fig)

late=[s for s in steps if s>=60000]
counts={}
for group in GROUPS:
    metric='wer_micro' if group.endswith('_en') else 'cer_micro'
    deltas=np.array([summaries[s,'IMF'][group][metric]-summaries[s,'FM'][group][metric] for s in late])
    counts[group]=dict(better=int((deltas < -1e-12).sum()),equal=int((abs(deltas)<=1e-12).sum()),worse=int((deltas>1e-12).sum()))
dns_delta=[target(s,'IMF','dnsmos_ovrl_mean')-target(s,'FM','dnsmos_ovrl_mean') for s in steps]
print(json.dumps(dict(steps=steps,late_steps=late,late_asr_counts=counts,dnsmos_delta_range=[min(dns_delta),max(dns_delta)],
                      latest_bootstrap=bootstrap),indent=2))

ci_lines=['| 组别 | 指标 | IMF − FM | 配对 bootstrap 95% 区间 |','|---|---|---:|---:|']
for r in bootstrap:
    ci_lines.append(f"| {LABELS[r['group']]} | {r['metric']} | {r['imf_minus_fm']:+.4f} | [{r['low95']:+.4f}, {r['high95']:+.4f}] |")

report=f'''# IMF 与 FM：相同训练 checkpoint 的性能对比

分析日期：2026-09-16 UTC。配对 **{len(steps)} 个完整 checkpoint（2k–110k）**，每个模型每个 checkpoint 均为固定 650 条。1k 为小样本 smoke，不混入；分析开始时 IMF 115k 尚未完成，固定截至 110k。以下数值均来自已有完整评测，本次没有重新训练或更改推理配置。

**结论：IMF 2 步较早达到较低识别错误率；后期准确率接近且有波动，不能说全面优于 FM。FM 10 步的目标音色 DNSMOS 在 {sum(x<0 for x in dns_delta)}/{len(steps)} 个配对 checkpoint 更高；到 110k，FM 的 WavLM/CAMPPlus 也在四个分组全部更高。当前表现是低步数推理与音质之间的取舍。**

## 比较口径

- FM：`{RUNS['FM'].name}`；IMF：`{RUNS['IMF'].name}`，即 `imf_h1_b48/version_0` 的 3e-4 基线。
- “同 checkpoint”按各自本轮 optimizer update 配对，均不计共同 Foundation 21k。相同初始化谱系、冻结数据、有效 batch 192、峰值 LR 3e-4；110k 前均处于相同计划的稳定学习率阶段。不是同一权重文件，也不是相同训练耗时比较。
- 推理采用已有设置：**FM 10 步 Euler，IMF 2 步 Euler**；temperature=1，相同逐条随机种子规则、token/tone、speaker 与参考录音；原始 24kHz Vocos。读取到的 Vocos 权重 SHA-256 为 `1d60c04156e59348566c65cab921591e1651ab48757ca3766caf9459475bd1c2`，原 FM 工作树与 IMF 快照一致。
- 固定目标中文 200、目标英文 200、目标混读 50、LJS 英文 200。逐对核验 650 个 ID、文本、speaker、token/tone、ASR 分母与 DNSMOS 协议一致；所有纳入结果均无评测失败。评测清单 SHA-256：`{protocol['manifest_sha256']}`。
- 中文/混读报 micro CER，英文报 micro WER，均为百分数，越低越好；WavLM、CAMPPlus、DNSMOS 越高越好。目标音色/音质汇总按 450 句加权，不含 LJS。四组参考录音不同，主要看同组模型差异。
- DNSMOS 是自动预测音质分数，不是人工 MOS，不能单凭它断言某种噪声已解决。总训练 loss 定义不同，不用于性能排名。

## 多 checkpoint 主表

**每格顺序均为 FM / IMF。** 全部 {len(steps)} 点及分组数据见 [CSV](imf_training_record_assets/fm_comparison_20260916/all_checkpoints.csv)。

{chr(10).join(table)}

IMF 在 2k 的目标英文 WER 从 FM 的 3.380% 降至 1.449%，中文 CER 从 3.016% 降至 0.820%，混读 CER 从 9.486% 降至 1.330%，早期优势很清楚。但 100k 三项错误率都反向变差，110k 又全部较低，说明后期不能挑一个点判定稳定领先。

60k–110k 共 {len(late)} 个配对点中，IMF 英文 WER 更低/相同/更高为 {counts['majestic_en']['better']}/{counts['majestic_en']['equal']}/{counts['majestic_en']['worse']}，中文 CER 为 {counts['majestic_zh']['better']}/{counts['majestic_zh']['equal']}/{counts['majestic_zh']['worse']}，混读 CER 为 {counts['majestic_mixed']['better']}/{counts['majestic_mixed']['equal']}/{counts['majestic_mixed']['worse']}。这些是相关 checkpoint 的描述性计数，不是独立实验次数。

目标 450 条的 DNSMOS OVRL：IMF − FM 的差值范围为 {min(dns_delta):+.4f} 至 {max(dns_delta):+.4f}；22/23 个点较低，唯一例外是 5k（IMF 高 0.0032）。110k 为 3.3044 对 3.3500，差 −0.0456。IMF 后期音质未呈现随训练步数持续追平的趋势。

![全部配对 checkpoint 的四组性能曲线](imf_training_record_assets/fm_comparison_20260916/checkpoint_curves.png)

## 110k 分语言与音色

每格仍是 **FM / IMF**。英文 ASR 栏为 WER，中文和混读为 CER。

{chr(10).join(group_table)}

110k 的主要音质差距出现在目标英文（OVRL 低约 0.077）与 LJS 英文（低约 0.060）；目标中文差距约 0.014。四组 WavLM 均低于 FM，CAMPPlus 也均略低，其中目标中文 CAMPPlus 几乎持平。

混读 CER 只覆盖当前归一化后的字符错误；已有英文片段 WER 在 110k 为 FM 6.25%、IMF 8.33%（仅 50 句中的少量英文词）。因此混读 CER 改善不能解释成中英两部分都更好。

## 最新点的不确定性

按同一文本配对重采样 4000 次，seed=20260916。错误率差值单位为百分点，其余为原分数。区间只反映固定评测文本的抽样不确定性，不含不同训练 seed/推理 seed 的变化，不作多重比较校正。

110k 四组 ASR 差值区间都跨过 0，因此该点错误率数值较低不足以确认稳定优势；四组 DNSMOS 差值区间均在 0 以下，目标中文上界非常接近 0。

{chr(10).join(ci_lines)}

## 推理效率与 duration 的边界

IMF 使用 2 次采样更新，FM 使用 10 次，采样更新次数减少 80%。**这不等于端到端快 5 倍**：声学网络路径、duration/prior、Vocos、I/O 与共享 GPU 负载都会影响时间。历史 `synthesis_seconds` 包括 waveform 写盘且测量环境未隔离，本报告不把它当受控延迟 benchmark。尚无 FM 同 checkpoint 2 步对照，因此这里证明的是现有 IMF-2 与 FM-10 配方的差异，不把收益全部归因于训练目标。

632 条 duration 诊断已导出 [CSV](imf_training_record_assets/fm_comparison_20260916/duration_diagnostics.csv)。其中相关性/MAE 都是相对于各模型自身产生的 MAS，对齐目标也会改变，不能当作共同真实时长标签上的准确率来排名。

## 可追溯数据

- [完整配对指标](imf_training_record_assets/fm_comparison_20260916/all_checkpoints.csv)
- [checkpoint 与评测文件哈希、协议](imf_training_record_assets/fm_comparison_20260916/provenance.json)
- [110k 配对区间](imf_training_record_assets/fm_comparison_20260916/latest_paired_bootstrap.csv)
- [本报告生成脚本](imf_training_record_assets/fm_comparison_20260916/build_report.py)
- [IMF 训练与恢复记录](imf_training_record_zh.md)、[FM 训练记录](majestic_training_record_zh.md)

原始结果位于两实验各自的 `eval/step_XXXXXXXX/summary.json` 与 `details.jsonl`。本报告复核各组 micro 错误率和音色/音质均值与逐句数据一致。
'''
(DOCS/'imf_vs_fm_checkpoints_zh.md').write_text(report)
print('\n'.join(table))
