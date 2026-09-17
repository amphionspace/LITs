"""Analyze frozen TensorBoard scalars and existing generated-audio evaluations."""
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
RAW=Path('/119010446/tts-assets/diagnostics/imf_tensorboard_20260916')
ROOT=Path('/119010446/tts-assets/training_runs')
RUNS={'imf':ROOT/'ljs_majestic_100h_imf_h1_b48_from21k_500ep_20260915',
      'fm':ROOT/'ljs_majestic_100h_backbone21k_20260914'}
DATA={m:{k:v for k,v in np.load(RAW/f'{m}_scalars.npz').items()} for m in RUNS}
d=DATA['imf'];cutoff=int(d['imf/train_u_mse_step'][-1,0])+1
WINDOWS=[(1000,10000),(10000,20000),(20000,40000),(40000,60000),(60000,80000),
         (80000,100000),(100000,120000),(120000,136000),(136000,cutoff+1)]

def write_csv(name,rows):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def window(model,tag,lo,hi):
    a=DATA[model][tag]
    return a[(a[:,0]+1>=lo)&(a[:,0]+1<hi),2]

inventory=[]
for model,tags in DATA.items():
    for tag,a in tags.items():
        v=a[:,2];finite=v[np.isfinite(v)]
        inventory.append(dict(model=model,tag=tag,count=len(v),first_step=int(a[0,0]),last_step=int(a[-1,0]),
            first=float(v[0]),last=float(v[-1]),minimum=float(finite.min()),maximum=float(finite.max()),
            median=float(np.median(finite)),nonfinite=int((~np.isfinite(v)).sum()),
            constant=bool(np.all(v==v[0])),zero_fraction=float(np.mean(v==0)),
            duplicate_steps=len(a)-len(np.unique(a[:,0])),backward_jumps=int((np.diff(a[:,0])<0).sum())))
write_csv('all_tag_inventory.csv',inventory)

CORE=['imf/train_u_mse_epoch','imf/val_u_mse_epoch','imf/train_v_mse_epoch','imf/val_v_mse_epoch',
      'imf/train_jvp_rms_epoch','imf/val_jvp_rms_epoch','imf/train_u_weighted_epoch','imf/val_u_weighted_epoch',
      'imf/train_v_weighted_epoch','imf/val_v_weighted_epoch',
      'sub_loss/train_dur_loss_epoch','sub_loss/val_dur_loss_epoch',
      'sub_loss/train_prior_loss_epoch','sub_loss/val_prior_loss_epoch',
      'sub_loss/train_diff_loss_epoch','sub_loss/val_diff_loss_epoch','loss/train_epoch','loss/val_epoch',
      'duration_mask/train_floor_binding_token_ratio_epoch','duration_mask/val_floor_binding_token_ratio_epoch',
      'duration_mask/train_ceiling_binding_token_ratio_epoch','duration_mask/val_ceiling_binding_token_ratio_epoch',
      'duration_mask/train_constrained_score_gap_per_frame_mean_epoch','duration_mask/val_constrained_score_gap_per_frame_mean_epoch',
      'mas_duration/train_speech_le2_ratio_epoch','mas_duration/val_speech_le2_ratio_epoch',
      'grad_norm/grad_2.0_norm_total','learning_rate/decoder_step']
window_rows=[];core_rows=[]
for model in DATA:
    for tag in CORE:
        if tag not in DATA[model]:continue
        a=DATA[model][tag]
        for step,wall,value in a[a[:,0]<cutoff]:
            core_rows.append(dict(model=model,tag=tag,completed_updates=int(step)+1,wall_time=wall,value=value))
        for lo,hi in WINDOWS:
            v=window(model,tag,lo,hi)
            if len(v):window_rows.append(dict(model=model,tag=tag,start_inclusive=lo,end_exclusive=hi,
                 count=len(v),mean=float(v.mean()),median=float(np.median(v)),minimum=float(v.min()),maximum=float(v.max())))
write_csv('core_series.csv',core_rows);write_csv('window_statistics.csv',window_rows)

lr_checks={}
for tag in [k for k in d if k.startswith('learning_rate/') and k.endswith('_step')]:
    a=d[tag];s=a[:,0]
    expected=np.where(s<1000,.0003*(.1+.9*(s+1)/1000),
                      np.where(s<136000,.0003,.0003+(.00002-.0003)*np.minimum(1,(s-136000+1)/34000)))
    lr_checks[tag]=float(abs(a[:,2]-expected).max())
assert max(lr_checks.values())<2e-11
grad=d['grad_norm/grad_2.0_norm_total']
grad_tags=[k for k in d if k.startswith('grad_norm/') and k!='grad_norm/grad_2.0_norm_total']
assert all(np.array_equal(d[k][:,0],grad[:,0]) for k in grad_tags)
summed=np.sqrt(sum(d[k][:,2]**2 for k in grad_tags))
grad_sum_max_relative=float(np.max(abs(summed-grad[:,2])/grad[:,2]))
gradient_groups=defaultdict(list)
for tag in grad_tags:
    name=tag.split('grad_2.0_norm/',1)[1]
    if name.startswith('encoder.proj_w.'):group='duration_predictor'
    elif name.startswith('encoder.'):group='prior_encoder'
    elif name.startswith('spk_emb.'):group='speaker_embedding'
    elif name.startswith('decoder.encoder.'):group='frame_encoder'
    elif name.startswith('decoder.estimator.interval_projector.'):group='interval_projector'
    elif name.startswith(('decoder.estimator.v_up_blocks.','decoder.estimator.v_final_')):group='v_head'
    elif name.startswith(('decoder.estimator.up_blocks.','decoder.estimator.final_')):group='u_head'
    else:group='shared_flow'
    gradient_groups[group].append(tag)
group_curves={g:np.sqrt(sum(d[k][:,2]**2 for k in keys)) for g,keys in gradient_groups.items()}
group_rows=[]
for group,v in group_curves.items():
    for lo,hi in WINDOWS:
        y=v[(grad[:,0]+1>=lo)&(grad[:,0]+1<hi)]
        if len(y):group_rows.append(dict(group=group,parameter_tensors=len(gradient_groups[group]),start=lo,end=hi,
                                        median=float(np.median(y)),maximum=float(y.max()),zero_fraction=float(np.mean(y==0))))
write_csv('gradient_groups.csv',group_rows)

quality=[]
for model,run in RUNS.items():
    for p in sorted((run/'eval').glob('step_*/summary.json')):
        step=int(p.parent.name.split('_')[1]);s=json.loads(p.read_text())
        if step>cutoff or s['status']!='complete' or s['samples']!=650:continue
        g=s['groups'];target=[k for k in g if k.startswith('majestic')]
        assert sum(x['evaluation_failures'] for x in g.values())==0
        quality.append(dict(model=model,step=step,
            target_dnsmos=sum(g[k]['samples']*g[k]['dnsmos_ovrl_mean'] for k in target)/450,
            target_wavlm=sum(g[k]['samples']*g[k]['wavlm_similarity_mean'] for k in target)/450,
            target_camp=sum(g[k]['samples']*g[k]['camp_similarity_mean'] for k in target)/450,
            en_wer_pct=100*g['majestic_en']['wer_micro'],zh_cer_pct=100*g['majestic_zh']['cer_micro'],
            mixed_cer_pct=100*g['majestic_mixed']['cer_micro'],summary=str(p)))
write_csv('generation_quality.csv',quality)
qindex={(r['model'],r['step']):r for r in quality}
common=sorted({r['step'] for r in quality if r['model']=='imf'}&{r['step'] for r in quality if r['model']=='fm'})
dns_wins=sum(qindex['fm',s]['target_dnsmos']>qindex['imf',s]['target_dnsmos'] for s in common)

def line(ax,model,tag,label,color,style='-',smooth=1):
    a=DATA[model][tag];a=a[a[:,0]<cutoff];x=(a[:,0]+1)/1000;y=a[:,2]
    if smooth>1:
        y=np.array([np.median(y[max(0,j-smooth+1):j+1]) for j in range(len(y))])
    ax.plot(x,y,label=label,color=color,ls=style,lw=1.6)

def decorate(ax,title,ylabel='',log=False):
    ax.set_title(title);ax.set_xlabel('Optimizer updates (k)');ax.set_ylabel(ylabel);ax.grid(alpha=.2)
    ax.axvline(24,color='gray',alpha=.4,ls=':');ax.axvline(136,color='purple',alpha=.5,ls=':')
    if log:ax.set_yscale('log')
    ax.legend(fontsize=8)

fig,axes=plt.subplots(3,2,figsize=(13,12),constrained_layout=True)
for ax,metric,title in [(axes[0,0],'u_mse','Raw u compound residual'),(axes[0,1],'v_mse','Raw v residual'),
                         (axes[1,0],'jvp_rms','Directional derivative (JVP), not optimizer gradient')]:
    line(ax,'imf',f'imf/train_{metric}_epoch','IMF train epoch','#2074af')
    line(ax,'imf',f'imf/val_{metric}_epoch','IMF validation pass','#e17c24')
    decorate(ax,title,log=metric=='u_mse')
for ax,metric,title in [(axes[1,1],'dur','Duration loss'),(axes[2,0],'prior','Prior loss (includes 0.91894 constant)')]:
    for m,c in [('imf','#e17c24'),('fm','#2074af')]:
        line(ax,m,f'sub_loss/train_{metric}_loss_epoch',m.upper()+' train',c,'--')
        line(ax,m,f'sub_loss/val_{metric}_loss_epoch',m.upper()+' validation',c)
    decorate(ax,title)
line(axes[2,1],'imf','loss/train_epoch','IMF train total','#2074af')
line(axes[2,1],'imf','loss/val_epoch','IMF validation total','#e17c24')
decorate(axes[2,1],'Total loss mainly reflects duration + prior, flow loss ~2')
fig.suptitle('TensorBoard loss audit | gray: resume 24k; purple: LR decay 136k',fontsize=14)
fig.savefig(OUT/'loss_curves.png',dpi=160);plt.close(fig)

fig,axes=plt.subplots(3,2,figsize=(13,12),constrained_layout=True)
line(axes[0,0],'imf','learning_rate/decoder_step','IMF decoder LR','#e17c24');decorate(axes[0,0],'WSD learning rate')
axes[0,1].plot((grad[:,0]+1)/1000,grad[:,2],alpha=.25,color='#2074af',lw=.6,label='Logged every 50 updates')
line(axes[0,1],'imf','grad_norm/grad_2.0_norm_total','Running median (20 points)','#153b5d',smooth=20)
axes[0,1].axhline(5,color='red',ls='--',label='Clip threshold 5');decorate(axes[0,1],'Total gradient norm before clipping')
for group,v in group_curves.items():
    smooth=np.array([np.median(v[max(0,j-19):j+1]) for j in range(len(v))])
    axes[1,0].plot((grad[:,0]+1)/1000,smooth,label=group,lw=1)
decorate(axes[1,0],'Gradient groups (20-point median)',log=True)
for ax,key,title in [(axes[1,1],'floor_binding_token_ratio','MAS lower-bound binding'),
                      (axes[2,0],'ceiling_binding_token_ratio','MAS upper-bound binding'),
                      (axes[2,1],'constrained_score_gap_per_frame_mean','MAS constraint score cost per frame')]:
    for m,c in [('imf','#e17c24'),('fm','#2074af')]:
        line(ax,m,'duration_mask/val_'+key+'_epoch',m.upper()+' validation',c)
    decorate(ax,title)
fig.suptitle('Optimization and alignment | same 632 validation recordings',fontsize=14)
fig.savefig(OUT/'optimization_alignment.png',dpi=160);plt.close(fig)

fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
for ax,key,title in [(axes[0,0],'target_dnsmos','Target 450: DNSMOS OVRL'),
                     (axes[0,1],'target_wavlm','Target 450: WavLM + ECAPA cosine'),
                     (axes[1,0],'en_wer_pct','Target English WER (%)'),
                     (axes[1,1],'zh_cer_pct','Target Chinese CER (%)')]:
    for m,c in [('imf','#e17c24'),('fm','#2074af')]:
        records=[qindex[m,s] for s in common]
        ax.plot([r['step']/1000 for r in records],[r[key] for r in records],marker='.',label=m.upper()+(' 2 steps' if m=='imf' else ' 10 steps'),color=c)
    decorate(ax,title)
fig.savefig(OUT/'quality_curves.png',dpi=160);plt.close(fig)

families=Counter(k.split('/')[0] for k in d)
imf_inventory=[r for r in inventory if r['model']=='imf']
val_steps=[r for r in imf_inventory if '/val_' in r['tag'] and r['tag'].endswith('_step')]
summary={'snapshot_training_updates':cutoff,'last_full_validation_updates':int(d['imf/val_u_mse_epoch'][-1,0])+1,
    'last_paired_generation_updates':max(common),'scalar_tags':len(d),'scalar_events':sum(len(v) for v in d.values()),
    'families':dict(families),'nonfinite_scalar_events':sum(r['nonfinite'] for r in imf_inventory),
    'constant_tags':sum(r['constant'] for r in imf_inventory),'validation_step_tags_with_restart':sum(r['backward_jumps']>0 for r in val_steps),
    'validation_u_step_duplicate_steps':len(d['imf/val_u_mse_step'])-len(np.unique(d['imf/val_u_mse_step'][:,0])),
    'learning_rate_max_errors':lr_checks,'gradient_logged_points':len(grad),'gradient_over_clip_points':int((grad[:,2]>5).sum()),
    'gradient_max':float(grad[:,2].max()),'gradient_max_updates':int(grad[np.argmax(grad[:,2]),0])+1,
    'gradient_median':float(np.median(grad[:,2])),'gradient_group_reconstruction_relative_error':grad_sum_max_relative,
    'always_zero_logged_gradient_tags':[r['tag'] for r in imf_inventory if r['tag'].startswith('grad_norm/') and r['zero_fraction']==1],
    'paired_generation_checkpoints':len(common),'fm_target_dnsmos_wins':dns_wins}
(OUT/'audit_summary.json').write_text(json.dumps(summary,indent=2))
for model in RUNS:
    (OUT/f'{model}_event_metadata.json').write_text((RAW/f'{model}_event_metadata.json').read_text())
print(json.dumps(summary,indent=2))

def mean(model,tag,lo,hi):return float(window(model,tag,lo,hi).mean())
columns=[('训练 u MSE','imf/train_u_mse_epoch'),('验证 u MSE','imf/val_u_mse_epoch'),
         ('训练 v MSE','imf/train_v_mse_epoch'),('验证 v MSE','imf/val_v_mse_epoch'),
         ('验证 JVP RMS','imf/val_jvp_rms_epoch'),('训练 duration loss','sub_loss/train_dur_loss_epoch'),
         ('验证 duration loss','sub_loss/val_dur_loss_epoch'),('验证 prior loss','sub_loss/val_prior_loss_epoch')]
table=['| 指标 | 20k–40k | 80k–100k | 120k–136k |','|---|---:|---:|---:|']
for name,tag in columns:
    table.append('| '+name+' | '+' | '.join(f'{mean("imf",tag,lo,hi):.5f}' for lo,hi in [(20000,40000),(80000,100000),(120000,136000)])+' |')
family_table=['| 指标族 | 条数 | 用途 / 主要误读 |','|---|---:|---|']
descriptions={
 'grad_norm':'497 条 = 496 个参数张量 + 1 个总范数；不是 497 种性能指标。',
 'mas_duration':'28 种 MAS 分配诊断 × train/val × step/epoch；不是生成音频真实音素时长。',
 'duration_mask':'17 种 mask/约束诊断 × train/val × step/epoch；当前 masked_* 多数是代码设为零。',
 'imf':'6 项 × train/val × step/epoch；优先看原始 u/v MSE 与 JVP。',
 'sub_loss':'duration、prior、diff × train/val × step/epoch。',
 'learning_rate':'4 个优化器参数组 + lr 别名，各自 step/epoch；目前值相同。',
 'loss':'train/val 总损失的 step/epoch。',
 'train':'duration/prior loss 权重，当前都固定为 1。',
 'epoch':'训练 epoch 计数。'}
for family,count in families.items():family_table.append(f'| `{family}` | {count} | {descriptions[family]} |')
quality_table=['| Step | Val u MSE ↓ | IMF DNSMOS ↑ | FM DNSMOS ↑ | IMF WavLM+ECAPA ↑ | FM WavLM+ECAPA ↑ |','|---:|---:|---:|---:|---:|---:|']
for s in [80000,100000,110000,120000,125000,130000,135000]:
    a=d['imf/val_u_mse_epoch'];v=float(a[a[:,0]+1==s,2][0]);i=qindex['imf',s];f=qindex['fm',s]
    quality_table.append(f'| {s//1000}k | {v:.4f} | {i["target_dnsmos"]:.4f} | {f["target_dnsmos"]:.4f} | {i["target_wavlm"]:.4f} | {f["target_wavlm"]:.4f} |')

report=f'''# IMF TensorBoard 指标审计

日期：2026-09-16 UTC。读取当前 `imf_h1_b48/version_0` 两份 event 文件的固定字节快照，覆盖训练至 **{cutoff:,} updates**、完整验证至 **139k**，并读取 FM 对照与共同完成至 **135k** 的 650 条生成评测。只读分析；没有更改训练、checkpoint 或原 TensorBoard 日志。所有训练步数在文中换算为完成的更新次数（TensorBoard 某些标量横轴少 1）。

**判断：没有已记录的数值崩溃证据，确有验证 step 横轴重叠问题；训练效果方面，u 的复合残差已进入平台并回升，duration 泛化出现轻度退化，生成音质长期未追平 FM。不能用接近常数的总 flow loss 宣告收敛，也不能据当前曲线断言必须立即停训。**

## 1. 730 条标量为何这么多

{chr(10).join(family_table)}

一共 {summary['scalar_events']:,} 个记录值，{summary['constant_tags']} 条标量在本次快照内完全不变。逐条统计见 [指标清单 CSV](imf_training_record_assets/tensorboard_audit_20260916/all_tag_inventory.csv)。生成评测的 CER/WER、相似度、DNSMOS **没有被当前 watcher 写入 TensorBoard**，只保存在 `eval/`；这比缺少更多训练标量更影响判断效果。

## 2. 确认的显示与解释问题

### 验证 `_step` 横轴不是训练 step，且续训后重叠

`imf/val_u_mse_step` 的横轴在续训前走到 234，续训后从 0 重启；235 个横轴位置重复，共 {summary['validation_step_tags_with_restart']} 条验证 step 指标出现回退。它们记录验证 batch 计数，不能与训练的 20k/100k 放在同一 step 轴上解读。恢复时按 global step 设置的 `purge_step=24001` 无法解决这种不同坐标的重叠。

**看验证趋势应选 `*_val_*_epoch` / `imf/val_*_epoch`**：它们的 step 单调、无重复，1000 步一次汇总完整验证 pass（这里的 `_epoch` 不表示一定等一个训练 epoch）。24k 在暂停前未完成验证，缺失是已知边界，不能拿状态文件携带的 23k 指标冒充 24k。`train_*_step` 与梯度每 50 步记录；`train_*_epoch` 是 340 updates/epoch 的汇总，24k 恢复所在 epoch 的后半段不能当作完整 340 步均值。

### `weighted≈1`、`diff≈2`、`loss≈3.18` 的含义

当前每个样本的原始平方误差和为 S，目标显示值为 `S / stopgrad(S + 0.01)`。当 S 很大时，u/v 各接近 1，总 flow loss 接近 2。分母停止梯度，因此参数仍然得到梯度；显示值接近常数不代表梯度为零。

所以总 loss 约等于 `2 + duration + prior`，后期总验证 loss 上升主要反映 duration。优先查看 `imf/val_u_mse_epoch`、`imf/val_v_mse_epoch`，不要用加权 loss 或 FM/IMF 总 loss 的绝对大小评判模型。

### 一些“正常值”不是效果证据

- `fm_fraction=0.5` 是规定一半样本走 r=t，不是学出来的结果。
- `duration_mask/*masked* = 0`：启用 constrained MAS 后，代码主动跳过上下界 outlier masking，不意味着所有时长目标都自然正确。
- `blank_* = 0`：135k 的 632 条验证诊断里根本没有 blank ID 0；这些比率分母被 clamp 后为零，不证明 blank 对齐好。
- `mas_duration/*mean`、`mel_per_token*` 部分几乎不变：固定数据总 Mel 帧数和 token 数基本决定了它们。
- TensorBoard 的 `speech_*` 只排除 blank 和 `_`，**还包含声调 token、标点与 `<sil>`**。例如验证 `speech_le2≈33.2%` 不能解释成 33.2% 的真正音素只有两帧。独立 duration 诊断的 `speech` 排除了这些特殊符号，口径不同。
- `*_max_epoch` / `*_min_epoch` 经跨卡与跨 batch 均值聚合，不是全验证集严格最大/最小值；各比率同样不是严格 token 加权的全数据比例。两者验证均为每卡 batch 16、相同的 632 条数据，但各模型自身产生的 MAS 目标可能不同。
- TensorBoard 的 Mel/对齐图片使用训练 **epoch** 作为横轴，而且 rank 0 的展示采样没有固定噪声种子，不应当成固定条件下的质量基准。

## 3. 确实值得关注的训练趋势

下面是区间内汇总点的均值，区间右端不包含；用于避开单个随机 batch 的波动，不代表多次独立训练。

{chr(10).join(table)}

### u 平台与回升：优先级最高

验证 u MSE 从早期大幅下降，但 80k–100k 均值约 0.413，120k–136k 为 0.489；训练 u 同期从 0.416 升至 0.500。训练、验证一起回升，更符合 u 目标的优化平台/波动，**不单是验证集过拟合的典型形状**。与此同时 v 持续缓慢改善，JVP RMS 没有重新爆涨。

u MSE 是 `u + (t-r)·JVP` 对速度目标的复合残差，v MSE 是另一个目标，二者数值不能当相同难度的任务比较。推理实际使用 u，因此 v 变好不足以说明 2 步生成也会变好。现在缺少对角 r=t / 非对角、区间长度 h、语言/时长等分组误差，无法把平台精确归因到哪个子问题。

验证会重新采样 t/r、噪声，还随机选择 streaming 分支，因此单点 u 有额外噪声。139k 的 u=0.3647 不能单独证明已经解决；125k 的 u=0.7218 也不代表该 checkpoint 音频必然最差。

### duration 泛化缓慢变差；prior 基本进入平台

验证 duration 在 40k–60k 均值 0.2110，120k–136k 为 0.2160；训练则约 0.1810 → 0.1796。约 2.4% 的验证相对上升属于轻度泛化退化信号，不是 duration 崩溃。FM 也有较小的同类趋势：同期验证约 0.2119/0.2147（20k–40k / 120k–136k）。在线 MAS 目标也在更新，因此不能把它完全归因于传统固定标签过拟合。

Prior loss 包含常数 `0.5·log(2π)≈0.91894`。验证约 0.969 并不是“误差接近 1”；扣除常数再乘 2，归一化 Mel 平方误差约 0.101。后期验证 prior 基本平台，训练仍缓慢下降，泛化差距略扩大，但幅度很小。

![训练与验证损失曲线](imf_training_record_assets/tensorboard_audit_20260916/loss_curves.png)

## 4. 梯度、学习率与 MAS 有没有硬故障

- 全部已记录 IMF 标量没有 NaN/Inf；已记录的 496 个参数梯度范数没有任何一条全程为零，后期各组均有梯度。梯度存在不能证明每个参数方向都有效学习。
- 总梯度范数来自 **裁剪前** hook。2783 个采样点中，中位数 {summary['gradient_median']:.3f}，最大 {summary['gradient_max']:.3f}，发生在 {summary['gradient_max_updates']:,} updates；只有一个记录点超过 clip=5。后期中位数约 1.1，未见梯度爆炸或整体消失。日志每 50 步采样，这不是全量更新的裁剪率。
- 参数范数按平方和重建总范数，相对偏差最大 {grad_sum_max_relative:.2g}，与已记录 total 一致。IMF 与 FM 梯度量级不同，目标归一化不同，不能把比值直接理解成不稳定程度。
- 四个优化器参数组学习率一致，所有记录点与 WSD 公式一致（最大浮点误差小于 2e-11）。136k 开始线性下降，快照最后约 **2.741e-4**，170k 计划到 2e-5；没发现续训后 LR 错位。下降阶段目前只有约 3k 步，尚不足以判定最终收益。
- 记录中的 `infeasible_sample_ratio`、floor/ceiling rescale 比率均为零，未见这些监测项表示的约束不可行问题；零时长 token 比率也为零。标准 MAS/约束本身会保证若干此类性质，不能据此宣布对齐准确。
- 120k–136k IMF 验证 floor binding 约 37.4%、ceiling binding 约 6.3%；FM 约 34.8%/7.1%。约束仍在明显影响 MAS，值得观察，但没有通用阈值证明这些占比异常。分母只含设有对应界限的 token，不是全部音素。
- 独立 135k duration 诊断中，真实语音 token 的预测时长标准差 / 自身 MAS 标准差约 **0.65–0.67**，与 20k 的 0.67–0.69 接近，仍有时长变化被压平的现象；FM 135k 约 0.63–0.67，属于两者共有的限制。MAS 不是人工真实时长，不能由此直接判定韵律错误或模型排名。

![优化与对齐诊断](imf_training_record_assets/tensorboard_audit_20260916/optimization_alignment.png)

## 5. loss 与生成质量没有一一对应

同 checkpoint 的目标 450 条（中文 200、英文 200、混读 50）均值；FM 10 步、IMF 2 步，其余固定评测协议一致。WavLM 指使用现成微调权重的 **WavLM + ECAPA-TDNN** 余弦相似度；DNSMOS 是自动预测分数。

{chr(10).join(quality_table)}

125k→130k，验证 u 从 0.722 降至 0.387，但 IMF DNSMOS 从 3.321 降至 3.284、相似度从 0.761 降至 0.751，说明 **不能按最低 u loss 自动选择听感最好的 checkpoint**。135k 相似度恢复至 0.764、DNSMOS 回升至 3.312，130k 不是已确认的持续退化起点。

在共同完成的 {len(common)} 对 checkpoint 中，FM 的目标 DNSMOS 有 {dns_wins} 个更高。后期实际生成音质未随 v loss 的缓慢下降而持续改善，这个脱节比加权 loss 接近常数更值得重视。

![实际生成评测曲线](imf_training_record_assets/tensorboard_audit_20260916/quality_curves.png)

## 6. 建议的阅读顺序与下一步

1. **主要看生成指标**：按 checkpoint 同时看分语言 CER/WER、WavLM+ECAPA、CAMPPlus、DNSMOS，并听固定样本。当前这些评测没有进 TensorBoard，后续应另加按训练 global step 记录的评测曲线。
2. **主要看六条训练诊断**：`imf/val_u_mse_epoch`、`imf/val_v_mse_epoch`、`imf/val_jvp_rms_epoch`、`sub_loss/val_dur_loss_epoch`、`sub_loss/val_prior_loss_epoch`、`grad_norm/grad_2.0_norm_total`；训练对应 epoch 曲线用作参照。
3. **控制项**：`learning_rate/decoder_step`、MAS infeasible/rescale、约束 binding 与 score gap。当前 LR 已开始下降，保留现有 170k 计划观察衰减阶段表现；现有证据不支持仅凭这些数值立即停训或再次加大学习率。
4. **下一轮诊断最有价值的新增项**：固定验证随机种子/噪声与 streaming 模式，拆分 diagonal/off-diagonal u 残差、不同 h 区间，以及真正语音 token 的语言分组时长指标。这是需要补充的证据，不是已经执行的训练改动。
5. 暂时隐藏验证 `*_step`、大部分单参数梯度、重复 LR 别名与恒零 mask 项即可显著减少阅读负担；原始记录仍保留用于定位故障。

## 7. 可复核证据

- [全部 IMF/FM 标量统计](imf_training_record_assets/tensorboard_audit_20260916/all_tag_inventory.csv)
- [关键原始曲线 CSV](imf_training_record_assets/tensorboard_audit_20260916/core_series.csv)
- [分窗口统计](imf_training_record_assets/tensorboard_audit_20260916/window_statistics.csv)
- [分组梯度](imf_training_record_assets/tensorboard_audit_20260916/gradient_groups.csv)
- [生成评测汇总与原文件路径](imf_training_record_assets/tensorboard_audit_20260916/generation_quality.csv)
- [审计摘要](imf_training_record_assets/tensorboard_audit_20260916/audit_summary.json)
- [event 提取脚本](imf_training_record_assets/tensorboard_audit_20260916/extract_events.py)、[分析脚本](imf_training_record_assets/tensorboard_audit_20260916/analyze.py)

代码依据：运行快照 `source/lits/models/base.py`（日志坐标、梯度、统计口径），`source/lits/models/components/improved_mean_flow.py`（目标与随机采样），`source/lits/models/lits.py`（prior/MAS/mask），`source/training/majestic_scratch/schedule.py`（LR），`source/training/majestic_scratch/watch_eval.py`（评测输出）。完整冻结的数值提取位于 `/119010446/tts-assets/diagnostics/imf_tensorboard_20260916/`。
'''
(OUT.parents[1]/'imf_tensorboard_audit_zh.md').write_text(report)
