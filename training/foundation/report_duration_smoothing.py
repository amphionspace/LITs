"""Render duration diagnostic tables and standalone plots from saved token pairs."""
import argparse
import json
from pathlib import Path

import numpy as np

from training.foundation.diagnose_duration_smoothing import pair_metrics


def main(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    summary = json.loads((root/'summary.json').read_text())
    metadata = json.loads((root/'metadata.json').read_text())
    rows = [json.loads(line) for line in (root/'samples.jsonl').read_text().splitlines()]
    content = ['# 197k checkpoint：Duration 平滑诊断', '',
        '本报告比较同一个 checkpoint 的 MAS 对齐时长和 duration predictor 输出。MAS 是模型派生参照，不是人工时长真值；分布压缩能支持相对于 MAS 的 oversmoothing 判断，不能单独证明机械感的因果来源。', '',
        '## 结果判断', '',
        '197k 在这套固定验证集上存在明显的、相对于 MAS 的 duration 分布压缩，中文更强。排除静音/标点/声调 token、按句统计、去掉整体语速差异之后仍然成立。', '',
        f'- 原始预测的 speech-token Std/MAS Std：中文 {summary["zh"]["speech"]["raw"]["std_ratio"]:.1%}，英文 {summary["en"]["speech"]["raw"]["std_ratio"]:.1%}。',
        f'- 实际推理时长的 speech-token Std/MAS Std：中文 {summary["zh"]["speech"]["inference"]["std_ratio"]:.1%}，英文 {summary["en"]["speech"]["inference"]["std_ratio"]:.1%}。压缩在取整/限幅之前已存在。',
        f'- 将每句预测与 MAS 分别归一化到相同均值后，推理时长的 Std 比仍只有中文 {summary["zh"]["speech"]["inference"]["sentence_mean_normalized"]["std_ratio"]:.1%}、英文 {summary["en"]["speech"]["inference"]["sentence_mean_normalized"]["std_ratio"]:.1%}，不能仅由整体语速解释。',
        '- 这支持优先诊断 duration 的路线；尚未证明 log-duration MSE 是唯一原因，或 duration 是机械感的主要原因。建议先做总时长匹配的 duration 替换试听实验，不据此直接改训练 loss。', '',
        '## 实验设置', '',
        f'- checkpoint：`{metadata["checkpoint"]}`。',
        f'- SHA-256：`{metadata["checkpoint_sha256"]}`。',
        f'- 样本：训练使用的固定验证子集，中文 Premium 和英文 HiFiTTS 各 256 条，共 {len(rows)} 条；speaker ID 均为 0。没有把训练样本或仅有文本的混合评测集纳入。',
        '- 使用已有数值 token/声调 IDs、原录音、原 Mel 归一化、当前 constrained MAS 和原 duration 计算路径；CPU FP32、eval、无梯度。',
        '- 不运行 flow、声码器、优化器；不修改 checkpoint。逐句检查 MAS 单调性、全帧覆盖和手算 duration loss 与原 forward 相等。',
        '- 本次 512 条样本均未触发 MAS floor/ceiling 可行性重缩放，也没有被 duration 监督掩码排除的有效 token；因此 speech 与 supervised_speech 结果相同。',
        '- raw = exp(logw)；ceil = 向上取整；inference = 取整加当前声调 token 时长限幅，length_scale=1。',
        f'- 额外推理补丁开关：`{metadata["duration_patches"]}`。',
        '- speech 子集排除静音、空白、未知、词分隔符、标点和独立声调符号；supervised_speech 进一步排除 duration loss 屏蔽的异常 MAS 目标。',
        '- Std 使用总体标准差（ddof=0），单位为 Mel 帧；每帧 16 ms。CV=Std/Mean。',
        '- 按句统计的中位数置信区间采用句子有放回 bootstrap 2,000 次，仅反映该样本集的抽样不确定性，不是跨说话人/章节独立性保证。', '',
        '## 全 token 统计（对应直接展平验证集 token 的比较）', '',
        '| 语言 | 预测形式 | token 数 | Corr | MAS Std | 预测 Std | Std 比 | MAS CV | 预测 CV | CV 比 | 预测/目标均值 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    def table(subset):
        lines=[]
        for lang,label in [('zh','中文'),('en','英文')]:
            for variant in ('raw','inference'):
                m=summary[lang][subset][variant]
                lines.append(f'| {label} | {variant} | {m["n"]} | {m["pearson"]:.3f} | {m["mas_std"]:.3f} | {m["predicted_std"]:.3f} | {m["std_ratio"]:.3f} | {m["mas_cv"]:.3f} | {m["predicted_cv"]:.3f} | {m["cv_ratio"]:.3f} | {m["mean_ratio"]:.3f} |')
        return lines
    content += table('all_tokens')
    for subset,title in [('speech','排除静音、标点、声调等符号'),('supervised_speech','仅保留实际参与 duration 监督的语音 token')]:
        content += ['',f'## {title}','','| 语言 | 预测形式 | token 数 | Corr | MAS Std | 预测 Std | Std 比 | MAS CV | 预测 CV | CV 比 | 预测/目标均值 |',
                    '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']+table(subset)
    content += ['', '## 句内节奏与语速控制', '',
        '以下均使用 speech 子集。逐句 Std/CV 比再取中位数，避免长句独占结果。均值归一化是分别把每句的 MAS 和预测时长除以各自句均值，再合并统计；它去掉了整体语速尺度差异。', '',
        '| 语言 | 形式 | 句内 Corr 中位数 | 句内 Std 比中位数 [95% CI] | 句内 CV 比中位数 [95% CI] | CV 比<1 的句子占比 | 均值归一化后 Std 比 |',
        '|---|---|---:|---|---|---:|---:|']
    for lang,label in [('zh','中文'),('en','英文')]:
        for variant in ('raw','inference'):
            m=summary[lang]['speech'][variant];s=m['per_sentence'];std=s['std_ratio'];cv=s['cv_ratio']
            fmt=lambda d:f'{d["median"]:.3f} [{d["median_bootstrap_ci95"][0]:.3f}, {d["median_bootstrap_ci95"][1]:.3f}]'
            content.append(f'| {label} | {variant} | {s["pearson"]["median"]:.3f} | {fmt(std)} | {fmt(cv)} | {cv["fraction_below_one"]:.1%} | {m["sentence_mean_normalized"]["std_ratio"]:.3f} |')
    content += ['', '## 去掉音素身份的均值差异', '',
        '对出现至少 10 次的 token ID，分别扣除该 token 的 MAS 平均时长和预测平均时长，再比较残差。此项检查同一音素在不同语境中的变化是否被压缩，不把“不同音素本来时长不同”误认为节奏变化。仍未控制所有语境、重音和真实说话人因素。', '',
        '| 语言 | 形式 | 去 token 均值后 Std 比 | 残差 Corr |', '|---|---|---:|---:|']
    for lang,label in [('zh','中文'),('en','英文')]:
        for variant in ('raw','inference'):
            m=summary[lang]['speech'][variant]['within_token_centered']
            content.append(f'| {label} | {variant} | {m["std_ratio"]:.3f} | {m["pearson"]:.3f} |')
    fig, axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    examples=[]
    for row_index,lang in enumerate(('zh','en')):
        selected=[r for r in rows if r['language']==lang]
        t,p,stats=[],[],[]
        for r in selected:
            mask=np.array(r['speech']);a=np.array(r['mas'])[mask];b=np.array(r['inference'])[mask]
            if len(a)<2:continue
            t.extend(a);p.extend(b);stats.append((r,pair_metrics(a,b)))
        t,p=np.array(t),np.array(p)
        ax=axes[row_index,0]
        ax.hexbin(t,p,gridsize=45,mincnt=1,bins='log',cmap='Blues')
        maximum=max(t.max(),p.max());ax.plot([0,maximum],[0,maximum],'--',color='orange',lw=1)
        ax.set(xlabel='MAS duration (frames)',ylabel='Inferred duration (frames)',title=f'{lang.upper()}: speech tokens')
        ax=axes[row_index,1];a=[m['mas_std'] for r,m in stats];b=[m['predicted_std'] for r,m in stats]
        ax.scatter(a,b,s=12,alpha=.5);maximum=max(max(a),max(b));ax.plot([0,maximum],[0,maximum],'--',color='orange')
        ax.set(xlabel='Sentence MAS std',ylabel='Sentence inferred std',title='Within-sentence variation')
        ax=axes[row_index,2];ratios=[m['cv_ratio'] for r,m in stats if m['cv_ratio'] is not None]
        ax.hist(ratios,bins=30,color='#2878a4');ax.axvline(1,color='orange',ls='--');ax.axvline(np.median(ratios),color='black')
        ax.set(xlabel='Inferred CV / MAS CV',ylabel='Sentences',title=f'Median CV ratio = {np.median(ratios):.3f}')
        usable=[(r,m) for r,m in stats if 20<=m['n']<=80 and m['cv_ratio'] is not None]
        examples.append(min(usable,key=lambda item:abs(item[1]['cv_ratio']-np.median(ratios)))[0])
    fig.savefig(root/'duration_distribution.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(14,7),constrained_layout=True)
    for ax,r in zip(axes,examples):
        mask=np.array(r['speech']);a=np.array(r['mas'])[mask];b=np.array(r['raw'])[mask];c=np.array(r['inference'])[mask]
        ax.plot(a,label='MAS',lw=1.8);ax.plot(b,label='Raw prediction',lw=1.5);ax.plot(c,label='Inference',lw=1.3,ls='--')
        ax.set(title=f'{r["language"].upper()} sample {r["sample_id"]}: near median CV ratio',xlabel='Speech-token index (non-speech tokens omitted)',ylabel='Duration (frames)');ax.legend()
    fig.savefig(root/'duration_examples.png',dpi=160);plt.close(fig)
    content += ['', '## 可视化', '', '![分布及句内变化](duration_distribution.png)', '',
        '下例从 20–80 个 speech token 的句子中，选择 CV 比最接近该语言中位数的句子；不是手选最差样本。横轴为排除特殊符号后的 token 顺序。', '',
        '![典型句子的时长序列](duration_examples.png)', '']
    for r in examples:content += [f'- {r["language"]} / sample {r["sample_id"]}：{r["text"]}']
    content += ['', '## 解释边界', '',
        '- Std 比明显低于 1 表示相对于 MAS 的绝对变化幅度缩小；CV 比低于 1 且均值归一化后仍缩小，才更支持节奏形状被压平，而不仅是整体说得更快。',
        '- Corr 衡量 token 长短排序的一致性；高相关并不排除幅度缩小，低相关也不单独等于 oversmoothing。',
        '- 统计分布压缩不等于已证明 MSE 是唯一原因。MAS 本身的噪声、单个 speaker ID 覆盖的真实音色/语速差异、不可由文本预测的韵律，以及训练数据分布都可能参与。',
        '- 本次没有生成对照音频或主观试听，因此不能把结果直接等同于“机械感主要来自 duration”。',
        '- 不使用简单放大所有 duration 方差作为修复结论：它可能放大预测错误、破坏总时长或停顿。先做相同文本/噪声/flow 设置、相同总时长下的预测 duration 与 MAS duration 听感对照，再决定目标函数或建模修改。', '',
        '## 原始结果与复现', '',
        '- [summary.json](summary.json)：全量统计，包括 ceil 中间形式。',
        '- [samples.jsonl](samples.jsonl)：每条样本的 token、MAS、raw/ceil/inference 时长、监督掩码。',
        '- [metadata.json](metadata.json)：checkpoint 身份、设置和执行时间。', '',
        '诊断脚本：`LITs/training/foundation/diagnose_duration_smoothing.py`；报告脚本：`LITs/training/foundation/report_duration_smoothing.py`。', '']
    (root/'REPORT.md').write_text('\n'.join(content))
    print(root/'REPORT.md')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
