"""Build a local listening page and figures for paired generation controls."""
import argparse
from collections import defaultdict
import html
import json
from pathlib import Path
import re
import unicodedata

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('output', type=Path)
a = p.parse_args()
root = a.output
rows = [json.loads(l) for l in (root/'asr.jsonl').read_text().splitlines()]
meta = json.loads((root/'metadata.json').read_text())
modes = ['original','reconstruction','mas_duration','predicted_duration']
labels = {'original':'原录音（24 kHz）', 'reconstruction':'真实 Mel → Vocos',
          'mas_duration':'模型生成：MAS 时长', 'predicted_duration':'模型生成：预测时长'}
groups = defaultdict(list)
samples = defaultdict(list)
for row in rows:
    assert Path(row['audio']).exists()
    groups[(row['source'],row['mode'])].append(row)
    samples[row['sample']].append(row)

summary = []
for (source,mode), items in groups.items():
    nchar = sum(v['reference_characters'] for v in items)
    edits = sum(round(v['cer']*v['reference_characters']) for v in items)
    words = lambda text: re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)*",unicodedata.normalize('NFKC',text).lower().replace('’',"'"))
    nwords = sum(len(words(v['ref_text'])) for v in items)
    word_edits = sum(round(v['wer']*len(words(v['ref_text']))) for v in items)
    summary.append(dict(source=source,mode=mode,count=len(items),cer=edits/max(1,nchar),
        wer=word_edits/nwords if nwords else None,
        average_duration=sum(v['duration'] for v in items)/len(items)))
(root/'summary.json').write_text(json.dumps(dict(metadata=meta,groups=summary),ensure_ascii=False,indent=2)+'\n')

cards = []
for sample, items in samples.items():
    bymode = {v['mode']:v for v in items}
    assert set(bymode) == set(modes)
    diagnostic = json.loads((root/sample/'diagnostic.json').read_text())
    arrays = np.load(root/sample/'tensors.npz')
    fig, axes = plt.subplots(4,1,figsize=(12,9),constrained_layout=True)
    for ax,key,title in zip(axes[:3],['real_mel','mas_mel','predicted_mel'],
                           ['Reference Mel','Generated: MAS durations, 10 steps','Generated: predicted durations, 10 steps']):
        mel = arrays[key][0]
        ax.imshow(mel,origin='lower',aspect='auto',extent=[0,mel.shape[-1]*0.016,0,100],vmin=-11.5,vmax=2)
        ax.set(title=title,ylabel='Mel bin',xlabel='Time (s)')
    for key,label in [('mas_duration_frames','MAS'),('predicted_duration_frames','Predicted')]:
        durations = np.array(diagnostic[key])
        axes[3].plot(np.arange(len(durations)+1),np.r_[0,durations.cumsum()]*0.016,label=label)
    axes[3].set(xlabel='Token boundary index',ylabel='Cumulative duration (s)',title='Token timing (MAS is model-derived)')
    axes[3].legend()
    fig.savefig(root/sample/'comparison.png',dpi=120)
    plt.close(fig)
    audio_cards = []
    for mode in modes:
        v = bymode[mode]
        metric = f"WER {v['wer']:.1%}" if v['source']=='HiFiTTS' else f"CER {v['cer']:.1%}"
        audio_cards.append(f'<div class="audio"><b>{labels[mode]}</b><p>{v["duration"]:.2f}s · {metric}</p>'
            f'<audio controls preload="none" src="{sample}/{mode}.wav"></audio>'
            f'<p class="transcript">ASR：{html.escape(v["asr_text"])}</p></div>')
    cards.append(f'<section><h2>{sample}</h2><p class="reference">{html.escape(items[0]["ref_text"])}</p>'
        f'<div class="grid">{"".join(audio_cards)}</div>'
        f'<p>预测/原音频时长比：{diagnostic["predicted_to_real_duration"]:.3f}；'
        f'归一化往返最大误差：{diagnostic["normalization_roundtrip_max_error"]:.2e}。</p>'
        f'<details><summary>音素、时长和频谱</summary><p>{html.escape(diagnostic["phonemes"])}</p>'
        f'<img src="{sample}/comparison.png" alt="Mel and token duration comparison">'
        f'<p><a href="{sample}/diagnostic.json">逐 token 时长与检查结果</a></p></details></section>')

table = []
for source in ['HiFiTTS','Premium']:
    for mode in modes:
        s = next(v for v in summary if v['source']==source and v['mode']==mode)
        metric = s['wer'] if source=='HiFiTTS' else s['cer']
        table.append(f'<tr><td>{"英文" if source=="HiFiTTS" else "中文"}</td><td>{labels[mode]}</td><td>{metric:.2%}</td><td>{s["count"]}</td></tr>')
page = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>25k 生成链路对照</title><style>
body{{font:16px/1.6 system-ui,sans-serif;margin:auto;padding:24px;max-width:1280px;background:#f3f5f7;color:#17212b}}
h1{{font-size:27px}}section{{background:white;padding:22px;margin:24px 0;border-radius:12px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}}
.audio{{padding:15px;background:#eef3f7;border-radius:8px}}audio{{width:100%}}.transcript{{font-size:14px}}
.reference{{font-size:19px}}img{{max-width:100%}}table{{border-collapse:collapse;background:white}}td,th{{padding:8px 18px;border:1px solid #ddd}}
</style><h1>25,000 步：生成链路对照试听</h1>
<p>8 条固定抽样：英文、中文各 4 条，每种语言含训练集 2 条及验证集 2 条。先听原录音和真实 Mel 重建，再比较两种模型生成。</p>
<p>模型生成均为非流式、10 步、temperature=1.0，并共享对应帧的初始噪声。MAS 时长由模型根据真实录音推导，不能视为人工标注的正确时长。</p>
<p>ASR 使用 Qwen3-ASR-1.7B、自动语言识别、无文本提示；英文汇总 WER，中文汇总 CER。8 条样本仅用于配对诊断，ASR 不代替试听。</p>
<table><tr><th>语言</th><th>对照</th><th>错误率</th><th>条数</th></tr>{''.join(table)}</table>
{''.join(cards)}<p><a href="summary.json">汇总 JSON</a> · <a href="metadata.json">实验设置</a> · <a href="asr.jsonl">逐条 ASR 结果</a></p></html>'''
(root/'index.html').write_text(page)
print(json.dumps(summary,ensure_ascii=False,indent=2))
print('Report:',root/'index.html')
