"""Report paired Chinese duration failure audit without inferring human ratings."""
import argparse
import base64
from collections import defaultdict
import html
import json
from pathlib import Path


LABELS={
    'original':'原录音','reconstruction':'真实 Mel 经 Vocos 重建',
    'predicted_matched_10':'预测 duration，匹配总长（A）','mas_10':'原 MAS duration（B）',
    'mas_30':'MAS：增加到 30 步 Euler','tone_only_mas_10':'MAS：只保留声调约束',
    'free_mas_10':'MAS：取消显式时长约束','blend50_10':'预测/MAS duration 各混合 50%',
    'mas_syllable_pred_internal_10':'保留 MAS 音节总长，改用预测的音节内部比例',
    'mas_body_pred_special_10':'采用预测的停顿/声调时长，保留 MAS 语音 token 比例',
}


def main(root):
    rows=[json.loads(l) for l in (root/'asr.jsonl').read_text().splitlines()]
    assert len(rows)==len({r['id'] for r in rows})==30
    details=json.loads((root/'details.json').read_text());by_id={str(d['sample_id']):d for d in details}
    groups=defaultdict(list);cases=defaultdict(dict)
    for r in rows:groups[r['mode']].append(r);cases[r['sample']][r['mode']]=r
    summary={}
    for mode in LABELS:
        data=groups[mode];assert len(data)==3
        chars=sum(r['reference_characters'] for r in data)
        edits=sum(round(r['cer']*r['reference_characters']) for r in data)
        summary[mode]=dict(samples=3,reference_characters=chars,edits=edits,cer=edits/chars)
    (root/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    text=['# 197k 中文 MAS 版本可懂度下降：核查与消融','',
        '## 结论','',
        '已排除本次 A/B 的时长展开、噪声复用及生成复现错误。三条中文中，前两条的严重内容错误可以被 ASR 复现；第三条主要版本均识别正确，因此不能把全部样本等同为同一种失败。', '',
        '主要证据指向 MAS 对齐/条件时序与生成质量不一致，而非“只要恢复时长方差就更自然”。MAS 在前两条把部分语音 token 挤到 2 帧，将另一些拉长到 27–34 帧；它的 prior MSE 更小，但生成可懂度更差。取消显式时长约束或仅调整音节内部比例没有解决问题。', '',
        '这尚不能唯一断言“MAS 路径一定错误”或“flow 本身正常”。MAS 来自模型的声学均值匹配，不是人工边界；正确音素边界仍需独立强制对齐或人工检查，MAS 与 flow 的适配问题也不能完全分离。', '',
        '## 已排除的实现问题','',
        '- 用同一 checkpoint 和原音重新计算 MAS：与上一轮保存的逐 token duration、原 model.forward 的对齐逐元素一致。',
        '- `repeat_interleave` 展开与原 `generate_path + bmm` 展开逐元素相等。',
        '- 重新生成原 A/B 的 Mel：最大误差均为 0。WAV 与原文件的差异不超过 PCM16 一个量化单位。',
        '- 每组总帧数相同，噪声哈希相同，speaker、Mel 归一化、声码器、非流式设置相同。',
        '- 此次只运行独立诊断，没有优化器更新，也没有改动后台合成或 48,000 步训练配置。','',
        '## 内容错误率对照','',
        '同一个本地 Qwen3-ASR-1.7B，BF16、SDPA、batch size 1、自动语言识别、不提供参考文本提示。CER 仅用于辅助衡量内容可识别性，不等价于人工可懂度或自然度。合并值按参考字符数加权；样本很小，不能外推整体质量。', '',
        '| 设置 | 中文 CER | 编辑数 / 参考字符 |','|---|---:|---:|']
    for mode,label in LABELS.items():
        s=summary[mode];text.append(f'| {label} | {s["cer"]:.1%} | {s["edits"]}/{s["reference_characters"]} |')
    text += ['', '原音与重建的错误完全一致：第二条 ASR 少识别句末“啊”。这三条里没有发现额外的声码器内容丢失证据。增加 Euler 步数没有稳定改善 MAS 版本；目前不支持“只是 10 步不够”的解释。', '',
        '预测/MAS 混合 50% 的结果也没有超过预测基线，不能据此部署 duration 插值修复。', '',
        '## MAS 的极端帧分配与 prior 误差','',
        '| 样本 | MAS 语音 token ≤2 帧比例 | 预测对应比例 | MAS 最大语音 token 时长 | 预测 prior MSE | MAS prior MSE | 自由 MAS prior MSE |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for d in details:
        v=d['variants'];a=v['predicted_matched'];b=v['mas'];f=v['free_mas']
        text.append(f'| {d["sample_id"]} | {b["speech_one_or_two_frame_fraction"]:.1%} | {a["speech_one_or_two_frame_fraction"]:.1%} | {b["max_speech_frames"]} 帧（{b["max_speech_frames"]*16} ms） | {a["prior_mse"]:.3f} | {b["prior_mse"]:.3f} | {f["prior_mse"]:.3f} |')
    text += ['', '例如第二句“是否可以说尚大夫……”：MAS 给“否”的韵母 ㄡ 27 帧（432 ms），却将后面“大夫”处多个声母/韵母分到每个 2 帧（32 ms）。仅凭时长不能证明边界错误，但结合合成结果，不能把这类 MAS 分布当作可靠韵律真值。', '',
        '自由 MAS 的 prior MSE 进一步下降，但 CER 没有改善：当前高斯 Mel 均值匹配目标与可懂度存在明显差距。它只能证明路径更符合该模型的打分，不能证明更符合真实音素边界。', '',
        '一个实现细节：当前 MAS 的声调 token 实际 floor 为 2 帧、ceiling 为 3 帧。虽然 `tone_floor_frames=1`，`_mas_floor_frames` 先应用全局 `duration_floor_min_frames=2` 再取最大值，实际下限变成 2；推理限幅仍为 1–3。这是已有规则的作用，本次没有修改，也没有证据表明单改它就能解决问题。', '',
        '## 每句结果','']
    for sid,items in cases.items():
        text += [f'### {sid}：{items["original"]["ref_text"]}','','| 版本 | CER | ASR 转写 |','|---|---:|---|']
        for mode in LABELS:
            r=items[mode];text.append(f'| {LABELS[mode]} | {r["cer"]:.1%} | {r["asr_text"].replace("|","/")} |')
        text += ['']
    text += ['## 后续判断顺序','',
        '1. 在出错的前两句上用独立音素强制对齐或人工音节边界验证 MAS，尤其检查被压到 32 ms 的音素、长韵母和停顿附近。当前没有独立音素边界真值。',
        '2. 若独立边界证实 MAS 错位，优先修复对齐目标/约束或改用可靠的 duration 监督，再研究是否需要分布/韵律损失。',
        '3. 若独立边界支持 MAS，继续检查 flow 对该时序条件的生成适配，以及训练到推理的差异。原音重建正常只能缩小范围，不能证明 flow 已被排除。',
        '4. 保留原始预测作为基线，不直接放大方差、移除全部 MAS 约束或采用 50% duration 插值作为正式修复。', '',
        '## 产物','',
        '- [试听与逐 token 表](listen.html)：30 条音频及 ASR 结果；HTML 内嵌音频，可以离线打开。',
        '- [summary.json](summary.json)：聚合 CER。',
        '- [asr.jsonl](asr.jsonl)：完整识别输出。',
        '- [details.json](details.json)：时长、边界、prior MSE 与实现核对。',
        '- [metadata.json](metadata.json)：checkpoint 与运行设置。', '']
    (root/'REPORT.md').write_text('\n'.join(text))
    def audio(path):return '<audio controls preload="none" src="data:audio/wav;base64,'+base64.b64encode(Path(path).read_bytes()).decode()+'"></audio>'
    page=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>197k 中文 Duration 失败核查</title>',
        '<style>body{max-width:1180px;margin:25px auto;padding:0 18px;font:16px/1.6 system-ui;color:#203044}table{width:100%;border-collapse:collapse}td,th{padding:9px;border-bottom:1px solid #d8e2eb;text-align:left}audio{width:280px;max-width:100%}section{margin:30px 0}details{margin-top:16px}summary{cursor:pointer}h1{font-size:27px}.scroll{overflow:auto}</style>',
        '<h1>197k 中文 Duration 失败核查</h1><p>固定同一噪声、模型与总时长，仅改变时序分配或指定的 Euler 步数。识别错误率用于辅助定位；自然度仍需试听判断。</p>',
        '<p>复核结果：原 A/B 已精确复现；原音与真实 Mel 重建的识别结果一致。前两句 MAS 版本内容错误明显，第三句各主要版本均能正确识别。</p>']
    for sid,items in cases.items():
        page += [f'<section><h2>{html.escape(items["original"]["ref_text"])}</h2><div class="scroll"><table><tr><th>设置</th><th>音频</th><th>CER</th><th>ASR</th></tr>']
        for mode,label in LABELS.items():
            r=items[mode];page.append(f'<tr><td>{label}</td><td>{audio(r["audio"])}</td><td>{r["cer"]:.1%}</td><td>{html.escape(r["asr_text"])}</td></tr>')
        d=by_id[sid];page += ['</table></div><details><summary>逐 token 时长与 MAS 边界（帧，每帧 16 ms）</summary><table><tr><th>序号</th><th>符号</th><th>预测</th><th>MAS</th><th>floor</th><th>ceiling（0=不设）</th></tr>']
        for i,symbol in enumerate(d['symbols']):
            a=d['variants']['predicted_matched']['durations'][i];b=d['variants']['mas']['durations'][i]
            page.append(f'<tr><td>{i}</td><td>{html.escape(symbol)}</td><td>{a}</td><td>{b}</td><td>{d["floors"][i]:.2f}</td><td>{d["ceilings"][i]:.2f}</td></tr>')
        page += ['</table></details></section>']
    page += ['<script>document.querySelectorAll("audio").forEach(a=>a.addEventListener("play",()=>document.querySelectorAll("audio").forEach(b=>{if(a!==b)b.pause();})));</script></html>']
    (root/'listen.html').write_text('\n'.join(page))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
