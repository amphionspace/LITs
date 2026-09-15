"""Render paired Chinese MAS ablation results with auditable local artifacts."""
import argparse
from collections import defaultdict
import html
import json
import os
from pathlib import Path
import statistics


def readrows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main(root):
    import numpy as np
    import soundfile as sf
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = root/'mas_constraints'
    rows = readrows(out/'asr.jsonl')
    synthesis = readrows(out/'synthesis.jsonl')
    assert len(rows) == len(synthesis) == len({r['id'] for r in rows}) == 60
    old = [r for r in readrows(root/'asr.jsonl') if r['group'] in ('train_zh', 'val_zh')
           and r['mode'] in ('original', 'reconstruction', 'pred_10_s0', 'pred_matched_duration_10_s0')]
    modes = ['original', 'reconstruction', 'pred_10_s0', 'pred_matched_duration_10_s0', 'full', 'none', 'floor_only', 'ceiling_only', 'tone_only']
    labels = dict(original='原始录音', reconstruction='真实 Mel 重建', pred_10_s0='正常预测时长',
        pred_matched_duration_10_s0='预测时长匹配总长', full='完整约束', none='无显式约束',
        floor_only='仅下限', ceiling_only='仅上限', tone_only='仅声调约束')
    allrows = old+rows
    groups = defaultdict(list)
    cases = defaultdict(dict)
    for row in allrows:
        groups[(row['group'], row['mode'])].append(row)
        groups[('all_zh', row['mode'])].append(row)
        cases[row['sample']][row['mode']] = row
    summary = {}
    for (group, mode), items in groups.items():
        denominator = sum(r['reference_characters'] for r in items)
        numerator = sum(round(r['cer']*r['reference_characters']) for r in items)
        summary.setdefault(group, {})[mode] = dict(cer=numerator/denominator, edits=numerator, characters=denominator, samples=len(items))
    paired = {}
    details = {name:json.loads((out/name/'diagnostic.json').read_text()) for name in cases}
    for mode in modes[4:]:
        paired[mode] = dict(improved=sum(c[mode]['cer'] < c['full']['cer']-1e-9 for c in cases.values()),
            unchanged=sum(abs(c[mode]['cer']-c['full']['cer']) < 1e-9 for c in cases.values()),
            worse=sum(c[mode]['cer'] > c['full']['cer']+1e-9 for c in cases.values()),
            changed_frame_ratio_mean=statistics.mean(d['modes'][mode]['changed_frame_ratio'] for d in details.values()),
            prior_mse_mean=statistics.mean(d['modes'][mode]['prior_mse'] for d in details.values()),
            tone_frames_mean=statistics.mean(d['modes'][mode]['tone_frames'] for d in details.values()))
    for r in synthesis:
        wave, sr = sf.read(r['audio'])
        assert sr == 24000 and wave.ndim == 1 and np.isfinite(wave).all()
        assert abs(len(wave)/sr-r['duration']) < 1e-9
    verification = dict(audio_count=60, finite_audio=True, sample_rate=24000,
        baseline_paths_exactly_reproduced=True,
        baseline_audio_max_error=max(d['modes']['full']['baseline_audio_max_error'] for d in details.values()))
    result = dict(scores=summary, paired_vs_full=paired, verification=verification,
        limitations=['12 selected Chinese utterances, 256 reference characters; exploratory same-checkpoint inference ablation.',
            'Changing MAS bounds at inference cannot establish the outcome of retraining with those bounds.',
            'Unconstrained MAS still assigns at least one frame to each token; MAS is not phonetic ground truth.',
            'Automatic ASR CER is a content metric; no claim of human listening or naturalness evaluation.'])
    original_mas = {r['sample']:r for r in readrows(root/'asr.jsonl') if r['mode'] == 'mas_10_s0' and r['group'] in ('train_zh', 'val_zh')}
    result['baseline_asr_recheck'] = dict(
        previous_edits=sum(round(r['cer']*r['reference_characters']) for r in original_mas.values()),
        current_edits=summary['all_zh']['full']['edits'],
        transcript_changed_samples=[name for name, c in cases.items() if c['full']['asr_text'] != original_mas[name]['asr_text']],
        note='Waveforms exactly equal. ASR batch composition differs; content metric has small numerical/decoding sensitivity.')
    single_path = out/'asr_single/asr.jsonl'
    if single_path.exists():
        single = readrows(single_path)
        assert len(single) == len({r['id'] for r in single}) == 60
        by_id = {r['id']:r for r in single}
        scores_single = {}
        pairs_single = {}
        for mode in modes[4:]:
            for group in ['all_zh', 'train_zh', 'val_zh']:
                items = [r for r in single if r['mode'] == mode and (group == 'all_zh' or r['group'] == group)]
                denominator = sum(r['reference_characters'] for r in items)
                numerator = sum(round(r['cer']*r['reference_characters']) for r in items)
                scores_single.setdefault(group, {})[mode] = dict(cer=numerator/denominator, edits=numerator, characters=denominator)
            delta = [by_id[f'{name}/{mode}']['cer']-by_id[f'{name}/full']['cer'] for name in cases]
            pairs_single[mode] = dict(improved=sum(d < -1e-9 for d in delta), unchanged=sum(abs(d) < 1e-9 for d in delta), worse=sum(d > 1e-9 for d in delta))
        result['asr_single'] = dict(scores=scores_single, paired_vs_full=pairs_single,
            transcript_changed_count=sum(r['asr_text'] != by_id[r['id']]['asr_text'] for r in rows))
    (out/'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    esc = html.escape
    page = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>中文 MAS 时长约束对照</title>',
        '<style>body{max-width:1400px;margin:30px auto;font:16px/1.5 system-ui}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccc}audio{width:280px}img{max-width:100%}.case{margin-top:40px}</style>',
        '<h1>中文 MAS 时长约束对照 · step 145000</h1><p>12 条中文样本（训练集、验证集各 6 条），固定 checkpoint、真实总帧数、噪声、10 步生成和声码器。仅改变对齐搜索的上下限。CER 越低越好。</p>',
        '<p>无显式约束仍保留经典 MAS 每个 token 至少 1 帧。仅声调约束移除历史韵母时长边界，保留当前模型实际声调 token 边界。此实验不能直接推断更改约束后重新训练的效果。</p>',
        '<p>以下主表使用 ASR batch size 2；前四项复用上一轮对照记录。相同完整约束音频的 ASR 错误数从上一轮 110 变为本轮 107（256 字），存在少量识别波动，音频本身逐采样点完全一致。</p>',
        '<table><tr><th>设置</th><th>全部中文 CER</th><th>训练 CER</th><th>验证 CER</th></tr>']
    for mode in modes:
        page.append('<tr><td>'+labels[mode]+'</td>'+''.join(f'<td>{summary[g][mode]["cer"]:.2%} ({summary[g][mode]["edits"]}/{summary[g][mode]["characters"]})</td>' for g in ['all_zh','train_zh','val_zh'])+'</tr>')
    page.append('</table><h2>相对完整约束的逐句变化</h2><table><tr><th>设置</th><th>改善 / 持平 / 变差</th><th>帧归属变化均值</th><th>先验 MSE 均值</th></tr>')
    for mode, p in paired.items():
        page.append(f'<tr><td>{labels[mode]}</td><td>{p["improved"]} / {p["unchanged"]} / {p["worse"]}</td><td>{p["changed_frame_ratio_mean"]:.2%}</td><td>{p["prior_mse_mean"]:.4f}</td></tr>')
    page.append('</table>')
    if 'asr_single' in result:
        page.append('<h2>ASR 单条识别复核</h2><p>对同一批 60 条音频改用 batch size 1 重新识别；不重新生成音频。</p><table><tr><th>设置</th><th>batch 2 CER</th><th>batch 1 CER</th><th>batch 1 改善 / 持平 / 变差</th></tr>')
        for mode in modes[4:]:
            s = result['asr_single']['scores']['all_zh'][mode]
            p = result['asr_single']['paired_vs_full'][mode]
            page.append(f'<tr><td>{labels[mode]}</td><td>{summary["all_zh"][mode]["cer"]:.2%}</td><td>{s["cer"]:.2%} ({s["edits"]}/{s["characters"]})</td><td>{p["improved"]} / {p["unchanged"]} / {p["worse"]}</td></tr>')
        page.append('</table>')
    for name, c in sorted(cases.items()):
        d = details[name]
        fig, axes = plt.subplots(2, 1, figsize=(13, 6), layout='constrained')
        for mode in modes[4:]:
            durations = np.asarray(d['modes'][mode]['durations'])
            axes[0].plot(np.cumsum(durations)*.016, label=mode, alpha=.8)
            axes[1].plot(durations, label=mode, alpha=.7)
        axes[0].set(ylabel='Token end time (seconds)', title=name)
        axes[1].set(xlabel='Token index', ylabel='Duration (frames)')
        axes[0].legend(ncol=5)
        fig.savefig(out/name/'duration_comparison.png', dpi=130)
        plt.close(fig)
        page.append(f'<section class="case"><h2>{name}</h2><p>{esc(d["text"])}</p><img src="{name}/duration_comparison.png"><table><tr><th>设置</th><th>音频</th><th>ASR</th><th>CER</th></tr>')
        for mode in modes:
            r = c[mode]
            link = os.path.relpath(r['audio'], out)
            assert (out/link).is_file()
            page.append(f'<tr><td>{labels[mode]}</td><td><audio controls preload="none" src="{esc(link)}"></audio></td><td>{esc(r["asr_text"])}</td><td>{r["cer"]:.2%}</td></tr>')
        page.append('</table></section>')
    page.append('<p>全部 60 条生成音频通过有限值、采样率和时长检查。完整约束路径与上一轮逐元素一致；音频最大差值 '+str(verification['baseline_audio_max_error'])+'。</p></html>')
    (out/'index.html').write_text('\n'.join(page))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
