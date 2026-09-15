"""Audition gallery and evidence bounds for Chinese accent investigation."""
import argparse
from collections import defaultdict
import html
import json
from pathlib import Path


def main(run):
    import numpy as np
    import soundfile as sf
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=run/'diagnostics/accent_2250'
    meta=json.loads((out/'metadata.json').read_text())
    rows=[json.loads(s) for s in (out/'asr.jsonl').read_text().splitlines()]
    assert len(rows)==len({r['id'] for r in rows})==meta['audio_count']==56
    cases=defaultdict(dict)
    groups=defaultdict(list)
    for r in rows:
        audio,sr=sf.read(r['audio'])
        assert sr==24000 and audio.ndim==1 and np.isfinite(audio).all()
        assert abs(len(audio)/sr-r['duration'])<1e-8
        cases[r['sample']][r['mode']]=r
        groups[(r['group'],r['mode'])].append(r)
    scores={}
    for (group,mode),items in groups.items():
        n=sum(r['reference_characters'] for r in items)
        e=sum(round(r['cer']*r['reference_characters']) for r in items)
        scores.setdefault(group,{})[mode]=dict(cer=e/n,edits=e,characters=n,samples=len(items))
    baseline=json.loads((run/'baseline/summary.json').read_text())
    current=json.loads((run/'eval/step_00002250/summary.json').read_text())
    formal=[json.loads(s) for s in (run/'eval/step_00002250/details.jsonl').read_text().splitlines()]
    assert all(r['speaker']==(0 if r['group']=='ljs_en' else 1) for r in formal)
    result=dict(source_inventory=dict(native_chinese=5,native_mixed=5,mixed_training_rows=0),
        training_language_row_share=dict(English=14423/23100,Chinese=8677/23100),
        training_weighted_hours=dict(English=24.175914282407756,Chinese=9.088444444444281),
        baseline_eval=baseline['groups'],current_eval=current['groups'],probe_scores=scores,
        verification=dict(audio_count=56,all_audio_finite=True,sample_rate=24000,diagnostic_speaker_id=1,
            formal_eval_speaker_mapping_correct=True),limitations=meta['limitations'])
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    labels=dict(reference='原始录音 / 合成训练数据的保留音频',initial='第一阶段初值',current='第二阶段 2250 步',
        restore_prior='仅回退文本/先验编码器',restore_duration='仅回退时长网络',restore_speaker='仅回退 speaker 向量',restore_decoder='仅回退解码器')
    details=json.loads((out/'details.json').read_text())
    esc=html.escape
    page=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>中文口音同句试听对照</title>',
        '<style>body{max-width:1400px;margin:28px auto;font:16px/1.5 system-ui;color:#20252b}table{border-collapse:collapse;width:100%}td,th{padding:8px;border:1px solid #ccc;vertical-align:top}audio{width:300px}img{max-width:100%}section{margin:32px 0}details{margin:16px 0}.note{background:#f3f5f8;padding:16px}</style>',
        '<h1>中文口音：同句试听对照</h1><div class="note">全部模型音频明确使用 speaker 1。固定文本、噪声、10 步生成、非流式设置和声码器。先听“原始录音 → 第一阶段 → 第二阶段”，再展开模块回退。自动 ASR 只核对内容，不能判断是否像外国口音；本报告不声称已做人工听感评测。</div>',
        '<h2>已核实的数据与结构</h2><ul><li>原始 TS-004 目录实际有 5 条中文和 5 条混读录音。目录名中的范围不是实际文件数量。当前训练包含纯中文和纯英文，句内混读训练行数为 0。</li>',
        '<li>原混合参考试验使用混合录音作为提示去生成纯英文；候选未通过筛选，不等于这 5 条混读原录音不合格。</li>',
        '<li>英文占训练清单行数约 62.4%、按采样清单累计的音频时长约 72.7%。这表明语言曝光不平衡，不能直接据此计算梯度贡献或认定它造成口音。</li>',
        '<li>speaker 向量同时参与文本编码/时长路径和解码器；它没有被结构上限制为只改变音色。共享网络的跨语言适配值得排查，尚未证明是根因。</li></ul>',
        '<p>以下模块回退只在独立诊断模型中进行。它会改变模块间的配合，不能当作重新训练实验或直接替换正式模型的依据。当前训练按用户要求继续。</p>',
        '<h2>已有完整评估：内容与音色</h2><table><tr><th>指标</th><th>初值</th><th>2250 步</th></tr>']
    for label,group,key in [('中文 CER','majestic_zh','cer_micro'),('混读 CER','majestic_mixed','cer_micro'),('中文 WavLM 相似度','majestic_zh','wavlm_similarity_mean'),('中文 CAM++ 相似度','majestic_zh','camp_similarity_mean')]:
        a=baseline['groups'][group][key];b=current['groups'][group][key]
        page.append(f'<tr><td>{label}</td><td>{a:.2%}</td><td>{b:.2%}</td></tr>' if key.endswith('micro') else f'<tr><td>{label}</td><td>{a:.3f}</td><td>{b:.3f}</td></tr>')
    page.append('</table><p>识别和音色相似度提升，与用户报告的韵律不自然可以同时存在。</p>')
    for sample,c in cases.items():
        row=c['reference']
        source='原始 TS-004 录音' if sample.startswith('native') else '未用于训练的 VoxCPM2 合成保留音频'
        page.append(f'<section><h2>{sample} · {source}</h2><p>{esc(row["ref_text"])}</p><table><tr><th>版本</th><th>音频</th><th>时长</th><th>ASR（内容核验）</th></tr>')
        def audio_row(mode):
            r=c[mode];link=f'{sample}/{mode}.wav'
            return f'<tr><td>{labels[mode]}</td><td><audio controls preload="none" src="{link}"></audio></td><td>{r["duration"]:.2f}s</td><td>{esc(r["asr_text"])}</td></tr>'
        for mode in ['reference','initial','current']:page.append(audio_row(mode))
        page.append('</table><details><summary>展开四种模块回退试听与时长图</summary><table>')
        for mode in ['restore_prior','restore_duration','restore_speaker','restore_decoder']:page.append(audio_row(mode))
        fig,ax=plt.subplots(figsize=(11,3),layout='constrained')
        for mode in meta['modes']:
            values=details[sample]['modes'][mode]['durations']
            ax.plot(np.cumsum(values)*.016,label=mode,alpha=.8)
        ax.set(xlabel='Token index',ylabel='Predicted token end time (s)',title=sample)
        ax.legend(ncol=3,fontsize=8)
        fig.savefig(out/sample/'timing.png',dpi=120);plt.close(fig)
        page.append(f'</table><img src="{sample}/timing.png"></details></section>')
    page.append('</html>')
    (out/'index.html').write_text('\n'.join(page))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,required=True)
    main(parser.parse_args().run_dir)
