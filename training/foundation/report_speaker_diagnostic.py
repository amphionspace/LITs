"""Build an auditable speaker-conditioning diagnostic table and listening page."""
import argparse
from collections import defaultdict
import html
import json
from pathlib import Path
import re
import statistics
import unicodedata


def read_rows(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def main(out):
    meta = json.loads((out/'metadata.json').read_text())
    rows = read_rows(out/'synthesis.jsonl')
    asr = {x['id']:x for x in read_rows(out/'asr.jsonl')}
    speakers = {x['id']:x for x in read_rows(out/'speaker_metrics.jsonl')}
    assert len(rows) == len(asr) == len(speakers) == 64
    groups = defaultdict(list)
    samples = defaultdict(list)
    for r in rows:
        groups[r['mode']].append(r)
        samples[r['sample']].append(r)
    result = {}
    for mode, group in groups.items():
        result[mode] = {}
        for language, source in [('en','HiFiTTS'),('zh','Premium')]:
            selected = [r for r in group if r['source']==source]
            units=[]
            for r in selected:
                ref=unicodedata.normalize('NFKC',r['ref_text']).lower()
                units.append(len(re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)*",ref.replace('’',"'"))) if language=='en'
                             else sum(unicodedata.category(c)[0] in ('L','N') for c in ref))
            errors=[asr[r['id']]['wer' if language=='en' else 'cer']*n for r,n in zip(selected,units)]
            result[mode][language+'_error_micro']=sum(errors)/sum(units)
        for key in ['wavlm_cosine_to_baseline','camp_cosine_to_baseline']:
            result[mode][key]=statistics.mean(speakers[r['id']][key] for r in group)
        for key in ['duration_ratio','encoder_mu_rmse','encoder_logw_rmse','normalized_mel_rmse','mel_relative_l2']:
            values=[r['diagnostics'][key] for r in group if key in r.get('diagnostics',{})]
            if values:
                result[mode][key]=dict(mean=statistics.mean(values),minimum=min(values),maximum=max(values),n=len(values))
    gradients=defaultdict(list)
    for name in samples:
        d=json.loads((out/name/'diagnostic.json').read_text())
        assert d['baseline_repeat_max_error']<1e-5
        for r in d['gradients']:
            gradients[r['loss']+'_'+('streaming' if r['streaming'] else 'full')].append(r)
    gradient_summary={k:dict(mean_row0_norm=statistics.mean(r['row_gradient_norms'][0] for r in v),
                             max_row1_norm=max(r['row_gradient_norms'][1] for r in v),
                             connected_count=sum(r['gradient_connected'] for r in v),n=len(v)) for k,v in gradients.items()}
    summary=dict(checkpoint_step=meta['checkpoint_step'],sample_count=len(samples),audio_count=len(rows),
                 modes=result,gradients=gradient_summary,trajectory=meta['trajectory'],limitations=meta['limitations'])
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    labels={'baseline':'原 speaker 0','scale_0p9':'speaker 0 × 0.9','scale_1p1':'speaker 0 × 1.1',
            'zero':'全部 speaker 条件置零','reserved_1':'替换为未训练的 speaker 1',
            'decoder_zero':'仅 decoder 条件置零（固定时长与 mu）',
            'different_noise':'原 speaker 0，换一份噪声','original':'真实原音频'}
    body=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>Speaker embedding 诊断</title>',
          '<style>body{max-width:1100px;margin:30px auto;font:16px/1.6 system-ui;padding:0 20px;color:#222}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;padding:7px;text-align:left}audio{width:100%}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}.card{border:1px solid #ddd;padding:12px;border-radius:8px}small{color:#555}</style>',
          f'<h1>Speaker embedding 诊断：step {meta["checkpoint_step"]}</h1>',
          '<p>固定 8 条样本（训练/验证各 2 英文、2 中文），10 步非流式推理。干预之间共享同一份初始噪声；另设换噪声对照。原训练继续运行，诊断未执行参数更新。</p>',
          '<p><strong>解释边界：</strong>所有训练样本都使用 speaker 0，speaker 1 未训练。非零梯度或改变声音只能证明条件路径有效，不能证明已学会可分离的说话人身份。置零和替换实验属于训练分布外干预。声纹余弦是辅助比较，不设身份判定阈值；ASR 数字来自小样本诊断，不能替代 450 条评估。</p>',
          '<table><tr><th>条件</th><th>英文 WER</th><th>中文 CER</th><th>WavLM 对 baseline</th><th>CAM++ 对 baseline</th></tr>']
    for mode,r in result.items():
        body.append(f'<tr><td>{labels[mode]}</td><td>{r["en_error_micro"]:.2%}</td><td>{r["zh_error_micro"]:.2%}</td><td>{r["wavlm_cosine_to_baseline"]:.4f}</td><td>{r["camp_cosine_to_baseline"]:.4f}</td></tr>')
    body.append('</table>')
    for sample,group in samples.items():
        body.append(f'<h2>{html.escape(sample)}</h2><p>{html.escape(group[0]["ref_text"])}</p><div class="grid">')
        for row in group:
            path=Path(row['audio']); assert path.exists()
            rel=path.relative_to(out).as_posix()
            metrics=speakers[row['id']]
            body.append(f'<div class="card"><strong>{labels[row["mode"]]}</strong><audio controls preload="none" src="{html.escape(rel)}"></audio><p>ASR：{html.escape(asr[row["id"]]["asr_text"])}</p><small>对 baseline：WavLM {metrics["wavlm_cosine_to_baseline"]:.3f}，CAM++ {metrics["camp_cosine_to_baseline"]:.3f}</small></div>')
        body.append('</div>')
    body.append('<p><a href="summary.json">汇总数据</a> · <a href="metadata.json">实验设置</a> · <a href="FINDINGS.md">诊断结论</a></p></html>')
    (out/'index.html').write_text('\n'.join(body))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    main(p.parse_args().output)
