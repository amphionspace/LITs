"""Generate matched-total-duration A/B audio without updating the model."""
import argparse
import base64
import hashlib
import html
import json
from pathlib import Path
import time

import numpy as np

from training.foundation.diagnose_duration_smoothing import pair_metrics


def allocate_frames(weights, total, tone_mask):
    """Scale a timing pattern to an exact integer budget, retaining tone bounds."""
    weights = np.asarray(weights, dtype=np.float64)
    upper = np.where(tone_mask, 3, total)
    assert len(weights) <= total <= upper.sum() and (weights > 0).all()
    low, high = 0., float(total)
    for _ in range(100):
        scale = (low+high)/2
        if np.clip(weights*scale, 1, upper).sum() < total:
            low = scale
        else:
            high = scale
    ideal = np.clip(weights*((low+high)/2), 1, upper)
    duration = np.floor(ideal).astype(np.int64)
    remaining = int(total-duration.sum())
    order = np.argsort(-(ideal-duration), kind='stable')
    eligible = order[duration[order] < upper[order]]
    assert 0 <= remaining <= len(eligible)
    duration[eligible[:remaining]] += 1
    assert duration.sum() == total and (duration >= 1).all() and (duration <= upper).all()
    return duration, dict(scale=(low+high)/2, integer_rounding='floor plus largest fractional remainders',
                          minimum=1, tone_maximum=3, total_frames=total)


def select_samples(records):
    chosen=[]
    for language in ('zh','en'):
        pool=[]
        for r in records:
            mask=np.array(r['speech'],dtype=bool)
            if r['language'] != language or not 3 <= r['mas_frames']*.016 <= 10 or mask.sum()<15:
                continue
            m=pair_metrics(np.array(r['mas'])[mask],np.array(r['inference'])[mask])
            if m and m['cv_ratio'] is not None:
                pool.append((r,m['cv_ratio']))
        used=set()
        for quantile in (.25,.5,.75):
            target=float(np.quantile([v for r,v in pool],quantile))
            r,value=min((item for item in pool if item[0]['sample_id'] not in used),key=lambda item:abs(item[1]-target))
            used.add(r['sample_id'])
            chosen.append(dict(r,selection_quantile=quantile,selection_cv_ratio=value))
    return chosen


def render_page(root, cases):
    def audio(path):
        encoded=base64.b64encode(path.read_bytes()).decode()
        return f'<audio controls preload="none" src="data:audio/wav;base64,{encoded}"></audio>'
    cards=[]
    for i,c in enumerate(cases):
        label='中文' if c['language']=='zh' else '英文'
        pair=[]
        for arm in ('A','B'):
            pair.append(f'<div><b>版本 {arm}</b>{audio(root/c["audio"][arm])}</div>')
        votes=''.join(f'<label><input type="radio" name="case_{i}" value="{v}">{t}</label>'
                      for v,t in [('A','A 更自然'),('B','B 更自然'),('tie','差不多'),('unclear','无法判断')])
        mapping='；'.join(f'{arm}：'+('预测时长，已匹配总长' if mode=='predicted_matched' else 'MAS 时长') for arm,mode in c['mapping'].items())
        cards.append(f'''<section><h2>{i+1}. {label} · {c['frames']*.016:.2f} 秒</h2>
<p class="text">{html.escape(c['text'])}</p><div class="pair">{''.join(pair)}</div>
<p>哪一版的节奏更自然、机械感更少？</p><div class="votes">{votes}</div>
<textarea id="note_{i}" placeholder="可选：停顿、重音、拖长或发音的具体差异"></textarea>
<details><summary>参考录音与真实 Mel 重建</summary><p>原录音音色可能与模型输出不同，主要用于参考节奏；重建用于检查声码器影响。</p>
<div class="pair"><div>原录音{audio(root/c['audio']['original'])}</div><div>真实 Mel 经 Vocos 重建{audio(root/c['audio']['reconstruction'])}</div></div></details>
<details><summary>听完后查看 A/B 身份</summary><p>{html.escape(mapping)}</p><p>样本 {c['sample_id']}；取自该语言 CV 比约第 {c['selection_quantile']*100:.0f} 百分位。</p></details></section>''')
    data=json.dumps([dict(case=i+1,sample_id=c['sample_id'],language=c['language']) for i,c in enumerate(cases)])
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>197k · Duration A/B 试听</title><style>
body{font:16px/1.65 system-ui,sans-serif;color:#203044;background:#f3f6fa;margin:auto;max-width:1040px;padding:28px}
h1{font-size:28px}h2{font-size:21px}.intro,section{background:white;border:1px solid #dae2ea;border-radius:12px;padding:22px;margin:18px 0}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:22px}audio{display:block;width:100%;margin:10px 0}.text{font-size:19px}.votes{display:flex;flex-wrap:wrap;gap:18px}
textarea{box-sizing:border-box;width:100%;min-height:68px;margin:12px 0;padding:10px;font:inherit;border:1px solid #bac8d5;border-radius:6px}
details{margin-top:14px}summary{cursor:pointer;color:#236b9f}button{background:#175b8e;color:white;border:0;border-radius:7px;padding:12px 20px;font:inherit;cursor:pointer}
@media(max-width:640px){body{padding:12px}.pair{grid-template-columns:1fr}}
</style><h1>197k · Duration A/B 试听</h1><div class="intro">
<p><strong>每组只有时长分配不同，总音频长度完全相同。</strong>文本、模型、speaker 0、初始噪声、10 步 Euler、非流式 flow 和 Vocos 都保持一致；A/B 顺序逐组打乱。</p>
<p>重点听句内长短、重音与停顿是否更自然，同时留意吞字、拖音或错音。原录音仅供参考。建议先听 A/B 再展开身份。</p>
<p>中文、英文各 3 条，覆盖较强、中等、较弱压缩。选择仅保存在此浏览器，不会上传；可下载评价 JSON。</p>
<button id="download">下载我的评价</button><span id="status" style="margin-left:16px"></span></div>'''+''.join(cards)
    page+='''<script>
const cases=CASE_DATA,key='lits-duration-ab-197000-20260914';
function current(){return cases.map((c,i)=>({...c,choice:document.querySelector(`input[name="case_${i}"]:checked`)?.value??null,note:document.getElementById(`note_${i}`).value}));}
function save(){try{localStorage.setItem(key,JSON.stringify(current()));document.getElementById('status').textContent='已保存在此浏览器';}catch(e){document.getElementById('status').textContent='可用下载按钮保存评价';}}
try{const saved=JSON.parse(localStorage.getItem(key)||'[]');saved.forEach((r,i)=>{if(!cases[i])return;const radio=document.querySelector(`input[name="case_${i}"][value="${r.choice}"]`);if(radio)radio.checked=true;document.getElementById(`note_${i}`).value=r.note||'';});}catch(e){}
document.querySelectorAll('input,textarea').forEach(e=>e.addEventListener('change',save));
document.querySelectorAll('audio').forEach(a=>a.addEventListener('play',()=>document.querySelectorAll('audio').forEach(b=>{if(a!==b)b.pause();})));
document.getElementById('download').onclick=()=>{const blob=new Blob([JSON.stringify({experiment:key,ratings:current()},null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='duration_ab_ratings.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);};
</script></html>'''.replace('CASE_DATA',data)
    (root/'listen.html').write_text(page)


def main(args):
    import soundfile as sf
    import soxr
    import torch
    from torch.nn import functional as F
    from lits.models.lits import LITS
    from lits.utils.model import fix_len_compatibility,sequence_mask
    from lits.utils.audio import mel_spectrogram
    from vocos.vocoder import load_vocos_vocoder

    torch.set_num_threads(4)
    root=args.output;root.mkdir(parents=True,exist_ok=False)
    start=time.time();repo=Path(__file__).resolve().parents[2]
    metadata=json.loads((args.diagnostic/'metadata.json').read_text())
    checkpoint=Path(metadata['checkpoint'])
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==metadata['checkpoint_sha256']
    records=[json.loads(line) for line in (args.diagnostic/'samples.jsonl').read_text().splitlines()]
    chosen=select_samples(records)
    (root/'selected_samples.json').write_text(json.dumps(chosen,ensure_ascii=False,indent=2)+'\n')
    model=LITS.load_from_checkpoint(str(checkpoint),map_location='cpu',weights_only=False).eval().requires_grad_(False)
    assert model.tone_floor_frames==1 and model.tone_ceiling_frames==3
    vocoder,cfg=load_vocos_vocoder(str(repo/'vocos/generator.ckpt'),torch.device('cpu'),repo)
    vocoder.eval().requires_grad_(False)
    assert (cfg.sampling_rate,cfg.num_mels,cfg.hop_size)==(24000,100,384)
    protocol=dict(checkpoint=str(checkpoint),checkpoint_sha256=metadata['checkpoint_sha256'],device='cpu',precision='float32',
        euler_steps=10,streaming=False,temperature=1.,speaker_id=0,samples=6,
        selection='Per language, nearest 25th/50th/75th percentile inference speech CV ratio among 3–10s validation recordings with >=15 speech tokens',
        duration_matching='Global scale of existing inference durations with min1/tone cap3, exact integer total by largest remainder',
        variable='Only expanded timing path; downstream conditioning encoder and flow outputs consequently change',
        pair_gain='One common attenuation per pair if needed to keep peak <=0.98; no per-arm loudness normalization',
        started_at_unix=start)
    (root/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    cases=[];rng=np.random.default_rng(1970002026)
    with torch.inference_mode():
        for index,r in enumerate(chosen):
            dest=root/f'{index+1:02d}_{r["language"]}_{r["sample_id"]}';dest.mkdir()
            x=torch.tensor([r['ids']]);xl=torch.tensor([x.shape[-1]])
            # Read tones from the same immutable foundation record rather than reconstructing them.
            import sqlite3
            with sqlite3.connect('file:'+metadata['database']+'?mode=ro',uri=True) as db:
                original_row=json.loads(db.execute('select payload from samples where id=?',(r['sample_id'],)).fetchone()[0])
            xt=torch.tensor([original_row['tones']]) if original_row['tones'] is not None else None
            sid=torch.tensor([0]);spk=model.spk_emb(sid)
            mu_x,logw,xmask=model.encoder(x,xl,spk,x_tones=xt)
            np.testing.assert_allclose(torch.exp(logw).flatten().numpy(),r['raw'],rtol=2e-5,atol=2e-5)
            mas=np.asarray(r['mas'],dtype=np.int64);frames=int(mas.sum())
            tone_mask=model._tone_mark_mask(x).flatten().numpy()
            matched,allocation=allocate_frames(r['inference'],frames,tone_mask)
            padded=int(fix_len_compatibility(frames));mask=sequence_mask(torch.tensor([frames]),padded)[:,None].float()
            seed=197000+index;torch.manual_seed(seed);z=torch.randn(1,100,padded)
            zhash=hashlib.sha256(z.numpy().tobytes()).hexdigest()
            waves={};mels={}
            for name,durations in [('predicted_matched',matched),('mas',mas)]:
                mu=torch.repeat_interleave(mu_x,torch.tensor(durations),dim=2)
                assert mu.shape[-1]==frames
                mu=F.pad(mu,(0,padded-frames))
                model.decoder.reset_encoder_cache()
                mel=model.get_mel(mu,mask,10,1.,spks=spk,streaming=False,z=z.clone())[:,:,:frames]
                wave=vocoder(mel).flatten().numpy()[:frames*384]
                assert len(wave)==frames*384 and np.isfinite(wave).all() and torch.isfinite(mel).all()
                assert hashlib.sha256(z.numpy().tobytes()).hexdigest()==zhash
                waves[name]=wave;mels[name]=mel.numpy()
                print(json.dumps(dict(sample=index+1,language=r['language'],mode=name,frames=frames,seconds=time.time()-start)),flush=True)
            peak=max(float(np.abs(w).max()) for w in waves.values());gain=min(1.,.98/max(peak,1e-9))
            modes=['predicted_matched','mas'];rng.shuffle(modes);mapping=dict(zip(('A','B'),modes))
            files={}
            for arm,name in mapping.items():
                path=dest/f'{arm}.wav';sf.write(path,waves[name]*gain,24000,subtype='PCM_16');files[arm]=str(path.relative_to(root))
            original,sr=sf.read(r['audio'],dtype='float32')
            if sr!=24000:original=soxr.resample(original,sr,24000,quality='HQ')
            assert original.ndim==1 and np.isfinite(original).all()
            real=mel_spectrogram(torch.from_numpy(original)[None],2048,100,24000,384,1536,0,12000)
            assert real.shape[-1]==frames
            recon=vocoder(real).flatten().numpy()[:frames*384]
            for name,wave in [('original',original),('reconstruction',recon)]:
                assert np.isfinite(wave).all()
                ref_gain=min(1.,.98/max(float(np.abs(wave).max()),1e-9))
                path=dest/f'{name}.wav';sf.write(path,wave*ref_gain,24000,subtype='PCM_16');files[name]=str(path.relative_to(root))
            speech=np.array(r['speech'])
            case=dict(sample_id=r['sample_id'],language=r['language'],text=r['text'],source_audio=r['audio'],frames=frames,
                selection_quantile=r['selection_quantile'],selection_cv_ratio=r['selection_cv_ratio'],
                seed=seed,noise_sha256=zhash,audio=files,mapping=mapping,pair_gain=gain,
                original_inference_frames=int(sum(r['inference'])),allocation=allocation,
                mas_durations=mas.tolist(),matched_predicted_durations=matched.tolist(),
                matched_speech_metrics=pair_metrics(mas[speech],matched[speech]),
                audio_sha256={k:hashlib.sha256((root/v).read_bytes()).hexdigest() for k,v in files.items()})
            np.savez_compressed(dest/'mel_outputs.npz',**mels)
            (dest/'case.json').write_text(json.dumps(case,ensure_ascii=False,indent=2)+'\n')
            cases.append(case)
            (root/'cases.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2)+'\n')
    for case in cases:
        a,sr=sf.read(root/case['audio']['A']);b,sr2=sf.read(root/case['audio']['B'])
        assert sr==sr2==24000 and len(a)==len(b)==case['frames']*384
        assert np.max(np.abs(a))<1 and np.max(np.abs(b))<1
    render_page(root,cases)
    protocol.update(completed_at_unix=time.time(),elapsed_seconds=time.time()-start,
        verification=dict(pairs=6,wav_files=24,matched_lengths=True,same_pair_noise=True,finite_audio=True,no_pcm_clipping=True),
        listening_result='Not rated: no human preference has been collected')
    (root/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    report=['# 197k：相同总时长的 Duration A/B 试听','', '[打开独立试听页](listen.html)', '',
        '中文、英文各 3 条。A/B 标签随机分配，身份可在页面内展开。音频已嵌入 HTML，下载该文件后可离线试听并导出评价。', '',
        '## 控制变量','',
        '- 同一个 197k checkpoint、同一文本与 speaker 0、同一初始高斯噪声、10 步 Euler、非流式、同一个 Vocos。',
        '- 每对输出总帧数与样本 MAS 总帧数严格相等，24 kHz WAV 样本数也完全相等。',
        '- 预测版本使用原推理 duration 的全局缩放加整数分配；所有 token 至少 1 帧，声调保持 1–3 帧。整数误差按最大小数余数分配，不使用逐 token MAS 值修正预测。',
        '- MAS 版本使用上一轮原模型对齐结果。两版仅更换展开路径，下游条件编码器与 flow 计算保持相同。',
        '- 如需避免削波，两版使用同一个衰减系数；不逐版独立调整响度。', '',
        '## 样本与匹配后变化幅度','',
        '| 序号 | 语言 | MAS 总长（秒） | 原预测总长（秒） | 匹配后 speech Std/MAS Std | 文本 |',
        '|---|---|---:|---:|---:|---|']
    for i,c in enumerate(cases):
        report.append(f'| {i+1} | {c["language"]} | {c["frames"]*.016:.3f} | {c["original_inference_frames"]*.016:.3f} | {c["matched_speech_metrics"]["std_ratio"]:.3f} | {c["text"].replace("|","/")} |')
    report += ['', '## 如何判断','',
        '先比较 A/B 的节奏起伏、重音与停顿是否自然，再记录吞字、拖音、断裂等问题。真实录音和真实 Mel 重建只作为辅助参考；原录音音色与模型输出可能不同。', '',
        '若 MAS 在总时长相同的条件下稳定改善自然度，说明 duration/对齐条件路径值得优先处理；这仍不能直接证明换一种 duration loss 就能获得同样改善。替换完整路径同时改变停顿和音素时序，不能只归因于语音 token 方差。', '',
        '当前已完成生成及数值/音频文件检查，尚无人工试听结论；六条单种子样本是探索性对照，不是总体质量评测。', '',
        '- [protocol.json](protocol.json)：设置与验证结果。',
        '- [cases.json](cases.json)：A/B 身份、逐 token duration、音频及噪声哈希。',
        '- [selected_samples.json](selected_samples.json)：样本选择及上一轮原始统计。','']
    (root/'REPORT.md').write_text('\n'.join(report))
    print(json.dumps(dict(status='complete',seconds=time.time()-start,page=str(root/'listen.html'))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--diagnostic',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
