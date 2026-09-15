"""Filter the audited second-stage recipe to unique MajesticVoice recordings."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from training.stage2.prepare import main as prepare_base, digest


def main(args):
    prepare_base(SimpleNamespace(run_dir=args.run_dir,checkpoint=args.checkpoint,max_steps=12000))
    run=args.run_dir; data=run/'data'
    plan=json.loads((run/'plan.json').read_text())
    splits={}
    for split in ['train','val','test']:
        rows=[r for r in map(json.loads,(data/f'{split}.jsonl').read_text().splitlines()) if r['speaker']==1]
        rows=list({(r['audio'],r.get('start'),r.get('end')):r for r in rows}.values())
        splits[split]=rows
        (data/f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        def line(r):
            fields=[r['audio'],'1']+([str(r['start']),str(r['end'])] if 'start' in r else [])+[r['text']]
            return '|'.join(fields)+'\n'
        (data/f'{split}.txt').write_text(''.join(map(line,rows)))
    assert len(splits['train'])==11034 and len(splits['val'])==232
    assert Counter(r['language'] for r in splits['train'])=={'Chinese':8278,'English':2756}
    all_rows=[r for rows in splits.values() for r in rows]
    assert all('/MajesticVoice/' in r['audio'] and Path(r['audio']).is_file() for r in all_rows)
    texts={r['text'] for r in all_rows}
    cache=[r for r in map(json.loads,(data/'text_preflight.jsonl').read_text().splitlines()) if r['text'] in texts]
    (data/'text_preflight.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in cache))
    keys={json.dumps([r['audio'],r.get('start'),r.get('end')],separators=(',',':')) for r in all_rows}
    units=[r for r in map(json.loads,(data/'mel_statistics_units.jsonl').read_text().splitlines()) if r['key'] in keys]
    (data/'mel_statistics_units.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in units))
    evaluation=[r for r in map(json.loads,(data/'eval_manifest.jsonl').read_text().splitlines()) if r['speaker']==1]
    assert len(evaluation)==450
    (data/'eval_manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in evaluation))
    heldout={tuple(r['text_key']) for s in ['val','test'] for r in splits[s]}|{tuple(r['token_ids']) for r in evaluation}
    assert not {tuple(r['text_key']) for r in splits['train']} & heldout
    assert not {r['audio'] for r in splits['train']} & {r['audio'] for s in ['val','test'] for r in splits[s]}
    protocol=json.loads((data/'eval_protocol.json').read_text())
    protocol.update(samples=450,groups=dict(Counter(r['group'] for r in evaluation)),manifest_sha256=digest(data/'eval_manifest.jsonl'))
    protocol['note']='Only speaker 1 evaluated; row 0 reference retained for common metrics-loader compatibility, never used as training data.'
    (data/'eval_protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    summary=dict(stage=2,recipe='MajesticVoice only, unique recordings',train_rows=11034,train_rows_per_speaker={'1':11034},
                 validation_rows=232,test_rows=len(splits['test']),language_counts={'Chinese':8278,'English':2756},
                 ljs_training_rows=0,train_heldout_audio_overlap=0,train_heldout_phoneme_overlap=0,
                 original_mixed_recordings_included=False)
    (data/'training_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    stats=json.loads((data/'mel_statistics.json').read_text());stats['manifest_sha256']=digest(data/'train.txt')
    (data/'mel_statistics.json').write_text(json.dumps(stats,indent=2)+'\n')
    plan.update(recipe='majestic_only_embedding_warmup_then_joint',active_speaker_ids=[1],speakers={'0':'reserved, unused','1':'MajesticVoice'},
        train_rows=11034,validation_rows=232,test_rows=len(splits['test']),train_rows_per_speaker={'1':11034},
        ljs_original_train_rows=0,ljs_historical_synthetic_train_rows=0,
        unique_audio_hours={'majestic_chinese':8.670755555555672,'majestic_english':2.9952888888888705},
        schedule_callback='training.majestic_finetune.callbacks.MajesticSchedule',
        learning_rates={'decoder':5e-6,'spk_emb':1e-4,'prior_encoder':5e-6,'duration_predictor':5e-6},
        schedule={'warmup_steps':50,'encoder_update_start_step':500,'encoder_warmup_steps':200,'final_lr_ratio':.2,
                  'embedding_adaptation_lr':5e-4,'freeze_before_500':'all non-embedding parameters'},
        evaluation_groups=protocol['groups'],audio_paths_checked=len({r['audio'] for r in all_rows}))
    plan['manifest_hashes']={name:digest(data/name) for name in plan['manifest_hashes']}
    (run/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    (run/'TRAINING_PLAN.md').write_text('''# 大气女声单音色微调

用户授权停止上一轮，从第一阶段 197000 checkpoint 重新初始化；只用大气女声中文和英文。
用户补充：前 500 步冻结其他模块，提高 speaker embedding 学习率，之后再联合小学习率微调。

- 全部训练、验证、测试样本为 speaker 1；保留 2×64 embedding 表兼容源模型，row 0 不接收监督，row 1 从源 row 0 复制。
- 训练使用 11034 条唯一音频：中文 8278 条约 8.67 小时，英文 2756 条约 3 小时。没有 LJS，也不重复采样补齐到 LJS 数量。沿用既有保留集划分。原始混读录音没有另行加入此次配方。
- 初始模型严格来自第一阶段 197k，Adam 和第二阶段步数归零，保留源 Mel 归一化、前端、MAS 和 Vocos。
- 前 500 次更新：仅 embedding 更新，前 50 步由约 5e-5 预热到 5e-4，随后保持；文本 encoder、duration、帧级 encoder、flow 的梯度在 DDP 同步后设为 None，权重及 Adam 状态均不更新。
- 第 500 步起：embedding 峰值降到 1e-4；所有其他声学模块峰值 5e-6，200 步从约 10% 渐进加入。之后按余弦下降至峰值的 20%。不冻结声学模块，不更新独立 Vocos。
- 四卡 BF16，每卡 batch 48，有效 batch 192；每 epoch 57 步，预算 12000 步，约 210 epoch。
- 每 250 步验证 232 条并保存 checkpoint；评估只针对大气女声：中文 200、英文 200、混读 50。首次每组 8 条检查链路，之后至少每隔 1000 步完整评估并保留试听；结束后评估最后模型。
- 在第 1/100/500 步验证只有 row 1 embedding 更新，其余参数精确不变；第 510 步分别验证文本 encoder、duration、帧级 encoder、flow 都已更新。目标是检验口音能否改善，自动内容和音色指标不作为口音评分。
''')
    print(json.dumps({'status':'prepared','run_dir':str(run),'train_rows':11034,'evaluation_samples':450}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    main(p.parse_args())
