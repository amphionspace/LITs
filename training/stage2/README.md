# 第二阶段：LJSpeech 0 / 大气女声 1

本轮从第一阶段 step 197000 迁移权重，保留 24 kHz、100 Mel、173 tokens、原有 MAS 设置和第一阶段 Mel 归一化。该 checkpoint 的固定英文 WER 1.35%、中文 CER 4.66%，用于选择发音基础；第二阶段另用目标音色协议验证，不能把这两个数字当作音色适配结果。

## 数据与目标

| 数据 | speaker | 不同训练样本 | 时长 |
|---|---:|---:|---:|
| LJSpeech 原始录音 | 0 | 11,550 | 21.05 小时 |
| MajesticVoice 中文 | 1 | 8,278 | 8.67 小时 |
| MajesticVoice 英文 | 1 | 2,756 | 3.00 小时 |

本轮使用 LJSpeech 原录音进行音色微调。MajesticVoice 两种语言都保留，让 speaker 1 在中文和英文中代表同一音色。MajesticVoice 训练池补采样到 11,550 行，与 LJSpeech 按 1:1 配平，共 23,100 行；按时长分桶并在四个 rank 之间分配。采样重复只用于训练权重，验证集不重复抽样。

继承已审核的划分，完整验证 464 条、内部测试 1,004 条，训练与保留集的音频路径和音素序列无交叉。固定外部评估 650 条，包括 LJS 英文 200、大气女声英文 200、中文 200、混读 50。当前训练清单为单语语句，混读作为能力检查，不承诺仅音色微调就能解决混读问题。

## 初始化与优化

两个 speaker embedding 都从第一阶段训练过的 row 0 初始化；第一阶段未训练的 row 1 不作为目标音色初值。随后两行通过各自音频监督独立更新。其余模型权重逐项迁移，Adam 状态和全局步数重新建立。

数据和模型共同使用源 checkpoint 的 Mel 均值、标准差，防止预训练权重面对突然变化的输入尺度。实际使用值、初始化哈希和所有清单哈希记录在 `plan.json`。

| 参数组 | 峰值学习率 | 更新策略 |
|---|---:|---|
| speaker embedding | 1e-4 | 从第 0 步更新 |
| decoder | 2e-5 | 从第 0 步更新 |
| 文本/声学先验 encoder | 2e-6 | 前 500 步不更新，再用 500 步逐渐加入 |
| duration predictor | 5e-6 | 前 500 步不更新，再用 500 步逐渐加入 |

前两组用 200 步从峰值的约 10% 预热，之后所有组沿余弦调度降至峰值的 20%。冻结阶段在 DDP 梯度同步后把 encoder/duration 的梯度设为 `None`，保证权重和 Adam 状态不更新，同时保留稳定的 DDP 计算图供后续解冻。不开启旧的自动回退和 plateau LR 回调，避免覆盖分组调度。

第一轮上限 6,000 optimizer steps。四张 GPU、BF16、每卡 batch 48，有效 batch 192，无梯度累积；每轮约 120 步，预算约 50 轮。这是第一轮适配预算，后续是否延长依据保存的发音、音色和验证结果决定。

## 验证与运行

- 启动前：98 个样本的缓存前端与原始 loader 等价检查；四卡分桶一致性；最长批次完整前向/反向；冻结、解冻和两行 speaker 梯度检查。预检权重丢弃。
- 正式启动：权重与初始化文件逐项一致，优化器为空，参数覆盖完整，归一化匹配。
- 第 100 步：确认 decoder 和两行 speaker 都更新，encoder/duration 未变；第 510 步确认后两组开始更新。
- 每 250 步：完整验证并保存 checkpoint，保留最近 5 份、last、best-prior。模型选择同时查看内容和音色，不只看 prior loss。
- 自动评估：首份 checkpoint 做每组 8 条链路检查，之后至少间隔 1,000 步做完整 650 条；训练结束评估最终模型。另对未更新的初始化模型做完整基线。
- 指标：英文 WER、中文/混读 CER、WavLM/CAM++ 对各自参考音色的相似度，以及 DNSMOS。音频与逐句 ASR 结果保留供试听。

当前运行目录：`/119010446/tts-assets/training_runs/ljs_majestic_stage2_from197k_20260913`。

```bash
PYTHONPATH=/119010446/LITs \
/119010446/tts-assets/chenmingjie/lits-tts/envs/lits/bin/python \
  /119010446/LITs/training/stage2/run.py \
  --run-dir /119010446/tts-assets/training_runs/ljs_majestic_stage2_from197k_20260913
```

启动器拒绝重复启动同一个目录。代码、配置和数据清单保存在 run 目录，实际训练状态以 `training_state.json` 为准。

## 延长训练实验（2026-09-13）

新实验目录：`/119010446/tts-assets/training_runs/ljs_majestic_stage2_long24k_from197k_20260913`。
按用户要求再次从第一阶段 197000 checkpoint 初始化，训练上限 24000 步。
`prepare.py --max-steps 24000` 将预算写入 `plan.json`，启动器据此同步设置 trainer 和余弦调度周期；默认仍为 6000 步。
与首轮的初始化权重、数据清单、峰值学习率、冻结策略和随机种子相同。
余弦衰减周期延长意味着预热之后相同步数的学习率也会不同；详细对照见新目录的 `experiment_comparison.json` 和 `TRAINING_PLAN.md`。
