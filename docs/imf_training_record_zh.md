# 当前 IMF：训练完成记录、数据与学习率

记录日期：2026-09-16。**当前主实验已正常完成 170,000 次 optimizer 更新、500 epoch，最终 650 条生成评测也已完成。** 本文记录已运行的实际配置。

## 1. 当前模型与产物

| 项目 | 实际值 |
| --- | --- |
| Run | `ljs_majestic_100h_imf_h1_b48_from21k_500ep_20260915` |
| 训练结束 | 2026-09-16 12:15:59 UTC，进程返回码 0 |
| 最终训练步数 | 170,000；不含此前 Stage 1 的 21,000 步 |
| 峰值 / 最终 LR | **3e-4 / 2e-5** |
| 默认推理 | IMF，2 步 Euler，时间网格 `[0, 0.5, 1]` |
| Speaker | `0=LJSpeech`；`1=MajesticVoice` |
| 声码器 | 原有 24 kHz Vocos，未训练、未替换 |
| 源码快照 revision | `206539cc333224989a2a640160de4d3da2916990` |

- 最终完整 checkpoint：[final.ckpt](/119010446/tts-assets/training_runs/ljs_majestic_100h_imf_h1_b48_from21k_500ep_20260915/checkpoints/final.ckpt)。它的 `global_step=170000`，四组 Adam 参数的更新计数均为 170000。
- 最终生成评测：[summary.json](/119010446/tts-assets/training_runs/ljs_majestic_100h_imf_h1_b48_from21k_500ep_20260915/eval/step_00170000/summary.json)、[逐句结果](/119010446/tts-assets/training_runs/ljs_majestic_100h_imf_h1_b48_from21k_500ep_20260915/eval/step_00170000/details.jsonl)。650 条全部完成，零评测失败。
- 最终 duration 检测：[duration_summary.json](/119010446/tts-assets/training_runs/ljs_majestic_100h_imf_h1_b48_from21k_500ep_20260915/eval/step_00170000/duration_summary.json)。

`final.ckpt` 的 SHA-256 为 `aad729abcad82577587ccfa9112462368f7300132ca8bc7e0554c39d14d4c431`。最终训练文件和评测用 170k 文件的序列化哈希不同；已逐项核对模型张量完全相同。

## 2. 现在的 IMF 是什么

当前实现是 **improved MeanFlow 对条件 Mel 生成的适配**，训练目标直接来自配对音频的 Mel。Majestic 音频虽由 VoxCPM2 合成，但本轮优化不使用教师采样轨迹或蒸馏 loss。

推理流程为：

`文本音素 + speaker ID → 文本编码与 duration 预测 → 展开到帧级条件 → IMF 生成 Mel → 固定 Vocos → 24 kHz WAV`

### 网络和目标

- 保留 LITs 文本/条件编码器、duration predictor、speaker embedding 和 causal U-Net 主体。
- IMF 的 `u` 为平均速度头，`v` 为辅助瞬时速度头；两者共享 down/mid 主干，使用各自的 up/final 分支。
- 训练同时计算 `u/v` 目标，JVP 的方向使用零区间处预测的 `v`。**推理只调用 `u`，不调用辅助 `v`。**
- 新增区间时间 MLP，末层以零初始化；`v` 分支从已加载的 FM 对应分支复制。
- `interval_time_scale=1.0`，run 名称里的 **h1 指这个区间时间尺度，不是一步推理**。保留 FM 原有绝对时间 embedding 的尺度。

| IMF 参数 | 值与含义 |
| --- | --- |
| `P_mean / P_std` | `-0.4 / 1.0`；采样两个 logit-normal 时间并排序 |
| `data_proportion` | `0.5`；每批一半样本使用零区间的 FM 对角目标 |
| `sigma_min` | `0`；线性 Mel—噪声路径 |
| `norm_p / norm_eps` | `1.0 / 0.01`；每个样本、每个头独立自适应加权 |
| `guidance_scale` | `1.0`；无 CFG、无条件丢弃 |
| 推理 `num_steps` | `2`，LITs 时间方向为 `0=噪声，1=Mel` |

估计器 dropout 关闭；JVP 与 IMF loss 内部使用 FP32、math SDPA，其余允许 BF16 mixed。padding 在噪声状态、目标和误差求和中均被 mask。训练保留整句上下文与流式模式的随机混合。

总训练目标：

```text
L = 1 × duration_loss + 1 × prior_loss + 1 × IMF_loss
IMF_loss = weighted_u_loss + weighted_v_loss
```

每个头的权重形式是 `S / stop_gradient((S + 0.01)^1)`，其中 `S` 是有效 Mel 元素的平方误差和。因此数值往往接近 1，两个头合计接近 2；**不能拿这个约等于 2 的 loss 与 FM MSE 比大小，也不能仅凭它平稳就判断收敛。** 应结合未加权 `u_mse/v_mse`、JVP RMS 和生成评测。

最终验证日志：`u_mse=0.211268`，`v_mse=0.091716`，`duration_loss=0.216129`，`prior_loss=0.968524`，`IMF_loss=1.999994`。duration/prior 权重始终为 1，没有后期衰减。

### 初始化与训练范围

基础权重来自 Stage 1 的 **21k FM backbone**。具体使用旧 FM 主实验的 step-0 初始化文件，复制全部已有张量，包括当时随机初始化的两个 speaker 向量，从而保证 FM/IMF 起点一致。随后新增 IMF 的区间分支和辅助头，Adam 与步数从零开始。

这不是从旧 FM 的最终 170k 权重继续训练。encoder、duration predictor、speaker embedding、decoder 从本轮第一步起全部更新，没有冻结或分组降 LR。

Vocos 不在训练模型的参数组或 checkpoint `state_dict` 中。评测快照与当前仓库的 `vocos/generator.ckpt` 哈希一致：`1d60c04156e59348566c65cab921591e1651ab48757ca3766caf9459475bd1c2`。

实现依据：[运行时 IMF 源码](/119010446/tts-assets/training_runs/ljs_majestic_100h_imf_h1_b48_from21k_500ep_20260915/source/lits/models/components/improved_mean_flow.py)、[初始化核验](/119010446/tts-assets/training_runs/ljs_majestic_100h_imf_h1_b48_from21k_500ep_20260915/initialization_verification.json)、[迁移核验](/119010446/tts-assets/training_runs/ljs_majestic_100h_imf_h1_b48_from21k_500ep_20260915/migration.json)。

## 3. 训练数据

### 实际进入训练的规模

这次是 **MajesticVoice 100.0015 小时 + LJSpeech 21.0475 小时，共 121.0490 小时**，共 65,282 条。数据项目目录里的 `MajesticVoice_200h_20260913` 是历史名称，实际目标音色训练配额为 100 小时。

| Speaker / 数据 | 语言 | 训练条数 | 训练小时 | 验证条数 | 内部测试条数 |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 / LJSpeech 原始录音 | 英文 | 11,550 | 21.0475 | 232 | 0 |
| 1 / MajesticVoice 合成录音 | 中文 | 25,323 | 50.0004 | 192 | 205 |
| 1 / MajesticVoice 合成录音 | 英文 | 16,515 | 25.0002 | 115 | 294 |
| 1 / MajesticVoice 合成录音 | 混读 | 11,894 | 25.0009 | 93 | 109 |
| 合计 | — | **65,282** | **121.0490** | **632** | **608** |

验证约 1.1755 小时，内部测试约 1.1044 小时。LJS 没有单独的内部 test 划分，另有 200 条固定文本生成评测。上表条数和小时数已从本轮 `train/val/test.jsonl` 重新统计。

冻结清单：[train.jsonl](/119010446/tts-assets/training_runs/ljs_majestic_100h_imf_h1_b48_from21k_500ep_20260915/data/train.jsonl)、[val.jsonl](/119010446/tts-assets/training_runs/ljs_majestic_100h_imf_h1_b48_from21k_500ep_20260915/data/val.jsonl)、[test.jsonl](/119010446/tts-assets/training_runs/ljs_majestic_100h_imf_h1_b48_from21k_500ep_20260915/data/test.jsonl)。

### 来源和筛选

- **LJSpeech** 使用原始录音的 24 kHz 版本，共 11,550 条训练录音，没有混入历史合成 LJS 数据。
- **MajesticVoice** 使用 VoxCPM2 合成并经过质量筛选的音频。文本来自既有 foundation TRAIN、Emilia2 真实语音转写，以及已批准的 GPT-4o / 人工审阅 DOTA 脚本；这次 IMF 没有重新生成、拼接或重划分文本。
- 中文/混读合成参考为 `ts004_cn_000001.wav`；英文合成与质检参考切换为已选定的英文 candidate `72003.wav`。这属于上游数据构建；LITs 推理本身用 learned speaker ID，不需要逐句参考音频。
- 主要准入阈值：CER≤2%，英文 WER≤2%；混读英文 WER=0、中文 CER≤2%；WavLM≥0.70、CAMPPlus≥0.68、DNSMOS OVRL≥3.3 / SIG≥3.5 / BAK≥3.8，并检查削波、静音和时长。中文存在“带声调拼音完全一致、原始 CER≤5%”的受控例外。这些是逐条筛选门槛，不是最终平均分或人工口音保证。
- 从合格音频池按来源、文本长度、英文词数等分层选择目标 50h / 25h / 25h，seed 为 `20260913`。本轮直接复用旧 FM 冻结的数据清单。

联合划分审计记录：跨 train/val/test 音频重叠为 0、音素序列重叠为 0、训练与外部生成评测音素重叠为 0。这里的去重与独立划分不代表文本一定没有自然重复或口吃。

### 采样与音频特征

训练按自然样本池抽取，每个 epoch 打乱并按长度分桶，无固定 speaker 配额、无 LJS 过采样。条数占比约为 **LJS 17.69% / Majestic 82.31%**，不能当作按小时比例或每个 batch 的固定比例。

4 卡 × 每卡 48 = 全局 batch 192，累积 1。每个 epoch 有 `floor(65282 / 192)=340` 次更新，尾部 2 条丢弃，500 epoch 对应 170,000 次更新。

音频为 24 kHz；Mel 100 维，FFT=2048，window=1536，hop=384（16ms），频率范围 0–12kHz。使用整句训练，`out_size=null`。沿用初始化 backbone 的归一化：`mean=-5.6042261124`，`std=3.2161116600`。

前端词表 173，speaker embedding 64 维。`n_tones=0` 表示没有单独 tone embedding，**不表示删除了声调**：声调仍在音素序列中，MAS 的 `mas_n_tones=5`。使用带音素时长上下界的在线 MAS，`use_precomputed_durations=false`；duration 监督来自当前模型的 MAS，不是逐句人工或 MFA 对齐真值。

数据依据：[训练统计](/119010446/tts-assets/training_runs/ljs_majestic_100h_imf_h1_b48_from21k_500ep_20260915/data/training_summary.json)、[采样配置](/119010446/tts-assets/training_runs/ljs_majestic_100h_imf_h1_b48_from21k_500ep_20260915/data/sampling_plan.json)、[划分审计](/119010446/tts-assets/training_runs/ljs_majestic_100h_imf_h1_b48_from21k_500ep_20260915/data/joint_split_audit.json)；上游生成配置见 [FM 训练记录](majestic_training_record_zh.md#6-当前-100h-目标音色数据如何生成)。

## 4. 实际训练配置与续训

| 项目 | 配置 |
| --- | --- |
| GPU / 分布式 | 4×A800 80GB，DDP |
| Batch | 每卡 48；累积 1；有效 batch 192 |
| 优化器 | Adam，betas=(0.9, 0.999)，eps=1e-8，weight_decay=0 |
| 梯度裁剪 | norm=5 |
| 精度 | BF16 mixed；IMF/JVP 部分 FP32 |
| 随机种子 | 模型与数据均为 20260913 |
| 参数组 | prior encoder、duration predictor、speaker embedding、decoder；同一 LR |
| 训练预算 | 500 epoch / 170k optimizer updates |
| 验证与常规保存 | 每 1,000 步 |
| 音频评测 | 1k 小样本 smoke、2k、后续每 5k、最终 170k |

实际过程（UTC）：

| 时间 | 事件 |
| --- | --- |
| 09-15 04:24:49 | 启动当前 3e-4 主实验 |
| 09-15 08:38:57 | 在 24k 暂停，保存模型与完整 Adam 状态 |
| 09-15 09:32:44 | 单独的 5e-4 实验在 5k 暂停 |
| 09-15 10:48:35 | 从主实验 24k checkpoint 恢复，继续原 3e-4 配方 |
| 09-16 12:15:59 | 主实验正常结束于 170k / 500 epoch |

恢复时已核对模型张量和全部 Adam 状态逐位一致，四组 LR 均为 `3e-4`；从 epoch 70 的第 200 个 batch（零基）接着读取，该 epoch 还剩 140 个 batch。没有重置 optimizer、没有重新 warmup、没有从 5e-4 分支接权重。24,001 和 24,100 步的 Adam 计数、LR 也核验通过。

原暂停 checkpoint 没有完整 RNG 状态，因此恢复后的随机轨迹不能声称与“从未暂停”逐位相同。最终所有 Adam state 的 step 都为 170000。

依据：[训练完成状态](/119010446/tts-assets/training_runs/ljs_majestic_100h_imf_h1_b48_from21k_500ep_20260915/training_state.json)、[完整启动/恢复命令](/119010446/tts-assets/training_runs/ljs_majestic_100h_imf_h1_b48_from21k_500ep_20260915/launch.json)、[续训恢复核验](/119010446/tts-assets/training_runs/ljs_majestic_100h_imf_h1_b48_from21k_500ep_20260915/resume_20260915/restoration.json)。

## 5. LR：确切数值、调度与 5e-4 尝试

### 完成训练的这条 run 始终使用峰值 3e-4

采用 **WSD：1k warmup → 稳定段 → 最后 20% 线性衰减**。四个参数组一起变化。

以下 `k` 表示第几次 optimizer 更新，从 **1** 开始，避免把 `global_step` 在更新前后的差一位混在一起：

```text
1 ≤ k ≤ 1000:
    lr(k) = 3e-4 × (0.1 + 0.9 × k / 1000)
1001 ≤ k ≤ 136000:
    lr(k) = 3e-4
136001 ≤ k ≤ 170000:
    lr(k) = 3e-4 + (2e-5 - 3e-4) × (k - 136000) / 34000
```

| 更新步数 k | LR | 阶段 |
| --- | --- | --- |
| 1 | 0.00003027 | warmup 起点，约峰值的 10%，不是从 0 开始 |
| 1,000 | 0.00030000 | 到达峰值 |
| 24,000 / 24,001 | 0.00030000 | 暂停前后保持一致 |
| 136,000 | 0.00030000 | 稳定段最后一步 |
| 140,000 | 0.00026706 | 线性衰减 |
| 150,000 | 0.00018471 | 线性衰减 |
| 160,000 | 0.00010235 | 线性衰减 |
| 170,000 | **0.00002000** | 目标终点，最终 optimizer 实际值已核对 |

中间数值由实际生效 callback 的公式计算，暂停恢复点与最终值另用 optimizer 状态核实。LR 在每个 batch 开始时由 `WarmupStableDecay` 按累计 `global_step` 写入 optimizer；checkpoint 中 `lr_schedulers` 为空不代表没使用调度。

调度来源：[运行快照 schedule.py](/119010446/tts-assets/training_runs/ljs_majestic_100h_imf_h1_b48_from21k_500ep_20260915/source/training/majestic_scratch/schedule.py)、[实际 Hydra 配置](/119010446/tts-assets/training_runs/ljs_majestic_100h_imf_h1_b48_from21k_500ep_20260915/resume_20260915/hydra/config.yaml)。

### 5e-4 是独立短实验，没有混入当前 170k 模型

独立 run：`ljs_majestic_100h_imf_h1_lr5e4_b48_from21k_500ep_20260915`。它从同一个初始化文件的 step 0 开始，初始张量一致、Adam 为空；主要实验变量是峰值 LR 从 `3e-4` 改为 `5e-4`。该 run 在 **5k 暂停**，没有跑到 170k。

可用的相同 2k、650 条完整评测如下；目标 DNSMOS 是目标中文/英文/混读共 450 句的加权均值：

| 2k checkpoint | 目标中文 CER% ↓ | 目标英文 WER% ↓ | 混读 CER% ↓ | 目标 DNSMOS OVRL ↑ |
| --- | --- | --- | --- | --- |
| 峰值 3e-4 | 0.820 | 1.449 | 1.330 | 3.2049 |
| 峰值 5e-4 | 0.714 | 1.255 | 1.241 | 3.2495 |

对应原始评测：[3e-4 / 2k](/119010446/tts-assets/training_runs/ljs_majestic_100h_imf_h1_b48_from21k_500ep_20260915/eval/step_00002000/summary.json)、[5e-4 / 2k](/119010446/tts-assets/training_runs/ljs_majestic_100h_imf_h1_lr5e4_b48_from21k_500ep_20260915/eval/step_00002000/summary.json)。

5e-4 在这个早期点的指标更好，但当时按实验决策暂停，之后恢复原 3e-4 主实验。没有完整的 5e-4 最终对照，因此**不能由此宣称 3e-4 是最优 LR，或 5e-4 最终一定更差**。本次记录只确认实际完成的是哪条训练。

依据：[LR 对照初始化核验](/119010446/tts-assets/training_runs/ljs_majestic_100h_imf_h1_lr5e4_b48_from21k_500ep_20260915/lr_comparison_initialization.json)、[5e-4 暂停核验](/119010446/tts-assets/training_runs/ljs_majestic_100h_imf_h1_lr5e4_b48_from21k_500ep_20260915/pause_verification.json)。

## 6. 当前最终 170k 的结果与限制

### 生成评测

固定 650 条文本，整句合成，IMF 2 步，temperature=1，固定样本种子，原有 Vocos；ASR 为 Qwen3-ASR-1.7B。下表仅为 **IMF 170k**，CER/WER 是 micro 错误率。DNSMOS 是自动评分，不是人工 MOS。

| 分组 | 句数 | CER% ↓ | WER% ↓ | WavLM+ECAPA ↑ | CAMPPlus ↑ | DNSMOS OVRL ↑ |
| --- | --- | --- | --- | --- | --- | --- |
| 目标中文 | 200 | 0.582 | — | 0.7806 | 0.7218 | 3.3685 |
| 目标英文 | 200 | 0.377 | 0.869 | 0.7588 | 0.7879 | 3.2274 |
| 目标混读 | 50 | 0.089 | 6.250 | 0.7290 | 0.6961 | 3.3446 |
| LJS 英文 | 200 | 0.503 | 1.497 | 0.6433 | 0.8373 | 3.0288 |

目标 450 句 DNSMOS OVRL 为 **3.3032**；同为 170k、10 步推理的 FM 为 **3.3614**。最终点上，IMF 目标英文与混读错误率较低，中文 CER 则为 **0.582% 对 FM 0.397%**；LJS 英文 WER 也更高。因此不能沿用 150k 时“目标中文也更好”的判断，更不能认定 IMF 全面超过 FM。

FM 对照来源：[FM 170k 完整评测](/119010446/tts-assets/training_runs/ljs_majestic_100h_backbone21k_20260914/eval/step_00170000/summary.json)。两模型最终评测的 650 条 ID、输入文本、音素和 ASR 分母已核对一致；IMF 的组均值和 micro 错误率已由逐句记录重算确认。

### Duration 检测

固定 632 条验证录音，统计语音 token，排除标点、静音与独立声调。下面使用 **raw `exp(logw)`**，尚未取整/限幅；比较对象是该模型自身 MAS。Std/CV/均值比是预测值与 MAS 对应统计的比值，MAE 单位为帧，1 帧=16ms。

| 分组 | 录音数 | Corr | Std 比 | CV 比 | 均值比 | MAE（帧） |
| --- | --- | --- | --- | --- | --- | --- |
| 目标中文 | 192 | 0.6757 | 0.6689 | 0.7089 | 0.9436 | 1.1066 |
| 目标英文 | 115 | 0.6580 | 0.6629 | 0.7401 | 0.8956 | 1.3081 |
| 目标混读 | 93 | 0.6870 | 0.6834 | 0.7328 | 0.9325 | 1.1846 |
| LJS 英文 | 232 | 0.6506 | 0.6530 | 0.7058 | 0.9252 | 1.3805 |

Std 比仍约为 0.65–0.69，预测时长变化仍比自身 MAS 平缓；中文相关性为 0.6757，不能解释为 duration 问题已经解决。MAS 会随模型变化，也不是人工真值，不能仅凭相关性排名真实节奏准确率。

### 评测文本重复的已知问题

中文生成评测有 200 条记录，但只有 115 条不同文本。`0064` 的原文含“导航导航”，`0114` 含“独立思独立思独立思考”；155k 的 FM 和 IMF 转写都保留这些重复，仍可得到 CER=0。**固定集错误率较低不等于没有重复或更自然。** 这是外部评测文本的核查结果，不是对训练集重复率的统计；也没有排除模型额外生成短音节重复的可能。

此前完整的 150k 对比和 155k 重复核查保留在 [FM / IMF 对比记录](fm_imf_final_comparison_zh.md)。本节是训练结束后的 170k 结果，两者统计时点不同。

## 7. 配置与使用时以什么为准

运行目录：`/119010446/tts-assets/training_runs/ljs_majestic_100h_imf_h1_b48_from21k_500ep_20260915`。

1. **有效模型配置**看最终 checkpoint 的 `hyper_parameters`、本轮 [Hydra 配置](/119010446/tts-assets/training_runs/ljs_majestic_100h_imf_h1_b48_from21k_500ep_20260915/resume_20260915/hydra/config.yaml) 和 `source/` 冻结源码。
2. `data/model_config.json` 随旧 FM 数据资产继承，里面仍有 `cfm.name=CFM` 等旧字段，**不能当作当前 IMF 的实际生效配置**。已核对最终 checkpoint 为 `IMF`、`interval_time_scale=1`、默认 2 步。
3. 直接调用 `model.synthesise(..., n_timesteps=None)` 会读取 checkpoint 的 2 步默认值。当前仓库 `infer_e2e.sh` 入口仍默认 10 步；使用该入口复现 2 步时应显式设置 `N_TIMESTEPS=2`。复现本评测还需要整句模式、FP32、temperature=1，以及相同文本音素和种子，不能只改步数。
4. 本文仅整理已完成实验，没有重新训练、改动数据、修改推理默认值或替换 Vocos。
