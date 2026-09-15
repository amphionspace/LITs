# LJSpeech + MajesticVoice：用 21k 声学基础初始化，重新联合训练

实际执行记录：当前使用 Stage 1 21k 声学权重初始化，重置 speaker，所有组峰值 LR 3e-4；冻结数据为 65,282 条，每轮 340 步，500 epoch 共 170,000 步。完整说明见 [Stage 1 与当前模型训练实录](../../docs/majestic_training_record_zh.md)。

## 初始化选择及依据

选中的实际 checkpoint：

`/119010446/tts-assets/training_runs/hifitts_premium_stage1_lr3e-4_20260911/eval/step_00021000/checkpoint.ckpt`

SHA-256：`1bbf20d7b741c11c52db74901a5bfd7655b6e4e38ffe610bcf1418454c7da86b`。

20k 常规 checkpoint 已不在保留集中，21k 是附近完整保存的评测快照。以下为旧运行真实验证日志：

| Step | Duration | Prior | Flow | Total |
|---:|---:|---:|---:|---:|
| 10k | 0.37250 | 1.00402 | 0.25541 | 1.63193 |
| 20k | 0.35476 | 1.00164 | 0.22280 | 1.57921 |
| 21k | 0.35269 | 1.00180 | 0.21931 | 1.57380 |
| 40k | 0.35207 | 1.00029 | 0.20417 | 1.55653 |
| 60k | 0.34652 | 0.99923 | 0.21195 | 1.55770 |
| 100k | 0.34206 | 0.99942 | 0.21220 | 1.55368 |

20k 后 prior/duration 已明显放缓；flow 的 21–30k、51–60k 窗口均值分别为 0.22049、0.20358，仍有改善。91–100k 为 0.21031，之后不是持续单调下降。这里使用 21k 是为了复用基础声学参数，不声称它已经具备可靠对齐、优于后期模型，或已测得具体训练加速倍数。

额外读取了 21k/41k/61k/99k/197k，在同一批 48 条新域验证样本上计算 prior 和 duration 指标，包含重置 speaker 两行的影响。这个小样本诊断没有做优化或 A/B 训练；MAS 是各模型派生目标，跨模型 Corr 不能当作独立对齐精度。报告见 [初始化选择依据](/119010446/tts-assets/data_generation/majestic_200h_20260913/reports/backbone_initialization_review/selection.json)。

## 加载与重置

- 加载文本编码器、prior/mu 投影、两层 Conv duration head、条件编码器和 flow 网络的权重；逐项校验与源 checkpoint 一致。
- 两个 64 维 speaker embedding 独立随机初始化，使用新训练种子；不复制旧 speaker row，不引入 CAM++。
- 新建 Adam，优化器状态为空；新调度从新 step 0 开始。`ckpt_path=null`，仅通过 `init_ckpt_path` 加载准备后的纯模型状态。
- 保留源 checkpoint 的 Mel mean/std，保持已有声学参数面对的归一化尺度。原先从零方案的“重估新训练集统计”不用于此初始化模式。
- 本次训练的所有模块仍从第一步联合更新，不加 embedding-only、prior warmup 或冻结阶段。

## 数据与采样

| Speaker | 数据 | 训练量 |
|---|---|---:|
| 0 | LJSpeech 原始录音 | 11,550 条，21.0475 小时 |
| 1 | MajesticVoice 中文 | 50 小时 |
| 1 | MajesticVoice 英文 | 25 小时 |
| 1 | MajesticVoice 中英混读 | 25 小时 |

合计约 121.05 小时。LJS 历史合成音频不纳入；复用其 232 条验证录音，既有划分没有单独 LJS 内部测试集。MajesticVoice val/test 额外计算，仍须满足全部九项配额；不能用 LJS 英文抵扣目标音色英文 25 小时。

按唯一训练样本自然混合、全局时长分桶，不做两 speaker 1:1 补采样。实际 65,282 条，目标音色/LJS 按条数 82.31%/17.69%，每轮 340 updates。最终 `data/sampling_plan.json` 记录实际样本占比、音频时长占比、每轮更新数和预算对应轮数。分桶后每批不要求固定比例。

## 训练配方

| 项目 | 当前配置 |
|---|---|
| 结构与前端 | 保留当前 LITs、173-token rhyme-body-tone、两层 Conv duration head |
| Speaker 条件 | 两行 64d learnable embedding；0=LJS、1=目标音色 |
| 声学参数 | 24 kHz，100 Mel，FFT 2048，window 1536，hop 384 |
| 对齐 | 当前 constrained online MAS；不提取固定 duration |
| Loss | duration log-MSE + Gaussian prior + flow matching，始终 1:1:1 |
| 优化器 | Adam，betas 0.9/0.999，eps 1e-8，weight decay 0 |
| LR | 所有组相同；1000 步约 3e-5 → 3e-4，稳定至总步数 80%，最后 20% 线性降至 2e-5 |
| 预算上限 | 500 epoch；实际 170,000 optimizer steps，按冻结清单计算 |
| 并行 | 4 GPU × batch 48 = 192，无梯度累积，BF16 mixed，clip norm 5 |
| 更新范围 | 全部声学模块从新 step 0 联合更新 |
| 停止与延长 | 500 epoch 为上限，根据生成与试听选择中间 checkpoint；不因单次随机 flow loss 自动停训 |

这不是完整 resume。旧 21k 的 Adam、旧 LR 数值和旧调度位置均不继承；也没有为 backbone 单独降低 LR 或提高 speaker LR。

## 验证与评测

每 1000 步计算完整固定验证集 loss、保存常规 checkpoint。1k 做每组 8 条的链路检查；2k 开始做完整评测，随后每 5k，覆盖 5k/10k/20k/50k 和最终预算节点。精确评测 checkpoint 单独保留，队列按顺序处理，避免被最近 checkpoint 清理覆盖。

固定生成评测共 650 条：LJS 英文 200、目标英文 200、目标中文 200、目标混读 50。speaker/语言参考分别匹配，参考文件及哈希冻结。目标英文使用已批准的英文参考，目标中文/混读使用原中文参考，LJS 使用自己的参考录音。

Duration 诊断按四组单独报告：原始 exp(logw)、向上取整、实际 tone 限幅后的推理时长；speech、实际受监督 speech、tone 和全 token 子集。统计 Corr、Std(pred)/Std(MAS)、CV ratio、均值比例、MAE、按句中位数和分位数、每句均值归一化后的结果，以及 tone 帧数直方图和约束命中统计。

当前 tone 的 MAS 实际下限通常为 2 帧、上限 3 帧；合法整数本身就在边界，不能仅因边界命中就判坏。保持原设置并检查实际分配。Corr/Std ratio 接近某个数字不单独作为验收标准，必须结合 CER/WER、漏字重字、停顿和试听。

VoxCPM2 当前接口没有与 LITs 173-token 序列逐项对应的 phoneme duration，因此仍使用在线 MAS；旧 checkpoint 也不作为固定 duration teacher。

## 自动交接与工程核验

训练目录：`/119010446/tts-assets/training_runs/ljs_majestic_100h_backbone21k_20260914`。

运行中的生成主进程会在交接时重新读取 `config.json.training`。既有 prepare/preflight/run 入口支持 `mode=backbone_init`，不用中断生成。英文候选现为最多 4 个、每轮 2 个，有合格结果即停止；全部质检阈值不变。

交接流程：全部配额满足 → 在途候选完成 → 释放生成 GPU → 全量数据与联合划分审计 → 校验源 checkpoint 哈希 → 准备纯模型初始化、重置 speaker → 最长真实批次 GPU 预检 → 正式四卡训练。

启动核验全部参数指纹、空 Adam、新 step 0、源归一化、固定 loss 权重。第 1/100 步检查各声学模块更新，第 100 步确认两行 speaker 均已更新。工程冒烟测试只做小样本单步更新，不构成 scratch/迁移 A/B 实验，也不替代正式数据完成后的 GPU 预检。

最新预算修订：`max_epochs=500`，准备时按 `floor(唯一训练条数 / 192) × 500` 计算 `max_steps`。实际 170,000 steps；已按最终清单冻结。所有组峰值 3e-4、末值 2e-5、预热 1000 steps；随后保持峰值，到总预算的 80% 开始线性衰减，最后一次更新达到末值。调度实现为 `training.majestic_scratch.schedule.WarmupStableDecay`。
