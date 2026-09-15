# HiFiTTS + Premium 两阶段训练

已完成的 3e-4、200,000-step 实验详见 [Stage 1 训练实录](../../docs/stage1_training_zh.md)，包含数据、模型、loss、优化器、评测及 197k 权重迁移依据。下文保留早期 1e-4 配方和阶段规划的历史记录；正式实验及后续配置以实录中注明的运行记录为准。

第一阶段从随机权重开始，使用 HiFiTTS 英文和 WenetSpeech4TTS Premium 中文学习发音、对齐和韵律。全部样本使用 speaker ID 0。模型保留两个 embedding 槽位以兼容当前条件网络，ID 1 仅供第二阶段的大气女声使用；第一阶段不使用该 ID，也不加载历史 checkpoint。

## 第一阶段数据

预处理结果在 `/119010446/tts-assets/data_24k/foundation/`。`dataset.sqlite` 为只读索引，保存绝对音频路径、原始文本、173-token 前端编码、声调、时长与划分。读取时通过 soxr HQ 转为 24 kHz；原音频不改写。Mel 使用 100 bins、FFT 2048、hop 384、window 1536、0–12 kHz。归一化采用固定种子从每个来源各取 2,048 条训练音频估计，并在本次运行全程固定，具体值见 `mel_statistics.json`。

| 来源 | 训练条数 | 训练小时 | 验证条数 | 测试条数 |
|---|---:|---:|---:|---:|
| HiFiTTS | 320,897 | 289.21 | 494 | 991 |
| Premium | 388,311 | 857.66 | 2,396 | 2,081 |

HiFiTTS 保留官方划分；Premium 按原录制 ID 哈希划分，避免同一录制的不同切片跨集合。保留 0.5–20 秒完整音频，不做破坏文本对应关系的截断；排除 12,983 条时长不符合要求的样本和 274 条空/未知编码样本。训练排除与验证、测试、已有大气女声保留集及固定外部评估文本的音素重叠，共 3,026 条；另删除 19 条验证/测试文本重叠。所有排除原因保存在 `exclusions.jsonl`，数量与来源汇总见 `summary.json`。

训练时对两个来源按样本数 1:1 配平，每轮遍历较大的中文池，并重复采样较小的英文池；丢弃不足一个全局 batch 的尾部。先在全局随机窗口中按时长分桶，再分配到四个 rank；每个 rank 的步数一致。验证固定抽取每个来源 256 条，完整保留集仍在数据库中。

## 启动与容量

环境：`/119010446/tts-assets/chenmingjie/lits-tts/envs/lits/bin/python`。

`start_pipeline.py` 依次运行统计、容量测试和正式训练。`benchmark.py` 使用接近 20 秒且音素较多的真实样本执行完整前向、反向及 Adam 更新，以峰值显存选择每卡 batch；容量测试的模型权重丢弃。`capacity.json` 保留每个候选的耗时与显存。正式训练配置、数据摘要及命令保存到 run 目录的 `launch.json`，源代码与配置快照在 `source/`，完整 Hydra 配置在 `.hydra/`。

正式训练使用四张 A800、BF16、Adam、梯度裁剪 5，无梯度累积。学习率用前 1,000 optimizer steps 从约 1e-5 预热到 1e-4，之后余弦下降到 2e-5，预算上限 200,000 steps。当前阶段不自动冻结 encoder、不自动回退到旧的 best-prior 模型。训练实际启动必须同时满足 `ckpt_path=null` 和 `init_ckpt_path=null`。

2026-09-11 的 LR 重训实验使用 `--peak-lr 0.0003 --final-lr 0.00006`：前 1,000 步从约 3e-5 预热到 3e-4，之后余弦下降到 6e-5。`run.py` 将这两个参数同时写入优化器和调度器配置，避免 callback 覆盖手动设置的 LR。CLI 默认值保留旧配方，重训必须显式传参。此次重训保持数据、有效 batch 576、随机种子、模型与评估设置，并从第 0 步使用 valid-frame padding loss 修复。

```bash
/119010446/tts-assets/chenmingjie/lits-tts/envs/lits/bin/python training/foundation/run.py \
  --run-dir /119010446/tts-assets/training_runs/hifitts_premium_stage1_lr3e-4_20260911 \
  --run-name hifitts_premium_stage1_lr3e-4_fromscratch \
  --peak-lr 0.0003 --final-lr 0.00006
```

历史 checkpoint 核验：归档 `lits-en-zh.ckpt`（step 195300）实际 Adam LR 为 2e-4；7 月 16 日 `balanced_16k_biaobei_ljs_voxcpm2en2756_bs96_acc1_workers16`（step 31350）为 3e-4。两者都在保存的优化器状态中确认，且没有保存的 scheduler。旧 16 kHz/80-mel/150-token 实验与当前 24 kHz/100-mel/173-token 实验存在配方差异，不能据此认定 LR 是生成质量问题的唯一原因。

每 10 步写 `training_state.json`；每 1,000 步验证并保存 checkpoint，常规 checkpoint 保留最近 3 份及 last，另保存最佳验证 prior。首次 checkpoint 进行每组 8 条评估，之后间隔至少 2,000 步做 450 条固定文本评估（英文 200、中文 200、混合 50），统一 speaker 0。评估使用 10 ODE steps + 24 kHz Vocos、Qwen3-ASR、DNSMOS；第一阶段不把与大气女声的音色相似度作为质量目标。评估进程使用 GPU 0，因此容量测试为 DDP 与评估保留显存余量。

## 第二阶段策略

2026-09-13 用户明确指定并授权第二阶段训练：LJSpeech 为 speaker 0，大气女声为 speaker 1。当前采用 LJSpeech 原录音与已验收的 MajesticVoice 中英文，两个说话人按样本数 1:1 配平。从第一阶段 step 197000 迁移权重，两个 speaker embedding 均以已训练的 ID 0 初始化，保留第一阶段 Mel 归一化；先适配音色，再用低学习率联合微调。具体数据、6,000 步首轮预算、分组学习率、预检和评估见 [第二阶段方案](../stage2/README.md)。

## 校验

`tests/test_foundation_data.py` 检查四卡分桶无跨 rank 的意外重复、步数一致、epoch 洗牌与来源配平。GPU 容量测试覆盖真实音频、重采样、Mel、文本/声调、MAS、前向、反向与优化器更新；正式四卡试跑另核实 loss 有限和训练进度增长。

## 本次运行

运行目录：`/119010446/tts-assets/training_runs/hifitts_premium_stage1_20260911`。每卡 batch 144，有效 batch 576；最长批次单卡实测峰值 67.37 GiB，为 DDP 和自动评估保留余量。训练中的显存和耗时随长度分桶变化。状态与启动核验见运行目录的 `training_state.json` 和 `startup_verification.json`。

对齐 Cython 内核启用 OpenMP，并对不会抛出 Python 异常的内部函数使用 `noexcept nogil`，避免每个 DP 单元反复获取 GIL。23 组路径/分数数组与原版逐元素一致；16 条长样本的 CPU 对齐耗时从 2.456 秒降至 0.029 秒。另有小规模穷举最优路径测试，连同分桶、声调和启动测试共 14 项通过。
