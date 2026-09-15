# LJSpeech + MajesticVoice 24 kHz 训练

这是两个说话人的中英文声学模型：speaker 0 为历史 LJSpeech 英文音色，speaker 1 为 MajesticVoice 的中文及新合成英文。Vocos 使用仓库的 24 kHz 预训练生成器，LITs 从随机权重开始。

训练目录：`/119010446/tts-assets/training_runs/ljs_majestic_24k_20260910_164600`，2026-09-10 16:46 UTC 启动。2026-09-11 按用户要求暂停训练和自动评估，尚未恢复。可恢复快照为 `checkpoints/paused_resume_20260911.ckpt`，保存于 step 27456（epoch 95），包含优化器与训练循环状态；停止时最后记录为 step 27550，尚未保存的约94步需在恢复后重跑。`training_state.json` 为训练状态，`validation.jsonl` 为验证历史，`tensorboard/` 保存曲线，`eval/latest_eval.json` 为最近评估状态。

首个 epoch 已完成（step 286），验证总 loss 为 3.44463，时长 / prior / flow 损失分别为 0.33584 / 1.03789 / 2.07090。第一轮 32 条评估的合成、ASR 和音色/音质计算均完成，执行失败数为零。此时模型尚未学会清晰发音，各组 CER 为 1.0；这是评估链路检查结果，不代表音质已合格。后续完整 650 条评估继续按 checkpoint 进度自动执行。

## 数据与准入

- 数据准备目录：`/119010446/tts-assets/data_24k/ljs_majestic`。
- 上述目录现在只保存训练/验证清单、统计和检查记录。音频已迁移到 `/ai_sds_wuzz/DATA_TTS/`：原始 LJS 的24 kHz版本在 `LJSpeech/LJSpeech_24k/wavs`，合成 LJS 的16/24 kHz版本在 `LJSpeech_Synthetic/16k/wavs` 和 `LJSpeech_Synthetic/24k/wavs`，MajesticVoice 保持原位置。
- 迁移按 SHA-256 核验39,978个音频文件，训练内容、顺序、时间片段、采样权重和 Mel 均值/标准差不变。清单的路径及文件哈希已更新，完整旧清单、新旧路径表和验证结果位于 `/119010446/tts-assets/data_operations/20260911/`。固定650条评估文本及其哈希不变，参考音频路径从 `eval_protocol.json` 读取。
- MajesticVoice 中文：`/ai_sds_wuzz/DATA_TTS/MajesticVoice`，9,426 条通过质检，约 9.8446 小时。
- MajesticVoice 英文：上述目录的 `english/`，必须完成生成与质检，并至少提供 2,756 条训练用的不同文本。
- 两份生成数据都必须有 `quality/final_audit.json`，状态为 `passed`。训练脚本不会使用未通过质检的候选。
- 历史 LJSpeech 训练清单 27,448 行、验证 232 行。100 个长音频路径对应多个起止时间不同的片段，必须保留五列清单，不能当成同一句的重复采样。

`prepare_training.py` 保留中文原始测试划分，从中文训练池抽取 200 条验证样本。英文保留单独验证和内部测试数据；固定外部评估还有 200 条英文文本。训练集与全部保留集按当前前端的音素序列检查交集，冲突的训练行排除并记录。两位说话人按样本数配平，MajesticVoice 池使用固定种子的有放回抽样。实际数量以 `training_summary.json` 为准。

2026-09-10 实际准备结果：英文 3,001 条通过音频验收，合计 3.3602 小时；其中 6 条不能被当前文本前端正确编码，单列于 `text_frontend_exclusions.json`，不进入训练。最终 MajesticVoice 训练池为中文 8,278 条、英文 2,756 条。排除与保留文本冲突的 54 行历史 LJS 数据后，两位说话人各采样 27,394 行，共 54,788 行；其中 LJS 英文 27,394、MajesticVoice 中文 20,665、英文 6,729 行。验证 464 行，内部测试 1,004 行；训练与保留集的音频路径和音素交集均为零，验证与内部测试的音素交集也为零。

清单支持 `wav|speaker_id|text` 和 `wav|speaker_id|start|end|text`，使用绝对路径。所有音频为 24 kHz 单声道 PCM16。LJS 原录音从 22.05 kHz 原始文件转换；历史合成英文可用版本为 16 kHz，上采样不会补回高频信息。

## 训练配置

参考归档 checkpoint 的数据组织及有效 batch，使用当前仓库的模型配置。旧 checkpoint 是 16 kHz、80 Mel、150 tokens，不能直接恢复为这次的 24 kHz、100 Mel、173-token 模型。

| 项目 | 配置 |
|---|---|
| 声学模型 | 当前 en-zh 的 RoPE encoder + duration predictor + Flow Matching |
| Mel | 24,000 Hz，100 bins；FFT 2048 / hop 384 / window 1536；0–12,000 Hz |
| 归一化 | 按实际训练行及时间片段计算加权 mean/std，排除验证和测试 |
| 词表 / speaker | 173 / 2；保留 inline tone 与 MAS 声调约束 |
| 初始化 | `ckpt_path=null init_ckpt_path=null` |
| 训练 | 2 张 A800，BF16，每卡 batch 32，累积 3，有效 batch 192 |
| 优化器 | Adam，lr=1e-4，梯度裁剪 5.0 |
| 预算 | 上限 200,000 optimizer steps |
| 验证 / 保存 | 每 epoch 验证；每 5 epoch 保存常规 checkpoint，保留最近 3 份及 best-prior |

沿用仓库的 encoder/duration plateau 降学习率回调。`training_state.json` 每 10 step 更新，`validation.jsonl` 记录各轮验证损失，非有限损失会使任务报错。

实际 Mel mean 为 `-5.159219438279001`，std 为 `2.413289874150853`；训练包含 38,428 个不同时间片段，去除采样重复后的音频约 40.4379 小时。统计和 MAS 长度检查报告见数据目录的 `mel_statistics.json`。

## 评估

固定 650 条评估输入取自历史实验保存的清单：LJSpeech 英文 200、MajesticVoice 英文 200、MajesticVoice 中文 200、中英混合 50。前两组使用同一批英文文本、不同 speaker ID。

`watch_eval.py` 复制保存完成的 checkpoint，记录 SHA256 后评估；第一份 checkpoint 每组取 8 条，共 32 条，用于检查完整评估链路。之后至少间隔 2,000 step 评估完整 650 条，训练结束时评估最新一份。评估与训练同时运行，评估按阶段串行使用 GPU 0。

- 当前 LITs 推理接口，固定种子和 10 ODE steps，24 kHz Vocos。
- Qwen3-ASR-1.7B：英文 WER、中文及混合 CER；保留原始识别文本。
- WavLM、CAM++：对两位说话人各自的固定参考音频计算相似度。
- DNSMOS：OVRL、SIG、BAK，并记录 P.808。

这是固定音色的文本评估，音色参考不是 Seed-TTS 原始的零样本提示说话人。每个 checkpoint 的音频、逐句结果、阶段日志和汇总保存在 `eval/step_XXXXXXXX/`；评估失败也会记录，不能把失败样本当作通过。

## 运行

训练环境使用已解压且完成导入核验的：

```text
/119010446/tts-assets/chenmingjie/lits-tts/envs/lits/bin/python
```

首次使用需编译仓库的 MAS 扩展：

```bash
cd /119010446/LITs/lits/utils/monotonic_align
/119010446/tts-assets/chenmingjie/lits-tts/envs/lits/bin/python setup.py build_ext --inplace
```

依次完成以下步骤，英文验收和最终 Mel 统计完成前不要启动训练：

```bash
cd /119010446/LITs/training/majestic_voice
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
TASK_PYTHON=/119010446/tts-assets/chenmingjie/lits-tts/envs/lits/bin/python
"$TASK_PYTHON" prepare_eval.py
"$TASK_PYTHON" prepare_training.py
"$TASK_PYTHON" compute_statistics.py
"$TASK_PYTHON" run_training.py --run-dir /119010446/tts-assets/training_runs/ljs_majestic_24k_RUN_ID
```

`run_training.py` 使用仓库现有 `training.sh`，同时启动评估监听进程。实际命令和进程号写入 `launch.json`，Hydra 保存完整配置。重复使用已启动的目录会被拒绝，恢复训练应显式使用该次运行的 checkpoint，不能误启动另一份从头训练。

本次补齐模型依赖的声调统计辅助函数，把环境变量传入的 speaker 数和 Mel 统计转换为数值，并在 MAS 转 NumPy 前转成 FP32 以兼容 BF16。14 项启动、声调与对齐测试通过；Vocos 已使用真实中英文音频完成重建预检，报告位于数据准备目录的 `vocos_preflight/report.json`。

启动验证还修复了当前 Matplotlib 已移除 `tostring_rgb` 导致的 Mel 绘图错误，改用 RGBA 缓冲区转 RGB；真实图像转换及随后完整启动验证已通过。早期失败尝试保留在 `ljs_majestic_24k_20260910_1645`，该尝试没有执行训练更新，当前运行是上面的 `164600` 目录。
