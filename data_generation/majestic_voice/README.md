# MajesticVoice 数据生成

用 `TS-004_大气女声` 的参考音频和 VoxCPM2，重合成原始文本集的全部 10,000 条文本。原始训练/测试划分为 9,000/1,000；原音频仅在准备阶段检查路径，不作为目标声音输入。中文已完成验收，保留 9,426 条。后续补齐英文，再按 `training/majestic_voice/README.md` 启动双音色训练和评估。

## 目录

- 代码：本目录。
- 模型：`/119010446/tts-assets/{VoxCPM2,Qwen3-ASR-1.7B,CAMPlus,WavLM-Large,DNSMOS}`。
- 数据：`/ai_sds_wuzz/DATA_TTS/MajesticVoice`。
- `reference/`：48 kHz 原始参考音频与去除韵律标记的转写。
- `candidates/wavs_{48k,24k}/`：候选隔离区；不直接用于训练。
- `wavs_{48k,24k}/{train,test}/`：通过全部质检的最终音频。
- `train.txt`、`test.txt`：`wavs_24k/train/0|文本` 格式；音频路径相对数据根目录，省略 `.wav`。`*_48k.txt` 对应 48 kHz。
- `quality/selected.jsonl`：每条入选音频对应的候选、ASR 转写、CER、相似度和 DNSMOS。
- `quality/candidates_scored.jsonl`：运行完成后的全部候选评分和拒绝原因。
- `pipeline_progress.json`、`quality/selection_summary.json`、`logs/`：进度、入库数量和日志。

## 流程与阈值

每张 GPU 启动一个单卡 vLLM-Omni 服务。每条文本每轮生成 2 个候选；ASR 和音色/音质检查独立消费已落盘候选，模型常驻。仅对没有合格候选的文本重试，最多 3 轮（每条最多 6 个候选）。全部轮次后仍不合格的文本从最终清单排除，明确报告数量和 ID。隔离区保留候选以便审计；最终文件使用硬链接，不重复占用音频空间。

固定门槛见 `quality_thresholds.json`：文本满足 CER ≤ 0.05 **或中文带声调拼音完全一致**；同时 WavLM ≥ 0.60、CAM++ ≥ 0.60、DNSMOS OVRL ≥ 3.0、SIG ≥ 3.2、BAK ≥ 3.5。用户确认接受声调一致的同音字，例如“假语村言”与“贾雨村言”；声调或音节不同不能通过同音字规则。合格候选按 `2×WavLM + CAM + 0.25×OVRL − 2×有效CER` 择优，带声调拼音完全一致时有效 CER 为 0，原始汉字 CER 始终保留。这是本项目的筛选配置，不是通用模型标准；运行过程中不自动降低门槛。

Qwen-ASR 不接收目标文本作为提示，避免影响转写。CER 对 Unicode NFKC、大小写、标点和空白作统一处理。同音字核验使用 pypinyin 0.55.0 的带声调音节序列（轻声记作 5），仅适用于纯中文文本；未知读音不放行，数字与英文仍按原 CER 处理。DNSMOS 按 16 kHz、9.01 秒窗口、1 秒步长计算，窗口数严格使用官方脚本公式；短音频重复到窗口长度，再取窗口平均值。P.808 分数同时记录，未作为硬门槛。

WavLM 采用 UltraEval-Audio 的 Seed-TTS SIM-O 路径（WavLM-Large + ECAPA）；CAM++ 使用 80 维均值归一化 Kaldi FBank。相似度相对同一个生成参考音频。`quality/calibration.json` 保存了 TS-004 和另一声音 TS-050 的本地校准结果。

音色评分仅对等长音频组批，避免填充改变评分；使用 TF32 加速矩阵运算，32 个分散候选的最大相似度差为 0.0000474，距准入阈值 0.002 内的结果会用严格 FP32 重新计算。DNSMOS 使用 8 个 CPU 工作线程，每个 ONNX 会话和 OpenBLAS 运算限制为单线程，避免小矩阵运算争抢 CPU。

## 环境与运行

TTS 环境：`tts-assets/.venv-vllm-voxcpm2`，Python 3.10、vLLM 0.20.0+cu129、vLLM-Omni 0.20.0、PyTorch/torchaudio 2.11.0+cu129、VoxCPM 2.0.3、Transformers 4.57.6、torchcodec 0.11.0。安装依赖检查通过。Omni 的 `output_modality.py` 在 Python 3.10 下需使用 `backports.strenum`，补丁记录在 `omni_py310.patch`。系统安装 ffmpeg。

ASR/控制脚本：`tts-assets/.venv-voxcpm2`（Qwen-ASR 0.0.6、PyTorch 2.10）。音色/DNSMOS：现有 `UltraEval-Audio/envs/metrics/bin/python`；未修改该环境。两个质检进程分别使用 GPU 1、GPU 0，TTS 服务预留显存供质检使用。

```bash
cd /119010446/LITs/data_generation/majestic_voice
/119010446/tts-assets/.venv-voxcpm2/bin/python prepare_data.py
bash serve.sh 0 8100 > /ai_sds_wuzz/DATA_TTS/MajesticVoice/logs/server-gpu0.log 2>&1 &
bash serve.sh 1 8101 > /ai_sds_wuzz/DATA_TTS/MajesticVoice/logs/server-gpu1.log 2>&1 &
# 两个 /health 接口就绪后执行：
/119010446/tts-assets/.venv-voxcpm2/bin/python run_pipeline.py
# 全部轮次完成后核对最终入库文件与分数：
/119010446/tts-assets/.venv-voxcpm2/bin/python audit_dataset.py
```

脚本支持中断后重跑：已完整保存的候选和已评分候选会跳过，最终入库清单按通过门槛的候选重建。不要同时运行两个控制进程。若修改参考、文本、模型或阈值，应使用新的数据输出目录，避免混入旧评分。

路径可通过 `MAJESTIC_VOICE_DATA_ROOT`、`MAJESTIC_VOICE_ASSETS_ROOT`、`ULTRAEVAL_AUDIO_ROOT`、`MAJESTIC_METRICS_PYTHON` 覆盖。

## 英文补齐

英文单独写入数据根目录的 `english/`。固定使用同一条 11.7815 秒中文 reference `ts004_cn_000001.wav`；32 条英文文本、每种参考 2 个候选的比较中，中文参考通过 18 条，中英混合参考通过 0 条，因此选用中文参考。比较结果保存在 `english_reference_probe/quality/probe_comparison.json`。

英文使用 `quality_thresholds_en.json`，额外要求 WER ≤ 0.05，不使用中文同音字豁免，其余音色和音质阈值相同。目标为最终训练集提供 2,756 条不同音素序列的合格英文，另留验证和测试样本；实际划分还要排除跨音色的保留文本及固定外部评估文本。

```bash
export MAJESTIC_VOICE_DATA_ROOT=/ai_sds_wuzz/DATA_TTS/MajesticVoice/english
export MAJESTIC_QUALITY_CONFIG=/119010446/LITs/data_generation/majestic_voice/quality_thresholds_en.json
/119010446/tts-assets/.venv-voxcpm2/bin/python run_pipeline.py
/119010446/tts-assets/.venv-voxcpm2/bin/python audit_dataset.py
```

若三轮后数量不足，完成验收后可执行 `prepare_english.py --root "$MAJESTIC_VOICE_DATA_ROOT" --extend-train 750 --extend-test 80` 添加备用文本，然后重跑上述两个命令。扩充会备份原请求和验收报告，保留原样本 ID、参考及评分。只在整轮候选全部评分后判断数量是否达标，最多仍为每条 6 个候选。

模型与算法来源：[VoxCPM](https://github.com/OpenBMB/VoxCPM)、[vLLM-Omni](https://github.com/vllm-project/vllm-omni)、[Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)、[DNS-Challenge](https://github.com/microsoft/DNS-Challenge)。
