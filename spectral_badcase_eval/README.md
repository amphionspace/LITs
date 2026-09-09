# 频谱 Badcase 评测

本目录提供离线评测工具，用于对比合成音频与参考（GT）音频，并定位频谱层面的 badcase。适用于两段 wav 长度不一致的场景。

参考音频来自 **Expresso `wav_24k_default`**（英语，24 kHz）或 **BZNSYP `wav_24k`**（中文，24 kHz）。GT 与合成音频采样率一致，可避免 LibriSpeech 式 16k→24k 上采样带来的 mel 对比伪影。

## 快速开始（英语 — Expresso default，100 条）

在仓库根目录执行：

```bash
SPK_ID=2 CKPT=/path/to/model.ckpt bash spectral_badcase_eval/run_pipeline.sh
```

若 checkpoint 为 tone-embedding 模型（如 `0727_en-zh_1199_inference.ckpt`），需加 `MODEL_LANG=en-zh-dict-toneless-rhyme`：

```bash
MODEL_LANG=en-zh-dict-toneless-rhyme SPK_ID=2 CKPT=/path/to/model.ckpt \
  bash spectral_badcase_eval/run_pipeline.sh
```

流程如下：

1. 从 `dataset/expresso/wav_24k_default/expresso_default_24k.txt`（共 1519 条）随机抽取 100 条
2. 对 `target_text.txt` 运行 `infer_e2e_zh.sh`（会去除 Expresso 的 `*emphasis*` 标记）
3. 构建 manifest 并运行 `evaluate_spectral_badcases.py`
4. 用 `evaluate_mas_duration_violations.py` 检查预测 token 时长是否超出训练 MAS 上下界

## 快速开始（中文 — BZNSYP，100 条）

在仓库根目录执行：

```bash
SPK_ID=0 CKPT=/path/to/model.ckpt bash spectral_badcase_eval/run_pipeline_zh.sh
```

tone-embedding checkpoint 同样需指定 `MODEL_LANG=en-zh-dict-toneless-rhyme`。

流程如下：

1. 从 `dataset/Dataset/zh/BZNSYP/zh_data.txt`（共 10000 条，`wav_24k/` 下 24 kHz wav）随机抽取 100 条
2. 对 `target_text.txt` 运行 `infer_e2e_zh.sh`
3. 构建 manifest（`language=zh`）并运行 `evaluate_spectral_badcases.py`
4. 用 zh354 MAS 上下界检查预测 token 时长

自定义 run id 或样本数：

```bash
NUM_SAMPLES=500 SEED=123 SPK_ID=0 CKPT=/path/to/model.ckpt \
  bash spectral_badcase_eval/run_pipeline_zh.sh bznsyp_rand500
```

仅随机抽样（不推理、不评测）：

```bash
python spectral_badcase_eval/scripts/sample_bznsyp.py \
  --output_dir spectral_badcase_eval/data/bznsyp_rand500 \
  --num_samples 500 --seed 123
```

MAS 时长检查与 `transsion` 分支训练逻辑一致：

- **下界**：LJSpeech ARPA `p01_ms`（× `arpa_duration_floor_scale`）或 zh354 `p01`（× `duration_floor_scale`），且不低于 `duration_floor_min_frames`
- **上界**：king_shahenda ARPA 上界统计，或 zh354 `p95`/`p99` 上界公式

上下界表优先从 checkpoint 超参解析，本地回退路径为 `spectral_badcase_eval/data/duration_stats/`。

说明：`sample_info.csv` 中的 `gt_speaker_id`（如 `ex01`）仅为 Expresso 录音的 GT 元数据；manifest 的 `speaker` 列始终是推理时通过 `SPK_ID` 传入的模型说话人 id。

自定义 run id（英语）：

```bash
SPK_ID=0 CKPT=/path/to/model.ckpt bash spectral_badcase_eval/run_pipeline.sh my_run_id
```

仅重抽样或仅重跑评测：

```bash
# 英语
python spectral_badcase_eval/scripts/sample_expresso.py \
  --output_dir spectral_badcase_eval/data/expresso_default_rand100 \
  --num_samples 100 --seed 42

python spectral_badcase_eval/scripts/prepare_manifest.py \
  --sample_info spectral_badcase_eval/data/expresso_default_rand100/sample_info.csv \
  --synth_dir spectral_badcase_eval/out/expresso_default_rand100/synth_wavs \
  --speaker 0 \
  --output spectral_badcase_eval/manifests/expresso_default_rand100.csv

# 中文
python spectral_badcase_eval/scripts/sample_bznsyp.py \
  --output_dir spectral_badcase_eval/data/bznsyp_rand100 \
  --num_samples 100 --seed 42

python spectral_badcase_eval/scripts/prepare_manifest.py \
  --sample_info spectral_badcase_eval/data/bznsyp_rand100/sample_info.csv \
  --synth_dir spectral_badcase_eval/out/bznsyp_rand100/synth_wavs \
  --speaker 0 \
  --language zh \
  --output spectral_badcase_eval/manifests/bznsyp_rand100.csv

# 仅 MAS 时长上下界检查
python spectral_badcase_eval/evaluate_mas_duration_violations.py \
  --manifest spectral_badcase_eval/manifests/expresso_default_rand100.csv \
  --checkpoint /path/to/model.ckpt \
  --output_dir spectral_badcase_eval/out/expresso_default_rand100/mas_duration \
  --model_lang en-zh-dict
```

## 评测指标

评测器提取 24 kHz log-mel 特征，并用 DTW 对齐 GT 与合成 mel。

**句级指标：**

- `utterance_mel_dtw`：log-mel 上的归一化 DTW 距离
- `utterance_mel_l1_on_path`、`utterance_mel_l2_on_path`、`utterance_mel_cosine_on_path`：沿 DTW 路径的距离
- `utterance_mcd`：沿 DTW 路径的 MFCC/MCD 风格失真
- `duration_ratio`：合成 mel 帧数 / GT mel 帧数

**token 级指标**（需提供 span 文件）：

- `local_mel_dtw`：GT token span 与合成 token span 的 DTW 距离
- `local_mcd`：token 级 MCD 风格失真
- `global_dtw_cost`：整句 DTW 代价中归因于该 GT token span 的平均值
- `global_coverage_ratio`：该 GT token 映射到的合成帧数占比
- `span_iou`：DTW 映射的合成区域与预测合成 token span 的重叠度
- `prior_mel_dtw`：GT token span 与 `mu_y`/prior span 的距离（可选）
- `duration_ratio`：合成 token 帧数 / GT token 帧数

漏读类 badcase 最常见的模式：

```text
duration_ratio 正常，但 local_mel_dtw_z / global_dtw_cost_z 偏高
```

含义：模型给了该 token 的帧，但这些帧在声学上与 GT token 不够接近。

## 输入 Manifest

支持 CSV、TSV、JSONL 或管道符分隔格式。

推荐 CSV 列：

```csv
utt_id,gt_wav,synth_wav,text,language,speaker,gt_spans,synth_spans,prior_mel
case001,/path/to/gt.wav,/path/to/synth.wav,who are you,en,0,/path/to/gt.tokens.tsv,/path/to/synth.tokens.tsv,/path/to/mu_y.npy
```

最简管道符格式：

```text
/path/to/gt.wav|/path/to/synth.wav|reference text
```

若无 `gt_spans` 和 `synth_spans`，仍会输出句级指标与可视化，但 token 级行为空。

## Span 文件

Span 文件可为 CSV、TSV 或 JSONL。必填字段为帧边界：

```csv
index,token,start_frame,end_frame
0,HH,120,125
1,UW1,125,134
```

或时间边界：

```csv
index,token,start_sec,end_sec
0,HH,1.920,2.000
1,UW1,2.000,2.144
```

`tools/visualize_token_mel.py` 生成的 token TSV 可作为合成 span 文件。

## 单独运行评测

在仓库根目录执行：

```bash
python spectral_badcase_eval/evaluate_spectral_badcases.py \
  --manifest path/to/manifest.csv \
  --output_dir path/to/spectral_eval_out
```

可选：用已知正常 run 做校准：

```bash
python spectral_badcase_eval/evaluate_spectral_badcases.py \
  --manifest path/to/manifest.csv \
  --output_dir path/to/spectral_eval_out \
  --calibration_token_csv path/to/good_run/token_metrics.csv
```

## 输出文件

每次 run 的输出都在 `spectral_badcase_eval/out/<run_id>/` 下，包含：

- `synth_wavs/`：本次评测的全部合成 wav（`1.wav`, `2.wav`, ...），不再写入 `infer_output/`
- `badcase_wavs/`：可疑 badcase 的 GT/合成 wav 软链接及对应文本，如 `001_<utt_id>_gt.wav`、`001_<utt_id>_synth.wav`、`001_<utt_id>_text.txt`，附 `index.tsv` 索引
- `utterance_metrics.csv`：每句一行
- `token_metrics.csv`：有 span 时每个 token 一行
- `flagged_tokens.csv`：`flag_reason` 非空的 token 行
- `run_summary.txt`：汇总摘要
- `mel_cache/`：缓存的 GT/合成 log-mel 数组
- `badcase_figures/`：按 `utterance_mel_dtw`（或 token anomaly score）排序的 top-K mel 对比图（默认 20 张）

可疑 badcase 选取规则与 figure 一致：优先 `max_token_anomaly_score`，否则按 `utterance_mel_dtw` 取 top-K；另包含 `num_flagged_tokens > 0` 或 MAS duration 违规的句子。`badcase_figures` 与 `badcase_wavs` 数量由 `TOP_K`（默认 20）统一控制。

## 前端 Flag：OOV 逐字母拼读

独立于频谱 badcase，pipeline 会扫描英文词是否因 OOV 走了逐字母拼读（`spell` fallback），但原词**不是全大写**（全大写如 `NASA` 视为有意拼读，不 flag）。

输出在 `out/<run_id>/frontend_flags/`：

- `spell_fallback.csv`：全部句子及 `has_spell_fallback` / `flagged_words`
- `spell_fallback_flagged_only.csv`：仅命中句子
- `flagged_words.txt`：本次 run 被 flag 的词，格式 `utt_id<TAB>word`，一行一条
- `flagged_wavs/`：命中句子的 GT/合成 wav 软链接及带标注的 `*_text.txt`

单独运行：

```bash
python spectral_badcase_eval/scripts/detect_spell_fallback.py \
  --sample_info spectral_badcase_eval/data/expresso_default_rand100/sample_info.csv \
  --synth_dir spectral_badcase_eval/out/expresso_default_rand100/synth_wavs \
  --output_dir spectral_badcase_eval/out/expresso_default_rand100/frontend_flags
```

## 注意事项

- DTW 可处理 GT/合成长度不等；除非已对齐，否则不要直接比较原始帧索引。
- token 级定位需提供可靠的 GT span（MFA 或 MAS）及合成 span（推理 duration/attention）。
- 阈值为稳健 z-score；对比不同 checkpoint 时，建议用正常 checkpoint 或人工验收 run 的 `--calibration_token_csv`。
- mel 前端用 NumPy/SciPy 本地实现，评测时不依赖训练环境。
