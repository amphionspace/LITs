# MFA → 354 中文 token 时长统计

用 [Montreal Forced Aligner (MFA)](https://montreal-forced-aligner.readthedocs.io/) 对中文语料做音素对齐，再把 MFA 音素时长聚合到 **354 词表**（21 声母 + 225 带调韵母 + `ㄦ˙`）的 mel frame 尺度（hop=384, sr=24k）。

## 目录结构

```
mfa_zh354_duration/
├── README.md                 # 本文件
├── run.py                    # 主入口：align + stats
├── map.py                    # 354 token ↔ MFA IPA 映射（含上下文规则）
├── mapping/
│   └── mfa_zh354_token_mapping.tsv   # 静态映射表（246 行）
└── runs/
    └── bznsyp_10k/           # 示例：BZNSYP 10000 句
        ├── summary.json
        ├── zh354_duration_stats.tsv  # 主结果
        ├── mfa_corpus/       # 送给 MFA 的 wav/lab（符号链接）
        └── mfa_align/        # MFA TextGrid 输出
```

## 环境依赖

```bash
source /chenmingjie/xingwen/get_mamba.sh
micromamba activate lits_5090

# MFA + 中文分词
micromamba install -c conda-forge montreal-forced-aligner
pip install spacy-pkuseg dragonmapper hanziconv textgrid
```

MFA 预训练模型（需提前下载到固定路径）：

| 组件 | 模型 | 默认路径 |
|------|------|----------|
| 词典 | `mandarin_china_mfa` | `/chenmingjie/xingwen/tools/MFA/pretrained_models/dictionary/mandarin_china_mfa.dict` |
| 声学模型 | `mandarin_mfa` | `/chenmingjie/xingwen/tools/MFA/pretrained_models/acoustic/mandarin_mfa.zip` |

下载示例：

```bash
mfa model download dictionary mandarin_china_mfa
mfa model download acoustic mandarin_mfa
```

## 输入数据格式

`--data-txt` 为 pipe 分隔文本，每行：

```
/path/to/utt.wav|speaker_id|中文文本
```

- 若第一列路径不存在，会回退到 `--wav-dir/<文件名>`。
- 文本经 354 cleaner（`en_zh_dict_mixed_rhyme_tone_cleaners`）转成注音 token 再与 MFA 对齐。

## 快速开始（复用已有 BZNSYP 10k 结果）

```bash
cd /chenmingjie/xingwen/multiling_up-to-date
PYTHONPATH=. python mfa_zh354_duration/run.py --stats --limit 10000 --run-name bznsyp_10k
```

主结果：`runs/bznsyp_10k/zh354_duration_stats.tsv`

## 换一批数据怎么跑

### 1. 准备语料

假设新数据为：

- 列表：`/path/to/my_corpus/meta.txt`（格式同上）
- 音频：`/path/to/my_corpus/wav_24k/`

### 2. 新建一次 run（align + stats）

```bash
cd /chenmingjie/xingwen/multiling_up-to-date

PYTHONPATH=. python mfa_zh354_duration/run.py \
  --data-txt /path/to/my_corpus/meta.txt \
  --wav-dir /path/to/my_corpus/wav_24k \
  --limit 5000 \
  --run-name my_corpus_5k \
  --align --stats \
  --num-jobs 4 \
  --single-speaker
```

输出在 `mfa_zh354_duration/runs/my_corpus_5k/`。

单 speaker 语料建议加 `--single-speaker`，否则 MFA 会按 speaker 分 job，只有一个 speaker 时即使设置了 `--num-jobs` 也会串行。
如果 MFA 卡在建库/MFCC 阶段，可加 `--temporary-directory /tmp/mfa_<run_name>` 避免复用全局 temporary directory。

也可用 `--out-dir /任意/路径` 代替 `--run-name`。

### 3. 只重跑统计（映射修好后，无需 re-align）

```bash
PYTHONPATH=. python mfa_zh354_duration/run.py \
  --data-txt /path/to/my_corpus/meta.txt \
  --wav-dir /path/to/my_corpus/wav_24k \
  --limit 5000 \
  --run-name my_corpus_5k \
  --stats
```

### 4. 重建 354↔MFA 映射表

```bash
PYTHONPATH=. python mfa_zh354_duration/run.py --rebuild-mapping
# 或
PYTHONPATH=. python mfa_zh354_duration/map.py
```

写入 `mapping/mfa_zh354_token_mapping.tsv`。

## 输出说明

### `zh354_duration_stats.tsv`

| 列 | 含义 |
|----|------|
| `token_354` | 354 词表 token |
| `count` | 成功对齐的次数 |
| `p5` / `p50` / `p95` / `p99` | 时长分位数（mel frames） |
| `p01` | 1% 分位，训练 floor 推荐列 |
| `suggested_floor` / `suggested_cap` | count≥20 时取 p01 / `max(1.5*p99, p95+6, 8)`，供推理或对照 |

### `summary.json`

本次 run 的语料路径、句数、匹配率等元信息。

## 映射规则（代码内，非 TSV）

静态 TSV 来自单字词典；以下上下文规则在 `map.py` → `contextual_mfa_phones()` 中处理：

| 场景 | MFA | 354 |
|------|-----|-----|
| j/q/x + üe | `ɕʷ/tɕʷ` + `e` | `ㄒ/ㄐ/ㄑ` + `ㄩㄝ*` |
| j/q/x + üan | `ɕʷ/tɕʷ` + `e` + `n` | `ㄒ/ㄐ/ㄑ` + `ㄩㄢ*` |
| 的/了（轻声） | `t/l` + `ə` | `ㄉ/ㄌ` + `ㄜ˙` |
| 儿化尾 | `ɻ` | `ㄦ˙` |

## 注意事项

- MFA 对齐耗时与语料量成正比（1 万句约 20+ 分钟）。
- `mfa_align/` 体积较大（约 500MB/万句），可按需删除后保留 TSV。
- 354 词表中的死音节（`ㄏㄇ*`、`ㄧㄞ*` 等）在一般语料中不会出现，stats 为空属正常。
