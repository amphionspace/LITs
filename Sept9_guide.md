# 中英 LITs 训练与推理指南

## 1. 简介

训练时，程序会读取音频和对应文本的token（可以是raw text，可以是phoneme。考虑到模型较小、汉语和英语发音和字符之间对应关系不规律，我们这里简单地使用phoneme），学习它们之间的对应关系。

模型整体流程如下：

```text
输入中文、英文或中英混合文字
  -> 文字规范化(text normalization)，例如把数字转换为便于朗读的形式。# 这个功能来自一个外部仓库
  -> G2P (grapheme to phoneme)，查词典得到phoneme序列
  -> 根据我们定义的词表，把phoneme序列变成数字编号(token ids)
  （以上为前端）
  -> 声学模型处理序列，生成mel spectrogram
  -> Vocos 把频谱转换成 wav 音频
```

项目主要提供三种工作方式：

1. 训练普通 LITs 模型。
2. 用普通 LITs 模型推理。普通模型也称为 teacher 模型。
3. 把普通模型压缩成推理更快的 IntMeanFlow student 模型，再使用 student 模型推理。

建议先完成普通模型训练和推理，确认结果正常后，再尝试模型压缩。

## 2. 环境准备

（这部分纯codex生成，实际操作也建议直接把这个小节丢给codex）

### 2.1 硬件建议

- 推荐使用 NVIDIA GPU 训练。
- 推理也推荐使用 GPU；CPU 可以运行部分代码，但通常会慢很多。
- 训练需要的显存取决于模型大小和 `batch_size`。如果显存不足，应先减小 `batch_size`。

### 2.2 创建 Python 环境

建议使用 Python 3.10。在仓库根目录执行：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r lits_requirements.txt
```

以后每次打开新的终端，都需要先进入仓库并启用环境：

```bash
cd /path/to/LITs
source .venv/bin/activate
```

### 2.3 初始化文字前端子模块

本项目使用一个 Git 子模块处理中文、英文、数字、货币和发音转换：

```bash
git submodule update --init Transsion_Multilingual_Text_Normalization_for_TTS
```

虽然这个外部子模块本身含有其他语种，本项目只使用其中的中文、英文和中英发音配置。

### 2.4 安装 ICU 并编译文字前端

文字前端依赖 ICU。先安装 ICU，然后把 `ICU_ROOT` 指向 ICU 的安装目录。例如：

```bash
export ICU_ROOT=/path/to/icu
bash install_e2e_tn.sh
```

编译成功后，应该生成：

```text
e2e_infer/bin/tts_cli
```

可以运行下面的检查：

```bash
bash verify_e2e_tn.sh en-zh-dict
```

如果提示找不到 ICU，请检查：

- `$ICU_ROOT/include/unicode/locid.h` 是否存在；
- `$ICU_ROOT/lib` 下是否存在 ICU 动态库；
- 是否在当前终端执行过 `export ICU_ROOT=...`。

## 3. 准备训练数据

（主要需要设置configs/data里面的yaml文件，具体可以问codex）

### 3.1 音频要求

默认训练配置使用：

- WAV 音频；
- 采样率 24000 Hz
- 100 维声音特征
- 每条音频配一段中文、英文或中英混合文字。

尽量保证：

- 音频清晰，没有明显背景噪声；
- 文字和录音内容完全一致；
- 不要在一条样本中放太长的录音；
- 相同说话人始终使用相同的说话人编号。

配置和实际音频需要保持采样率一致，否则会报错。

**注意**：训练时需要设置mel频谱参数，形如：

```python
n_fft: 2048
n_feats: 100
sample_rate: 24000
hop_length: 384
win_length: 1536
f_min: 0
f_max: 12000
```
这个参数必须和下游的vocoder保持一致。如果沿用仓库里已有的24k vocos的话，不用改。

### 3.2 训练清单格式

训练清单是普通文本文件，每行对应一条音频。默认配置建议使用：

```text
/path/to/audio.wav|说话人编号|文字
```

例如：

```text
/data/wavs/000001.wav|0|你好，欢迎使用语音合成系统。
/data/wavs/000002.wav|0|Today is a beautiful day.
/data/wavs/000003.wav|1|请在 five minutes 后提醒我。
```

项目中有两个格式示例：

```text
data/filelists/templates/train.txt
data/filelists/templates/valid.txt
```

注意：示例中的音频路径只是占位符，不能直接用于训练。

程序也支持以下格式：

```text
/path/to/audio.wav|文字
/path/to/audio.wav|说话人编号|开始秒数|结束秒数|文字
```

但为了减少配置错误，新手建议统一使用 `音频路径|说话人编号|文字`。

### 3.3 纯文本如何转换成 G2P 结果

训练清单最后一列可以有两种内容：

- 原始文字，例如 `你好，welcome to LITs.`；
- 已经转换好的 phoneme，例如中文注音 token 与英文 ARPAbet token。

默认配置使用：

```yaml
cleaners: [en_zh_dict_mixed_rhyme_body_tone_cleaners]
```

因此，**如果清单里是已经完成数字、货币、缩写等文字规范化的普通纯文本，不必先手工转换 G2P**。训练读取每条样本时会自动执行：

```text
纯文本
  -> 中文查汉字—拼音词典，英文查 CMUdict
  -> 拼音转为注音 token，英文转为 ARPAbet
  -> 映射为模型 token IDs
```

如果文本中还有 `25`、`3.5%`、货币或复杂缩写，应先使用第 2 节编译的 TN 前端完成文字规范化，再生成正式训练清单；G2P 词典本身不能可靠决定这些符号该怎样朗读。

例如清单中的：

```text
/data/wavs/000003.wav|1|请在 five minutes 后提醒我。
```

会在内存中转换为类似下面的发音序列。实际结果以当前词典和规则为准：

```text
ㄑ ㄧ ㄥ ˇ ... F AY1 V ...
```

#### 3.3.1 当前使用哪些词典

默认词典来自旧分支保留下来的中英 G2P 前端：

- 中文主词典：`lits/text/sources/chinese_lexicon.txt`；
- 中文读音覆盖：`lits/text/sources/user_dict.txt`；
- 英文主词典：`temp_cmu_g2p/data/cmudict-0.7b`；
- 英文生词补充词典：`temp_cmu_g2p/data/supplement_lexicon.json`。

中文词典每行格式为 `词<TAB>带声调数字的拼音`：

```text
重庆<TAB>chong2 qing4
```

英文补充词典是 JSON，键使用英文词，值使用 ARPAbet：

```json
{
  "entries": {
    "MYPRODUCT": ["M", "AY1", "P", "R", "AA2", "D", "AH0", "K", "T"]
  }
}
```

英文查找顺序是 CMUdict、补充词典、逐字母拼读。完全无法处理的词可能被跳过，所以正式训练前必须检查确认不存在unknown token。

#### 3.3.2 使用仓库外部词典

不需要覆盖仓库中的词典文件，可以通过环境变量指定外部文件：

```bash
export LITS_ZH_LEXICON=/data/dicts/chinese_lexicon.txt
export LITS_ZH_USER_DICT=/data/dicts/zh_user_dict.txt
export LITS_CMUDICT=/data/dicts/cmudict-0.7b
export LITS_EN_SUPPLEMENT=/data/dicts/en_supplement.json
```

随后在同一个终端执行训练即可。未设置的路径继续使用仓库内置词典。

注意：这些词典在每个训练进程首次使用时加载并缓存。因此应当先设置环境变量，再启动训练；训练已经开始后再修改环境变量不会自动刷新词典。

修改中文词典后可以先校验拼音是否合法：

```bash
python lits/text/sources/validate_chinese_lexicon.py \
  --lexicon /data/dicts/chinese_lexicon.txt \
  --user-dict /data/dicts/zh_user_dict.txt
```

#### 3.3.3 推荐先生成一份可检查的 phoneme 清单

即使最终选择训练时自动 G2P，也建议先离线转换一份清单，用于人工抽查读音：

```bash
python scripts/prepare_g2p_filelist.py \
  /data/en-zh/train.raw.txt \
  /data/en-zh/train.phoneme.txt \
  --cmudict /data/dicts/cmudict-0.7b \
  --en-supplement /data/dicts/en_supplement.json \
  --zh-lexicon /data/dicts/chinese_lexicon.txt \
  --zh-user-dict /data/dicts/zh_user_dict.txt
```

脚本会保留路径、说话人编号和时间字段，只替换最后一列文字。例如：

```text
输入：/data/wavs/000002.wav|0|Hello world.
输出：/data/wavs/000002.wav|0|HH AH0 L OW1 _ W ER1 L D _ .
```

如果全部使用内置词典，可以省略四个词典参数：

```bash
python scripts/prepare_g2p_filelist.py \
  /data/en-zh/train.raw.txt \
  /data/en-zh/train.phoneme.txt
```

离线 phoneme 清单不能继续使用默认的 G2P cleaner，否则 phoneme 可能被重复处理。训练这种清单时要改用直通 cleaner：

```bash
TRAIN_FILELIST=/data/en-zh/train.phoneme.txt \
VALID_FILELIST=/data/en-zh/valid.phoneme.txt \
N_SPKS=3 \
bash training.sh trainer.devices='[0]' \
  'data.cleaners=[zh_en_phoneme_passthrough_cleaners]'
```

无论采用“训练时自动 G2P”还是“先生成 phoneme 清单”，训练和推理都必须使用同一套中英词典、音素体系、声调表示和 token 表。否则同一个词在训练与推理中可能得到不同发音。

### 3.4 划分训练集和验证集

至少准备两个清单：

- 训练集：用于模型学习，通常占数据的 95% 左右；
- 验证集：用于观察模型对未直接训练样本的效果，通常占 5% 左右。

同一条音频不能同时出现在训练集和验证集中。

假设文件路径为：

```text
/data/en-zh/train.txt
/data/en-zh/valid.txt
```

### 3.5 说话人数量

`N_SPKS` 表示说话人总数。

- 只有一个说话人时设为 `1`，清单中的编号使用 `0`。
- 有三个说话人时设为 `3`，编号必须位于 `0`、`1`、`2` 之中。

不要跳号，例如只有两个说话人时不要使用编号 `0` 和 `5`。

## 4. 训练普通 LITs 模型

### 4.1 最简单的启动命令

在仓库根目录执行：

```bash
TRAIN_FILELIST=/data/en-zh/train.txt \
VALID_FILELIST=/data/en-zh/valid.txt \
N_SPKS=3 \
bash training.sh trainer.devices='[0]'
```

这几个参数的含义是：

- `TRAIN_FILELIST`：训练清单路径；
- `VALID_FILELIST`：验证清单路径；
- `N_SPKS`：说话人数量；
- `trainer.devices='[0]'`：使用编号为 0 的 GPU。

只有一个说话人时，可以写：

```bash
TRAIN_FILELIST=/data/en-zh/train.txt \
VALID_FILELIST=/data/en-zh/valid.txt \
N_SPKS=1 \
bash training.sh trainer.devices='[0]'
```

### 4.2 显存不足怎么办

默认 `batch_size` 是 32，配置位于：

```text
configs/data/en-zh_24k.yaml
```

可以在启动时临时减小，不需要修改配置文件：

```bash
TRAIN_FILELIST=/data/en-zh/train.txt \
VALID_FILELIST=/data/en-zh/valid.txt \
N_SPKS=3 \
bash training.sh trainer.devices='[0]' data.batch_size=8
```

如果仍然显存不足，可以继续改成 `4` 或 `2`。`batch_size` 越小，一次处理的样本越少，训练通常也会更慢。

### 4.3 使用多张 GPU

例如使用 GPU 0 和 GPU 1：

```bash
TRAIN_FILELIST=/data/en-zh/train.txt \
VALID_FILELIST=/data/en-zh/valid.txt \
N_SPKS=3 \
bash training.sh trainer.devices='[0,1]' trainer.strategy=ddp
```

第一次训练建议先用一张 GPU 跑通，确认清单、音频和文字都能读取，再切换到多卡训练。

### 4.4 从已有训练继续

训练中断后，可以用上一次保存的 checkpoint 继续：

```bash
TRAIN_FILELIST=/data/en-zh/train.txt \
VALID_FILELIST=/data/en-zh/valid.txt \
N_SPKS=3 \
bash training.sh \
  trainer.devices='[0]' \
  ckpt_path=/path/to/last.ckpt
```

这里的 `ckpt_path` 会恢复模型和训练进度。

如果只想加载模型参数，但不恢复已有的训练状态，可以使用：

```bash
init_ckpt_path=/path/to/model.ckpt
```

这两种方式不要混用。

### 4.5 训练输出在哪里

默认输出目录由 `configs/paths/default.yaml` 和 Hydra 配置决定，一般位于仓库的 `logs/` 目录中。训练开始后，终端会打印实际输出路径。

每次的输出会包含：

- `.ckpt` 模型文件；
- 本次训练使用的配置；
- 训练和验证日志；
- 说话人编号与真实说话人的对应关系。

## 5. 准备用于推理的文件

普通模型推理需要准备两个输入文件：

1. 训练好的 LITs checkpoint，例如 `/models/lits-en-zh.ckpt`；
2. 待合成文本，例如 `/data/input.txt`。

仓库已通过 Git LFS 保存 24 kHz Vocos checkpoint：`vocos/generator.ckpt`。如果该文件只有一百多字节，需要先执行 `git lfs pull --include=vocos/generator.ckpt`。设置 `VOCOS_CHECKPOINT` 可以覆盖这个默认权重。

输入文本每行是一句话：

```text
你好，欢迎使用中英语音合成系统。
Hello, nice to meet you.
今天下午 three o'clock 开会。
```

不要在每行前面添加文件名或说话人编号。推理脚本会按行生成音频。

## 6. 使用普通模型推理

### 6.1 推荐命令

```bash
CKPT=/models/lits-en-zh.ckpt \
VOCOS_CHECKPOINT=/models/vocos-generator.ckpt \
SPK_ID=0 \
bash infer_e2e.sh en-zh-dict /data/input.txt demo
```

参数说明：

- `CKPT`：LITs 模型路径；
- `VOCOS_CHECKPOINT`：Vocos 模型路径；
- `SPK_ID`：使用哪个说话人的声音；
- `en-zh-dict`：处理普通中文、英文和中英混合文本，推荐新手使用；
- `/data/input.txt`：输入文本；
- `demo`：本次任务名称，可自行修改。

输出默认位于：

```text
infer_output/demo/
```

其中常见文件包括：

- `1.wav`、`2.wav`：生成的语音；
- `normalized.txt`：数字、符号等经过整理后的文字；
- `phonemes.txt`：程序使用的发音表示；
- `token_ids.jsonl`：模型实际读取的编号；
- `meta.txt`：生成结果列表。

### 6.2 如何选择 SPK_ID

`SPK_ID` 必须和训练时清单里的说话人编号一致。

例如训练数据使用：

```text
0 -> 中文女声
1 -> 英文男声
2 -> 英文女声
```

推理时设置 `SPK_ID=1` 就会使用编号 1 对应的声音。

如果编号超出模型训练时的范围，推理会报错，或者产生不可用结果。

### 6.3 `en-zh-dict` 和 `en-zh` 的区别

新手请直接使用：

```text
en-zh-dict
```

它接受正常文字，例如：

```text
今天温度是 25 摄氏度。
Hello world.
```

`en-zh` 主要用于已经提前转换成拼音或发音符号的输入，不适合普通文本。

### 6.4 指定输出目录

```bash
OUTPUT_DIR=/data/output/run1 \
CKPT=/models/lits-en-zh.ckpt \
VOCOS_CHECKPOINT=/models/vocos-generator.ckpt \
SPK_ID=0 \
bash infer_e2e.sh en-zh-dict /data/input.txt run1
```

### 6.5 CPU 推理

程序会在没有 CUDA 时使用 CPU，但速度可能很慢。建议同时关闭半精度：

```bash
FP16=0 \
CKPT=/models/lits-en-zh.ckpt \
VOCOS_CHECKPOINT=/models/vocos-generator.ckpt \
SPK_ID=0 \
bash infer_e2e.sh en-zh-dict /data/input.txt cpu-demo
```

## 7. 时长控制：句首静音、最短时长和最长时长

在介绍具体设置前，先解释本节所说的“给发音分配时长”是什么意思。

一段录音在模型内部会被切成许多很短的时间片，也就是“帧”。与此同时，输入文字会被转换成一串发音单位。例如“你好”不会只被看成两个汉字，而会被转换成声母、韵母、声调等模型能够识别的单位。

训练时，程序只知道整句话的文字和整段录音，并不知道录音中的第几毫秒对应“你”的声母、第几毫秒对应“好”的韵母。因此，程序需要按照发音顺序，把所有音频帧分别交给这些发音单位。这个过程就是本节所说的“分配发音”，更准确地说，是“为每个发音单位分配一段音频和持续时长”。

例如，一句话转换后有四个发音单位，录音被切成 20 帧，程序可能得到这样的结果：

```text
发音单位：  A       B          C       D
音频帧数：  3 帧    6 帧       4 帧    7 帧
时间长度：  48 ms   96 ms      64 ms   112 ms
```

这些帧必须按顺序分配，而且通常要覆盖整段有效录音。如果某个发音得到的帧太少，它可能听起来像被吞掉；得到的帧太多，它可能被拉得很长。录音开头如果有静音而又没有专门的静音单位，这些静音帧还可能被错误地分给第一个真正的发音。

模型训练完成后，会学习根据文字预测类似的时长。推理时没有参考录音，模型会先预测每个发音应该持续多少帧，再根据这些时长生成声音。因此，训练阶段的分配是否合理，会直接影响最终语音的节奏、停顿和清晰度。

语音模型不仅要读对字，还要决定每个发音持续多久。本项目旧分支设计了专门的时长控制，主要解决三类问题：

- 录音开头的静音被错误分给第一个发音；
- 某些元音或中文韵母太短，听起来像被吞掉；
- 分隔符和标点太长，导致停顿拖沓。

这部分分为“训练时控制”和“推理时修补”两层。两者作用不同，不能互相替代。

### 7.1 句首 `<sil>` 吸收静音

训练录音开头经常有一小段静音。如果输入的第一个 token 就是真正的发音，模型对齐文字和音频时，可能把这段静音算到第一个发音上。例如：

```text
录音：  [静音][静音][你][好]
错误：  [   你持续很久   ][好]
```

旧分支的做法是在每句话最前面自动加入一个特殊 token：

```text
<sil> 你 好
```

`<sil>` 表示静音，不需要发出实际读音。训练对齐时，它可以接收开头多出来的静音帧：

```text
录音：  [静音][静音][你][好]
对齐：  [  <sil>  ][你][好]
```

这样做的目的不是删除静音，而是防止第一个辅音或元音被错误拉长。

中英符号表中已经保留 `<sil>`。旧设计还会让 `<sil>` 不受普通发音的最长时长限制，因此它能够按实际录音吸收句首静音。

当前 `clean-main` 已经接通句首 `<sil>`：训练配置默认启用 `prepend_sil: true`，训练与推理都会在每条完整句子开头添加一次，tone ID 为 0，并与 blank token 分开处理。

旧 checkpoint 如果训练时没有使用 `<sil>`，推理时应设置 `PREPEND_SIL=0`。训练和推理必须保持一致，不能一边开启、一边关闭。

### 7.2 最短时长控制

最短时长的作用是避免重要发音短到几乎听不见。

训练配置中的主要参数位于：

```text
configs/model/lits.yaml
```

当前配置包括：

```yaml
duration_constrained_mas: true
duration_floor_scale: 1.0
duration_floor_min_frames: 2
arpa_duration_floor_scale: 0.5
```

可以简单理解为：

- `duration_floor_min_frames`：一个受保护发音至少占多少帧；
- `duration_floor_scale`：中文统计下限的缩放比例；
- `arpa_duration_floor_scale`：英文发音统计下限的缩放比例。

“帧”是模型内部计算声音长度的单位。当前配置中一帧对应：

```text
384 / 24000 秒，约 16 毫秒
```

因此 2 帧约为 32 毫秒，5 帧约为 80 毫秒。

上下限不是对所有 token 使用同一个固定数字。项目还保存了中文和英文的时长统计文件，让不同发音使用各自较合理的范围：

```text
lits/text/char_symbols/ljspeech_mfa_arpa_phone_duration_stats.csv
lits/text/char_symbols/king_shahenda_mfa_arpa_phone_duration_stats_cleaned.csv
mfa_zh354_duration/runs/chuanyin_biaobei/zh354_duration_stats.tsv
mfa_zh354_duration/runs/chuanyin_biaobei/zh354_duration_stats_cleaned.tsv
```

### 7.3 最长时长控制

最长时长用于防止某个发音或停顿占用过多音频。例如，句首静音如果错误分给第一个字，第一个字可能会被拉得很长；标点和词间分隔符过长，则会让句子听起来断断续续。

旧分支的训练方案会把每个 token 的最短和最长帧数直接交给文字—音频对齐算法。对齐算法在寻找路径时就遵守限制，而不是对齐结束后再简单裁剪。

这个区别很重要：

- 对齐过程中控制：多出来的帧可以重新分给更合适的 token；
- 对齐后直接裁剪：总帧数可能对不上，也可能破坏相邻发音。

如果整句话太短，无法同时满足所有最短时长，底层实现会按比例缩小下限；如果所有最长时长加起来仍不足以覆盖整段音频，也会放宽上限，避免训练直接失败。

**当前 `clean-main` 保留了 `maximum_path_constrained` 的上下限底层实现和对应测试，但 `lits/models/lits.py` 的训练主流程目前仍调用普通 `maximum_path`。** 所以配置文件中虽然有上下限参数，完整的“训练对齐时强制上下限”目前并未接通。后续恢复时，需要把中文、英文统计值转换为每个 token 的上下限，再传给 `maximum_path_constrained`。

### 7.4 当前可直接使用的推理时长修补

当前分支已经保留推理阶段的中英时长修补。普通模型和 student 模型都可以通过环境变量开启：

```bash
INFER_DURATION_PATCHES=1 \
CKPT=/models/lits-en-zh.ckpt \
VOCOS_CHECKPOINT=/models/vocos-generator.ckpt \
SPK_ID=0 \
bash infer_e2e.sh en-zh-dict /data/input.txt duration-demo
```

它目前会处理：

- 英文元音过短：预测不超过 3 帧时，提高到 5 帧；
- 中文零声母、单韵母音节过短，例如“嗯”“啊”等，提高到 5 帧；
- 中文字间 `_` 分隔符过长：限制到最多 15 帧；
- 中文标点附近的 `_` 过长：限制到最多 30 帧；
- 中文逗号、分号、冒号过长：限制到最多 3 帧。

相关默认值位于：

```text
lits/utils/infer_duration_floor.py
```

英文 `JH`、`CH` 也预留了最短时长控制，但默认下限为 0，表示当前没有启用。

这些规则只修改模型预测出的时长，不会重新训练模型。如果问题来自训练时错误的文字—音频对齐，推理修补只能减轻现象，不能完全替代带 `<sil>` 和上下限约束的重新训练。

student 模型同样可以开启：

```bash
INFER_DURATION_PATCHES=1 \
STUDENT_CKPT=/models/student.pt \
VOCOS_CHECKPOINT=/models/vocos-generator.ckpt \
SPK_ID=0 \
bash meanflow_distill/infer_distilled.sh \
  en-zh-dict /data/input.txt distilled-duration-demo
```

### 7.5 新手应该如何选择

建议按以下顺序操作：

1. 第一次训练先检查原始音频，尽量减少过长的句首静音。
2. 先使用默认配置训练一个小规模模型，确认数据和说话人编号正确。
3. 推理时发现少量吞音或停顿过长，可以尝试 `INFER_DURATION_PATCHES=1`。
4. 如果大量句子的第一个发音被拉长，应优先恢复并使用句首 `<sil>` 训练方案，而不是不断增大推理修补值。
5. 恢复完整上下限训练后，应重新训练模型；不要默认旧 checkpoint 与新 token、对齐方式兼容。
6. 修改任何帧数前，先把帧换算成毫秒并试听一批固定句子，避免修复一个问题的同时制造新的节奏问题。

## 8. 可选：训练更快的 student 模型

普通模型验证正常后，可以进行 IntMeanFlow 蒸馏。可以把它理解成：让一个推理步骤更少的 student 模型学习普通 teacher 模型的输出。

训练 student 模型需要：

- 已训练好的普通 LITs 模型；
- 单独准备的训练清单和验证清单；
- 可用的 GPU。

启动命令：

```bash
TEACHER_CKPT=/models/lits-en-zh.ckpt \
TRAIN_MANIFEST=/data/en-zh/distill-train.txt \
VAL_MANIFEST=/data/en-zh/distill-valid.txt \
CONFIG=en-zh \
bash meanflow_distill/run_distill.sh
```

默认配置位于：

```text
meanflow_distill/configs/en-zh.yaml
```

输出默认位于：

```text
meanflow_distill/runs/
```

如果训练时显存不足，可以先在配置中减小 `batch_size`。第一次使用时，不建议修改时间步、缓存长度等高级参数。

## 9. 使用 student 模型推理

student 模型使用下面的入口：

```bash
STUDENT_CKPT=/models/student.pt \
VOCOS_CHECKPOINT=/models/vocos-generator.ckpt \
SPK_ID=0 \
bash meanflow_distill/infer_distilled.sh \
  en-zh-dict /data/input.txt distilled-demo
```

输出默认位于：

```text
meanflow_distill/infer_output/distilled-demo/
```

默认使用适合连续生成的分块方式。如果只想进行完整句子的离线推理，可以使用：

```bash
STREAMING_MODE=non_streaming \
STUDENT_CKPT=/models/student.pt \
VOCOS_CHECKPOINT=/models/vocos-generator.ckpt \
SPK_ID=0 \
bash meanflow_distill/infer_distilled.sh \
  en-zh-dict /data/input.txt distilled-offline
```

不要随意修改 `T_GRID` 或 `N_TIMESTEPS`。student 模型通常只在训练时使用的设置下表现正常。

## 10. 常见问题

### 10.1 提示缺少 `TRAIN_FILELIST` 或 `VALID_FILELIST`

原因：没有设置训练清单路径。

处理方法：

```bash
export TRAIN_FILELIST=/data/en-zh/train.txt
export VALID_FILELIST=/data/en-zh/valid.txt
export N_SPKS=3
bash training.sh trainer.devices='[0]'
```

### 10.2 提示音频采样率不是 24000

原因：训练配置要求 24 kHz，但音频使用了其他采样率。

处理方法：先把所有训练音频统一转换成 24000 Hz，或者系统性修改训练和 Vocos 配置。新手不建议混用多个采样率。

### 10.3 提示找不到 `tts_cli`

原因：文字前端没有编译。

处理方法：

```bash
git submodule update --init Transsion_Multilingual_Text_Normalization_for_TTS
export ICU_ROOT=/path/to/icu
bash install_e2e_tn.sh
```

### 10.4 提示缺少 checkpoint

本仓库不包含训练好的 LITs 权重，需要自行提供普通推理的 `CKPT` 或 student 推理的 `STUDENT_CKPT`。24 kHz Vocos 权重已保存在 `vocos/generator.ckpt`，`VOCOS_CHECKPOINT` 只用于覆盖默认权重。

路径必须指向真实存在的文件。

### 10.5 提示说话人编号错误

检查训练清单中的说话人编号，并确认：

- 编号从 0 开始；
- 最大编号小于 `N_SPKS`；
- 推理时的 `SPK_ID` 是训练时出现过的编号。

### 10.6 训练显存不足

按以下顺序处理：

1. 减小 `data.batch_size`；
2. 先使用一张 GPU 验证数据；
3. 缩短异常长的训练音频；
4. 确认没有其他程序占用大量显存。

例如：

```bash
bash training.sh trainer.devices='[0]' data.batch_size=4
```

前提是当前终端已经设置好训练清单和 `N_SPKS`。

### 10.7 生成结果与文字不一致

优先检查：

- 输入文字是否有错字；
- 训练清单文字是否和录音一致；
- 是否错误使用了 `en-zh`，普通文本应使用 `en-zh-dict`；
- checkpoint 是否来自当前中英 token 配置；
- `SPK_ID` 是否正确。

## 11. 开始训练前的检查清单

在正式训练前逐项确认：

- [ ] 已启用 Python 环境；
- [ ] 已安装 `lits_requirements.txt`；
- [ ] TN 子模块已经初始化；
- [ ] `e2e_infer/bin/tts_cli` 已编译；
- [ ] 所有音频都是 24000 Hz；
- [ ] 训练清单和验证清单格式正确；
- [ ] 文字与音频内容一致；
- [ ] `N_SPKS` 与说话人编号一致；
- [ ] 已决定模型输出目录和备份方式。

## 12. 开始推理前的检查清单

- [ ] LITs checkpoint 路径存在；
- [ ] `vocos/generator.ckpt` 是已由 Git LFS 下载的真实权重；
- [ ] 输入文本每行是一句话；
- [ ] 使用 `en-zh-dict` 处理普通文本；
- [ ] `SPK_ID` 是模型训练过的说话人；
- [ ] `bash verify_e2e_tn.sh en-zh-dict` 能通过。

完成这些检查后，再执行推理命令，可以减少大部分常见错误。
