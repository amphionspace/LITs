# Frontend Rules（文本前端规则层）

本目录保留 LITs 的模型侧标点辅助、英文查词规则，以及中文变调的 Python **参考/回退实现**。生产运行时的 TN 与中文变调已经统一到 `frontend/`：原始文本使用 `data/<locale>`，词典查音后的拼音使用 `data/zh_g2p`。当统一动态库/CLI 未安装时，Python 变调规则才作为兼容回退。

## 架构

```
原始文本
    │
    ▼
[TextNormalizer data/<locale>]  ← 数字、货币、车牌与已合并的通用清理
    │
    ▼
[frontend_rules/punctuation]  ← 标点增删/规范化（common + 语言 locale）
    │
    ▼
[lexicon G2P / pinyin 转写]  ← 英文查词时应用 g2p_homograph
    │
    ▼
[TextNormalizer data/zh_g2p]  ← 中文变调（汉字查音之后，主路径）
    │
    ├── runtime 不可用 → frontend_rules/g2p_sandhi（Python 回退）
    │
    ▼
Bopomofo / ARPAbet token 序列
```

### 与 TN 的关系

| 模块 | 路径 | 是否可与 TN 合并 | 说明 |
|------|------|------------------|------|
| 标点规则 | `rules/punctuation/*.json` | **已部分合并** | TN 主规则已吸收通用/locale 清理；本目录仍服务模型侧 tokenizer 与兼容入口 |
| 变调规则 | `rules/g2p_sandhi/zh.json` | **已由统一引擎接管** | 主路径为 `frontend/rules_v2/zh_g2p_sandhi.json`；本文件是 Python 参考/回退 |
| 英文多读音 | `rules/g2p_homograph/en.json` | **否** | CMUdict variant 轻量消歧（前后词） |
| 英文大小写缩略词 | `rules/g2p_acronym_case/en.json` | **否** | 小写按单词读、大写按逐字母读（查词典时） |

## 目录结构

```
frontend_rules/
├── README.md                 # 本文件：规则行为说明
├── rules/
│   ├── punctuation/          # 标点预处理
│   │   ├── common.json       # 全语言通用（preprocess_text）
│   │   ├── zh.json           # 中文全角→半宽等
│   │   ├── ru.json / ar.json / bn.json
│   ├── g2p_sandhi/
│   │   └── zh.json           # 中文变调与分词合并
│   ├── g2p_homograph/
│   │   └── en.json           # 英文 CMUdict 多读音
│   ├── g2p_acronym_case/
│   │   └── en.json           # 大小写敏感缩略词（us/US、ai/AI）
│   └── abbreviation/
│       └── en.json           # 字母点缩写 U.S. → US
├── engine/                   # （实现位于 ops/ + pipeline.py）
│   ├── loader.py
│   ├── pipeline.py
│   └── ops/
│       ├── punctuation.py
│       └── sandhi.py
└── docs/
    └── schema.json           # JSON 规则文档 schema（草案）
```

英文 homograph 执行骨架在 `temp_cmu_g2p/homograph.py`（查词典时调用）。  
大小写缩略词：`temp_cmu_g2p/acronym_case.py`，在 homograph 之前按表面大小写选 variant。  
缩写合并：`temp_cmu_g2p/abbreviation.py`，在 `tokenize_english_text` 之前执行。

## 英文缩写（`rules/abbreviation/en.json`）

在英文 G2P 分词**之前**，将 `U.S.` / `U.S.A.` / `Ph.D.` 合并为无点的单词 token（`US` / `USA` / `PhD`），**缩写内部的句号不再作为停顿标点**；仅句末 `.` 保留。

| 输入 | 分词 | 句末标点 |
|------|------|----------|
| `The U.S. economy grew.` | `The` `US` `economy` `grew` | 一个 `.` |

## 英文 homograph 轻量规则（`rules/g2p_homograph/en.json`）

多读音词不再固定用 CMUdict 主条目（variant 0），而是按**前/后词**选 variant。

| 词 | 条件 | 读音 |
|----|------|------|
| `LIVE` | 前词是冠词（the, a…） | `L AY1 V` |
| `LIVE` | 前词是代词 / 助动词，或 `n't` 缩写 | `L IH1 V` |
| `LIVE` | 下一词以 **-ly** 结尾 | `L IH1 V`（任意副词：naturally, happily…） |
| `LIVE` | 下一词是介词（in, on, at…） | `L IH1 V` |
| `LIVE` | 前词非冠词且句末/标点后 | `L IH1 V`（**任意**主语名词：organisms, bacteria…） |
| `LIVE` | 默认 | `L AY1 V`（live music 等） |

词表只保留**封闭类**（冠词、代词、助动词、介词），不罗列 organisms、music、naturally 等内容词。

**扩展：** `when` 支持 `next_suffix` / `prev_suffix`（如 `ly`、`n't`）、`prev_matches`（正则）。

## 英文大小写缩略词（`rules/g2p_acronym_case/en.json`）

部分词既可作普通单词，也可作逐字母缩略词。CMUdict 查词键**不区分大小写**，单靠词典无法区分 `us` 与 `US`。本表根据**表面大小写**统一驱动两条英文前端：

| 后端 | 执行位置 | `mode: word` | `mode: spell` |
|------|----------|--------------|---------------|
| CMUdict 查词 | `temp_cmu_g2p/acronym_case.py` | CMUdict variant（默认 0） | variant 1 或逐字母兜底 |
| Transsion G2P | `lits/text/acronyms.py` | 走正常 G2P | 走缩写拼读 |

共享逻辑在 `ops/acronym_case.py`；**规则用 `mode`，不用 `pick`**，以便两台机器读同一份 JSON。

| 词 | 小写 | 大写 |
|----|------|------|
| `us` / `US` | `AH1 S` | `Y UW2 EH1 S` |
| `ai` / `AI` | `AY1` | `EY1 AY1` |

规则格式：

```json
"US": {
  "lower": { "mode": "word", "cmudict": { "pick": 0 } },
  "upper": { "mode": "spell", "cmudict": { "pick": 1 } }
}
```

- `mode`：`word`（整词）或 `spell`（逐字母），**两端通用**
- `cmudict.pick`：**仅 CMUdict 端**可选，variant 顺序与默认不一致时覆盖（默认 word→0、spell→1）
- 大写无 CMUdict variant 时 CMUdict 端自动走逐字母兜底

表中词优先于 `common_acronyms.txt`。混合大小写（如 `Us`）不匹配，回退各自默认逻辑。

## 标点规则行为（`rules/punctuation/`）

### common.json — 全局 `preprocess_text`

在 `lits.text.text_to_sequence()` 进入各 language cleaner 之前执行。

| 阶段 | 行为 | 示例 |
|------|------|------|
| NFKC 规范化 | Unicode 兼容分解再组合 | 全角/半角变体统一 |
| 删除 emoji | 移除常见 emoji 区段 | `你好😀世界` → `你好世界` |
| Markdown 星号 | 去掉 `**bold**` / `*italic*` / 列表 `*` 标记，保留词 | `*重点*内容` → `重点内容` |
| 控制字符 | 删除 C0/C1、bidi、零宽、BOM 等 | |
| 波浪号 | `~` / `～` / `∼` → `!` | `你好～` → `你好!` |
| 换行/空格断句 | 真实换行：左段无句末标点则补 `。`；仅当左右均为汉字时在空格处插入 `，` | `你好 世界` → `你好， 世界` |
| 结构性引号删除 | 删除成对引号；保留英文词内撇号 | `don't` 保留；`"你好"` → `你好` |
| 斜杠段清理 | 合并空 `/` 段 | `a / / b` → `a / b` |
| 破折号删除 | 各类 dash 变体替换为空格 | `你好—世界` → `你好 世界` |
| 空白折叠 | 多空格合一并 strip | |
| 尾部标点去重 | 仅当尾部为**同一**标点重复时折叠 | `!!!` → `!`；`/.` 不合并 |
| 补句末标点 | 无句末标点时补 `.`；词/音素结尾补 ` .`（避免 `OW1.` 粘连） | `你好` → `你好 .` |

**不插入逗号的空格**：英文词内空格（`Hello there`）、中英/英英交界（`GT 十 Pro`）。

### zh.json — 中文标点

| 行为 | 说明 |
|------|------|
| 省略号统一 | `...`、`⋯` 等 → 单字符 `…` |
| NFKC 规范化 | 兼容全角/半角变体 |
| 全角→半角映射 | `，`→`,`、`。`→`.`、`！`→`!` 等（见 JSON `char_maps.zh_punct_to_half`） |
| `~`/`～` | 映射为 `!` |

**结构性标点**（`()[]<>"'`）：在 hanzi lexicon 路径中**不读出**，查找前剥离；去掉括号时若外侧缺停顿则插入 `，`：

- `现行犯（如正在实施犯罪）` → `现行犯，如正在实施犯罪`

**停顿性标点**（`,.!?;:`、`…`）：保留在 token 流中，用 `_` 边界分隔，并**切断三三变调组**。

### ru.json / ar.json / bn.json

- **俄语**：删除 `«»„""''()` 等结构性标点；句读去重；省略号统一。
- **阿拉伯语**：波斯字母 `ی/ک` → 阿拉伯 `ي/ك`；删除 tatweel `ـ`。
- **孟加拉语**：NFC 规范化 + 结构性引号删除。

## 变调参考/回退行为（`rules/g2p_sandhi/zh.json`）

在 `chinese_lexicon.txt` / `user_dict.txt` 查得带调拼音之后、转 Bopomofo 之前（或 Bopomofo 路径上 per-word）应用。

### 分词合并（word_stages）

变调前将孤立字并入邻词，避免错误边界：

| 规则 | 行为 | 示例 |
|------|------|------|
| merge_yi | 重叠式中的 `一`、数字串中的 `一` 与后继字合并 | `看`+`一`+`看` → `看一看` |
| merge_bu | 孤立 `不` 并入后词 | `不`+`怕` → `不怕` |
| merge_er | 孤立 `儿` 并入前词 | `玩`+`儿` → `玩儿` |

### 三三变调（third_tone_sandhi）

**连续两个第三声**：前一个改为第二声。

- 拼音：`ni3 hao3` → `ni2 hao3`
- 两字词 Bopomofo：`ㄕˇ ㄐㄧㄝˋ` 中若前两音节均为 `ˇ`，首音节 `ˇ`→`ˊ`
- **三字全三声**（triple_third_tone_bopomofo）：前三字均为三声时，**前两字**改二声

**变调边界**（不跨边界连读）：`,.!?;:`、`…`、`_`、`/`、`|`，以及 ARPAbet token。

示例：`chuan3 . ci3` 中句号两侧**不**做三三变调。

### 「不」变调（bu_sandhi）

| 条件 | 读音 |
|------|------|
| 整词全是「不」 | 不变调 |
| `不字` | 不变调 |
| `X不Y` 三字且中间为「不」（如「好不好」「或不焉」） | 中间「不」读轻声 |
| 词尾「不」且**后接问号**（如「好不？」「来不？」「可以不？」） | 尾字「不」默认读轻声 |
| `bu_question_keep_tone_words` 内短语且后接问号（如「偏不？」） | 保持原调（不读轻声） |
| 「不」后接第四声 | 读第二声（bú） | `不怕` bú pà |

默认覆盖所有「A 不？」省略疑问（`来不？`、`明白不？`、`不？` 等）。分词为 `可以`+`不` 时，以前词+「不」组成短语判断是否在例外词表；不在例外中则单独「不」亦读轻声。例外词表 `word_lists.bu_question_keep_tone_words` 中的「不」字均**需**以四声强调。

### 「一」变调（yi_sandhi）

| 条件 | 读音 |
|------|------|
| 含数字的一串（除「一」外有 `isnumeric` 字符） | 首「一」依后字：后接四/轻声→二声，否则→四声；其余「一」→一声 |
| 重叠式 `X一X`（如「看一看」） | 中间「一」→轻声 |
| `第一` | 「一」→一声（yī） |
| `一月`/`一日`/`一号` | 首「一」→一声 |
| 「一」后接第四声 | →第二声（yí） | `一段`、`一个`（「个」本调去声，连读轻声仍按去声变调） |
| 「一」后接非四声且非标点 | →第四声（yì） | `一天` |
| 「一」后接轻声 | 按后字**本调**判断上两行（非按表面轻声） | `一个` yí ge |
| 「一」后接标点 | 保持一声 |

### 「儿」化（er_sandhi）

词尾「儿」→轻声，**例外词不变**：`女儿`、`老儿`、`男儿`、`少儿`、`小儿`。

### 语气词标点变调（interjection_tones）

配置在 `tone_rules.interjection_tones`（JSON 数组，可扩展多个字）。根据**紧随其后的标点**（跳过空格）变调。

**「啊」**（`a_interjection`）

| 后续标点 | 读音 | 示例 |
|----------|------|------|
| `?` / `？` | 第二声（á） | `啊？` |
| `!` / `！` | 第四声（à） | `啊！` |
| 其他 | 不变 | `啊，` |

**「嗯」**（`en_interjection`）

| 后续标点 | 读音 | 示例 |
|----------|------|------|
| `?` / `？` | 第二声（én） | `嗯？` |
| 其他 | 第四声（èn） | `嗯，`、`嗯！` |

## Python 调用入口

```python
from lits.text.frontend_rules import PunctuationEngine, G2PSandhiEngine

# 全局预处理（等同 preprocess_text）
PunctuationEngine("common").apply(text)

# 中文标点规范化
PunctuationEngine("zh").apply(text)

# 变调（通常在 language_cleaners / mandarin 内部调用）
engine = G2PSandhiEngine("zh")
engine.merge_words(words)
engine.apply_word_sandhi_bopomofo(word, bopomofos)
engine.apply_third_tone_sandhi_to_pinyin_tokens(tokens)
```

对外兼容：`lits.text.language_cleaners.preprocess_text`、`normalize_clause_break_punct` 等仍可使用，内部已委托本层。

## 修改规则

1. 生产 TN/变调规则以 `frontend/rules_v2/<locale>.full.json` 与 `frontend/rules_v2/zh_g2p_sandhi.json` 为准。
2. `rules/punctuation/<locale>.json` 只在 LITs 模型侧仍需的辅助步骤中维护。
3. `rules/g2p_sandhi/zh.json` 应与 C++ 规则保持交叉验证，用作无运行库环境的回退。
4. 修改 Python 回退规则后需重启进程（规则有 `lru_cache`）。

## JSON 格式（frontend_rules_v1）

参考 TN 的 `rules/*.json`，但使用独立 `format` 字段，避免被 TN 引擎误加载：

```json
{
  "format": "frontend_rules_v1",
  "module": "punctuation",
  "locale": "common",
  "metadata": { "tn_merge_target": null },
  "resources": { "char_sets": {}, "char_maps": {} },
  "pipeline": { "stages": [ { "id": "...", "op": "...", "params": {} } ] }
}
```

变调文档额外包含 `pipeline.word_stages`、`syllable_stages`、`pinyin_token_stages`、`interjection_stages`。
