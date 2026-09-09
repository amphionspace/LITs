# `lits/text/sources` 资源说明

本目录存放中文 G2P、英文前端等推理所需的词典与配置。正常推理只会**加载**这些文件，不会对整本词典做逐条校验。

## 词典校验（`check_pinyin`）

历史上 `Frontend_chinese` 在加载 `chinese_lexicon.txt` 时会对全部约 66 万条拼音调用 `check_pinyin()`（验证能否转成注音/Bopomofo），冷启动需额外 **7–11 秒**。

现已改为：

| 场景 | 行为 |
|------|------|
| **正常推理** | 直接按 `词<TAB>拼音` 读入词典，不做全量校验 |
| **修改词典后** | 手动运行校验脚本 |

```bash
# 在项目根目录执行
python lits/text/sources/validate_chinese_lexicon.py

# 指定词典路径
python lits/text/sources/validate_chinese_lexicon.py --lexicon /path/to/chinese_lexicon.txt

# 跳过可选的 user_dict.txt
python lits/text/sources/validate_chinese_lexicon.py --skip-user-dict
```

脚本只加载 `pinyin_2_bpmf.txt` 等拼音表，不会再次加载整本 `chinese_lexicon.txt`，全量校验约数秒。退出码：`0` 全部通过，`1` 存在无效条目。

词典格式：`词<TAB>拼音`，多音节用空格分隔，如 `天气\tian1 qi4`。

## 文件清单

### 推理在用

| 文件 | 用途 | 引用位置 |
|------|------|----------|
| `chinese_lexicon.txt` | 汉字 → 拼音主词典（最长匹配分词） | `mandarin.py`, `language_cleaners.py` |
| `pinyin_2_bpmf.txt` | 拼音 → 注音（Bopomofo）映射 | `mandarin.py`, `language_cleaners.py` |
| `bpmf_2_pinyin.txt` | 注音 → 拼音反向表 | `mandarin.py` |
| `common_acronyms.txt` | 英文缩写词表（按字母读） | `acronyms.py` |
| `contraction_phonemes.py` | 英文缩写（如 don't、I'm）音素修正 | `g2p/english.py` |

### 可选（存在则加载）

| 文件 | 用途 |
|------|------|
| `user_dict.txt` | 用户覆盖读音，覆盖 `chinese_lexicon.txt` 中同词条目 |
| `jieba_user_dict.txt` | 仅当 `LITS_USE_JIEBA=1` 时供 jieba 分词使用（默认关闭 jieba） |

### 工具脚本

| 文件 | 说明 |
|------|------|
| `validate_chinese_lexicon.py` | 离线校验词典拼音（修改词典后运行） |

## 英文 CMUdict 说明

端到端推理中的英文文本 G2P（`en-zh-dict`）使用 **`temp_cmu_g2p/`** 包内的 `data/cmudict-0.7b`，与本目录无关。
