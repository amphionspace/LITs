# 内置中英文字前端

本目录保存 LITs 中英文流程使用的文字前端源码、规则和词典。它来自原项目
`hhk1994/Transsion_Multilingual_Text_Normalization_for_TTS` 的中英文运行子集，
对应上游提交：

```text
e280475dfd2f51741dcc5406121065141b75f8ad
```

该版本号也记录在 `frontend/UPSTREAM_COMMIT` 中，便于以后核对来源和同步更新。

## 保留内容

这里只保留三个运行配置：

- `data/zh`：中文文字规范化，例如处理数字、日期、时间和货币；
- `data/en`：英文文字规范化；
- `data/en-zh-g2p`：把中文、英文和中英混合文本转换成模型使用的发音序列。

同时保留了构建这些功能所需的 C++ 源码、JSON 规则和中英文词典。

## 已删除内容

本目录不包含阿拉伯语、孟加拉语和俄语的 backend、规则、数据与测试，也不包含俄语 MorphoDiTa 模型。

这些文件现在是主仓库中的普通文件，不再使用 Git submodule。克隆 LITs 主仓库后，不需要另外初始化或下载文字前端仓库。

## 编译方法

文字前端依赖 ICU。在 LITs 仓库根目录执行：

```bash
export ICU_ROOT=/path/to/icu
bash install_e2e_tn.sh
```

编译成功后会生成：

```text
e2e_infer/bin/tts_cli
```

可以运行下面的检查：

```bash
bash verify_e2e_tn.sh en-zh-dict
```

检查会分别验证中文文字规范化、英文文字规范化和中英 G2P。

## 目录说明

```text
frontend/
├── build.sh              # 编译中英文字前端
├── UPSTREAM_COMMIT       # 记录复制时使用的上游提交
├── data/                 # zh、en 和 en-zh-g2p 运行配置及词典
├── rules_v2/             # 中文、英文和中文变调规则
├── unified/              # 统一 C++ 入口及中英文 backend
├── third_party/          # 构建所需的第三方头文件
└── tts_normalizer_engine.*
```

如需更新文字前端，应先在原项目中确认新的中英文实现，再有选择地同步到本目录。不要直接复制整个多语种仓库，以免重新引入无关语种代码和资源。
