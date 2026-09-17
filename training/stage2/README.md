# Stage 2：当前联合训练入口

当前配方从 foundation 21k 声学权重初始化，重置两个 speaker embedding，使用冻结的 LJSpeech 原始录音（约 21h）和 MajesticVoice（中文 50h、英文 25h、混读 25h）。自然样本混合、在线 MAS、全模型联合更新；完整配方见 [联合训练说明](../majestic_scratch/README.md)。

```bash
export PYTHONPATH=/119010446/LITs
python training/stage2/prepare.py
python training/stage2/preflight.py --run-dir /path/to/new-run
python training/stage2/run.py --run-dir /path/to/new-run
```

准备入口读取数据集根目录的 `config.json.training`；正式启动前必须通过数据审计和 GPU 预检。已启动的 run 不能再次作为新训练启动。

`data.py` 提供缓存文本与 Mel loader；`evaluate_checkpoint.py` 使用冻结评估清单。具体联合训练实现仍放在 `training/majestic_scratch/`：这是历史命名，当前支持 `backbone_init`，且运行中的训练、评估和已有配置依赖这些模块路径，因此保留名称。

旧 197k 初始化的分阶段适配入口及独立单音色微调流程已移除；历史结果与配置保留在训练实录和各 run 的 `source/` 快照中。此目录的启动命令只表示当前联合训练流程。
