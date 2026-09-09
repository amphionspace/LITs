# Chinese-English IntMeanFlow Distillation

This directory trains a low-step Chinese-English LITs student from an external teacher checkpoint. No checkpoints or production manifests are stored in the repository.

The deployment-aligned path uses the same chunked KV-cache geometry for teacher targets, student trajectories, mean-flow loss, and causal inference. Defaults are defined in `configs/en-zh.yaml` (`chunk_size=100`, `decoder_left_frames=20`, `pre_lookahead_len=3`).

## Train

```bash
TEACHER_CKPT=/models/lits-en-zh.ckpt \
TRAIN_MANIFEST=/data/en-zh/train.txt \
VAL_MANIFEST=/data/en-zh/valid.txt \
CONFIG=en-zh \
bash meanflow_distill/run_distill.sh
```

Environment variables override YAML values. Outputs go to `meanflow_distill/runs/`, which is ignored by Git.

## Infer

```bash
STUDENT_CKPT=/models/student.pt \
VOCOS_CHECKPOINT=/models/vocos-generator.ckpt \
SPK_ID=0 \
bash meanflow_distill/infer_distilled.sh en-zh-dict /data/input.txt demo
```

Use `STREAMING_MODE=non_streaming` for full-utterance decoding. The student metadata supplies its time grid unless `T_GRID` is explicitly set.
