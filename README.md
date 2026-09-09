# LITs Chinese-English Pipeline

This branch contains the source skeleton for training and inference of a Chinese-English LITs model. It intentionally excludes trained acoustic checkpoints, distilled checkpoints, production manifests, generated audio, and experiment outputs.

The current design is:

```text
raw Chinese/English text
  -> Transsion C++ TN profiles (`zh` / `en`)
  -> C++ `en-zh-g2p` profile
  -> JSON Text2Id token/tone IDs
  -> LITs acoustic model
  -> Vocos vocoder
```

The production frontend is single-pass and the Python runtime consumes numeric IDs. The legacy Python Chinese-English cleaner remains available for training manifests and diagnostics. The frontend/runtime boundary is documented in `docs/frontend_runtime_contract.md`.

## Repository scope

Included:

- Chinese-English acoustic-model training code and Hydra configuration
- teacher/original inference
- deployment-aligned IntMeanFlow distillation and distilled inference
- vendored 24 kHz Vocos source, training configuration, and generator checkpoint
- Chinese-English frontend adapters, token inventories, tests, and small fixtures

Not included:

- LITs acoustic-model or distilled-model weights
- training/validation datasets or production manifests
- generated audio, diagnostic output, or benchmark artifacts
- pipeline code and resources for languages outside Chinese and English

The retained `Transsion_Multilingual_Text_Normalization_for_TTS` git submodule is an external multilingual project. This repository invokes only its `zh`, `en`, and `en-zh-g2p` profiles; cloning that submodule may still download profiles outside this repository's language scope.

## Setup

```bash
git submodule update --init Transsion_Multilingual_Text_Normalization_for_TTS
python -m venv .venv
source .venv/bin/activate
pip install -r lits_requirements.txt
```

Build the C++ frontend after installing ICU:

```bash
export ICU_ROOT=/path/to/icu
bash install_e2e_tn.sh
```

## Data contract

Training manifests are external files. Supported rows are:

```text
/path/to/audio.wav|text
/path/to/audio.wav|speaker_id|text
/path/to/audio.wav|speaker_id|start_seconds|end_seconds|text
```

For the default multi-speaker Chinese-English configuration, use `wav|speaker_id|text`. See `data/filelists/templates/` for non-runnable examples. Audio must match the configured 24 kHz sample rate.

## Acoustic-model training

Set the external manifests and speaker count, then launch Hydra training:

```bash
TRAIN_FILELIST=/data/en-zh/train.txt \
VALID_FILELIST=/data/en-zh/valid.txt \
N_SPKS=3 \
bash training.sh trainer.devices='[0]'
```

The default `configs/experiment/en-zh.yaml` uses the 173-token rhyme-body-tone inventory. If a dataset uses a different retained Chinese-English inventory, override the cleaner and `model.n_vocab` together.

## Teacher inference

Required external artifact:

- a LITs checkpoint compatible with the selected Chinese-English inventory

The bundled `vocos/generator.ckpt` is used by default. It is a 24 kHz generator compatible with 100-bin mel output.

```bash
CKPT=/models/lits-en-zh.ckpt \
SPK_ID=0 \
bash infer_e2e.sh en-zh-dict /data/input.txt demo
```

Supported model language names are `en-zh-dict`, `en-zh`, and their rhyme-body-tone compatibility aliases. Output is written under `infer_output/` unless `OUTPUT_DIR` is set.

## IntMeanFlow distillation

The distillation path incorporates the deployment-aligned KV-cache design: teacher targets, student trajectories, and mean-flow loss use the same chunked decoder geometry as causal deployment inference.

Provide all model/data assets from outside the repository:

```bash
TEACHER_CKPT=/models/lits-en-zh.ckpt \
TRAIN_MANIFEST=/data/en-zh/distill-train.txt \
VAL_MANIFEST=/data/en-zh/distill-valid.txt \
CONFIG=en-zh \
bash meanflow_distill/run_distill.sh
```

Distilled inference uses the same frontend and bundled Vocos checkpoint by default:

```bash
STUDENT_CKPT=/models/student.pt \
SPK_ID=0 \
bash meanflow_distill/infer_distilled.sh en-zh-dict /data/input.txt distilled-demo
```

## Vocos training

The `vocos/` directory contains the 24 kHz source, `vocos/config.yaml`, and the Git-LFS-managed `vocos/generator.ckpt`. Update the external `filelist_path` values in the configuration before training. After cloning, run `git lfs pull --include=vocos/generator.ckpt` if the checkpoint remains a small pointer file.

## Validation without weights

```bash
python -m compileall -q lits vocos meanflow_distill infer_e2e.py inference_stream.py
pytest -q tests/test_runtime_text2id.py tests/test_infer_chunking.py \
  tests/test_single_pass_frontend.py tests/test_tone_embedding.py tests/test_lr_downshift.py
```

`tests/test_cpp_frontend.py` additionally requires the initialized TN submodule and a built `e2e_infer/bin/tts_cli`. Full synthesis requires an external LITs checkpoint; the Vocos checkpoint is bundled through Git LFS.
