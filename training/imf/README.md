# Two-step iMF on Stage 2 audio

Select `model/cfm=imf` to use improved MeanFlow. The default `model/cfm=default`
still constructs the original CFM decoder and uses its original loss/sampler.
This is direct training on paired text/audio, with no teacher trajectories,
distillation targets, or teacher sampling loop.

## Implementation sources

This is a PyTorch/Mel adaptation of the **official repository implementation**:

- [Training `imf.py`, main at bf60cd7](https://github.com/Lyy-iiis/imeanflow/blob/bf60cd7cb653f6628e59d48034b333c5eba445e2/imf.py).
- [Model/defaults, same revision](https://github.com/Lyy-iiis/imeanflow/tree/bf60cd7cb653f6628e59d48034b333c5eba445e2).
- [PyTorch sampling, torch at 0468798](https://github.com/Lyy-iiis/imeanflow/blob/04687983e821b3ad01f54f03dc44194a33c20c54/imf.py).

The official `torch` branch supplies **inference only**. The training algorithm
is ported from the official JAX `main` branch, not inferred solely from the
paper's simplified single-head pseudocode. The upstream MIT notice is retained
in [UPSTREAM_LICENSE](UPSTREAM_LICENSE).

The port retains:

- Two ordered logit-normal time draws (`P_mean=-0.4`, `P_std=1`). The first
  `floor(batch_size * data_proportion)` samples use `r=t`; the default proportion
  is 0.5, as upstream.
- A shared trunk and separate average-velocity `u` and auxiliary instantaneous-
  velocity `v` heads. For the JVP direction, `v` is evaluated at zero interval.
- `V = u + (t-r) * stop_gradient(JVP(u; v_pred, 1, 0))`, with regression to
  `noise - Mel`. The conditional sample velocity is **not** the JVP direction.
- Per-example squared-error sums, separately adaptively weighted for each head:
  `S / stop_gradient((S + norm_eps) ** norm_p)`, then averaged over examples.
  Defaults are `norm_p=1`, `norm_eps=0.01`; the auxiliary head has its own loss.
- Sampling uses only `u`, once per time interval. The auxiliary `v` tail is not
  executed in either whole-utterance or cached streaming inference.

## LITs adaptations

The new decoder retains LITs' Conformer condition encoder and causal U-Net.
It does not replace them with the image DiT. Down/mid blocks are shared; the
existing up/final blocks form the `u` tail and are copied into a new `v` tail.
This differs from upstream's eight Transformer layers per head.

LITs' pretrained absolute-time embedding remains in place. An interval MLP
with zero final projection is added to it. Upstream's image backbone instead
uses in-context tokens and omits explicit absolute-time conditioning. Keeping
the existing embedding allows exact copying of FM tensors and preserves the
old network's diagonal output at initialization (after the time/sign mapping).

The objective follows upstream's `t=1` noise, `t=0` data convention. LITs'
external sampler/cache interface remains `s=0` noise, `s=1` Mel, with `s=1-t`
and negated velocity. This reversal is explicit in `forward_uv`; FM weights
themselves are not negated. iMF uses the exact linear path (`sigma_min=0`),
whereas existing FM defaults retain `sigma_min=1e-4`.

This first preset implements the exact **no-CFG specialization** (`omega=1`,
no condition dropout). LITs text/speaker conditions remain supplied normally.
It does not add image-class/null-class tokens, flexible guidance, or guidance
intervals; `guidance_scale != 1` is rejected instead of silently ignored.

Mel padding is masked in the noisy state, targets, and both error sums before
adaptive weighting. Each utterance receives equal weight, like each fixed-size
image upstream. Unweighted errors are also logged as valid-element MSE, so
sequence length and the nearly constant adaptively weighted loss do not hide
training behavior. Existing duration/prior objectives continue to train.

Estimator dropout is disabled to match upstream and make the JVP and normal
forward describe the same deterministic field. The condition encoder executes
once per loss call, retaining its existing dropout and streaming/full-context
mixture; its output is held fixed in the JVP direction but receives ordinary
gradients from the trainable `u/v` forward. JVP and estimator loss evaluation
use FP32 with math SDPA; BF16 mixed precision can still be used elsewhere.
Validation explicitly supports Lightning's `inference_mode` by creating normal
tensors inside the JVP context. No gradient through the JVP is retained.

## Configuration and initialization

Use the frozen Stage 2 data directory; no regeneration or re-splitting is needed.
For example, the existing dataset is:

```bash
export STAGE2_DATA=/119010446/tts-assets/training_runs/ljs_majestic_100h_backbone21k_20260914/data
export TRAIN_FILELIST="$STAGE2_DATA/train.txt"
export VALID_FILELIST="$STAGE2_DATA/val.txt"
export N_SPKS=2
```

`experiment=en-zh-imf-stage2` selects the natural-utterance Stage 2 loader and
iMF. It requires an explicit `init_ckpt_path` and `trainer.max_steps`; selecting
this preset does not silently choose a long training budget. A configuration
preview (does not train) is:

```bash
python -m lits.train experiment=en-zh-imf-stage2 \
  init_ckpt_path=/path/to/stage1_21000.ckpt \
  trainer.max_steps=1000 --cfg job
```

To actually train after choosing a budget and measuring capacity, remove
`--cfg job`, set `trainer.devices`, and select `data.batch_size`. The preset's
batch of 8 is provisional; JVP has additional compute/memory cost. This feature
does **not** automatically launch, stop, or resume any existing run. The old
`training/majestic_scratch/run.py` remains the FM recipe; use the explicit Hydra
iMF preset for this path.

On a weights-only warm start:

1. Checkpoint Mel mean/std are applied to both the dataset and the model before
   construction, avoiding a mismatch with new environment defaults.
2. All matching FM backbone tensors load strictly; only the interval MLP and
   auxiliary tail are new. Missing unrelated keys or incompatible shapes fail.
3. The auxiliary tail is copied from the **loaded** FM up/final blocks.
4. Speaker rows are reset by this Stage 2 preset, matching the existing 21k
   experiment. Set `init_reset_speaker_embeddings=false` to retain them.
5. Adam, scheduler, and step counters are fresh. Full iMF resume uses
   `ckpt_path` with the generic `experiment=en-zh model/cfm=imf` configuration,
   matching saved architecture and sampling options, and `init_ckpt_path=null`.

An iMF checkpoint warm start loads both heads and the learned interval branch
strictly; it never reinitializes those learned parameters as if it were FM.

## Two-step inference and evaluation

Both `model/cfm=imf` and the Stage 2 preset use `[0, 0.5, 1]`, the official
uniform two-step sampler expressed in LITs time. No previous distillation grid,
loss weights, teacher settings, or streaming-distillation flags are imported.
The checkpoint stores the iMF sampling configuration.

`model.synthesise(..., n_timesteps=None)` uses the checkpoint's default step
count (two for iMF, ten for FM). The shared Stage 2 evaluator now reads the
checkpoint budget too and records `flow_steps`/`flow_objective` in synthesis
results. Explicitly requesting another positive step count uses a uniform grid;
to change the default, set both `num_steps` and `sampling_time_grid` consistently.
Streaming retains one independent KV/conv cache per step; changing the time
grid midway through an utterance is rejected.

Monitor `imf/train_u_mse`, `imf/train_v_mse`, their validation counterparts,
weighted head losses, JVP RMS, and the sampled diagonal fraction. The weighted
loss values are not numerically comparable with the previous FM MSE. Generated
two-step CER/WER, timbre, Chinese fluency, and runtime remain the quality checks.

## Validation

Run the self-contained CPU regression suite from the repository root:

```bash
OMP_NUM_THREADS=2 python -m pytest -q tests/test_improved_mean_flow.py tests/test_flow_padding.py
```

The tests cover analytic loss/gradient agreement, predicted-v JVP directions,
padding invariance, diagonal sampling, strict FM migration, iMF checkpoint
reload, the two-step interval solver, streaming caches, and FP32/BF16 backward.

A bounded real-data check is available without any optimizer update:

```bash
python -m training.imf.smoke \
  --checkpoint /path/to/stage1_21000.ckpt \
  --data-dir "$STAGE2_DATA" --device cuda:0 \
  --output /path/to/imf_smoke.json
```

It selects short validation examples from both speakers, verifies all migrated
backbone tensors, runs full duration/prior/iMF backward, checks gradient groups,
then validates checkpoint reload and exactly two synthesis calls. It clears
nonpersistent position and streaming caches before comparing fresh-utterance
outputs; the existing encoders' caches can otherwise retain mixed-precision
values from earlier forwards. The temporary checkpoint is removed automatically.

The implementation check on September 15 passed with the real Stage 1 21k
checkpoint and the frozen Stage 2 data: all seven gradient groups were finite
and nonzero, fresh-cache checkpoint reload produced identical Mel, and sampling
used exactly two u calls with zero v-tail calls. There were **zero optimizer
updates**. This verifies the software path, not trained iMF audio quality or a
production batch-size/throughput claim.
