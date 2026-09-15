# Stage 2 teacher → two-step student

This recipe distills the completed Stage 2 FM model at step 170,000 on the
full MajesticVoice **training** split: 53,732 prompts, approximately 100 hours
of source recordings (Chinese 50h, English 25h, mixed 25h). The held-out
Majestic validation split has 400 prompts. LJSpeech is excluded from this
training run; the existing four-group 650-item speech evaluation is retained.

The JSONL loader consumes the audited `ids`, `tones`, and `speaker` fields
directly. It does not re-tokenize or truncate long prompts. The recordings are
not read during distillation: the frozen teacher generates trajectories online.

## Network and objective

The student starts from the **completed Stage 2** teacher, including both
speaker rows. Only the decoder estimator and interval projection are trained.
Text, duration, speaker, and decoder condition encoders remain frozen. The
teacher stays in evaluation mode and has no trainable parameters.

The interval network is the existing IntMeanFlow wrapper:

```
concat(sinusoidal(r), sinusoidal(t)) → Linear([0, I] at initialization)
                                   → pretrained time_mlp → decoder
```

Both sinusoidal inputs retain the pretrained scale **1000**. This is the old
two-endpoint encoding, not the iMF interval-derivative formulation. The initial
embedding equals the teacher's embedding at `t`; this does not imply that the
student's entire coarse trajectory equals the teacher's fine trajectory.
There is one velocity output and no auxiliary v head or JVP.

Teacher Euler sampling uses 16 steps; the student uses `[0, 0.5, 1]`. The loss
is `endpoint_MSE + 0.5 * trajectory_MSE + mean_velocity_MSE`, masked over valid
Mel frames. ODE states and loss reductions are FP32; network computation uses
BF16. The initial run uses whole-utterance decoding. Streaming KV-cache
distillation can be selected with `streaming: true` in the run plan.

## Streaming experiment

The streaming run uses the same data, initialization, loss, `[0, 0.5, 1]` time
grid and training budget. It follows the older recipe's 100-frame chunks and
20-frame decoder left context, with a full-utterance bidirectional condition
encoder. This is streaming acoustic decoding after full-text conditioning.
The legacy pre-lookahead setting is 3; with full condition precomputation,
the emitted decoder chunks still contain 100 new frames, with the remainder
merged into the last chunk.

Both teacher and student train through `forward_streaming`, carrying attention
and convolution caches. These direct method calls bypass DDP.forward, so the
streaming trainer broadcasts initialization and explicitly averages all
gradients across ranks before clipping and Adam. Audits check identical
student parameter hashes across ranks at updates 1 and 100.

Streaming evaluation uses chunk-outer, ODE-step-inner decoding with a separate
cache per solver step, then chunked Vocos with 8 Mel frames of context and a
Hann waveform crossfade, matching `inference_stream.py`. Checkpoint loading
restores the streaming geometry from `distill_args`; `--distilled` selects
streaming synthesis automatically. Teacher baselines pass `--streaming`.
The preflight compares training and inference trajectories for single chunks,
multiple chunks and an odd-length tail, and checks exact waveform save/load
round trips. Nonstreaming checkpoints and their results remain separate.

### Parallel evaluation of streaming training

Set `parallel_streaming: true` in the run plan to remove the serial chunk loop
from training. `parallel_streaming.py` evaluates each decoder layer across the
utterance, preserving the exact visibility of the cached implementation. Its
attention mask intersects the network's static chunk boundary, the actual
processing chunk boundary at each resolution, and the frame-level left window.
Padding follows the cached decoder: convolution masks apply, but attention
does not introduce an additional padding mask.

At every transposed-convolution boundary, the already-emitted last sample of
the previous chunk excludes the next chunk's first input. A differentiable
boundary overwrite reproduces that behavior; a plain whole-utterance forward
would leak future context there. The supported kernel/stride/padding and chunk
alignment are checked explicitly. Deployment continues to use the existing KV
cache path; weights, time grid, loss, chunk size and context limits are unchanged.

Differential checks cover the teacher and trained student, short and merged-tail
chunks, masked frames, complete 16-step/2-step objectives and gradients. FP32
matches to numerical precision. BF16 has normal kernel-rounding differences;
dropout remains enabled during training, with a different random-mask ordering
under parallel execution. Longest-prompt capacity and two-rank resume audits are
required before the supervisor accepts `performance_preflight.json`.

To resume, the plan supplies `resume_checkpoint` and `resume_global_step`.
The supervisor verifies saved optimizer state and computes the remaining budget,
so resuming 1k with `max_steps: 10000` performs 9k additional updates. The trainer
checks immutable distillation settings, restores the distributed sampler's
epoch/batch position, and records optimizer counters in `resume_state.json`.
Legacy checkpoints do not store per-rank RNG states, so this is not a bitwise
continuation of their random noise/dropout sequence. TensorBoard purges events
after the restored step; completed evaluations are retained and not rerun.

## Budget

- Learning rate: `2e-5`, AdamW, no weight decay; clip gradient norm at 1.
- Two GPUs, batch 16 per GPU, effective batch 32, no accumulation.
- 10,000 updates: roughly 320,000 prompt presentations / 53,732 = 5.96 passes.
  The old 9,796-prompt, batch-4, 10k run processed about 4.08 passes.
- Temperature 1.0 matches Stage 2 evaluation; the old small experiment used 0.667.
- Seed 20260915. Checkpoint and deterministic validation every 1,000 updates.
- Speech smoke evaluation at 1k (8 per group); full 650-item evaluation at 5k
  and 10k. Full teacher 2/16-step baselines run after the first smoke evaluation.
  Existing Stage 2 10-step results remain the historical reference.

## Launch and verification

The external run directory contains `plan.json`, train/validation JSONLs,
teacher/vocoder hashes and all outputs. Run `stage2_preflight` with the same
teacher, data and precision, then:

```bash
python -m meanflow_distill.stage2_run --run-dir "$DISTILL_RUN" --preflight-ddp
python -m meanflow_distill.stage2_run --run-dir "$DISTILL_RUN"
```

The supervisor requires successful capacity/round-trip and two-GPU preflight
reports. It refuses to start over an existing training state, locks the shared
run directory, and records hostname as well as PIDs. Run it from this worktree
using the LITs Python environment. Evaluation uses a separate process on GPU 1.

Startup audits verify exact FM initialization, time-embedding initialization,
teacher trajectory agreement with production decoding, finite gradients,
parameter updates in every estimator group, frozen teacher/conditioning
weights, and an exact save/load inference round trip. The formal trainer repeats
the parameter audit at steps 1 and 100. Save files are published atomically.

Student files carry the teacher architecture hyperparameters, the complete
student state, optimizer state, time grid, metrics and distillation arguments.
Load them with `meanflow_distill.stage2_support.load_student`; do not interpret
them as plain FM Lightning checkpoints. The common evaluator accepts
`--distilled`, an explicit vocoder, and the frozen evaluation data directory.
