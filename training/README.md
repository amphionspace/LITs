# Maintained training code

| Path | Purpose |
| --- | --- |
| `foundation/` | HiFiTTS/Premium indexing, statistics, capacity, training and fixed checkpoint evaluation |
| `stage2/` | Current joint-training public entry points, cached data and evaluation adapter |
| `majestic_scratch/` | Current 21k-backbone joint implementation, natural sampling, schedule and queued duration/evaluation checks |
| `common/` | Training status and shared synthesis/ASR/quality evaluation |

`majestic_scratch` is an active module name, not a retired experiment. Keep it stable for the live training/evaluation processes and saved configurations. The iMF feature branch adds `training/imf` and its Hydra preset without replacing FM.

Removed: obsolete single-voice and staged adaptation launchers, the former synthesis/QC orchestration, one-time duration/accent/model-health probes and report generators, reference/rescore/benchmark scripts, and temporary tests added during implementation. Current quota audits, text preparation, data loaders, preflight gates, checkpoint evaluation and the original repository tests remain.

Historical experiment reports and their evidence are retained. To replay a retired recipe, use the corresponding Git revision or the immutable `source/` snapshot saved inside its run directory. Do not run a historical launch command against the current entry points.

This cleanup changes source layout and supported entry points; it does not rewrite existing run data, checkpoints, frozen manifests or running training parameters.
