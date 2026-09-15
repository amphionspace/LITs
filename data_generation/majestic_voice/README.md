# Shared synthesis and evaluation support

This directory retains only the operations used by the current collection and checkpoint evaluators:

- `serve.sh` / `deploy.yaml`: local VoxCPM2 service.
- `synthesize.py`: atomic 48/24 kHz audio validation and writing (`save_audio`).
- `quality_worker.py`: shared CER/WER, normalization and tone comparison.
- `quality_metrics.py`: speaker similarity and DNSMOS.
- `omni_py310.patch` / `environment_versions.json`: service environment provenance.

The former requests-file producer, standalone QC worker, retry selector and historical dataset conversion scripts have been removed. The current collection entry point is [majestic_200h](../majestic_200h/README.md); thresholds live in the external dataset's frozen `config.json`.
