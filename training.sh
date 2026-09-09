#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PROJECT_ROOT="${PROJECT_ROOT:-$REPO_ROOT}"

: "${TRAIN_FILELIST:?Set TRAIN_FILELIST to an external Chinese-English training manifest}"
: "${VALID_FILELIST:?Set VALID_FILELIST to an external Chinese-English validation manifest}"

python "$REPO_ROOT/lits/train.py" experiment=en-zh "$@"
