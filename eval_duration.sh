#!/bin/bash
# Duration evaluation pipeline:
#   1. Run inference on plain-text eval file (one utterance per line)
#   2. Compare inferred audio duration against reference audio by line order
#
# Usage:
#   SPK_ID=N ./eval_duration.sh <checkpoint> [infer_id]

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$REPO_ROOT/data_for_test/duration_eval"
EVAL_TEXTS="$EVAL_DIR/eval_texts.txt"
REF_DIR="$EVAL_DIR/ref_audio"

TEST_CHECKPOINT="${1:?Usage: SPK_ID=N $0 <checkpoint> [infer_id]}"
SPK_ID="${SPK_ID:?SPK_ID is required (e.g. SPK_ID=1 $0 model.ckpt)}"
INFER_ID=${2:-"duration_eval"}
INFER_DIR="$REPO_ROOT/infer_output/$INFER_ID"

echo "============================================"
echo "  Duration Evaluation Pipeline"
echo "  Eval texts:     $EVAL_TEXTS"
echo "  Ref audio dir:  $REF_DIR"
echo "  Infer output:   $INFER_DIR"
echo "  Checkpoint:     $TEST_CHECKPOINT"
echo "  Speaker id:     $SPK_ID"
echo "============================================"

echo ""
echo "[Step 1/2] Running inference..."
python infer_e2e.py \
  --model_lang en-zh-dict \
  --checkpoint "$TEST_CHECKPOINT" \
  --input_txt "$EVAL_TEXTS" \
  --spk_id "$SPK_ID" \
  --output_dir "$INFER_DIR" \
  --output_txt "$INFER_DIR/meta.txt" \
  --output_sample_rate 24000 \
  --num_decoding_left_chunks -1 \
  --max_infer_tokens 0 \
  --length_scale 1

echo ""
echo "[Step 2/2] Evaluating duration..."
python "$EVAL_DIR/eval_duration.py" \
  --ref_dir "$REF_DIR" \
  --infer_dir "$INFER_DIR" \
  --output "$INFER_DIR/duration_results.tsv"

echo ""
echo "Done. Results saved to: $INFER_DIR/duration_results.tsv"
