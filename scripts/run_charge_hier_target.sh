#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/outputs_charge_target}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/$RUN_ID}"
LOG_DIR="$OUTPUT_DIR/logs"
LOG_FILE="$LOG_DIR/charge_hier_target.log"
PID_FILE="$OUTPUT_DIR/run.pid"
STATUS_FILE="$OUTPUT_DIR/status.txt"
INNER_SCRIPT="$OUTPUT_DIR/run_inner.sh"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
CHARGE_DATA_DIR="${CHARGE_DATA_DIR:-data/processed_110_paper}"
BERT_DIR="${BERT_DIR:-chinese-bert-wwm-ext}"
REUSE_FLAT_DIR="${REUSE_FLAT_DIR:-$ROOT_DIR/outputs_overnight/latest/deep_models}"

EPOCHS="${EPOCHS:-4}"
MAX_LENGTH="${MAX_LENGTH:-256}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
DEVICE="${DEVICE:-cuda}"
TARGET_MODELS="${TARGET_MODELS:-rcnn}"
TARGET_SEEDS="${TARGET_SEEDS:-42 2024 3407}"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_charge_hier_target.sh
  scripts/run_charge_hier_target.sh --foreground

Environment overrides:
  EPOCHS=4 MAX_LENGTH=256 TARGET_MODELS="rcnn fc" TARGET_SEEDS="42 2024 3407"
  REUSE_FLAT_DIR=/path/to/previous/deep_models
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

FOREGROUND=0
if [[ "${1:-}" == "--foreground" ]]; then
  FOREGROUND=1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing Python environment: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -d "$ROOT_DIR/$BERT_DIR" ]]; then
  echo "Missing local BERT directory: $ROOT_DIR/$BERT_DIR" >&2
  exit 1
fi
if [[ ! -d "$ROOT_DIR/$CHARGE_DATA_DIR" ]]; then
  echo "Missing charge processed data: $ROOT_DIR/$CHARGE_DATA_DIR" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
ln -sfn "$OUTPUT_DIR" "$RUN_ROOT/latest"

cat > "$INNER_SCRIPT" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT_DIR"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg

finish() {
  rc=\$?
  if [[ \$rc -eq 0 ]]; then
    echo "COMPLETED \$(date '+%F %T')" > "$STATUS_FILE"
  else
    echo "FAILED rc=\$rc \$(date '+%F %T')" > "$STATUS_FILE"
  fi
  exit \$rc
}
trap finish EXIT

echo "RUNNING \$(date '+%F %T')" > "$STATUS_FILE"
echo "== start =="; date
echo "== git =="; git log --oneline -1 || true; git status -sb || true
echo "== gpu =="; nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader || true

mkdir -p "$OUTPUT_DIR"
if [[ -f "$REUSE_FLAT_DIR/metrics.json" ]]; then
  ln -sfn "$REUSE_FLAT_DIR" "$OUTPUT_DIR/deep_models"
  echo "Reusing flat models: $REUSE_FLAT_DIR"
else
  echo "Reusable flat models not found; training flat FC/RCNN first."
  "$PYTHON_BIN" scripts/train_deep_models.py \
    --data-dir "$CHARGE_DATA_DIR" \
    --output-dir "$OUTPUT_DIR/deep_models" \
    --models fc rcnn \
    --device "$DEVICE" \
    --pretrained-model "$BERT_DIR" \
    --epochs "$EPOCHS" \
    --max-length "$MAX_LENGTH" \
    --train-batch-size "$TRAIN_BATCH_SIZE" \
    --eval-batch-size "$EVAL_BATCH_SIZE" \
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
    --loss weighted_ce \
    --sampler none \
    --label-smoothing 0.05 \
    --pin-memory on \
    --persistent-workers off
fi

read -r -a MODEL_ARGS <<< "$TARGET_MODELS"
read -r -a SEED_ARGS <<< "$TARGET_SEEDS"

"$PYTHON_BIN" scripts/train_charge_hier_multitask.py \
  --data-dir "$CHARGE_DATA_DIR" \
  --output-dir "$OUTPUT_DIR/charge_hier_multitask" \
  --flat-dir "$OUTPUT_DIR/deep_models" \
  --models "\${MODEL_ARGS[@]}" \
  --seeds "\${SEED_ARGS[@]}" \
  --device "$DEVICE" \
  --pretrained-model "$BERT_DIR" \
  --epochs "$EPOCHS" \
  --max-length "$MAX_LENGTH" \
  --train-batch-size "$TRAIN_BATCH_SIZE" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
  --loss weighted_ce \
  --sampler none \
  --label-smoothing 0.05 \
  --class-weight-max 3.0 \
  --coarse-loss-weight 0.3 \
  --consistency-loss-weight 0.2 \
  --optimize-profile windows_4060ti_best \
  --pin-memory on \
  --persistent-workers off \
  --prefetch-factor 2

"$PYTHON_BIN" scripts/make_results_table.py \
  --output-dir "$OUTPUT_DIR" \
  --save-path "$OUTPUT_DIR/results_table.csv"

"$PYTHON_BIN" scripts/show_final_results.py \
  --output-dir "$OUTPUT_DIR" \
  --export-dir "$OUTPUT_DIR/final_report" \
  --skip-table-refresh

echo "== requirement check =="
cat "$OUTPUT_DIR/final_report/requirements_check.md" || true
echo "== end =="; date
EOF

chmod +x "$INNER_SCRIPT"

if [[ "$FOREGROUND" -eq 1 ]]; then
  echo "Running charge hierarchy target in foreground."
  echo "Output dir: $OUTPUT_DIR"
  "$INNER_SCRIPT" 2>&1 | tee "$LOG_FILE"
else
  nohup "$INNER_SCRIPT" > "$LOG_FILE" 2>&1 &
  pid=$!
  echo "$pid" > "$PID_FILE"
  echo "Started charge hierarchy target run in background."
  echo "PID: $pid"
  echo "Output dir: $OUTPUT_DIR"
  echo "Log: $LOG_FILE"
  echo "Status: $STATUS_FILE"
fi
