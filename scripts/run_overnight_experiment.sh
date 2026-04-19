#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/outputs_overnight}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/$RUN_ID}"
LOG_DIR="$OUTPUT_DIR/logs"
LOG_FILE="$LOG_DIR/overnight.log"
PID_FILE="$OUTPUT_DIR/run.pid"
STATUS_FILE="$OUTPUT_DIR/status.txt"
COMMAND_FILE="$OUTPUT_DIR/command.txt"
INNER_SCRIPT="$OUTPUT_DIR/run_inner.sh"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
DATA_DIR="${DATA_DIR:-../2018数据集/2018数据集}"
CHARGE_DATA_DIR="${CHARGE_DATA_DIR:-data/processed_110_paper}"
LAW_DATA_DIR="${LAW_DATA_DIR:-data/processed_law}"
BERT_DIR="${BERT_DIR:-chinese-bert-wwm-ext}"

EPOCHS="${EPOCHS:-4}"
MAX_LENGTH="${MAX_LENGTH:-256}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
DEVICE="${DEVICE:-cuda}"
DEEP_MODELS="${DEEP_MODELS:-fc rcnn}"

quote() {
  printf "%q" "$1"
}

usage() {
  cat <<'EOF'
Usage:
  scripts/run_overnight_experiment.sh
  scripts/run_overnight_experiment.sh --foreground

Environment overrides:
  EPOCHS=4 TRAIN_BATCH_SIZE=4 EVAL_BATCH_SIZE=8 GRADIENT_ACCUMULATION_STEPS=2
  DEVICE=cuda DEEP_MODELS="fc rcnn"
  OUTPUT_DIR=/path/to/output RUN_ID=custom_name
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

mkdir -p "$RUN_ROOT"
shopt -s nullglob
for old_pid_file in "$RUN_ROOT"/*/run.pid; do
  old_pid="$(cat "$old_pid_file" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "A previous experiment is still running: pid=$old_pid"
    echo "PID file: $old_pid_file"
    echo "Stop it or wait for it to finish before starting another full run."
    exit 1
  fi
done
shopt -u nullglob

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

if [[ ! -d "$ROOT_DIR/$LAW_DATA_DIR" ]]; then
  echo "Missing law processed data: $ROOT_DIR/$LAW_DATA_DIR" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

ROOT_Q="$(quote "$ROOT_DIR")"
PYTHON_Q="$(quote "$PYTHON_BIN")"
DATA_Q="$(quote "$DATA_DIR")"
CHARGE_Q="$(quote "$CHARGE_DATA_DIR")"
LAW_Q="$(quote "$LAW_DATA_DIR")"
OUTPUT_Q="$(quote "$OUTPUT_DIR")"
BERT_Q="$(quote "$BERT_DIR")"
DEVICE_Q="$(quote "$DEVICE")"
EPOCHS_Q="$(quote "$EPOCHS")"
MAX_LENGTH_Q="$(quote "$MAX_LENGTH")"
TRAIN_BATCH_Q="$(quote "$TRAIN_BATCH_SIZE")"
EVAL_BATCH_Q="$(quote "$EVAL_BATCH_SIZE")"
GRAD_ACCUM_Q="$(quote "$GRADIENT_ACCUMULATION_STEPS")"

cat > "$INNER_SCRIPT" <<EOF
#!/usr/bin/env bash
set -euo pipefail

cd $ROOT_Q
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg

STATUS_FILE=$(quote "$STATUS_FILE")
COMMAND_FILE=$(quote "$COMMAND_FILE")

finish() {
  rc=\$?
  if [[ \$rc -eq 0 ]]; then
    echo "COMPLETED \$(date '+%F %T')" > "\$STATUS_FILE"
  else
    echo "FAILED rc=\$rc \$(date '+%F %T')" > "\$STATUS_FILE"
  fi
  exit \$rc
}
trap finish EXIT

echo "RUNNING \$(date '+%F %T')" > "\$STATUS_FILE"

echo "== start =="
date
echo "== git =="
git log --oneline -1 || true
git status -sb || true
echo "== gpu =="
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader || true
echo "== python =="
$PYTHON_Q - <<'PY'
import torch, transformers
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("transformers", transformers.__version__)
PY

cat > "\$COMMAND_FILE" <<'CMD'
python scripts/run_pipeline.py \
  --data-dir DATA_DIR \
  --processed-dir CHARGE_DATA_DIR \
  --law-processed-dir LAW_DATA_DIR \
  --output-dir OUTPUT_DIR \
  --device DEVICE \
  --pretrained-model BERT_DIR \
  --deep-models DEEP_MODELS \
  --epochs EPOCHS \
  --max-length MAX_LENGTH \
  --train-batch-size TRAIN_BATCH_SIZE \
  --eval-batch-size EVAL_BATCH_SIZE \
  --gradient-accumulation-steps GRADIENT_ACCUMULATION_STEPS \
  --loss weighted_ce \
  --sampler none \
  --label-smoothing 0.05 \
  --pin-memory on \
  --persistent-workers off \
  --fallback-to-flat \
  --include-law \
  --skip-prepare \
  --skip-law-prepare
CMD

echo "== run_pipeline =="
$PYTHON_Q scripts/run_pipeline.py \\
  --data-dir $DATA_Q \\
  --processed-dir $CHARGE_Q \\
  --law-processed-dir $LAW_Q \\
  --output-dir $OUTPUT_Q \\
  --device $DEVICE_Q \\
  --pretrained-model $BERT_Q \\
  --deep-models $DEEP_MODELS \\
  --epochs $EPOCHS_Q \\
  --max-length $MAX_LENGTH_Q \\
  --train-batch-size $TRAIN_BATCH_Q \\
  --eval-batch-size $EVAL_BATCH_Q \\
  --gradient-accumulation-steps $GRAD_ACCUM_Q \\
  --loss weighted_ce \\
  --sampler none \\
  --label-smoothing 0.05 \\
  --pin-memory on \\
  --persistent-workers off \\
  --fallback-to-flat \\
  --include-law \\
  --skip-prepare \\
  --skip-law-prepare

echo "== final artifacts =="
find $OUTPUT_Q -maxdepth 3 -type f \\( \\
  -name 'results_table.csv' -o \\
  -name 'results_table_contrast.csv' -o \\
  -name 'results_table_intermediate.csv' -o \\
  -name 'final_summary.md' -o \\
  -name 'requirements_check.md' -o \\
  -name 'auc_summary.csv' \\
\\) | sort
echo "== end =="
date
EOF

chmod +x "$INNER_SCRIPT"
ln -sfn "$OUTPUT_DIR" "$RUN_ROOT/latest"

if [[ "$FOREGROUND" -eq 1 ]]; then
  echo "Running in foreground."
  echo "Output dir: $OUTPUT_DIR"
  "$INNER_SCRIPT" 2>&1 | tee "$LOG_FILE"
else
  nohup "$INNER_SCRIPT" > "$LOG_FILE" 2>&1 &
  pid=$!
  echo "$pid" > "$PID_FILE"
  echo "Started overnight experiment in background."
  echo "PID: $pid"
  echo "Output dir: $OUTPUT_DIR"
  echo "Log: $LOG_FILE"
  echo "Status: $STATUS_FILE"
  echo
  echo "Monitor with:"
  echo "  tail -f $(quote "$LOG_FILE")"
  echo "  cat $(quote "$STATUS_FILE")"
fi
