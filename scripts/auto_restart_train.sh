#!/bin/bash
# Auto-restart training loop: keeps restarting from latest.pt after DDP crashes
# until training completes naturally (epoch 700 or early stop).
#
# Usage: bash scripts/auto_restart_train.sh [GPU_IDS] [RESUME_CKPT]
#   GPU_IDS: comma-separated GPU IDs, default "0,1,2,3,4,5"
#   RESUME_CKPT: checkpoint to resume from, default = latest latest.pt
#
# Restarts at most MAX_RESTARTS times (default 30).
# Waits 30 seconds between restarts to let GPU memory release.
#
# Detects completion by looking for "Training complete" in the log.
# Detects crash by looking for "RuntimeError" / "ChildFailedError" /
# "Connection closed by peer" in the log.

set -u

GPUS="${1:-0,1,2,3,4,5}"
MAX_RESTARTS=30
SLEEP_BETWEEN=30

# Determine resume checkpoint
if [ -n "${2:-}" ]; then
  RESUME_CKPT="$2"
else
  RESUME_CKPT=$(ls -t /workspace/output/checkpoints/train/*/latest.pt 2>/dev/null | head -1)
fi

if [ -z "$RESUME_CKPT" ]; then
  echo "No latest.pt found, starting fresh training"
  RESUME_FLAG=""
else
  echo "Initial resume from: $RESUME_CKPT"
  RESUME_FLAG="--resume $RESUME_CKPT"
fi

cd /workspace

for i in $(seq 1 $MAX_RESTARTS); do
  echo ""
  echo "============================================================"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Restart attempt $i/$MAX_RESTARTS"
  echo "GPUs: $GPUS"
  echo "Resume: ${RESUME_CKPT:-<fresh>}"
  echo "============================================================"

  LOG_FILE="output/logs/train_autorestart_$(date +%Y%m%d_%H%M%S).log"
  echo "Log: $LOG_FILE"

  PYTHONPATH=/workspace python -m src.train \
    --config config/default.yaml \
    --gpus "$GPUS" \
    $RESUME_FLAG \
    > "$LOG_FILE" 2>&1
  EXIT_CODE=$?

  echo "Process exit code: $EXIT_CODE"

  # Check if training completed naturally
  if grep -q "Training complete" "$LOG_FILE" 2>/dev/null; then
    echo ""
    echo "============================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] TRAINING COMPLETE"
    echo "Final log: $LOG_FILE"
    echo "============================================================"
    # Print final 30 lines for summary
    tail -30 "$LOG_FILE"
    exit 0
  fi

  # Check for early stop
  if grep -qE "Early stopping|early stop|no improvement for [0-9]+ epoch" "$LOG_FILE" 2>/dev/null \
     && grep -q "Training complete" "$LOG_FILE" 2>/dev/null; then
    echo "Training stopped early (no improvement) — treating as complete"
    tail -30 "$LOG_FILE"
    exit 0
  fi

  # Crash: find newest latest.pt for next restart
  RESUME_CKPT=$(ls -t /workspace/output/checkpoints/train/*/latest.pt 2>/dev/null | head -1)
  if [ -z "$RESUME_CKPT" ]; then
    echo "ERROR: No latest.pt found after crash — cannot restart"
    echo "Last 50 lines of log:"
    tail -50 "$LOG_FILE"
    exit 1
  fi

  echo "Crash detected. Will restart from: $RESUME_CKPT"
  echo "Last 10 lines of crashed run:"
  tail -10 "$LOG_FILE"

  echo "Waiting $SLEEP_BETWEEN seconds for GPU memory release..."
  sleep $SLEEP_BETWEEN
done

echo ""
echo "============================================================"
echo "Reached MAX_RESTARTS=$MAX_RESTARTS"
echo "Last checkpoint: $RESUME_CKPT"
echo "Last log: $LOG_FILE"
echo "============================================================"
exit 1
