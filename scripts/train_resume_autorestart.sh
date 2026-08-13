#!/bin/bash
# Resume training from latest checkpoint with auto-restart on failure.
# Handles the known gloo TCP connection drop issue by relaunching.
#
# Usage: bash scripts/train_resume_autorestart.sh

cd /workspace

export NCCL_SHM_DISABLE=1
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=7200          # 2 hours (raised from 30 min default)
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export GLOO_TIMEOUT_SECONDS=7200  # gloo transport timeout
export PYTHONUNBUFFERED=1

RESUME_CKPT="/workspace/output/checkpoints/train/20260629_165231/best.pt"
MAX_RESTARTS=20
RESTART_COUNT=0

mkdir -p output/logs

while [ $RESTART_COUNT -lt $MAX_RESTARTS ]; do
    echo ""
    echo "============================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Restart attempt $((RESTART_COUNT+1))/$MAX_RESTARTS"
    echo "Resuming from: $RESUME_CKPT"
    echo "============================================================"

    # Always resume from best.pt (epoch 449, denoising_loss=0.0570; EMA loss=0.0429)
    echo "Using best.pt as resume checkpoint: $RESUME_CKPT"

    torchrun --nproc_per_node=7 -m src.train \
        --config config/default.yaml \
        --gpus 0,1,2,3,5,6,7 \
        --resume "$RESUME_CKPT" \
        2>&1 | tee -a output/logs/train_residual_prior.log

    EXIT_CODE=${PIPESTATUS[0]}
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Exit code: $EXIT_CODE"

    if [ $EXIT_CODE -eq 0 ]; then
        echo "Training completed successfully."
        break
    fi

    RESTART_COUNT=$((RESTART_COUNT+1))
    echo "Training crashed. Waiting 30s before restart..."
    sleep 30
done

if [ $RESTART_COUNT -ge $MAX_RESTARTS ]; then
    echo "Reached max restarts ($MAX_RESTARTS). Giving up."
    exit 1
fi
