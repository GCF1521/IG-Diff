#!/bin/bash
# v8.1: resume from v8 latest.pt + stronger prior_adherence loss (lambda 0.3 → 1.5)
# Uses 4 GPUs (0-3) because GPUs 4-7 have leaked CUDA contexts
export NCCL_SHM_DISABLE=1
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_TIMEOUT=1800

torchrun --nproc_per_node=4 -m src.train \
    --config config/default.yaml \
    --gpus 0,1,2,3 \
    --resume /workspace/output/checkpoints/train/20260628_192357/latest.pt
