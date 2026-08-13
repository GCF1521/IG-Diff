#!/bin/bash
# Training launch script with debug info
export NCCL_SHM_DISABLE=1
export NCCL_DEBUG=INFO          # Show NCCL errors in detail
export TORCH_DISTRIBUTED_DEBUG=DETAIL  # Capture full tracebacks
export NCCL_TIMEOUT=1800       # 30 min timeout (default)

torchrun --nproc_per_node=8 -m src.train \
    --config config/default.yaml \
    --gpus 0,1,2,3,4,5,6,7
