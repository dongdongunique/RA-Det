#!/bin/bash
################################################################################
# Loss Normalization Training Script
#
# Usage:
#   bash main.sh --gpus NUM --niter NUM
################################################################################

set -e

################################################################################
# Configuration
################################################################################

NUM_GPUS=8
NITER=1
EPS_VALUE=16
EPS_DECIMAL=0.06274509803921569
MARGIN_VALUE=1.0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(dirname "$SCRIPT_DIR")"
EXPERIMENT_DIR="${WORK_DIR}/experiment_results"

################################################################################
# Parse Arguments
################################################################################

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --niter)
            NITER="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

################################################################################
# Setup
################################################################################

cd "$WORK_DIR"
mkdir -p "$EXPERIMENT_DIR"

echo "========================================================================"
echo "LOSS NORMALIZATION TRAINING"
echo "========================================================================"
echo "Num GPUs:       $NUM_GPUS"
echo "Iterations:     $NITER"
echo "Output Dir:     $EXPERIMENT_DIR"
echo "========================================================================"

################################################################################
# Training
################################################################################

CHECKPOINT_DIR="${EXPERIMENT_DIR}/checkpoints"

CMD="torchrun --nproc_per_node=$NUM_GPUS train.py \
    --config ensemble_vitl16_raw_lpd_discrepancy \
    --eps $EPS_DECIMAL \
    --margin $MARGIN_VALUE \
    --niter $NITER \
    --checkpoint-dir \"$CHECKPOINT_DIR\" \
    --four-branch-ensemble \
    --normalize-loss"

echo "Running: $CMD"
eval "$CMD 2>&1 | tee \"${EXPERIMENT_DIR}/training.log\""

echo "Done! Results in: $EXPERIMENT_DIR"
