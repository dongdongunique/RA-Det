#!/bin/bash
################################################################################
# Validation Script for Trained Checkpoint
#
# Uses train.py --validate-checkpoint to ensure same initialization as training.
#
# Usage:
#   bash validate.sh [--gpus NUM] [--checkpoint PATH]
################################################################################

set -e

################################################################################
# Configuration
################################################################################

NUM_GPUS=8
CHECKPOINT_PATH="/mnt/shared-storage-user/chenyunhao/LeakyCLIP/ai_generated_image_detection/releases/experiment_results_reference/checkpoints/checkpoint_best.pt"
CONFIG="ensemble_vitl16_raw_lpd_discrepancy"
EPS_VALUE=16
EPS_DECIMAL=0.06274509803921569
MARGIN_VALUE=1.0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(dirname "$SCRIPT_DIR")"

################################################################################
# Parse Arguments
################################################################################

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT_PATH="$2"
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

echo "========================================================================"
echo "VALIDATION (using train.py initialization)"
echo "========================================================================"
echo "Num GPUs:       $NUM_GPUS"
echo "Checkpoint:     $CHECKPOINT_PATH"
echo "Config:         $CONFIG"
echo "========================================================================"

################################################################################
# Validation - uses train.py --validate-checkpoint for exact same init as training
################################################################################

CMD="torchrun --nproc_per_node=$NUM_GPUS train.py \
    --config $CONFIG \
    --eps $EPS_DECIMAL \
    --margin $MARGIN_VALUE \
    --four-branch-ensemble \
    --normalize-loss \
    --validate-checkpoint \"$CHECKPOINT_PATH\""

echo "Running: $CMD"
eval "$CMD"

echo "Done!"