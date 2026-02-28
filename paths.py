"""
Path configuration for LeakyCLIP release.

This file manages all paths used in the release package.
All paths are relative to the release root directory.
"""

import os

# Get the release root directory (where this file is located)
RELEASE_ROOT = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# Data Paths (should be configured by user)
# ============================================================================

# Training data: ProGAN (Wang2020 format)
# Default: uses original LeakyCLIP data path
# User should set this via environment variable or modify this file
PROGAN_TRAIN_DATA_PATH = os.environ.get(
    "PROGAN_TRAIN_DATA_PATH",
    "/mnt/shared-storage-user/chenyunhao/LeakyCLIP/data/progan_training_data/progan_train"
)

# Validation data: AIGCTestset
# Default: uses original LeakyCLIP data path
# User should set this via environment variable or modify this file
AIGCTEST_DATA_PATH = os.environ.get(
    "AIGCTEST_DATA_PATH",
    "/mnt/shared-storage-user/chenyunhao/LeakyCLIP/data/AIGCTestset/test"
)

# ============================================================================
# Model Paths
# ============================================================================

# DINO model weights (relative to release)
DINO_WEIGHTS_DIR = os.path.join(RELEASE_ROOT, "models", "dino")

# DINOv3 repository
DINOV3_REPO_DIR = os.path.join(RELEASE_ROOT, "models", "dinov3_repo")

# DINOv2 repository
DINOV2_REPO_DIR = os.path.join(RELEASE_ROOT, "models", "dinov2")

# ============================================================================
# Checkpoint Paths
# ============================================================================

# Default checkpoint directory
DEFAULT_CHECKPOINT_DIR = os.path.join(RELEASE_ROOT, "checkpoints")

# ============================================================================
# Helper Functions
# ============================================================================

def get_data_path(data_type="progan"):
    """
    Get data path by type.

    Args:
        data_type: "progan" for training, "aigctest" for validation

    Returns:
        Absolute path to data directory
    """
    if data_type == "progan":
        return PROGAN_TRAIN_DATA_PATH
    elif data_type == "aigctest":
        return AIGCTEST_DATA_PATH
    else:
        raise ValueError(f"Unknown data type: {data_type}")


def set_data_path(data_type, path):
    """
    Set data path by type (modifies global variable).

    Args:
        data_type: "progan" or "aigctest"
        path: Absolute path to data directory
    """
    global PROGAN_TRAIN_DATA_PATH, AIGCTEST_DATA_PATH

    if data_type == "progan":
        PROGAN_TRAIN_DATA_PATH = path
    elif data_type == "aigctest":
        AIGCTEST_DATA_PATH = path
    else:
        raise ValueError(f"Unknown data type: {data_type}")


if __name__ == "__main__":
    print("LeakyCLIP Release Paths:")
    print(f"  RELEASE_ROOT: {RELEASE_ROOT}")
    print(f"  PROGAN_TRAIN_DATA_PATH: {PROGAN_TRAIN_DATA_PATH}")
    print(f"  AIGCTEST_DATA_PATH: {AIGCTEST_DATA_PATH}")
    print(f"  DINO_WEIGHTS_DIR: {DINO_WEIGHTS_DIR}")
    print(f"  DINOV3_REPO_DIR: {DINOV3_REPO_DIR}")
    print(f"  DINOV2_REPO_DIR: {DINOV2_REPO_DIR}")
