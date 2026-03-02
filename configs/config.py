"""
Configuration file for LeakyCLIP training experiments.

This is the release version with only the recommended configuration.
"""

import os
import sys

# Add release root to path for importing paths.py
RELEASE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RELEASE_ROOT not in sys.path:
    sys.path.insert(0, RELEASE_ROOT)

from paths import PROGAN_TRAIN_DATA_PATH, AIGCTEST_DATA_PATH


# ============================================================================
# Experiment Configurations
# ============================================================================

EXPERIMENT_CONFIGS = {
    # ========================================================================
    # Ensemble with Discrepancy Loss and Loss Normalization (Recommended)
    # ========================================================================
    "ensemble_vitl16_raw_lpd_discrepancy": {
        "name": "ensemble_vitl16_raw_lpd_discrepancy",
        "model_name": "dinov3_vitl16",  # 1024-dim embeddings

        "decoder_type": "unet",
        "decoder_kwargs": {
            "strategy_channels": 15,  # MultiScaleRaw with level 0
            "base_channels": 64,
            "num_levels": 5,
            "num_heads": 8,
            "use_attention": True,
            "bottleneck_size": 14,
        },

        # Foundation branch: MultiScaleRaw strategy
        "use_multi_scale": True,
        "strategy_type": "multi_scale_raw",
        "strategy_levels": [0],
        "strategy_smooth_sigma": 2.0,
        "strategy_noise_std": 0.1,

        # Scratch branch: LPD strategy
        "use_lpd_strategy": True,
        "lpd_strategy_kernel_sizes": [3],

        "attack_eps": 16/255,
        "lr": 0.0001,
        "weight_decay": 0.01,
        "niter": 30,
        "batch_size": 32,

        # Epsilon randomization for domain generalization
        "eps_randomization": True,
        "eps_min": 4/255,   # ~4/255 minimum perturbation
        "eps_max": 32/255,  # ~32/255 maximum perturbation
        "eps_schedule": "random",  # Random sampling per batch

        "data_mode": "wang2020",
        "progan_train_data_path": PROGAN_TRAIN_DATA_PATH,

        "training_mode": "ensemble",
        "lambda_classification": 1.0,

        # Loss configuration - DISCREPANCY LOSS
        "loss_type": "discrepancy",
        "margin": 1,

        "checkpoint_dir": "./checkpoints/ensemble_vitl16_raw_lpd_discrepancy",
        "save_every": 5,
    },
}


# ============================================================================
# Validation Configurations
# ============================================================================

VALIDATION_CONFIGS = {
    "progan_validation": {
        "name": "progan_validation",
        "dataroot": os.path.join(AIGCTEST_DATA_PATH, "progan"),
        "data_mode": "wang2020",
        "batch_size": 32,
    },

    "all_generators": {
        "name": "all_generators",
        "dataroot": AIGCTEST_DATA_PATH,
        "data_mode": "wang2020",
        "batch_size": 32,
    },
}


def get_config(config_name: str) -> dict:
    """Get experiment configuration by name."""
    if config_name not in EXPERIMENT_CONFIGS:
        available = list(EXPERIMENT_CONFIGS.keys())
        raise ValueError(f"Unknown config: {config_name}. Available: {available}")
    return EXPERIMENT_CONFIGS[config_name]


def get_validation_config(config_name: str) -> dict:
    """Get validation configuration by name."""
    if config_name not in VALIDATION_CONFIGS:
        available = list(VALIDATION_CONFIGS.keys())
        raise ValueError(f"Unknown validation config: {config_name}. Available: {available}")
    return VALIDATION_CONFIGS[config_name]


def get_config_with_eps(config_name: str, eps_value: float, niter: int = 1, margin: float = 0.1) -> dict:
    """Get experiment configuration with custom epsilon value."""
    config = get_config(config_name).copy()
    config['attack_eps'] = eps_value
    eps_num = int(round(eps_value * 255))
    base_name = config['name']
    config['name'] = f"{base_name}_eps{eps_num}"
    config['checkpoint_dir'] = f"./ablation_experiment/eps_experiment/eps_{eps_num}/checkpoints"
    config['niter'] = niter
    config['margin'] = margin
    return config


def get_config_with_margin(config_name: str, margin_value: float, niter: int = 1, eps: float = 16/255) -> dict:
    """Get experiment configuration with custom margin value."""
    config = get_config(config_name).copy()
    config['margin'] = margin_value
    config['attack_eps'] = eps
    margin_str = str(margin_value).replace(".", "_")
    base_name = config['name']
    config['name'] = f"{base_name}_margin{margin_str}"
    config['checkpoint_dir'] = f"./ablation_experiment/margin_experiment/margin_{margin_str}/checkpoints"
    config['niter'] = niter
    return config


if __name__ == "__main__":
    print("Available experiment configurations:")
    for name, config in EXPERIMENT_CONFIGS.items():
        print(f"  - {name}")
        print(f"    Model: {config['model_name']}")
        print(f"    EPS: {config['attack_eps']:.4f}")
        print(f"    Training mode: {config['training_mode']}")
        print()
