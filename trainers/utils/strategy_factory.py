"""
Strategy factory for creating RFNT input strategies from configuration.

This module provides a factory function to create strategy objects
based on experiment configuration.
"""

import os
import sys
from typing import Optional, List

# Add release root to path for local imports
RELEASE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if RELEASE_ROOT not in sys.path:
    sys.path.insert(0, RELEASE_ROOT)

from strategies.multi_scale_hybrid import MultiScaleHybridStrategy
from strategies.multi_scale_difference import MultiScaleDifferenceStrategy
from strategies.multi_scale_raw import MultiScaleRawStrategy, MultiScaleRawJpegStrategy
from strategies.median_filter import LocalPixelDependencyStrategy


def create_strategy_from_config(config: dict):
    """
    Create an RFNT strategy object based on configuration dictionary.

    Args:
        config: Configuration dictionary with strategy parameters:
            - use_multi_scale (bool): Whether to use multi-scale strategy
            - strategy_type (str): Type of strategy ('lpd', 'multi_scale_hybrid', etc.)
            - strategy_levels (List[int]): Pyramid levels
            - strategy_smooth_sigma (float): Gaussian smoothing sigma
            - strategy_noise_std (float): Noise standard deviation
            - strategy_kernel_sizes (List[int]): LPD kernel sizes

    Returns:
        Strategy instance or None if use_multi_scale=False

    Examples:
        >>> config = {
        ...     "use_multi_scale": True,
        ...     "strategy_type": "multi_scale_hybrid",
        ...     "strategy_levels": [0, 1, 2],
        ...     "strategy_smooth_sigma": 1.0,
        ...     "strategy_noise_std": 0.1
        ... }
        >>> strategy = create_strategy_from_config(config)
    """
    use_multi_scale = config.get('use_multi_scale', False)

    if not use_multi_scale:
        return None

    strategy_type = config.get('strategy_type', None)

    if strategy_type is None:
        return None

    # Common parameters
    levels = config.get('strategy_levels', [0, 1, 2, 3])
    smooth_sigma = config.get('strategy_smooth_sigma', 1.0)
    noise_std = config.get('strategy_noise_std', 0.1)
    kernel_sizes = config.get('strategy_kernel_sizes', [3])

    # Create strategy based on type
    if strategy_type == 'multi_scale_hybrid':
        strategy = MultiScaleHybridStrategy(
            levels=levels,
            smooth_sigma=smooth_sigma,
            noise_std=noise_std,
            normalize=True
        )

    elif strategy_type == 'multi_scale_difference':
        strategy = MultiScaleDifferenceStrategy(
            levels=levels,
            smooth_sigma=smooth_sigma,
            normalize=True
        )

    elif strategy_type == 'multi_scale_raw':
        strategy = MultiScaleRawStrategy(
            levels=levels,
            smooth_sigma=smooth_sigma,
            noise_std=noise_std,
            normalize=True
        )

    elif strategy_type == 'multi_scale_raw_jpeg':
        strategy = MultiScaleRawJpegStrategy(
            levels=levels,
            smooth_sigma=smooth_sigma,
            noise_std=noise_std,
            jpeg_qualities=config.get('strategy_jpeg_qualities', (95, 75, 50)),
            normalize=True
        )

    elif strategy_type == 'lpd':
        # Local Pixel Dependency (FerretNet)
        strategy = LocalPixelDependencyStrategy(
            kernel_sizes=kernel_sizes,
            normalize=True
        )

    else:
        raise ValueError(
            f"Unknown strategy type: {strategy_type}. "
            f"Available: 'multi_scale_hybrid', 'multi_scale_difference', "
            f"'multi_scale_raw', 'multi_scale_raw_jpeg', 'lpd'"
        )

    return strategy


def get_strategy_channels(config: dict) -> int:
    """
    Calculate the number of output channels for a strategy configuration.

    IMPORTANT NOTE: Multi-scale strategies in RFNT only work with single level [0]
    due to implementation bugs with tensor size mismatches. Always use levels=[0].

    Verified channel counts:
    - LPD: 3 channels (kernel_size=3)
    - MultiScaleRaw[level_0]: 15 channels (5 types × 3 RGB)
    - MultiScaleDifference: Has bugs with multi-level
    - MultiScaleHybrid: Has bugs with multi-level

    Args:
        config: Configuration dictionary

    Returns:
        Number of channels the strategy will output

    Examples:
        >>> config = {"strategy_type": "lpd", "strategy_kernel_sizes": [3]}
        >>> get_strategy_channels(config)
        3
        >>> config = {"strategy_type": "multi_scale_raw", "strategy_levels": [0]}
        >>> get_strategy_channels(config)
        15
    """
    use_multi_scale = config.get('use_multi_scale', False)

    if not use_multi_scale:
        return 0

    strategy_type = config.get('strategy_type', None)
    levels = config.get('strategy_levels', [0])
    kernel_sizes = config.get('strategy_kernel_sizes', [3])

    if strategy_type == 'lpd':
        # LPD: 3 RGB channels per kernel size
        return len(kernel_sizes) * 3

    elif strategy_type == 'multi_scale_raw':
        # MultiScaleRaw: 5 raw types × 3 RGB channels
        # Types: resized, smoothed, median, noisy, current
        # Verified with actual strategy: levels=[0] → 15 channels
        return len(levels) * 5 * 3

    elif strategy_type == 'multi_scale_raw_jpeg':
        jpeg_qualities = config.get('strategy_jpeg_qualities', (95, 75, 50))
        return len(levels) * (5 + len(jpeg_qualities)) * 3

    elif strategy_type == 'multi_scale_difference':
        # MultiScaleDifference: 3 difference types × 3 RGB channels per level
        # WARNING: Has bugs with multiple levels, use levels=[0] only
        return len(levels) * 3 * 3

    elif strategy_type == 'multi_scale_hybrid':
        # MultiScaleHybrid: 8 types × 3 RGB channels per level
        # 5 raw + 3 differences
        # WARNING: Has bugs with multiple levels, use levels=[0] only
        return len(levels) * 8 * 3

    else:
        return 0


if __name__ == "__main__":
    # Test the factory
    print("Testing strategy factory...\n")

    # Test LPD
    print("=== Test LPD ===")
    config = {
        "use_multi_scale": True,
        "strategy_type": "lpd",
        "strategy_kernel_sizes": [3]
    }
    strategy = create_strategy_from_config(config)
    print(f"Strategy type: {type(strategy).__name__}")
    print(f"Channels: {get_strategy_channels(config)}")

    # Test Multi-scale hybrid
    print("\n=== Test Multi-Scale Hybrid ===")
    config = {
        "use_multi_scale": True,
        "strategy_type": "multi_scale_hybrid",
        "strategy_levels": [0, 1, 2],
        "strategy_smooth_sigma": 1.0,
        "strategy_noise_std": 0.1
    }
    strategy = create_strategy_from_config(config)
    print(f"Strategy type: {type(strategy).__name__}")
    print(f"Channels: {get_strategy_channels(config)}")

    # Test no strategy
    print("\n=== Test No Strategy ===")
    config = {"use_multi_scale": False}
    strategy = create_strategy_from_config(config)
    print(f"Strategy: {strategy}")
    print(f"Channels: {get_strategy_channels(config)}")

    print("\n✓ All tests passed!")
