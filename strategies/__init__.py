"""
Input strategies for AI detection models.

This module implements various input preprocessing strategies for both
foundation model-based and scratch training approaches.
"""

from .base import BaseInputStrategy
from .gaussian_noise import GaussianNoiseStrategy, MultiSigmaNoiseStrategy, AdaptiveNoiseStrategy
from .gaussian_pyramid import GaussianPyramidStrategy, MultiScaleGaussianStrategy
from .laplacian_pyramid import (
    LaplacianPyramidStrategy,
    LaplacianPyramidProcessedStrategy,
    LaplacianPyramidMultiScaleStrategy
)
from .smoothed_difference import SmoothedDifferenceStrategy, AdaptiveSmoothedStrategy
from .multi_scale_difference import MultiScaleDifferenceStrategy
from .multi_scale_raw import MultiScaleRawStrategy, MultiScaleRawProcessedStrategy, MultiScaleRawJpegStrategy
from .multi_scale_hybrid import MultiScaleHybridStrategy, MultiScaleHybridProcessedStrategy
from .median_filter import (
    MedianFilterStrategy,
    MedianFilterDifferenceStrategy,
    LaplacianPyramidDifferenceStrategy,
    LocalPixelDependencyStrategy
)
from .low_level_features import GaussianDifferenceStrategy, ResizeDifferenceStrategy
from .dino_intermediate import DinoIntermediateStrategy

__all__ = [
    'BaseInputStrategy',
    # Gaussian noise strategies
    'GaussianNoiseStrategy',
    'MultiSigmaNoiseStrategy',
    'AdaptiveNoiseStrategy',
    # Gaussian pyramid strategies (for foundation models)
    'GaussianPyramidStrategy',
    'MultiScaleGaussianStrategy',
    # Laplacian pyramid strategies (for scratch models)
    'LaplacianPyramidStrategy',
    'LaplacianPyramidProcessedStrategy',
    'LaplacianPyramidMultiScaleStrategy',
    # Smoothed difference strategies
    'SmoothedDifferenceStrategy',
    'AdaptiveSmoothedStrategy',
    # Multi-scale difference strategies
    'MultiScaleDifferenceStrategy',
    # Multi-scale raw strategies (for foundation models)
    'MultiScaleRawStrategy',
    'MultiScaleRawProcessedStrategy',
    'MultiScaleRawJpegStrategy',
    # Multi-scale hybrid strategies (combines raw + differences)
    'MultiScaleHybridStrategy',
    'MultiScaleHybridProcessedStrategy',
    # Median filter strategies (for scratch models)
    'MedianFilterStrategy',
    'MedianFilterDifferenceStrategy',
    'LaplacianPyramidDifferenceStrategy',
    'LocalPixelDependencyStrategy',
    'GaussianDifferenceStrategy',
    'ResizeDifferenceStrategy',
    'DinoIntermediateStrategy',
]
