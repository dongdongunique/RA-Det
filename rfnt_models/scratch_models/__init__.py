"""
Scratch model implementations for AI detection.

This module contains models trained from scratch on Laplacian pyramid
differences and other preprocessed inputs.
"""

from .base import BaseScratchModel
from .laplacian_pyramid import (
    LaplacianPyramidCNN,
    MultiScaleLaplacianNet,
    LaplacianTransformer
)
from .multi_scale_difference_cnn import (
    MultiScaleDifferenceCNN,
    MultiScaleDifferenceNet
)

__all__ = [
    'BaseScratchModel',
    # Laplacian pyramid models
    'LaplacianPyramidCNN',
    'MultiScaleLaplacianNet',
    'LaplacianTransformer',
    # Multi-scale difference models
    'MultiScaleDifferenceCNN',
    'MultiScaleDifferenceNet',
]