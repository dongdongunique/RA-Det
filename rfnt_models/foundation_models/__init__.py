"""
Foundation model implementations for AI detection.

This module contains foundation model-based detectors that use pretrained
encoders with trainable classifiers.
"""

from .base import BaseFoundationModel
from .gaussian_pyramid import (
    SplitChannelFoundationModel,
    GaussianPyramidFoundationModel,
    ScaleAwareFoundationModel
)
from .dino_intermediate import DinoIntermediateRINEModel
from .intermediate_base import BaseIntermediateFoundationModel

__all__ = [
    'BaseFoundationModel',
    # Generic foundation models
    'SplitChannelFoundationModel',
    # Gaussian pyramid models (backward compatibility)
    'GaussianPyramidFoundationModel',
    'ScaleAwareFoundationModel',
    'DinoIntermediateRINEModel',
    'BaseIntermediateFoundationModel',
]
