"""
Dataset classes for AI detection models.

This module provides dataset loaders for training and evaluation
with support for different input strategies.
"""

from .progan import AdaptiveAIGCDataset, ProGANTrainingDataset, ProGANDataloader
from .aigctest import AIGCTestDataset, CrossGeneratorEvaluator, create_evaluation_dataloader

__all__ = [
    'AdaptiveAIGCDataset',
    'ProGANTrainingDataset',
    'ProGANDataloader',
    'AIGCTestDataset',
    'CrossGeneratorEvaluator',
    'create_evaluation_dataloader',
]