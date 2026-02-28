"""
Base input strategy for AI detection models.

This module defines the abstract base class for all input preprocessing strategies
used in AI-generated image detection.
"""

from abc import ABC, abstractmethod
from typing import List
import torch
import torch.nn.functional as F


class BaseInputStrategy(ABC):
    """Abstract base class for input preprocessing strategies"""

    @abstractmethod
    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        """Apply preprocessing to input image

        Args:
            image: Input image tensor of shape (C, H, W) or (B, C, H, W)

        Returns:
            Preprocessed image tensor
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return strategy name for logging"""
        pass

    @abstractmethod
    def get_output_channels(self) -> int:
        """Return number of output channels for model input"""
        pass

    @abstractmethod
    def get_model_type(self) -> str:
        """Return whether this strategy is for 'scratch' or 'foundation' models"""
        pass