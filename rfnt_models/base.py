"""
Base classes for trainable AI detection models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import torch
import torch.nn as nn


class BaseDetectionModel(nn.Module, ABC):
    """Base class for all trainable AI detection models"""

    def __init__(self, num_classes: int = 2,pretrained=False):
        super().__init__()
        self.num_classes = num_classes
        self.model_type = None

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model"""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return model name for logging"""
        pass

    def get_config(self) -> Dict[str, Any]:
        """Return model configuration"""
        return {
            'model_type': self.model_type,
            'num_classes': self.num_classes
        }