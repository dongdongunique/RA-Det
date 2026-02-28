"""
Base class for models trained from scratch.
"""

import torch
import torch.nn as nn
from typing import Dict, Any
from ..base import BaseDetectionModel


class BaseScratchModel(BaseDetectionModel):
    """Base class for models trained from scratch"""

    def __init__(self, input_channels: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.input_channels = input_channels
        self.backbone = None
        self.classifier = None

    def build_classifier(self, feature_dim: int):
        """Build classification head"""
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(feature_dim, self.num_classes)
        )