"""
Base classes for intermediate-layer foundation models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any

import torch

from .base import BaseFoundationModel


class BaseIntermediateFoundationModel(BaseFoundationModel, ABC):
    """Base class for models using intermediate encoder representations."""

    @abstractmethod
    def get_intermediate_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """Return intermediate tokens as (B, N, D)."""
        raise NotImplementedError

    @abstractmethod
    def aggregate_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Aggregate tokens to (B, D)."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.get_intermediate_tokens(x)
        features = self.aggregate_tokens(tokens)
        return self.classifier(features)

    def get_config(self) -> Dict[str, Any]:
        base = super().get_config()
        base.update({
            "model_type": self.model_type,
        })
        return base
