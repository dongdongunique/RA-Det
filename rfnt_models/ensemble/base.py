"""
Base classes for flexible ensemble classifier.

Provides BaseBranch abstract class that all ensemble branches must inherit from.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import List


class BaseBranch(nn.Module):
    """
    Abstract base class for all ensemble branches.

    Each branch has:
    - A unique name/key for identification
    - Its own input processing logic
    - Its own output logits

    Subclasses must implement:
    - forward(): Extract inputs from kwargs and compute logits
    - get_input_keys(): Declare required input keys

    Example:
        class MyBranch(BaseBranch):
            def __init__(self, name="my_branch"):
                super().__init__(name)
                self.mlp = nn.Linear(64, 1)

            def forward(self, **kwargs):
                x = kwargs.get("my_input")  # Extract by key
                return self.mlp(x)

            def get_input_keys(self):
                return ["my_input"]
    """

    def __init__(self, name: str):
        """
        Initialize the branch.

        Args:
            name: Unique identifier for this branch (used for routing and logging)
        """
        super().__init__()
        self.name = name

    @abstractmethod
    def forward(self, **kwargs) -> torch.Tensor:
        """
        Forward pass for this branch.

        Each branch extracts its required inputs from kwargs by name.
        This allows flexible input routing without hardcoding signatures.

        Args:
            **kwargs: All inputs from the ensemble forward pass

        Returns:
            Logits tensor of shape [B, 1] (or [B] for binary classification)

        Example:
            def forward(self, **kwargs):
                multi_scale_images = kwargs.get("multi_scale_raw_images")
                return self.model(multi_scale_images)
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement forward()")

    def get_input_keys(self) -> List[str]:
        """
        Return list of input keys this branch requires.

        This is used for validation and debugging to ensure all required
        inputs are provided to the ensemble.

        Returns:
            List of string keys that this branch looks for in kwargs

        Example:
            def get_input_keys(self):
                return ["multi_scale_raw_images", "lpd_features"]
        """
        return []

    def extra_repr(self) -> str:
        """Extra representation for print(model)"""
        return f"name={self.name}"
