"""
Fusion strategies for flexible ensemble classifier.

All fusion strategies work with an arbitrary number of branches (N-branch compatible).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, List


class FusionStrategy(ABC):
    """
    Abstract base class for fusion strategies.

    All fusion strategies must work with an arbitrary number of branches,
    not just 2 or 3. The logits_dict contains logits from all branches.
    """

    @abstractmethod
    def forward(self, logits_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Fuse logits from all branches.

        Args:
            logits_dict: Dictionary mapping branch names to logits [B, 1]

        Returns:
            Fused logits [B, 1]
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement forward()")

    def get_parameters(self) -> List[nn.Parameter]:
        """
        Return learnable parameters for this fusion strategy.

        Used for optimizer parameter registration.

        Returns:
            List of learnable parameters (empty for non-learned fusion)
        """
        return []


class MaxFusion(FusionStrategy):
    """
    Maximum of all branch probabilities.

    Takes the maximum probability across all branches.
    Useful for evaluation (non-differentiable approximation during training).
    """

    def forward(self, logits_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Max fusion: select the highest probability from all branches.

        Args:
            logits_dict: Dict of branch logits

        Returns:
            Fused logits via max probability
        """
        if not logits_dict:
            raise ValueError("Cannot fuse empty logits_dict")

        # Convert to probabilities and stack
        probs = torch.stack([torch.sigmoid(l) for l in logits_dict.values()], dim=-1)  # [B, N]

        # Take max across branches
        max_probs, _ = torch.max(probs, dim=-1)  # [B]

        # Clamp to valid range and convert back to logits
        max_probs = torch.clamp(max_probs, min=1e-7, max=1 - 1e-7)
        logits = torch.log(max_probs / (1 - max_probs))

        return logits.unsqueeze(-1)  # [B, 1]


class AvgFusion(FusionStrategy):
    """
    Average of all branch logits.

    Simple equal-weight averaging across all branches.
    """

    def forward(self, logits_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Average fusion: mean of all branch logits.

        Args:
            logits_dict: Dict of branch logits

        Returns:
            Averaged logits
        """
        if not logits_dict:
            raise ValueError("Cannot fuse empty logits_dict")

        # Stack and average
        stacked = torch.stack(list(logits_dict.values()), dim=-1)  # [B, 1, N]
        return stacked.mean(dim=-1, keepdim=True)  # [B, 1]


class SumFusion(FusionStrategy):
    """
    Sum of all branch logits.

    Sums logits from all branches (scales output with number of branches).
    """

    def forward(self, logits_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Sum fusion: sum of all branch logits.

        Args:
            logits_dict: Dict of branch logits

        Returns:
            Summed logits
        """
        if not logits_dict:
            raise ValueError("Cannot fuse empty logits_dict")

        # Stack and sum
        stacked = torch.stack(list(logits_dict.values()), dim=-1)  # [B, 1, N]
        return stacked.sum(dim=-1, keepdim=True)  # [B, 1]


class LogitWeightedFusion(FusionStrategy):
    """
    Logit-weighted softmax fusion (confidence-based).

    Uses the logits themselves as weights via softmax.
    Higher logit = more confident = more weight in the ensemble.

    Args:
        temperature: Temperature for softmax (lower = sharper weighting)
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, logits_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Logit-weighted fusion: confidence-based dynamic weighting.

        Args:
            logits_dict: Dict of branch logits

        Returns:
            Weighted sum of logits
        """
        if not logits_dict:
            raise ValueError("Cannot fuse empty logits_dict")

        # Stack logits: [B, 1, N]
        stacked = torch.stack(list(logits_dict.values()), dim=-1)

        # Use logits as weights via softmax (higher logit = more weight)
        weights = F.softmax(stacked / self.temperature, dim=-1)  # [B, 1, N]

        # Weighted combination
        return (stacked * weights).sum(dim=-1, keepdim=True)  # [B, 1]


class LearnedWeightFusion(FusionStrategy, nn.Module):
    """
    Learnable fixed weights for each branch.

    Each branch gets a learnable weight parameter.
    Weights are normalized via softmax so they sum to 1.

    Args:
        num_branches: Number of branches (will be updated dynamically)
        initial_value: Initial value for all weights (0 = equal weights after softmax)
    """

    def __init__(self, num_branches: int = 0, initial_value: float = 0.0):
        FusionStrategy.__init__(self)
        nn.Module.__init__(self)

        self.num_branches = num_branches
        # Initialize to equal weights (logits that softmax to 1/N)
        self.weights = nn.Parameter(torch.full((num_branches,), initial_value))

    def forward(self, logits_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Learned-weight fusion: fixed learnable weights per branch.

        Args:
            logits_dict: Dict of branch logits

        Returns:
            Weighted sum of logits with learned weights
        """
        if not logits_dict:
            raise ValueError("Cannot fuse empty logits_dict")

        num_branches = len(logits_dict)

        # Resize weights if needed (when branches are added/removed)
        if num_branches != self.num_branches:
            self._resize_weights(num_branches)

        # Stack logits: [B, 1, N]
        stacked = torch.stack(list(logits_dict.values()), dim=-1)

        # Get softmax-normalized weights
        normalized_weights = F.softmax(self.weights[:num_branches], dim=0)  # [N]

        # Apply weights
        return (stacked * normalized_weights.view(1, -1, 1)).sum(dim=-1, keepdim=True)

    def _resize_weights(self, new_size: int):
        """Resize weight parameter when branch count changes."""
        if new_size == 0:
            return

        old_weights = self.weights.data
        new_weights = torch.full((new_size,), 0.0, device=old_weights.device)

        # Preserve old weights
        min_size = min(len(old_weights), new_size)
        new_weights[:min_size] = old_weights[:min_size]

        self.weights = nn.Parameter(new_weights)
        self.num_branches = new_size

    def get_parameters(self) -> List[nn.Parameter]:
        """Return learnable weights."""
        return [self.weights]


class ProductFusion(FusionStrategy):
    """
    Product of all branch probabilities (AND-like fusion).

    Multiplies all probabilities together. Requires all branches to be
    confident for high ensemble confidence.

    Uses odds ratio normalization to keep probabilities in valid range.
    """

    def forward(self, logits_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Product fusion: product of all branch probabilities.

        Args:
            logits_dict: Dict of branch logits

        Returns:
            Fused logits via probability product
        """
        if not logits_dict:
            raise ValueError("Cannot fuse empty logits_dict")

        # Convert to probabilities
        probs = [torch.sigmoid(l) for l in logits_dict.values()]

        # Product of probabilities
        ensemble_prob = probs[0]
        for p in probs[1:]:
            ensemble_prob = ensemble_prob * p

        # Product of complements
        complement_prod = torch.ones_like(ensemble_prob)
        for p in probs:
            complement_prod = complement_prod * (1 - p)

        # Normalize with odds ratio
        ensemble_prob = ensemble_prob / (ensemble_prob + complement_prod + 1e-10)
        ensemble_prob = torch.clamp(ensemble_prob, min=1e-7, max=1 - 1e-7)

        # Convert back to logits
        logits = torch.log(ensemble_prob / (1 - ensemble_prob))
        return logits.unsqueeze(-1)


class AttentionFusion(FusionStrategy, nn.Module):
    """
    Attention-based fusion over all branch logits.

    Projects logits through a small attention network to compute
    dynamic attention weights.

    Args:
        feature_dim: Dimension for query/key projection
        num_heads: Number of attention heads (currently uses 1 for simplicity)
    """

    def __init__(self, feature_dim: int = 64, num_heads: int = 1):
        FusionStrategy.__init__(self)
        nn.Module.__init__(self)

        self.feature_dim = feature_dim
        self.num_heads = num_heads

        # Project single logit to feature space
        self.to_query = nn.Linear(1, feature_dim)
        self.to_key = nn.Linear(1, feature_dim)
        self.to_value = nn.Linear(1, feature_dim)
        self.output_proj = nn.Linear(feature_dim, 1)

    def forward(self, logits_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Attention fusion: attention-weighted combination of logits.

        Args:
            logits_dict: Dict of branch logits

        Returns:
            Attention-weighted logits
        """
        if not logits_dict:
            raise ValueError("Cannot fuse empty logits_dict")

        # Stack: [B, N, 1]
        stacked = torch.stack(list(logits_dict.values()), dim=1)
        B, N, _ = stacked.shape

        # Create queries, keys, values
        Q = self.to_query(stacked)  # [B, N, feature_dim]
        K = self.to_key(stacked)
        V = self.to_value(stacked)

        # Attention scores
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.feature_dim)
        attn_weights = F.softmax(scores, dim=-1)  # [B, N, N]

        # Apply attention
        attended = torch.bmm(attn_weights, V)  # [B, N, feature_dim]
        output = self.output_proj(attended)  # [B, N, 1]

        # Average over branches for final output
        return output.mean(dim=1)  # [B, 1]

    def get_parameters(self) -> List[nn.Parameter]:
        """Return attention parameters."""
        return list(self.parameters())


# Fusion strategy registry
FUSION_REGISTRY = {
    "max": MaxFusion,
    "avg": AvgFusion,
    "sum": SumFusion,
    "logit_weighted": LogitWeightedFusion,
    "learned_weight": LearnedWeightFusion,
    "product": ProductFusion,
    "attention": AttentionFusion,
}


def get_fusion_strategy(name: str, **kwargs) -> FusionStrategy:
    """
    Factory function to create fusion strategies.

    Args:
        name: Fusion strategy name (must be in FUSION_REGISTRY)
        **kwargs: Additional arguments for the fusion strategy

    Returns:
        FusionStrategy instance

    Raises:
        ValueError: If name is not in registry

    Examples:
        >>> get_fusion_strategy("avg")
        AvgFusion()

        >>> get_fusion_strategy("logit_weighted", temperature=0.5)
        LogitWeightedFusion(temperature=0.5)

        >>> get_fusion_strategy("learned_weight", num_branches=3)
        LearnedWeightFusion(num_branches=3)
    """
    if name not in FUSION_REGISTRY:
        raise ValueError(
            f"Unknown fusion strategy: '{name}'. "
            f"Available: {list(FUSION_REGISTRY.keys())}"
        )
    return FUSION_REGISTRY[name](**kwargs)


def list_fusion_strategies() -> List[str]:
    """Return list of available fusion strategy names."""
    return list(FUSION_REGISTRY.keys())
