"""
Flexible Ensemble Classifier Module

This module provides a flexible ensemble classifier architecture that supports
an arbitrary number of classifier branches with generic fusion strategies.

Components:
- BaseBranch: Abstract base class for all ensemble branches
- FusionStrategy: Abstract base class for fusion methods
- Concrete branches: FoundationBranch, ScratchBranch, L2DistanceBranch, etc.
- Fusion strategies: MaxFusion, AvgFusion, SumFusion, LogitWeightedFusion, etc.
- FlexibleEnsembleClassifier: Main ensemble class with dynamic branch registration
- BackwardCompatibleEnsemble: Wrapper mimicking old EnsembleClassifier API

Usage:
    from rfnt_models.ensemble import FlexibleEnsembleClassifier, FoundationBranch, ScratchBranch, L2DistanceBranch

    # Create ensemble
    ensemble = FlexibleEnsembleClassifier(fusion_method="logit_weighted")

    # Register branches
    ensemble.register_branch(FoundationBranch(...))
    ensemble.register_branch(ScratchBranch(...))
    ensemble.register_branch(L2DistanceBranch(...))

    # Forward pass
    ensemble_logits, branch_logits = ensemble(
        return_all_logits=True,
        multi_scale_raw_images=msr,
        lpd_features=lpd,
        l2_distance=l2_dist
    )
"""

from .base import BaseBranch
from .branches import FoundationBranch, ScratchBranch, L2DistanceBranch, EmbeddingDiffBranch
from .fusion import (
    FusionStrategy,
    MaxFusion,
    AvgFusion,
    SumFusion,
    LogitWeightedFusion,
    LearnedWeightFusion,
    ProductFusion,
    AttentionFusion,
    get_fusion_strategy
)
from .classifier import (
    FlexibleEnsembleClassifier,
    BackwardCompatibleEnsemble,
    FourBranchEnsemble,
    ThreeBranchEnsembleNoLPD,
)

__all__ = [
    # Base classes
    "BaseBranch",
    "FusionStrategy",

    # Concrete branches
    "FoundationBranch",
    "ScratchBranch",
    "L2DistanceBranch",
    "EmbeddingDiffBranch",

    # Fusion strategies
    "MaxFusion",
    "AvgFusion",
    "SumFusion",
    "LogitWeightedFusion",
    "LearnedWeightFusion",
    "ProductFusion",
    "AttentionFusion",
    "get_fusion_strategy",

    # Ensemble classifiers
    "FlexibleEnsembleClassifier",
    "BackwardCompatibleEnsemble",
    "FourBranchEnsemble",
    "ThreeBranchEnsembleNoLPD",
]
