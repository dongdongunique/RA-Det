"""
Flexible ensemble classifier supporting arbitrary number of branches.

Provides:
- FlexibleEnsembleClassifier: Main class with dynamic branch registration
- BackwardCompatibleEnsemble: Wrapper mimicking old EnsembleClassifier API
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Union, Optional


from .base import BaseBranch
from .fusion import get_fusion_strategy, FusionStrategy, MaxFusion


class FlexibleEnsembleClassifier(nn.Module):
    """
    Flexible ensemble classifier supporting arbitrary number of branches.

    This class allows dynamic registration of branches through register_branch().
    All fusion methods work with any number of branches (N-branch compatible).

    Features:
    - Dynamic branch registration via register_branch()
    - Generic fusion over N branches
    - Returns individual branch logits + ensemble logits
    - Easy to extend with new branch types

    Example:
        >>> from rfnt_models.ensemble import (
        ...     FlexibleEnsembleClassifier, FoundationBranch,
        ...     ScratchBranch, L2DistanceBranch
        ... )
        >>>
        >>> # Create ensemble
        >>> ensemble = FlexibleEnsembleClassifier(fusion_method="logit_weighted")
        >>>
        >>> # Register branches
        >>> ensemble.register_branch(FoundationBranch(
        ...     name="foundation", encoder_name="dinov3_vitl16", num_scales=5
        ... ))
        >>> ensemble.register_branch(ScratchBranch(
        ...     name="scratch", input_channels=3, architecture="resnet34"
        ... ))
        >>> ensemble.register_branch(L2DistanceBranch(
        ...     name="l2_distance", hidden_dims=[128, 64, 1]
        ... ))
        >>>
        >>> # Forward with all inputs
        >>> ensemble_logits, branch_logits = ensemble(
        ...     return_all_logits=True,
        ...     multi_scale_raw_images=msr,
        ...     lpd_features=lpd,
        ...     l2_distance=l2_dist
        ... )

    Args:
        fusion_method: Fusion strategy name (default: "logit_weighted")
        temperature: Temperature for logit_weighted fusion (default: 1.0)
        calibrate_logits: Enable per-branch logit calibration
        calibration_init: Initial temperature for calibration (default: 1.0)
        calibration_learned: Learn calibration parameters if True
        calibration_clip: Clamp range for temperature scaling
        **fusion_kwargs: Additional arguments for fusion strategy
    """

    def __init__(
        self,
        fusion_method: str = "logit_weighted",
        temperature: float = 1.0,
        calibrate_logits: bool = False,
        calibration_init: float = 1.0,
        calibration_learned: bool = True,
        calibration_clip: Tuple[float, float] = (0.05, 5.0),
        **fusion_kwargs
    ):
        super().__init__()

        if calibration_init <= 0:
            raise ValueError("calibration_init must be > 0")

        self.fusion_method = fusion_method
        self.temperature = temperature
        self.calibrate_logits = calibrate_logits
        self.calibration_init = calibration_init
        self.calibration_learned = calibration_learned
        self.calibration_clip = calibration_clip

        # Store branches in ModuleDict for proper parameter registration
        self.branches: nn.ModuleDict = nn.ModuleDict()

        # Optional per-branch calibration parameters
        self.logit_temps: nn.ParameterDict = nn.ParameterDict()
        self.logit_biases: nn.ParameterDict = nn.ParameterDict()

        # Create fusion strategy
        self.fusion_strategy = self._create_fusion_strategy(fusion_method, **fusion_kwargs)


    def _create_fusion_strategy(self, method: str, **kwargs) -> FusionStrategy:
        """Create fusion strategy instance."""
        if method == "learned_weight":
            # LearnedWeightFusion needs num_branches, will be updated on branch registration
            return get_fusion_strategy(method, num_branches=0, **kwargs)
        elif method == "logit_weighted":
            return get_fusion_strategy(method, temperature=self.temperature, **kwargs)
        elif method == "attention":
            return get_fusion_strategy(method, **kwargs)
        else:
            return get_fusion_strategy(method, **kwargs)

    def register_branch(self, branch: BaseBranch) -> "FlexibleEnsembleClassifier":
        """
        Register a new branch to the ensemble.

        Args:
            branch: BaseBranch instance to add

        Returns:
            self for method chaining

        Raises:
            ValueError: If branch name already exists

        Example:
            >>> ensemble = FlexibleEnsembleClassifier()
            >>> ensemble.register_branch(FoundationBranch(...))
            >>> ensemble.register_branch(ScratchBranch(...))
        """
        if branch.name in self.branches:
            raise ValueError(f"Branch '{branch.name}' already registered")

        self.branches[branch.name] = branch

        # Update fusion strategy if needed (e.g., for learned weights)
        if hasattr(self.fusion_strategy, "_resize_weights"):
            self.fusion_strategy._resize_weights(len(self.branches))

        return self

    def get_branch_names(self) -> List[str]:
        """Return list of registered branch names."""
        return list(self.branches.keys())

    def get_num_branches(self) -> int:
        """Return number of registered branches."""
        return len(self.branches)

    def forward(
        self,
        return_all_logits: bool = True,
        use_max_for_eval: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward pass through all branches and fuse results.

        Args:
            return_all_logits: If True, return (ensemble_logits, branch_logits_dict)
                              If False, return only ensemble_logits
            use_max_for_eval: If True, use max fusion (for evaluation)
            **kwargs: Inputs for all branches (routed by branch input keys)

        Returns:
            ensemble_logits: [B, 1] Fused output
            branch_logits_dict: Dict mapping branch names to logits (if return_all_logits)

        Raises:
            RuntimeError: If no branches registered
        """
        if len(self.branches) == 0:
            raise RuntimeError(
                "No branches registered. Use register_branch() to add branches."
            )

        # Handle use_max_for_eval by temporarily switching fusion
        original_fusion = self.fusion_strategy
        if use_max_for_eval and not isinstance(self.fusion_strategy, MaxFusion):
            self.fusion_strategy = MaxFusion()

        # Collect logits from all branches
        branch_logits = {}
        for branch_name, branch in self.branches.items():
            try:
                branch_logits[branch_name] = branch(**kwargs)
            except Exception as e:
                raise RuntimeError(
                    f"Error in branch '{branch_name}': {e}. "
                    f"Ensure all required inputs are provided: {branch.get_input_keys()}"
                )

        # Fuse using the configured strategy
        ensemble_logits = self.fusion_strategy.forward(branch_logits)

        # Restore original fusion
        if use_max_for_eval and not isinstance(original_fusion, MaxFusion):
            self.fusion_strategy = original_fusion

        if return_all_logits:
            return ensemble_logits, branch_logits
        return ensemble_logits

    def get_name(self) -> str:
        """Return model name for logging."""
        branch_names = "+".join(self.get_branch_names())
        return f"FlexibleEnsemble({branch_names},fusion={self.fusion_method})"

    def extra_repr(self) -> str:
        """Extra representation for print(model)."""
        lines = [
            f"fusion_method={self.fusion_method}",
            f"num_branches={len(self.branches)}",
            f"branches={list(self.branches.keys())}"
        ]
        return "\n  ".join(lines)


class BackwardCompatibleEnsemble(FlexibleEnsembleClassifier):
    """
    Backward-compatible wrapper that mimics the old EnsembleClassifier API.

    This allows existing code to work without modification while using
    the new flexible architecture internally.

    The old EnsembleClassifier had:
    - Fixed 2 branches: foundation and scratch
    - Specific forward signature: (multi_scale_raw_images, lpd_features, use_max_for_eval)
    - Specific return format: (ensemble_logits, foundation_logits, scratch_logits)

    Example:
        >>> # Old code - still works!
        >>> classifier = BackwardCompatibleEnsemble(
        ...     encoder_name="dinov3_vitl16",
        ...     fusion_method="logit_weighted"
        ... )
        >>> ensemble_logits, foundation_logits, scratch_logits = classifier(
        ...     multi_scale_raw_images=msr,
        ...     lpd_features=lpd,
        ...     use_max_for_eval=False
        ... )

    Args:
        encoder_name: DINOv3 model name
        num_scales: Number of multi-scale groups
        channels_per_scale: Channels per scale (default: 3)
        images_per_scale: Images per scale (default: 5)
        feature_fusion: Feature fusion method for foundation (default: "attention")
        classifier_hidden_dims: Classifier hidden dims for foundation
        lpd_channels: LPD feature channels for scratch (default: 3)
        resnet_variant: ResNet variant for scratch (default: "resnet34")
        dropout: Dropout rate (default: 0.1)
        temperature: Temperature for softmax weighting (default: 1.0)
        fusion_method: Ensemble fusion method (default: "logit_weighted")
    """

    def __init__(
        self,
        encoder_name: str = "dinov3_vitl16",
        num_scales: int = 5,
        channels_per_scale: int = 3,
        images_per_scale: int = 5,
        feature_fusion: str = "attention",
        classifier_hidden_dims: List[int] = None,
        lpd_channels: int = 3,
        resnet_variant: str = "resnet34",
        dropout: float = 0.1,
        temperature: float = 1.0,
        fusion_method: str = "logit_weighted"
    ):
        if classifier_hidden_dims is None:
            classifier_hidden_dims = [1024, 512]

        super().__init__(fusion_method=fusion_method, temperature=temperature)

        # Store config for get_name()
        self.encoder_name = encoder_name
        self.num_scales = num_scales
        self.lpd_channels = lpd_channels
        self.resnet_variant = resnet_variant

        # Import branch classes here to avoid circular imports
        from .branches import FoundationBranch, ScratchBranch

        # Create and register the two standard branches
        foundation_branch = FoundationBranch(
            name="foundation",
            encoder_name=encoder_name,
            num_scales=num_scales,
            channels_per_scale=channels_per_scale,
            images_per_scale=images_per_scale,
            feature_fusion=feature_fusion,
            classifier_hidden_dims=classifier_hidden_dims,
            dropout=dropout,
            num_classes=1
        )

        scratch_branch = ScratchBranch(
            name="scratch",
            input_channels=lpd_channels,
            architecture=resnet_variant,
            pretrained=False,
            num_classes=1
        )

        self.register_branch(foundation_branch)
        self.register_branch(scratch_branch)

    def forward(
        self,
        multi_scale_raw_images: torch.Tensor,
        lpd_features: torch.Tensor,
        use_max_for_eval: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Backward-compatible forward matching old EnsembleClassifier signature.

        Args:
            multi_scale_raw_images: Multi-scale raw concatenated images
            lpd_features: LPD strategy features
            use_max_for_eval: If True, use max fusion for evaluation
            **kwargs: Additional inputs (for future extensibility)

        Returns:
            ensemble_logits: [B, 1] Ensemble classification logits
            foundation_logits: [B, 1] Foundation branch logits (for backward compatibility)
            scratch_logits: [B, 1] Scratch branch logits (for backward compatibility)
        """
        # Route inputs correctly
        ensemble_logits, branch_logits = super().forward(
            return_all_logits=True,
            use_max_for_eval=use_max_for_eval,
            multi_scale_raw_images=multi_scale_raw_images,
            lpd_features=lpd_features,
            **kwargs
        )

        # Return in old format for backward compatibility
        return (
            ensemble_logits,
            branch_logits.get("foundation", ensemble_logits),
            branch_logits.get("scratch", ensemble_logits)
        )

    def get_name(self) -> str:
        """Return name matching old format for logging compatibility."""
        return (
            f"ensemble_{self.fusion_method}_{self.encoder_name}_"
            f"foundation_{self.num_scales}scales_"
            f"scratch_{self.resnet_variant}_lpd{self.lpd_channels}_temp{self.temperature}"
        )


class FourBranchEnsemble(FlexibleEnsembleClassifier):
    """
    Four-branch ensemble with all available signals.

    Branches:
    1. Foundation: Frozen DINOv3 on multi-scale RAW images
    2. Scratch: ResNet on LPD (Local Pixel Dependency) features
    3. L2 Distance: MLP on scalar L2 distance between embeddings
    4. Embedding Diff: MLP on raw embedding difference vector

    Example:
        >>> from rfnt_models.ensemble import FourBranchEnsemble
        >>>
        >>> classifier = FourBranchEnsemble(
        ...     encoder_name="dinov3_vitl16",
        ...     fusion_method="logit_weighted"
        ... )
        >>>
        >>> # Forward with all inputs
        >>> ensemble_logits, branch_logits = classifier(
        ...     return_all_logits=True,
        ...     multi_scale_raw_images=msr,
        ...     lpd_features=lpd,
        ...     l2_distance=l2_dist,
        ...     embedding_diff=embed_diff
        ... )

    Args:
        encoder_name: DINOv3 model name (default: "dinov3_vitl16")
        num_scales: Number of multi-scale groups (default: 5)
        channels_per_scale: Channels per scale (default: 3)
        images_per_scale: Images per scale (default: 5)
        feature_fusion: Feature fusion method for foundation (default: "attention")
        classifier_hidden_dims: Classifier hidden dims for foundation
        lpd_channels: LPD feature channels for scratch (default: 3)
        resnet_variant: ResNet variant for scratch (default: "resnet34")
        l2_hidden_dims: Hidden dims for L2 branch (default: [128, 64, 1])
        embedding_dim: Embedding dimension for diff branch (default: 1024)
        diff_hidden_dims: Hidden dims for diff branch (default: [512, 256, 1])
        dropout: Dropout rate (default: 0.1)
        temperature: Temperature for softmax weighting (default: 1.0)
        fusion_method: Ensemble fusion method (default: "logit_weighted")
    """

    def __init__(
        self,
        encoder_name: str = "dinov3_vitl16",
        num_scales: int = 5,
        channels_per_scale: int = 3,
        images_per_scale: int = 5,
        feature_fusion: str = "attention",
        classifier_hidden_dims: list = None,
        lpd_channels: int = 3,
        resnet_variant: str = "resnet34",
        l2_hidden_dims: list = None,
        embedding_dim: int = 1024,
        diff_hidden_dims: list = None,
        dropout: float = 0.1,
        temperature: float = 1.0,
        fusion_method: str = "logit_weighted"
    ):
        if classifier_hidden_dims is None:
            classifier_hidden_dims = [1024, 512]
        if l2_hidden_dims is None:
            l2_hidden_dims = [128, 64, 1]
        if diff_hidden_dims is None:
            diff_hidden_dims = [512, 256, 1]

        super().__init__(fusion_method=fusion_method, temperature=temperature)

        # Store config for get_name()
        self.encoder_name = encoder_name
        self.num_scales = num_scales
        self.lpd_channels = lpd_channels
        self.resnet_variant = resnet_variant
        self.embedding_dim = embedding_dim

        # Import branch classes
        from .branches import FoundationBranch, ScratchBranch, L2DistanceBranch, EmbeddingDiffBranch

        # Register all 4 branches
        # 1. Foundation branch
        self.register_branch(FoundationBranch(
            name="foundation",
            encoder_name=encoder_name,
            num_scales=num_scales,
            channels_per_scale=channels_per_scale,
            images_per_scale=images_per_scale,
            feature_fusion=feature_fusion,
            classifier_hidden_dims=classifier_hidden_dims,
            dropout=dropout,
            num_classes=1
        ))

        # 2. Scratch branch
        self.register_branch(ScratchBranch(
            name="scratch",
            input_channels=lpd_channels,
            architecture=resnet_variant,
            pretrained=False,
            num_classes=1
        ))

        # 3. L2 Distance branch
        self.register_branch(L2DistanceBranch(
            name="l2_distance",
            hidden_dims=l2_hidden_dims,
            dropout=dropout
        ))

        # 4. Embedding Difference branch
        self.register_branch(EmbeddingDiffBranch(
            name="embedding_diff",
            feature_dim=embedding_dim,
            hidden_dims=diff_hidden_dims,
            dropout=dropout
        ))

    def forward(
        self,
        multi_scale_raw_images: torch.Tensor,
        lpd_features: torch.Tensor,
        l2_distance: torch.Tensor,
        embedding_diff: torch.Tensor,
        use_max_for_eval: bool = False,
        **kwargs
    ) -> tuple:
        """
        Forward pass for 4-branch ensemble.

        Args:
            multi_scale_raw_images: Multi-scale raw concatenated images
            lpd_features: LPD strategy features
            l2_distance: L2 distance between clean and noisy embeddings [B] or [B, 1]
            embedding_diff: Raw embedding difference vector [B, feature_dim]
            use_max_for_eval: If True, use max fusion for evaluation
            **kwargs: Additional inputs

        Returns:
            ensemble_logits: [B, 1] Ensemble classification logits
            branch_logits: Dict mapping branch names to their logits
        """
        ensemble_logits, branch_logits = super().forward(
            return_all_logits=True,
            use_max_for_eval=use_max_for_eval,
            multi_scale_raw_images=multi_scale_raw_images,
            lpd_features=lpd_features,
            l2_distance=l2_distance,
            embedding_diff=embedding_diff,
            **kwargs
        )

        return ensemble_logits, branch_logits

    def get_name(self) -> str:
        """Return model name for logging."""
        return (
            f"four_branch_ensemble_{self.fusion_method}_{self.encoder_name}_"
            f"foundation_{self.num_scales}scales_"
            f"scratch_{self.resnet_variant}_"
            f"l2_lpd{self.lpd_channels}_"
            f"diff{self.embedding_dim}_temp{self.temperature}"
        )


class ThreeBranchEnsembleNoLPD(FlexibleEnsembleClassifier):
    """
    Three-branch ensemble without LPD scratch branch.

    Branches:
    1. Foundation: Frozen DINOv3 on multi-scale RAW images
    2. L2 Distance: MLP on scalar L2 distance between embeddings
    3. Embedding Diff: MLP on raw embedding difference vector
    """

    def __init__(
        self,
        encoder_name: str = "dinov3_vitl16",
        num_scales: int = 5,
        channels_per_scale: int = 3,
        images_per_scale: int = 5,
        feature_fusion: str = "attention",
        classifier_hidden_dims: list = None,
        l2_hidden_dims: list = None,
        embedding_dim: int = 1024,
        diff_hidden_dims: list = None,
        dropout: float = 0.1,
        temperature: float = 1.0,
        fusion_method: str = "logit_weighted",
    ):
        if classifier_hidden_dims is None:
            classifier_hidden_dims = [1024, 512]
        if l2_hidden_dims is None:
            l2_hidden_dims = [128, 64, 1]
        if diff_hidden_dims is None:
            diff_hidden_dims = [512, 256, 1]

        super().__init__(fusion_method=fusion_method, temperature=temperature)

        self.encoder_name = encoder_name
        self.num_scales = num_scales
        self.embedding_dim = embedding_dim

        from .branches import FoundationBranch, L2DistanceBranch, EmbeddingDiffBranch

        self.register_branch(FoundationBranch(
            name="foundation",
            encoder_name=encoder_name,
            num_scales=num_scales,
            channels_per_scale=channels_per_scale,
            images_per_scale=images_per_scale,
            feature_fusion=feature_fusion,
            classifier_hidden_dims=classifier_hidden_dims,
            dropout=dropout,
            num_classes=1,
        ))

        self.register_branch(L2DistanceBranch(
            name="l2_distance",
            hidden_dims=l2_hidden_dims,
            dropout=dropout,
        ))

        self.register_branch(EmbeddingDiffBranch(
            name="embedding_diff",
            feature_dim=embedding_dim,
            hidden_dims=diff_hidden_dims,
            dropout=dropout,
        ))

    def forward(
        self,
        multi_scale_raw_images: torch.Tensor,
        l2_distance: torch.Tensor,
        embedding_diff: torch.Tensor,
        use_max_for_eval: bool = False,
        **kwargs,
    ) -> tuple:
        ensemble_logits, branch_logits = super().forward(
            return_all_logits=True,
            use_max_for_eval=use_max_for_eval,
            multi_scale_raw_images=multi_scale_raw_images,
            l2_distance=l2_distance,
            embedding_diff=embedding_diff,
            **kwargs,
        )
        return ensemble_logits, branch_logits

    def get_name(self) -> str:
        return (
            f"three_branch_no_lpd_{self.fusion_method}_{self.encoder_name}_"
            f"foundation_{self.num_scales}scales_"
            f"l2_diff{self.embedding_dim}_temp{self.temperature}"
        )
