"""
Concrete branch implementations for flexible ensemble classifier.

Provides:
- FoundationBranch: Wrapper around SplitChannelFoundationModel (DINOv3 on multi-scale RAW)
- ScratchBranch: Wrapper around MultiScaleDifferenceCNN (ResNet on LPD features)
- L2DistanceBranch: MLP on L2 distance between embeddings
"""

import torch
import torch.nn as nn
from typing import List

# Import from local rfnt_models
from rfnt_models.foundation_models.gaussian_pyramid import SplitChannelFoundationModel
from rfnt_models.scratch_models.multi_scale_difference_cnn import MultiScaleDifferenceCNN

from .base import BaseBranch


class FoundationBranch(BaseBranch):
    """
    Foundation model branch using frozen DINOv3 encoder.

    Processes multi-scale RAW images through the SplitChannelFoundationModel,
    which uses a frozen DINOv3 encoder with a trainable classifier head.

    Args:
        name: Branch name (default: "foundation")
        encoder_name: DINOv3 model name (e.g., "dinov3_vitl16")
        num_scales: Number of multi-scale groups
        channels_per_scale: Channels per scale (default: 3 for RGB)
        images_per_scale: Number of image types per scale (5 for MultiScaleRaw)
        feature_fusion: Fusion method for multi-scale features
        classifier_hidden_dims: Hidden dimensions for classifier MLP
        dropout: Dropout rate
        num_classes: Number of output classes (default: 1 for binary)
    """

    def __init__(
        self,
        name: str = "foundation",
        encoder_name: str = "dinov3_vitl16",
        num_scales: int = 5,
        channels_per_scale: int = 3,
        images_per_scale: int = 5,
        feature_fusion: str = "attention",
        classifier_hidden_dims: List[int] = None,
        dropout: float = 0.1,
        num_classes: int = 1
    ):
        super().__init__(name)

        if classifier_hidden_dims is None:
            classifier_hidden_dims = [1024, 512]

        self.encoder_name = encoder_name
        self.num_scales = num_scales
        self.channels_per_scale = channels_per_scale
        self.images_per_scale = images_per_scale
        self.feature_fusion = feature_fusion
        self.classifier_hidden_dims = classifier_hidden_dims
        self.dropout = dropout
        self.num_classes = num_classes

        self.model = SplitChannelFoundationModel(
            encoder_name=encoder_name,
            num_scales=num_scales,
            channels_per_scale=channels_per_scale,
            images_per_scale=images_per_scale,
            feature_fusion=feature_fusion,
            classifier_hidden_dims=classifier_hidden_dims,
            dropout=dropout,
            num_classes=num_classes
        )

    def forward(self, **kwargs) -> torch.Tensor:
        """
        Forward pass for foundation branch.

        Args:
            **kwargs: Must contain 'multi_scale_raw_images' key

        Returns:
            Logits [B, 1]
        """
        multi_scale_raw_images = kwargs.get("multi_scale_raw_images")
        if multi_scale_raw_images is None:
            raise ValueError("FoundationBranch requires 'multi_scale_raw_images' input")

        output = self.model(multi_scale_raw_images)
        # Ensure output is [B, 1]
        return output.reshape(-1, 1)

    def get_input_keys(self) -> List[str]:
        """Return required input keys."""
        return ["multi_scale_raw_images"]


class ScratchBranch(BaseBranch):
    """
    Scratch model branch using ResNet on LPD features.

    Processes Local Pixel Dependency (LPD) features through a ResNet backbone
    trained from scratch.

    Args:
        name: Branch name (default: "scratch")
        input_channels: Number of input channels (LPD feature channels)
        architecture: ResNet variant (resnet18, resnet34, resnet50, etc.)
        pretrained: Whether to use pretrained weights (default: False for scratch)
        num_classes: Number of output classes (default: 1 for binary)
    """

    def __init__(
        self,
        name: str = "scratch",
        input_channels: int = 3,
        architecture: str = "resnet34",
        pretrained: bool = False,
        num_classes: int = 1
    ):
        super().__init__(name)

        self.input_channels = input_channels
        self.architecture = architecture
        self.pretrained = pretrained
        self.num_classes = num_classes

        self.model = MultiScaleDifferenceCNN(
            input_channels=input_channels,
            architecture=architecture,
            pretrained=pretrained,
            num_classes=num_classes
        )

    def forward(self, **kwargs) -> torch.Tensor:
        """
        Forward pass for scratch branch.

        Args:
            **kwargs: Must contain 'lpd_features' key

        Returns:
            Logits [B, 1]
        """
        lpd_features = kwargs.get("lpd_features")
        if lpd_features is None:
            raise ValueError("ScratchBranch requires 'lpd_features' input")

        output = self.model(lpd_features)
        # Ensure output is [B, 1]
        return output.reshape(-1, 1)

    def get_input_keys(self) -> List[str]:
        """Return required input keys."""
        return ["lpd_features"]


class L2DistanceBranch(BaseBranch):
    """
    L2 distance branch using MLP on embedding distance.

    Processes the L2 distance between clean and noisy embeddings through
    a small MLP. This branch learns to directly use the distance signal,
    which varies across different generators (domain generalization).

    Args:
        name: Branch name (default: "l2_distance")
        hidden_dims: Hidden layer dimensions (default: [128, 64, 1])
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(
        self,
        name: str = "l2_distance",
        hidden_dims: List[int] = None,
        dropout: float = 0.1
    ):
        super().__init__(name)

        if hidden_dims is None:
            hidden_dims = [128, 64, 1]

        self.hidden_dims = hidden_dims
        self.dropout = dropout

        # Build MLP: 1 -> 128 -> 64 -> 1
        layers = []
        prev_dim = 1  # Input is scalar L2 distance

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # No BatchNorm or Dropout on final layer
            if i < len(hidden_dims) - 1:
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU(inplace=True))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

        # Initialize final layer bias for better training dynamics
        # Start with negative bias so that at initialization (large distances),
        # the model predicts "real" (negative logits), which is WRONG.
        # The model will then learn to flip this: small distance → real, large → fake
        # This ensures very low accuracy (<1%) at the start of training.
        nn.init.constant_(self.mlp[-1].bias, -2.0)

    def forward(self, **kwargs) -> torch.Tensor:
        """
        Forward pass for L2 distance branch.

        Args:
            **kwargs: Must contain 'l2_distance' key

        Returns:
            Logits [B, 1]
        """
        l2_distance = kwargs.get("l2_distance")
        if l2_distance is None:
            raise ValueError("L2DistanceBranch requires 'l2_distance' input")

        # Handle both [B] and [B, 1] shapes
        if l2_distance.dim() == 1:
            l2_distance = l2_distance.unsqueeze(-1)

        output = self.mlp(l2_distance)
        # Ensure output is [B, 1] (handles [B, 1, 1] -> [B, 1] or [B] -> [B, 1])
        return output.reshape(-1, 1)

    def get_input_keys(self) -> List[str]:
        """Return required input keys."""
        return ["l2_distance"]


class EmbeddingDiffBranch(BaseBranch):
    """
    Embedding difference branch using MLP on embedding difference vector.

    Processes the raw difference vector between clean and noisy embeddings.
    This captures more information than just scalar L2 distance.

    Args:
        name: Branch name (default: "embedding_diff")
        feature_dim: Dimension of embeddings (default: 1024 for DINOv3 ViT-L/16)
        hidden_dims: Hidden layer dimensions (default: [512, 256, 1])
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(
        self,
        name: str = "embedding_diff",
        feature_dim: int = 1024,
        hidden_dims: List[int] = None,
        dropout: float = 0.1
    ):
        super().__init__(name)

        if hidden_dims is None:
            hidden_dims = [512, 256, 1]

        self.feature_dim = feature_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout

        # Build MLP: feature_dim -> 512 -> 256 -> 1
        layers = []
        prev_dim = feature_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if i < len(hidden_dims) - 1:
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU(inplace=True))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

    def forward(self, **kwargs) -> torch.Tensor:
        """
        Forward pass for embedding difference branch.

        Args:
            **kwargs: Must contain 'embedding_diff' key

        Returns:
            Logits [B, 1]
        """
        embedding_diff = kwargs.get("embedding_diff")
        if embedding_diff is None:
            raise ValueError("EmbeddingDiffBranch requires 'embedding_diff' input")

        output = self.mlp(embedding_diff)
        # Ensure output is [B, 1]
        return output.reshape(-1, 1)

    def get_input_keys(self) -> List[str]:
        """Return required input keys."""
        return ["embedding_diff"]
