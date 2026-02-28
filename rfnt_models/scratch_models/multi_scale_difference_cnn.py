"""
Scratch models trained on multi-scale differences.

These models are trained from scratch to process multi-scale differences,
which capture high-frequency artifacts, scale-dependent patterns, and
smoothed patterns at multiple scales.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List, Dict, Any
from .base import BaseScratchModel

# Try to import timm for Xception, fallback if not available
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


class MultiScaleDifferenceCNN(BaseScratchModel):
    """
    CNN model trained from scratch on multi-scale differences.

    Processes THREE types of differences at each scale using a standard
    CNN backbone (ResNet or EfficientNet).
    """

    def __init__(self,
                 input_channels: int = 36,  # 9 channels per level * 4 levels (default)
                 architecture: str = "resnet18",
                 pretrained: bool = False,
                 **kwargs):
        """
        Initialize multi-scale difference CNN

        Args:
            input_channels: Number of input channels
                           9 = 3 differences * 3 channels
                           Total = 9 * num_levels
            architecture: Backbone architecture ("resnet18", "resnet34", "efficientnet_b0")
            pretrained: Whether to use pretrained weights
        """
        super().__init__(input_channels=input_channels, **kwargs)
        self.architecture = architecture
        self.pretrained = pretrained
        self.model_type = "multi_scale_difference_cnn"

        self._build_model()

    def _build_model(self):
        """Build the CNN backbone"""
        if self.architecture == "resnet18":
            self.backbone = models.resnet18(pretrained=self.pretrained)
            # Modify first conv layer for different input channels
            original_conv = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                self.input_channels,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None
            )
            # Initialize new weights
            if self.input_channels > 3:
                with torch.no_grad():
                    self.backbone.conv1.weight[:, :3, :, :] = original_conv.weight
                    for i in range(3, self.input_channels):
                        nn.init.kaiming_normal_(self.backbone.conv1.weight[:, i, :, :])

        elif self.architecture == "resnet34":
            self.backbone = models.resnet34(pretrained=self.pretrained)
            # Modify first conv layer
            original_conv = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                self.input_channels,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None
            )
            if self.input_channels > 3:
                with torch.no_grad():
                    self.backbone.conv1.weight[:, :3, :, :] = original_conv.weight
                    for i in range(3, self.input_channels):
                        nn.init.kaiming_normal_(self.backbone.conv1.weight[:, i, :, :])

        elif self.architecture == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(pretrained=self.pretrained)
            # Modify first conv layer
            original_conv = self.backbone.features[0][0]
            self.backbone.features[0][0] = nn.Conv2d(
                self.input_channels,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None
            )
            if self.input_channels > 3:
                with torch.no_grad():
                    self.backbone.features[0][0].weight[:, :3, :, :] = original_conv.weight
                    for i in range(3, self.input_channels):
                        nn.init.kaiming_normal_(self.backbone.features[0][0].weight[:, i, :, :])

        else:
            raise ValueError(f"Unsupported architecture: {self.architecture}")

        # Replace final classification layer
        if hasattr(self.backbone, 'fc'):  # ResNet
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, self.num_classes)
        elif hasattr(self.backbone, 'classifier'):  # EfficientNet
            if hasattr(self.backbone.classifier, '1'):  # EfficientNet has sequential classifier
                in_features = self.backbone.classifier[1].in_features
                self.backbone.classifier[1] = nn.Linear(in_features, self.num_classes)
            else:
                in_features = self.backbone.classifier.in_features
                self.backbone.classifier = nn.Linear(in_features, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.backbone(x)

    def get_name(self) -> str:
        """Return model name"""
        return f"multi_scale_diff_cnn_{self.architecture}_{self.input_channels}ch"


class MultiScaleDifferenceNet(BaseScratchModel):
    """
    Multi-scale network specifically designed for multi-scale differences.

    Uses a multi-branch architecture to process different scales separately
    before fusing the features.
    """

    def __init__(self,
                 input_channels: int = 36,  # 9 channels per level * 4 levels (default)
                 num_scales: int = 4,
                 base_filters: int = 64,
                 **kwargs):
        """
        Initialize multi-scale difference network

        Args:
            input_channels: Number of input channels (9 * num_scales)
            num_scales: Number of difference scales
            base_filters: Base number of filters
        """
        super().__init__(input_channels=input_channels, **kwargs)
        self.num_scales = num_scales
        self.base_filters = base_filters
        self.model_type = "multiscale_difference_net"

        self._build_model()

    def _build_model(self):
        """Build multi-scale architecture"""
        # Entry processing
        self.entry_conv = nn.Sequential(
            nn.Conv2d(self.input_channels, self.base_filters, 3, padding=1),
            nn.BatchNorm2d(self.base_filters),
            nn.ReLU(inplace=True)
        )

        # Multi-scale processing branches
        self.scale_branches = nn.ModuleList()
        for i in range(self.num_scales):
            branch = nn.Sequential(
                nn.Conv2d(self.base_filters, self.base_filters * 2, 3, padding=1),
                nn.BatchNorm2d(self.base_filters * 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.base_filters * 2, self.base_filters * 2, 3, padding=1),
                nn.BatchNorm2d(self.base_filters * 2),
                nn.ReLU(inplace=True)
            )
            self.scale_branches.append(branch)

        # Feature fusion
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(self.base_filters * 2 * self.num_scales,
                     self.base_filters * 4, 3, padding=1),
            nn.BatchNorm2d(self.base_filters * 4),
            nn.ReLU(inplace=True)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.base_filters * 4, self.base_filters * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(self.base_filters * 2, self.num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Entry processing
        x = self.entry_conv(x)

        # Process through scale branches
        scale_outputs = []
        for branch in self.scale_branches:
            scale_out = branch(x)
            scale_outputs.append(scale_out)

        # Concatenate multi-scale features
        fused = torch.cat(scale_outputs, dim=1)
        fused = self.fusion_conv(fused)

        # Classification
        logits = self.classifier(fused)
        return logits

    def get_name(self) -> str:
        """Return model name"""
        return f"multiscale_diff_net_{self.num_scales}scale_{self.base_filters}filters"
