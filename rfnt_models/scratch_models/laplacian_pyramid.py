"""
Scratch models trained on Laplacian pyramid differences.

These models are trained from scratch to process Laplacian pyramid
differences, which capture high-frequency details and artifacts.
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


# FerretNet Components
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 0):
        super().__init__()
        self.depth_wise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding=padding, groups=in_channels, bias=False
        )
        self.point_wise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, bias=False
        )

    def forward(self, x):
        x = self.depth_wise(x)
        x = self.point_wise(x)
        return x


class DilatedConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            dilation=1,
            bias=False
        )
        self.conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=2,
            groups=in_channels,
            dilation=2,
            bias=False
        )
        self.conv3 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            dilation=1,
            bias=False
        )

    def forward(self, x):
        x = self.conv1(x) + self.conv2(x)
        x = self.conv3(x)
        return x


class DSBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.layers = nn.Sequential(
            DilatedConv2d(in_channels, out_channels, stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SeparableConv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels)
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.layers(x) + self.shortcut(x)
        x = self.relu(x)
        return x


class LaplacianPyramidCNN(BaseScratchModel):
    """CNN model trained from scratch on Laplacian pyramid differences"""

    def __init__(self,
                 input_channels: int = 12,  # 3 channels * 4 levels
                 architecture: str = "resnet18",
                 pretrained: bool = False,
                 **kwargs):
        """
        Initialize Laplacian pyramid CNN

        Args:
            input_channels: Number of input channels (3 * num_levels)
            architecture: Backbone architecture
            pretrained: Whether to use pretrained weights
        """
        super().__init__(input_channels=input_channels, **kwargs)
        self.architecture = architecture
        self.pretrained = pretrained
        self.model_type = "laplacian_pyramid_cnn"

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

        elif self.architecture == "xception":
            if not TIMM_AVAILABLE:
                raise ImportError(
                    "timm library is required for Xception. "
                    "Install it with: pip install timm"
                )
            self.backbone = timm.create_model(
                'xception',
                pretrained=self.pretrained,
                num_classes=self.num_classes
            )
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
            if self.input_channels > 3:
                with torch.no_grad():
                    self.backbone.conv1.weight[:, :3, :, :] = original_conv.weight
                    for i in range(3, self.input_channels):
                        nn.init.kaiming_normal_(self.backbone.conv1.weight[:, i, :, :])

        else:
            raise ValueError(f"Unsupported architecture: {self.architecture}")

        # Replace final classification layer
        if hasattr(self.backbone, 'fc'):  # ResNet
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, self.num_classes)
        elif hasattr(self.backbone, 'classifier'):  # EfficientNet or Xception
            if hasattr(self.backbone.classifier, '1'):  # EfficientNet has sequential classifier
                in_features = self.backbone.classifier[1].in_features
                self.backbone.classifier[1] = nn.Linear(in_features, self.num_classes)
            else:
                in_features = self.backbone.classifier.in_features
                self.backbone.classifier = nn.Linear(in_features, self.num_classes)
        elif hasattr(self.backbone, 'head'):  # Some timm models use 'head'
            in_features = self.backbone.head.in_features
            self.backbone.head = nn.Linear(in_features, self.num_classes)
        elif hasattr(self.backbone, 'classifier'):  # Xception in timm
            if isinstance(self.backbone.classifier, nn.Linear):
                in_features = self.backbone.classifier.in_features
                self.backbone.classifier = nn.Linear(in_features, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.backbone(x)

    def get_name(self) -> str:
        """Return model name"""
        return f"laplacian_cnn_{self.architecture}_{self.input_channels}ch"


class MultiScaleLaplacianNet(BaseScratchModel):
    """Multi-scale network specifically designed for Laplacian pyramid"""

    def __init__(self,
                 input_channels: int = 12,  # 3 channels * 4 levels
                 num_scales: int = 4,
                 base_filters: int = 64,
                 **kwargs):
        """
        Initialize multi-scale Laplacian network

        Args:
            input_channels: Number of input channels
            num_scales: Number of pyramid scales
            base_filters: Base number of filters
        """
        super().__init__(input_channels=input_channels, **kwargs)
        self.num_scales = num_scales
        self.base_filters = base_filters
        self.model_type = "multiscale_laplacian_net"

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
        return f"multiscale_laplacian_net_{self.num_scales}scale_{self.base_filters}filters"


class LaplacianTransformer(BaseScratchModel):
    """Vision Transformer for Laplacian pyramid differences"""

    def __init__(self,
                 input_channels: int = 12,
                 patch_size: int = 16,
                 embed_dim: int = 384,
                 num_heads: int = 6,
                 num_layers: int = 6,
                 **kwargs):
        """
        Initialize Laplacian Transformer

        Args:
            input_channels: Number of input channels
            patch_size: Patch size for ViT
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
        """
        super().__init__(input_channels=input_channels, **kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.model_type = "laplacian_transformer"

        self._build_model()

    def _build_model(self):
        """Build Transformer architecture"""
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            self.input_channels,
            self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )

        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, 196, self.embed_dim))  # 14*14 for 224 input
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # Classification head
        self.classifier = nn.Linear(self.embed_dim, self.num_classes)

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # [B, embed_dim, H/patch, W/patch]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]

        # Add position embedding
        x = x + self.pos_embed

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Transformer encoding
        x = self.transformer(x)

        # Classification using CLS token
        cls_token_final = x[:, 0]
        logits = self.classifier(cls_token_final)

        return logits

    def get_name(self) -> str:
        """Return model name"""
        return f"laplacian_vit_patch{self.patch_size}_embed{self.embed_dim}_layers{self.num_layers}"


class FerretNet(BaseScratchModel):
    """FerretNet - CNN with Local Pixel Dependency and dilated separable convolutions"""

    def __init__(self,
                 input_channels: int = 3,
                 dim: int = 96,
                 depths: List[int] = None,
                 **kwargs):
        """
        Initialize FerretNet

        Args:
            input_channels: Number of input channels (default: 3 for RGB)
            dim: Base dimension for feature maps (default: 96)
            depths: List of depths for each stage (default: [2, 2])
        """
        super().__init__(input_channels=input_channels, **kwargs)
        self.dim = dim
        self.depths = depths if depths is not None else [2, 2]
        self.model_type = "ferret_net"

        self._build_model()

    def _build_model(self):
        """Build FerretNet architecture"""
        self.cbr1 = nn.Sequential(
            nn.Conv2d(self.input_channels, self.dim // 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.dim // 2),
            nn.ReLU(inplace=True)
        )
        self.cbr2 = nn.Sequential(
            nn.Conv2d(self.dim // 2, self.dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim),
            nn.ReLU(inplace=True)
        )

        # Build feature extractor with DSBlocks
        current_dim = self.dim
        self.feature = nn.ModuleList()

        for depth in self.depths:
            # First block with stride 2
            self.feature.append(DSBlock(current_dim, current_dim * 2, stride=2))
            current_dim = current_dim * 2

            # Remaining blocks with stride 1
            for _ in range(depth - 1):
                self.feature.append(DSBlock(current_dim, current_dim, stride=1))

        # Final convolution
        self.final_conv = nn.Conv2d(current_dim, current_dim, 1, 1, bias=False)

        # Global average pooling and classifier
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.logit = nn.Sequential(
            nn.Dropout(0.2, inplace=True),
            nn.Linear(current_dim, self.num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Apply LPD (Local Pixel Dependency) - using median filter as default
        # x = x - self.lpd(x)  # Uncomment when LPD function is available

        x = self.cbr1(x)
        x = self.cbr2(x)

        # Feature extraction
        for block in self.feature:
            x = block(x)

        x = self.final_conv(x)

        # Global pooling and classification
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.logit(x)

        return x

    def get_name(self) -> str:
        """Return model name"""
        depths_str = '_'.join(map(str, self.depths))
        return f"ferret_net_dim{self.dim}_depths{depths_str}_{self.input_channels}ch"