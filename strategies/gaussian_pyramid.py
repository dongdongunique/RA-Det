"""
Gaussian pyramid input strategy for foundation models.

This strategy creates a Gaussian pyramid representation for foundation models,
which contains actual images at different scales (not differences).
"""

import torch
import torch.nn.functional as F
from typing import List, Union, Dict
from .base import BaseInputStrategy


def _gaussian_kernel_2d(kernel_size: int, sigma: float) -> torch.Tensor:
    """Create 2D Gaussian kernel"""
    k = (kernel_size - 1) // 2
    x = torch.arange(-k, k + 1, dtype=torch.float32)
    y = torch.arange(-k, k + 1, dtype=torch.float32)
    x, y = torch.meshgrid(x, y, indexing='xy')
    kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel


class GaussianPyramidStrategy(BaseInputStrategy):
    """Create Gaussian pyramid representation for foundation models

    The Gaussian pyramid contains smoothed/blurred images at different scales,
    which are compatible with foundation models pretrained on natural images.
    All levels are resized to the foundation model's expected input size.
    """

    def __init__(self,
                 levels: List[int] = [0, 1, 2, 3],
                 input_size: int = 224,  # Standard CLIP input size
                 smooth_sigma: float = 1.0,
                 normalize: bool = True):
        """
        Initialize Gaussian pyramid strategy

        Args:
            levels: List of pyramid levels to include (0 is original resolution)
            input_size: Size to resize all images to (e.g., 224 for CLIP)
            smooth_sigma: Sigma for Gaussian smoothing before downsampling
            normalize: Whether to normalize images to [0, 1] range
        """
        self.levels = levels
        self.input_size = input_size
        self.smooth_sigma = smooth_sigma
        self.normalize = normalize

    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply Gaussian pyramid preprocessing

        Args:
            image: Tensor of shape (C, H, W) or (B, C, H, W)

        Returns:
            Concatenated pyramid levels along channel dimension
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Build Gaussian pyramid
        pyramid = self._build_gaussian_pyramid(image)

        # Select requested levels and resize to input_size
        processed_levels = []
        for level in self.levels:
            if level < len(pyramid):
                level_tensor = pyramid[level]
                # Resize to foundation model input size
                resized = F.interpolate(level_tensor, size=(self.input_size, self.input_size),
                                      mode='bilinear', align_corners=False)

                # Ensure values are in valid range
                if self.normalize:
                    resized = torch.clamp(resized, 0.0, 1.0)

                processed_levels.append(resized)

        if squeeze_output:
            # Remove batch dimension from each level
            processed_levels = [lvl.squeeze(0) for lvl in processed_levels]
            # Concatenate along channel dimension
            return torch.cat(processed_levels, dim=0)
        else:
            # Keep batch dimension and concatenate along channel dimension
            return torch.cat(processed_levels, dim=1)

    def _build_gaussian_pyramid(self, image: torch.Tensor) -> List[torch.Tensor]:
        """Build Gaussian pyramid from input image with proper smoothing"""
        pyramid = [image]
        current = image

        # Create Gaussian kernel
        kernel_size = 5
        kernel = _gaussian_kernel_2d(kernel_size=kernel_size, sigma=self.smooth_sigma)
        kernel = kernel.view(1, 1, kernel_size, kernel_size).to(image.device)

        # Build pyramid until we can't downsample further
        min_size = 8  # Minimum spatial size
        while current.shape[-2] > min_size and current.shape[-1] > min_size:
            # Expand kernel to match input channels for depthwise convolution
            in_channels = current.shape[1]
            expanded_kernel = kernel.repeat(in_channels, 1, 1, 1)
            # Apply Gaussian smoothing before downsampling to avoid aliasing
            # Use groups for depthwise convolution (same kernel for each channel)
            smoothed = F.conv2d(current, expanded_kernel, groups=in_channels, padding=2)
            # Downsample by factor of 2
            current = F.avg_pool2d(smoothed, kernel_size=2, stride=2)
            pyramid.append(current)

        return pyramid

    def get_name(self) -> str:
        """Return strategy name for logging"""
        levels_str = "_".join(map(str, self.levels))
        return f"gaussian_pyramid_levels_{levels_str}_size{self.input_size}_sigma{self.smooth_sigma:.1f}"

    def get_output_channels(self) -> int:
        """Return total number of output channels (all levels concatenated)"""
        return 3 * len(self.levels)  # 3 channels per level * number of levels

    def get_model_type(self) -> str:
        """Return model type"""
        return "foundation"


class MultiScaleGaussianStrategy(GaussianPyramidStrategy):
    """Enhanced Gaussian pyramid with different processing for each scale"""

    def __init__(self,
                 levels: List[int] = [0, 1, 2, 3],
                 input_size: int = 224,
                 scale_processing: Dict[int, str] = None,  # {level: processing_type}
                 **kwargs):
        """
        Initialize multi-scale Gaussian strategy

        Args:
            levels: Pyramid levels to use
            input_size: Target input size
            scale_processing: Dict mapping levels to processing types
                "original": Just the Gaussian blurred image
                "edges": Apply edge detection
                "details": Enhance details
            **kwargs: Additional arguments
        """
        super().__init__(levels=levels, input_size=input_size, **kwargs)
        self.scale_processing = scale_processing or {0: "original", 1: "original", 2: "edges", 3: "details"}

    def preprocess(self, image: torch.Tensor) -> List[torch.Tensor]:
        """Apply multi-scale Gaussian pyramid with different processing"""
        if image.dim() == 3:
            image = image.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Build Gaussian pyramid
        pyramid = self._build_gaussian_pyramid(image)

        # Process each level differently
        processed_levels = []
        for level in self.levels:
            if level < len(pyramid):
                level_tensor = pyramid[level]

                # Apply level-specific processing
                processing_type = self.scale_processing.get(level, "original")
                processed = self._apply_processing(level_tensor, processing_type)

                # Resize to input size
                resized = F.interpolate(processed, size=(self.input_size, self.input_size),
                                      mode='bilinear', align_corners=False)

                # Ensure valid range
                if self.normalize:
                    resized = torch.clamp(resized, 0.0, 1.0)

                processed_levels.append(resized)

        if squeeze_output:
            processed_levels = [lvl.squeeze(0) for lvl in processed_levels]

        return processed_levels

    def _apply_processing(self, image: torch.Tensor, processing_type: str) -> torch.Tensor:
        """Apply specific processing to a level"""
        if processing_type == "original":
            return image
        elif processing_type == "edges":
            # Apply Sobel edge detection
            sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]],
                                 dtype=image.dtype, device=image.device)
            sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]],
                                 dtype=image.dtype, device=image.device)

            # Apply to each channel
            edges_x = F.conv2d(image, sobel_x.repeat(image.shape[1], 1, 1, 1), padding=1, groups=image.shape[1])
            edges_y = F.conv2d(image, sobel_y.repeat(image.shape[1], 1, 1, 1), padding=1, groups=image.shape[1])
            edges = torch.sqrt(edges_x**2 + edges_y**2)
            return edges
        elif processing_type == "details":
            # Enhance details using unsharp masking
            blurred = F.gaussian_blur(image, kernel_size=5, sigma=2.0)
            details = image + 1.5 * (image - blurred)
            return torch.clamp(details, 0.0, 1.0)
        else:
            return image

    def get_name(self) -> str:
        """Return strategy name"""
        levels_str = "_".join(map(str, self.levels))
        processing_str = "_".join([f"L{lvl}{typ}" for lvl, typ in self.scale_processing.items()])
        return f"multiscale_gaussian_{processing_str}_size{self.input_size}"