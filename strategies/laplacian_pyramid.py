"""
Laplacian pyramid input strategy for AI detection models.

This strategy creates a Laplacian pyramid representation of images, which captures
details at multiple scales and is effective for detecting AI-generated artifacts.
Each pyramid level has different spatial dimensions.
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


class LaplacianPyramidStrategy(BaseInputStrategy):
    """Create Laplacian pyramid representation for multi-scale analysis

    The Laplacian pyramid captures the difference between Gaussian pyramid levels,
    highlighting details and artifacts at different scales. Each level has different
    spatial dimensions (height and width), so we return them as a dictionary or list.
    """

    def __init__(self,
                 levels: List[int] = [0, 1, 2, 3],
                 normalize: bool = True,
                 smooth_sigma: float = 1.0,
                 return_format: str = "list"):  # "list", "dict", "processed"
        """
        Initialize Laplacian pyramid strategy

        Args:
            levels: List of pyramid levels to include (0 is original resolution)
            normalize: Whether to normalize each level
            smooth_sigma: Sigma for Gaussian smoothing before downsampling
            return_format: How to return the pyramid levels
                - "list": Return list of tensors at different scales
                - "dict": Return dict mapping level to tensor
                - "processed": Process all levels to same size and concatenate
        """
        self.levels = levels
        self.normalize = normalize
        self.smooth_sigma = smooth_sigma
        self.return_format = return_format

    def preprocess(self, image: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor], Dict[int, torch.Tensor]]:
        """
        Apply Laplacian pyramid preprocessing

        Args:
            image: Tensor of shape (C, H, W) or (B, C, H, W)

        Returns:
            Laplacian pyramid representation (format depends on return_format)
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Build Laplacian pyramid
        pyramid = self._build_laplacian_pyramid(image)

        # Select requested levels
        selected_levels = {}
        for level in self.levels:
            if level < len(pyramid):
                selected_levels[level] = pyramid[level]

        # Return based on format
        if self.return_format == "dict":
            result = selected_levels
        elif self.return_format == "list":
            result = [selected_levels[level] for level in self.levels if level in selected_levels]
        elif self.return_format == "processed":
            # Resize all levels to original size and concatenate
            target_size = image.shape[-2:]
            processed_levels = []
            for level in self.levels:
                if level in selected_levels:
                    level_tensor = selected_levels[level]
                    # Resize to target size
                    resized = F.interpolate(level_tensor, size=target_size,
                                          mode='bilinear', align_corners=False)
                    processed_levels.append(resized)
            # Concatenate along channel dimension
            result = torch.cat(processed_levels, dim=1)
        else:
            raise ValueError(f"Unknown return_format: {self.return_format}")

        if squeeze_output and isinstance(result, torch.Tensor):
            result = result.squeeze(0)

        return result

    def _build_laplacian_pyramid(self, image: torch.Tensor) -> List[torch.Tensor]:
        """Build Laplacian pyramid from input image"""
        gaussian_pyramid = self._build_gaussian_pyramid(image)
        laplacian_pyramid = []

        # Build Laplacian pyramid
        for i in range(len(gaussian_pyramid) - 1):
            # Get current level
            current = gaussian_pyramid[i]

            # Get next level and upsample
            next_level = gaussian_pyramid[i + 1]
            upsampled = F.interpolate(next_level,
                                    size=current.shape[-2:],
                                    mode='bilinear',
                                    align_corners=False)

            # Compute Laplacian (difference)
            laplacian = current - upsampled

            if self.normalize:
                # Normalize to zero mean, unit variance
                laplacian = (laplacian - laplacian.mean()) / (laplacian.std() + 1e-8)

            laplacian_pyramid.append(laplacian)

        # Add the last Gaussian level (coarsest)
        if self.normalize:
            last_level = gaussian_pyramid[-1]
            last_level = (last_level - last_level.mean()) / (last_level.std() + 1e-8)
            laplacian_pyramid.append(last_level)
        else:
            laplacian_pyramid.append(gaussian_pyramid[-1])

        return laplacian_pyramid

    def _build_gaussian_pyramid(self, image: torch.Tensor) -> List[torch.Tensor]:
        """Build Gaussian pyramid from input image with proper smoothing"""
        pyramid = [image]
        current = image

        # Create Gaussian kernel
        kernel_size = 5
        kernel = _gaussian_kernel_2d(kernel_size=kernel_size, sigma=self.smooth_sigma)
        kernel = kernel.view(1, 1, kernel_size, kernel_size).to(image.device)
        kernel = kernel.repeat(current.shape[1], 1, 1, 1)

        # Build pyramid until we can't downsample further
        min_size = 8  # Minimum spatial size
        while current.shape[-2] > min_size and current.shape[-1] > min_size:
            # Apply Gaussian smoothing before downsampling to avoid aliasing
            smoothed = F.conv2d(current, kernel, groups=current.shape[1], padding=2)
            # Downsample by factor of 2
            current = F.avg_pool2d(smoothed, kernel_size=2, stride=2)
            pyramid.append(current)

        return pyramid

    def get_name(self) -> str:
        """Return strategy name for logging"""
        levels_str = "_".join(map(str, self.levels))
        return f"laplacian_pyramid_{self.return_format}_levels_{levels_str}_sigma{self.smooth_sigma:.1f}"

    def get_output_channels(self) -> int:
        """Return number of output channels"""
        if self.return_format == "processed":
            return 3 * len(self.levels)
        else:
            return 3  # Individual levels have 3 channels each

    def get_model_type(self) -> str:
        """Return model type - strategy is compatible with both foundation and scratch models"""
        return "universal"


class LaplacianPyramidProcessedStrategy(LaplacianPyramidStrategy):
    """Laplacian pyramid strategy that always returns processed (concatenated) output

    This is specifically for scratch models that need a fixed-size input.
    All pyramid levels are resized to the original resolution and concatenated.
    """

    def __init__(self,
                 levels: List[int] = [0, 1, 2, 3],
                 normalize: bool = True,
                 smooth_sigma: float = 1.0):
        super().__init__(
            levels=levels,
            normalize=normalize,
            smooth_sigma=smooth_sigma,
            return_format="processed"
        )

    def get_name(self) -> str:
        """Return strategy name for logging"""
        levels_str = "_".join(map(str, self.levels))
        return f"laplacian_processed_levels_{levels_str}_sigma{self.smooth_sigma:.1f}"

    def get_model_type(self) -> str:
        """Return model type - strategy is compatible with both foundation and scratch models"""
        return "universal"


class LaplacianPyramidMultiScaleStrategy(LaplacianPyramidStrategy):
    """Laplacian pyramid strategy that returns multi-scale output for foundation models

    Foundation models can process different scales separately, so we return
    the pyramid levels at their native resolutions.
    """

    def __init__(self,
                 levels: List[int] = [0, 1, 2, 3],
                 normalize: bool = True,
                 smooth_sigma: float = 1.0,
                 return_format: str = "list"):
        super().__init__(
            levels=levels,
            normalize=normalize,
            smooth_sigma=smooth_sigma,
            return_format=return_format
        )

    def get_name(self) -> str:
        """Return strategy name for logging"""
        levels_str = "_".join(map(str, self.levels))
        return f"laplacian_multiscale_{self.return_format}_levels_{levels_str}_sigma{self.smooth_sigma:.1f}"

    def get_model_type(self) -> str:
        """Return model type"""
        return "foundation"