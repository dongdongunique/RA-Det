"""
Multi-Scale Difference input strategy for AI detection models.

This strategy computes THREE types of differences at each scale:
1. Real - Smoothed(Real): Captures high-frequency artifacts
2. Real - Resized(Real): Captures scale-dependent patterns
3. Real - Resized(Smoothed(Real)): Captures smoothed patterns at scale

This provides rich multi-scale features for detecting AI-generated artifacts.
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


class MultiScaleDifferenceStrategy(BaseInputStrategy):
    """
    Multi-scale difference strategy that computes THREE types of differences:

    1. Real - Smoothed(Real): High-frequency artifacts
    2. Real - Resized(Real): Scale-dependent patterns
    3. Real - Resized(Smoothed(Real)): Smoothed patterns at scale

    Returns concatenated differences for CNN processing.
    """

    def __init__(self,
                 levels: List[int] = [0, 1, 2, 3],
                 smooth_sigma: float = 1.0,
                 normalize: bool = True):
        """
        Initialize multi-scale difference strategy

        Args:
            levels: List of pyramid levels (0=original, 1=half size, 2=quarter size, etc.)
            smooth_sigma: Sigma for Gaussian smoothing
            normalize: Whether to normalize differences
        """
        self.levels = levels
        self.smooth_sigma = smooth_sigma
        self.normalize = normalize

    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-scale difference preprocessing

        Args:
            image: Tensor of shape (C, H, W) or (B, C, H, W)
                  Image should already be preprocessed (resized, normalized, etc.)

        Returns:
            Concatenated differences: (C * len(levels) * 3, H, W)
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        original_size = image.shape[-2:]

        # Create Gaussian kernel
        kernel_size = 5
        kernel = _gaussian_kernel_2d(kernel_size=kernel_size, sigma=self.smooth_sigma)
        kernel = kernel.view(1, 1, kernel_size, kernel_size).to(image.device)
        kernel = kernel.repeat(image.shape[1], 1, 1, 1)

        all_differences = []

        for level in self.levels:
            if level == 0:
                # Level 0: Original size
                target_size = original_size
                current = image
            else:
                # Downsample to target size
                target_size = (original_size[0] // (2 ** level),
                              original_size[1] // (2 ** level))
                current = F.interpolate(image, size=target_size, mode='bilinear', align_corners=False)

            # Difference 1: Real - Smoothed(Real)
            # Smooth the image at current scale
            smoothed = F.conv2d(current, kernel, groups=current.shape[1], padding=2)
            #print("smoothed shape: ",smoothed.shape)
            diff1 = current - smoothed

            # Difference 2: Real - Resized(Real)
            # Resize back to original size
            resized_back = F.interpolate(current, size=original_size, mode='bilinear', align_corners=False)
            diff2 = image - resized_back

            # Difference 3: Real - Resized(Smoothed(Real))
            # Smooth first, then resize back
            smoothed_resized = F.interpolate(smoothed, size=original_size, mode='bilinear', align_corners=False)
            diff3 = image - smoothed_resized

            # Normalize each difference
            if self.normalize:
                diff1 = (diff1 - diff1.mean()) / (diff1.std() + 1e-8)
                diff2 = (diff2 - diff2.mean()) / (diff2.std() + 1e-8)
                diff3 = (diff3 - diff3.mean()) / (diff3.std() + 1e-8)

            all_differences.append(diff1)
            all_differences.append(diff2)
            all_differences.append(diff3)

        # Concatenate all differences: diff1_l0, diff2_l0, diff3_l0, diff1_l1, diff2_l1, diff3_l1, ...
        result = torch.cat(all_differences, dim=1)

        if squeeze_output:
            result = result.squeeze(0)

        return result

    def get_name(self) -> str:
        """Return strategy name for logging"""
        levels_str = "_".join(map(str, self.levels))
        return f"multi_scale_3diff_levels_{levels_str}_sigma{self.smooth_sigma:.1f}"

    def get_output_channels(self) -> int:
        """Return number of output channels"""
        # 3 differences per level * 3 channels per level * number of levels
        return 3 * len(self.levels) * 3

    def get_model_type(self) -> str:
        """Return model type - strategy is compatible with both foundation and scratch models"""
        return "universal"


