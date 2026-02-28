"""
Multi-Scale Hybrid input strategy combining raw images and difference features.

This strategy combines:
1. Raw images: resized, smoothed, median filtered, noisy, current
2. Difference features from multi_scale_difference: Real-Smoothed, Real-Resized, Real-Resized(Smoothed)

This hybrid approach leverages both raw content and difference-based artifact detection,
providing comprehensive features for detecting AI-generated images across different generators.
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


class MultiScaleHybridStrategy(BaseInputStrategy):
    """
    Multi-scale hybrid strategy combining raw images and difference features.

    This strategy provides:
    1. Raw images: resized, smoothed, median filtered, noisy, current
    2. Difference features: Real-Smoothed, Real-Resized, Real-Resized(Smoothed)

    The difference features are computed against the ORIGINAL scale image,
    while raw images are processed at each scale.

    Returns concatenated raw images and differences for foundation encoder processing.
    """

    def __init__(self,
                 levels: List[int] = [0, 1, 2, 3],
                 smooth_sigma: float = 1.0,
                 noise_std: float = 0.1,
                 normalize: bool = True):
        """
        Initialize multi-scale hybrid strategy

        Args:
            levels: List of pyramid levels (0=original, 1=half size, 2=quarter size, etc.)
            smooth_sigma: Sigma for Gaussian smoothing
            noise_std: Standard deviation for noise perturbation
            normalize: Whether to normalize images/differences
        """
        self.levels = levels
        self.smooth_sigma = smooth_sigma
        self.noise_std = noise_std
        self.normalize = normalize

    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-scale hybrid preprocessing

        Args:
            image: Tensor of shape (C, H, W) or (B, C, H, W)

        Returns:
            Concatenated raw images and differences: (C * len(levels) * 11, H, W)
            where 11 = 5 raw types + 3 difference types + 3 median differences per scale
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

        all_images = []

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

            # ========== RAW IMAGES (5 types) ==========
            # 1. Resized: Downsample then upsample back to target size
            resized = F.interpolate(current, size=target_size, mode='bilinear', align_corners=False)
            if self.normalize:
                resized = (resized * 2.0) - 1.0

            # 2. Smoothed: Apply Gaussian blur at current scale
            smoothed = F.conv2d(current, kernel, groups=current.shape[1], padding=2)
            if self.normalize:
                smoothed = (smoothed * 2.0) - 1.0

            # 3. Median Filtered: Apply median filter
            median_filtered = self._median_filter(current, kernel_size=3)
            if self.normalize:
                median_filtered = (median_filtered * 2.0) - 1.0

            # 4. Original + Noise: Add noise to current image
            noise = torch.randn_like(current) * self.noise_std
            noisy_original = current + noise
            # Clamp to valid range
            if self.normalize:
                noisy_original = torch.clamp(noisy_original, -1.0, 1.0)

            # 5. Current: Original image at this scale
            current_processed = current
            if self.normalize:
                current_processed = (current_processed * 2.0) - 1.0

            # ========== DIFFERENCE FEATURES (6 types) ==========
            # Note: Differences are computed against ORIGINAL scale image (not current scale)
            # This matches the multi_scale_difference strategy approach

            # Current scale image for processing
            current_for_diff = current

            # 6. Difference: Resized - Current (current scale)
            diff_resized = resized - current_processed

            # 7. Difference: Smoothed - Current (current scale)
            diff_smoothed = smoothed - current_processed

            # 8. Difference: Median - Current (current scale)
            diff_median = median_filtered - current_processed

            # 9. Difference 1 (from multi_scale_difference): Real - Smoothed(Real)
            # Smooth the current scale image
            smoothed_for_diff = F.conv2d(current_for_diff, kernel, groups=current_for_diff.shape[1], padding=2)
            diff1 = current_for_diff - smoothed_for_diff

            # 10. Difference 2 (from multi_scale_difference): Real - Resized(Real)
            # Resize current scale image back to original size, then downsample back
            resized_back = F.interpolate(current_for_diff, size=original_size, mode='bilinear', align_corners=False)
            # Resize back to current scale for difference calculation
            resized_for_diff = F.interpolate(resized_back, size=target_size, mode='bilinear', align_corners=False)
            diff2 = current_for_diff - resized_for_diff

            # 11. Difference 3 (from multi_scale_difference): Real - Resized(Smoothed(Real))
            # Smooth first, then resize back to original, then downsample to current scale
            smoothed_resized = F.interpolate(smoothed_for_diff, size=original_size, mode='bilinear', align_corners=False)
            smoothed_resized_for_diff = F.interpolate(smoothed_resized, size=target_size, mode='bilinear', align_corners=False)
            diff3 = current_for_diff - smoothed_resized_for_diff

            # Normalize differences if requested
            if self.normalize:
                # Normalize each difference
                diff1 = (diff1 - diff1.mean()) / (diff1.std() + 1e-8)
                diff2 = (diff2 - diff2.mean()) / (diff2.std() + 1e-8)
                diff3 = (diff3 - diff3.mean()) / (diff3.std() + 1e-8)

            # Append all 11 types to the list
            all_images.extend([
                resized,           # 1. Raw: Resized
                smoothed,          # 2. Raw: Smoothed
                median_filtered,   # 3. Raw: Median Filtered
                noisy_original,    # 4. Raw: Noisy
                current_processed, # 5. Raw: Current
                diff_resized,      # 6. Diff: Resized - Current
                diff_smoothed,     # 7. Diff: Smoothed - Current
                diff_median,       # 8. Diff: Median - Current
            ])

        # Concatenate all images along channel dimension
        result = torch.cat(all_images, dim=1)

        if squeeze_output:
            result = result.squeeze(0)

        return result

    def _median_filter(self, x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
        """
        Apply median filter to input tensor using unfold operation

        Args:
            x: Input tensor of shape (B, C, H, W)
            kernel_size: Size of the median filter kernel

        Returns:
            Median filtered tensor of same shape as input
        """
        padding = kernel_size // 2
        batch_size, channels, height, width = x.shape

        # 1. Unfold
        unfolded = F.unfold(x, kernel_size=kernel_size, padding=padding, stride=1)

        # 2. Reshape to separate channels from the kernel pixels
        unfolded = unfolded.view(batch_size, channels, kernel_size**2, -1)

        # 3. Calculate Median
        median_values = torch.median(unfolded, dim=2)[0]

        # 4. Reshape back to image
        output = median_values.view(batch_size, channels, height, width)

        return output

    def get_name(self) -> str:
        """Return strategy name for logging"""
        levels_str = "_".join(map(str, self.levels))
        return (f"multi_scale_hybrid_sigma{self.smooth_sigma:.1f}_noise{self.noise_std:.2f}"
                f"_levels_{levels_str}")

    def get_output_channels(self) -> int:
        """Return number of output channels"""
        # 11 types per level * 3 channels per level * number of levels
        return 3 * len(self.levels) * 11

    def get_model_type(self) -> str:
        """Return model type - strategy is designed for foundation models"""
        return "foundation"


class MultiScaleHybridProcessedStrategy(MultiScaleHybridStrategy):
    """
    Multi-scale hybrid strategy that processes all scales to same size and concatenates.

    This version resizes all multi-scale images to the original resolution before
    concatenating, making it compatible with scratch models that need fixed-size input.
    """

    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-scale hybrid preprocessing with all scales resized to original size

        Args:
            image: Tensor of shape (C, H, W) or (B, C, H, W)

        Returns:
            Concatenated raw images and differences all at original resolution
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

        all_images = []

        for level in self.levels:
            if level == 0:
                # Level 0: Original size
                current = image
            else:
                # Downsample to target size
                target_size = (original_size[0] // (2 ** level),
                              original_size[1] // (2 ** level))
                current = F.interpolate(image, size=target_size, mode='bilinear', align_corners=False)

            # ========== RAW IMAGES (5 types) ==========
            # All resized to original size
            # 1. Resized
            resized = F.interpolate(current, size=original_size, mode='bilinear', align_corners=False)
            if self.normalize:
                resized = (resized * 2.0) - 1.0

            # 2. Smoothed
            smoothed = F.conv2d(current, kernel, groups=current.shape[1], padding=2)
            smoothed = F.interpolate(smoothed, size=original_size, mode='bilinear', align_corners=False)
            if self.normalize:
                smoothed = (smoothed * 2.0) - 1.0

            # 3. Median Filtered
            median_filtered = self._median_filter(current, kernel_size=3)
            median_filtered = F.interpolate(median_filtered, size=original_size, mode='bilinear', align_corners=False)
            if self.normalize:
                median_filtered = (median_filtered * 2.0) - 1.0

            # 4. Original + Noise
            noise = torch.randn_like(current) * self.noise_std
            noisy_original = current + noise
            if self.normalize:
                noisy_original = torch.clamp(noisy_original, -1.0, 1.0)

            # 5. Current
            current_processed = current
            if self.normalize:
                current_processed = (current_processed * 2.0) - 1.0

            # ========== DIFFERENCE FEATURES (6 types) ==========
            # Current scale image for processing
            current_for_diff = current

            # 6. Resized - Current
            diff_resized = resized - F.interpolate(current_processed, size=original_size, mode='bilinear', align_corners=False)

            # 7. Smoothed - Current
            diff_smoothed = smoothed - F.interpolate(current_processed, size=original_size, mode='bilinear', align_corners=False)

            # 8. Median - Current
            diff_median = median_filtered - F.interpolate(current_processed, size=original_size, mode='bilinear', align_corners=False)

            # 9. Real - Smoothed(Real)
            smoothed_for_diff = F.conv2d(current_for_diff, kernel, groups=current_for_diff.shape[1], padding=2)
            diff1 = F.interpolate(current_for_diff, size=original_size, mode='bilinear', align_corners=False) - \
                    F.interpolate(smoothed_for_diff, size=original_size, mode='bilinear', align_corners=False)

            # 10. Real - Resized(Real)
            resized_back = F.interpolate(current_for_diff, size=original_size, mode='bilinear', align_corners=False)
            diff2 = F.interpolate(current_for_diff, size=original_size, mode='bilinear', align_corners=False) - resized_back

            # 11. Real - Resized(Smoothed(Real))
            smoothed_resized = F.interpolate(smoothed_for_diff, size=original_size, mode='bilinear', align_corners=False)
            diff3 = F.interpolate(current_for_diff, size=original_size, mode='bilinear', align_corners=False) - smoothed_resized

            # Normalize differences if requested
            if self.normalize:
                diff1 = (diff1 - diff1.mean()) / (diff1.std() + 1e-8)
                diff2 = (diff2 - diff2.mean()) / (diff2.std() + 1e-8)
                diff3 = (diff3 - diff3.mean()) / (diff3.std() + 1e-8)

            # Append all 11 types to the list
            all_images.extend([
                resized,           # 1
                smoothed,          # 2
                median_filtered,   # 3
                noisy_original,    # 4
                current_processed, # 5
                diff_resized,      # 6
                diff_smoothed,     # 7
                diff_median,       # 8
                diff1,             # 9
                diff2,             # 10
                diff3              # 11
            ])

        # Concatenate all images along channel dimension
        result = torch.cat(all_images, dim=1)

        if squeeze_output:
            result = result.squeeze(0)

        return result

    def get_name(self) -> str:
        """Return strategy name for logging"""
        levels_str = "_".join(map(str, self.levels))
        return (f"multi_scale_hybrid_processed_sigma{self.smooth_sigma:.1f}_noise{self.noise_std:.2f}"
                f"_levels_{levels_str}")

    def get_model_type(self) -> str:
        """Return model type - compatible with both foundation and scratch models"""
        return "universal"
