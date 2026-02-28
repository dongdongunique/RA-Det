"""
Multi-Scale Raw input strategy for foundation models.

This strategy returns RAW images with 8 types per scale:
1. Resized (downsampled and upsampled back)
2. Smoothed (Gaussian blur)
3. Median Filtered (median filtering)
4. Original + Noise (noise perturbation)
5. Current (original image at scale)
6. Difference: Resized - Current
7. Difference: Smoothed - Current (Gaussian filter difference)
8. Difference: Median Filtered - Current (Median filter difference)

This provides multi-scale raw inputs for foundation encoders.
"""

from io import BytesIO
from typing import List, Optional, Sequence, Union, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
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


def _jpeg_compress_batch(image: torch.Tensor, quality: int) -> torch.Tensor:
    if image.dim() == 3:
        image = image.unsqueeze(0)

    device = image.device
    img = image.detach().cpu().float()

    input_in_minus_one_one = img.min().item() < 0.0
    if input_in_minus_one_one:
        img = (img + 1.0) / 2.0

    img = img.clamp(0.0, 1.0)
    img_u8 = (img * 255.0).round().to(torch.uint8)

    quality = max(1, min(95, int(quality)))
    compressed = []

    for sample in img_u8:
        sample_np = sample.permute(1, 2, 0).numpy()
        pil_img = Image.fromarray(sample_np)
        buffer = BytesIO()
        pil_img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        jpeg_img = Image.open(buffer).convert("RGB")
        jpeg_np = np.array(jpeg_img, dtype=np.float32) / 255.0
        jpeg_tensor = torch.from_numpy(jpeg_np).permute(2, 0, 1)
        compressed.append(jpeg_tensor)

    result = torch.stack(compressed, dim=0)
    if input_in_minus_one_one:
        result = result * 2.0 - 1.0

    return result.to(device)


class MultiScaleRawStrategy(BaseInputStrategy):
    """
    Multi-scale raw strategy that returns RAW images with differences:

    At each scale, creates 8 types of images:
    1. Resized: Downsampled and upsampled back to original size
    2. Smoothed: Gaussian blurred at the scale
    3. Median Filtered: Median filtered image
    4. Original + Noise: Noise-perturbed original
    5. Current: Original image at scale
    6. Difference: Resized - Current
    7. Difference: Smoothed - Current (Gaussian filter difference)
    8. Difference: Median Filtered - Current (Median filter difference)

    Returns concatenated raw images for foundation encoder processing.
    """

    def __init__(self,
                 levels: List[int] = [0, 1, 2, 3],
                 smooth_sigma: float = 1.0,
                 noise_std: float = 0.1,
                 normalize: bool = True):
        """
        Initialize multi-scale raw strategy

        Args:
            levels: List of pyramid levels (0=original, 1=half size, 2=quarter size, etc.)
            smooth_sigma: Sigma for Gaussian smoothing
            noise_std: Standard deviation for noise perturbation
            normalize: Whether to normalize images to [-1, 1] range
        """
        self.levels = levels
        self.smooth_sigma = smooth_sigma
        self.noise_std = noise_std
        self.normalize = normalize

    def preprocess(self, image: torch.Tensor, external_noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply multi-scale raw preprocessing

        Args:
            image: Tensor of shape (C, H, W) or (B, C, H, W)
            external_noise: Optional external noise tensor from decoder [B, C, H, W]
                           If provided, will be used instead of random noise

        Returns:
            Concatenated raw images: (C * len(levels) * 8, H, W)
            where 8 = 8 image types per scale (5 raw + 3 differences)
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

            # Normalize if requested
            if self.normalize:
                # Normalize to [-1, 1] range
                current = (current * 2.0) - 1.0

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

            # 4. Original + Noise: Add noise to original image
            # Use external noise from decoder if provided (for level 0 only)
            if external_noise is not None and level == 0:
                # Resize external noise to match current scale if needed
                if external_noise.shape[-2:] != current.shape[-2:]:
                    noise = F.interpolate(external_noise, size=current.shape[-2:], mode='bilinear', align_corners=False)
                else:
                    noise = external_noise
                
            else:
                # Fall back to random noise
                noise = torch.randn_like(current) * self.noise_std

            noisy_original = current + noise
            # Clamp to valid range
            if self.normalize:
                noisy_original = torch.clamp(noisy_original, -1.0, 1.0)

            # 5. Current: Original image at this scale
            current_processed = current
            if self.normalize:
                current_processed = (current_processed * 2.0) - 1.0

            # 6-8. Differences: Resized - Current, Smoothed - Current, Median - Current
            # Note: Current here is before normalization to get proper difference
            current_unnorm = current
            if self.normalize:
                # Need unnormalized current for proper difference calculation
                current_unnorm = (current * 2.0) - 1.0

            diff_resized = resized - current_unnorm
            diff_smoothed = smoothed - current_unnorm
            diff_median = median_filtered - current_unnorm

            # Append all 8 types to the list
            all_images.extend([
                resized,           # 1
                smoothed,          # 2
                median_filtered,   # 3
                noisy_original,    # 4
                current_processed, # 5
            ])

        # Concatenate all images along channel dimension
        result = torch.cat(all_images, dim=1)
        #print(result.shape)
        if squeeze_output:
            result = result.squeeze(0)

        return result

    def get_name(self) -> str:
        """Return strategy name for logging"""
        levels_str = "_".join(map(str, self.levels))
        return (f"multi_scale_raw_sigma{self.smooth_sigma:.1f}_noise{self.noise_std:.2f}"
                f"_levels_{levels_str}")

    def get_output_channels(self) -> int:
        """Return number of output channels"""
        # 5 image types per level (resized, smoothed, median_filtered, noisy_original, current)
        # * 3 channels per image * number of levels
        return 3 * len(self.levels) * 5

    def get_images_per_scale(self) -> int:
        """Return number of image types per scale"""
        return 5

    def _median_filter(self, x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
        """
        Apply median filter to input tensor using unfold operation

        Args:
            x: Input tensor of shape (B, C, H, W)
            kernel_size: Size of the median filter kernel

        Returns:
            Median filtered tensor of same shape as input
        """
        B, C, H, W = x.shape
        padding = kernel_size // 2
        
        # Paper specifies constant padding with value 0 (Algorithm 1, Line 4)
        padded_image = F.pad(x, (padding, padding, padding, padding), mode='constant', value=0)

        # Extract local patches
        # Output shape: (B, C * kernel_size * kernel_size, L), where L = H * W
        patches = F.unfold(padded_image, kernel_size=kernel_size)

        # Reshape to separate channels and the spatial kernel elements
        # Shape: (B, C, kernel_size * kernel_size, H * W)
        patches = patches.view(B, C, kernel_size * kernel_size, H * W)

        # Identify the center index of the flattened kernel
        center_idx = (kernel_size * kernel_size) // 2

        # Algorithm 1, Line 8: Zero out the center pixel in each patch
        # We clone to avoid in-place modification errors if gradients differ, 
        # though usually safe here.
        patches = patches.clone()
        patches[:, :, center_idx, :] = -torch.inf

        # Algorithm 1, Line 10: Compute median along patch dimension
        # values, indices = torch.median(input, dim)
        median_values, _ = torch.median(patches, dim=2)

        # Algorithm 1, Line 12: Reshape back to original image dimensions
        median_map = median_values.view(B, C, H, W)

        return median_map

    def get_model_type(self) -> str:
        """Return model type - strategy is designed for foundation models"""
        return "foundation"


class MultiScaleRawJpegStrategy(BaseInputStrategy):
    """
    Multi-scale raw strategy with JPEG-compressed variants per scale.

    At each scale, creates 5 raw types plus JPEG-compressed versions
    of the current image at the specified quality levels.
    """

    def __init__(
                 self,
                 levels: List[int] = [0, 1, 2, 3],
                 smooth_sigma: float = 1.0,
                 noise_std: float = 0.1,
                 jpeg_qualities: Sequence[int] = (95, 75, 50),
                 normalize: bool = True):
        self.levels = levels
        self.smooth_sigma = smooth_sigma
        self.noise_std = noise_std
        self.normalize = normalize
        self.jpeg_qualities = [int(q) for q in jpeg_qualities]

    def preprocess(self, image: torch.Tensor, external_noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if image.dim() == 3:
            image = image.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        original_size = image.shape[-2:]

        kernel_size = 5
        kernel = _gaussian_kernel_2d(kernel_size=kernel_size, sigma=self.smooth_sigma)
        kernel = kernel.view(1, 1, kernel_size, kernel_size).to(image.device)
        kernel = kernel.repeat(image.shape[1], 1, 1, 1)

        all_images = []

        for level in self.levels:
            if level == 0:
                target_size = original_size
                current = image
            else:
                target_size = (original_size[0] // (2 ** level),
                              original_size[1] // (2 ** level))
                current = F.interpolate(image, size=target_size, mode='bilinear', align_corners=False)

            if self.normalize:
                current = (current * 2.0) - 1.0

            resized = F.interpolate(current, size=target_size, mode='bilinear', align_corners=False)
            if self.normalize:
                resized = (resized * 2.0) - 1.0

            smoothed = F.conv2d(current, kernel, groups=current.shape[1], padding=2)
            if self.normalize:
                smoothed = (smoothed * 2.0) - 1.0

            median_filtered = self._median_filter(current, kernel_size=3)
            if self.normalize:
                median_filtered = (median_filtered * 2.0) - 1.0

            if external_noise is not None and level == 0:
                if external_noise.shape[-2:] != current.shape[-2:]:
                    noise = F.interpolate(external_noise, size=current.shape[-2:], mode='bilinear', align_corners=False)
                else:
                    noise = external_noise
            else:
                noise = torch.randn_like(current) * self.noise_std

            noisy_original = current + noise
            if self.normalize:
                noisy_original = torch.clamp(noisy_original, -1.0, 1.0)

            current_processed = current
            if self.normalize:
                current_processed = (current_processed * 2.0) - 1.0

            jpeg_variants = []
            for quality in self.jpeg_qualities:
                jpeg_variants.append(_jpeg_compress_batch(current, quality))

            all_images.extend([
                resized,
                smoothed,
                median_filtered,
                noisy_original,
                current_processed,
            ])
            all_images.extend(jpeg_variants)

        result = torch.cat(all_images, dim=1)
        if squeeze_output:
            result = result.squeeze(0)

        return result

    def get_name(self) -> str:
        levels_str = "_".join(map(str, self.levels))
        qualities_str = "_".join(str(q) for q in self.jpeg_qualities)
        return (f"multi_scale_raw_jpeg_sigma{self.smooth_sigma:.1f}_noise{self.noise_std:.2f}"
                f"_jpeg{qualities_str}_levels_{levels_str}")

    def get_output_channels(self) -> int:
        return 3 * len(self.levels) * self.get_images_per_scale()

    def get_images_per_scale(self) -> int:
        return 5 + len(self.jpeg_qualities)

    def _median_filter(self, x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
        B, C, H, W = x.shape
        padding = kernel_size // 2

        padded_image = F.pad(x, (padding, padding, padding, padding), mode='constant', value=0)
        patches = F.unfold(padded_image, kernel_size=kernel_size)
        patches = patches.view(B, C, kernel_size * kernel_size, H * W)

        center_idx = (kernel_size * kernel_size) // 2
        patches = patches.clone()
        patches[:, :, center_idx, :] = -torch.inf

        median_values, _ = torch.median(patches, dim=2)
        median_map = median_values.view(B, C, H, W)

        return median_map

    def get_model_type(self) -> str:
        return "foundation"


class MultiScaleRawProcessedStrategy(MultiScaleRawStrategy):
    """
    Multi-scale raw strategy that processes all scales to same size and concatenates.

    This version resizes all multi-scale images to the original resolution before
    concatenating, making it compatible with scratch models that need fixed-size input.
    """

    def __init__(self,
                 levels: List[int] = [0, 1, 2, 3],
                 smooth_sigma: float = 1.0,
                 noise_std: float = 0.1,
                 normalize: bool = True):
        """
        Initialize multi-scale raw processed strategy

        Args:
            levels: List of pyramid levels
            smooth_sigma: Sigma for Gaussian smoothing
            noise_std: Standard deviation for noise perturbation
            normalize: Whether to normalize images
        """
        super().__init__(
            levels=levels,
            smooth_sigma=smooth_sigma,
            noise_std=noise_std,
            normalize=normalize
        )

    def preprocess(self, image: torch.Tensor, external_noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply multi-scale raw preprocessing with all scales resized to original size

        Args:
            image: Tensor of shape (C, H, W) or (B, C, H, W)
            external_noise: Optional external noise tensor from decoder [B, C, H, W]
                           If provided, will be used instead of random noise

        Returns:
            Concatenated raw images all at original resolution
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

            # Normalize if requested
            if self.normalize:
                current = (current * 2.0) - 1.0

            # 1. Resized: Downsample then upsample back to original size
            resized = F.interpolate(current, size=original_size, mode='bilinear', align_corners=False)
            if self.normalize:
                resized = (resized * 2.0) - 1.0

            # 2. Smoothed: Apply Gaussian blur at current scale, then resize to original
            smoothed = F.conv2d(current, kernel, groups=current.shape[1], padding=2)
            smoothed = F.interpolate(smoothed, size=original_size, mode='bilinear', align_corners=False)
            if self.normalize:
                smoothed = (smoothed * 2.0) - 1.0

            # 3. Median Filtered: Apply median filter, then resize to original
            median_filtered = self._median_filter(current, kernel_size=3)
            median_filtered = F.interpolate(median_filtered, size=original_size, mode='bilinear', align_corners=False)
            if self.normalize:
                median_filtered = (median_filtered * 2.0) - 1.0

            # 4. Original + Noise: Add noise to original image
            # Use external noise from decoder if provided (for level 0 only)
            if external_noise is not None and level == 0:
                # Resize external noise to match current scale if needed
                if external_noise.shape[-2:] != current.shape[-2:]:
                    # First interpolate to current scale, then to original size
                    noise_scaled = F.interpolate(external_noise, size=current.shape[-2:], mode='bilinear', align_corners=False)
                    noise = F.interpolate(noise_scaled, size=original_size, mode='bilinear', align_corners=False)
                else:
                    noise = F.interpolate(external_noise, size=original_size, mode='bilinear', align_corners=False)
            else:
                # Fall back to random noise
                noise = torch.randn_like(current) * self.noise_std
                noise = F.interpolate(noise, size=original_size, mode='bilinear', align_corners=False)

            noisy_original = current + noise
            if self.normalize:
                noisy_original = torch.clamp(noisy_original, -1.0, 1.0)

            # 5. Current: Original image at this scale (already resized to original)
            current_processed = current
            if self.normalize:
                current_processed = (current_processed * 2.0) - 1.0

            # 6-8. Differences: Resized - Current, Smoothed - Current, Median - Current
            # Note: Current here is before normalization to get proper difference calculation
            current_unnorm = current
            if self.normalize:
                # Need unnormalized current for proper difference calculation
                current_unnorm = (current * 2.0) - 1.0

            diff_resized = resized - current_unnorm
            diff_smoothed = smoothed - current_unnorm
            diff_median = median_filtered - current_unnorm

            # Append all 8 types to the list
            all_images.extend([
                resized,           # 1
                smoothed,          # 2
                median_filtered,   # 3
                noisy_original,    # 4
                current_processed, # 5
            ])

        # Concatenate all images along channel dimension
        result = torch.cat(all_images, dim=1)

        if squeeze_output:
            result = result.squeeze(0)

        return result

    def get_name(self) -> str:
        """Return strategy name for logging"""
        levels_str = "_".join(map(str, self.levels))
        return (f"multi_scale_raw_processed_sigma{self.smooth_sigma:.1f}_noise{self.noise_std:.2f}"
                f"_levels_{levels_str}")

    def get_model_type(self) -> str:
        """Return model type - compatible with both foundation and scratch models"""
        return "universal"
