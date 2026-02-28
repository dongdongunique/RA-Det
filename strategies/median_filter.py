"""
Median filter input strategies for scratch models.

This module provides median filter-based preprocessing strategies that are
effective for detecting AI-generated artifacts while preserving edges.
"""

import torch
import torch.nn.functional as F
from typing import List, Union
from .base import BaseInputStrategy


class MedianFilterStrategy(BaseInputStrategy):
    """Apply median filter to images for scratch training

    Median filtering is effective at removing outliers and noise while
    preserving edges, which can help detect AI-generated artifacts.
    """

    def __init__(self,
                 kernel_sizes: List[int] = [3, 5, 7],
                 mode: str = "filtered"):  # "filtered", "concat", "difference"
        """
        Initialize median filter strategy

        Args:
            kernel_sizes: List of kernel sizes for median filtering (odd numbers)
            mode: How to process the images
                - "filtered": Return median filtered images only
                - "concat": Return [original, filtered] for each kernel size
                - "difference": Return (original - filtered) for each kernel size
        """
        # Ensure all kernel sizes are odd
        self.kernel_sizes = [k if k % 2 == 1 else k + 1 for k in kernel_sizes]
        self.mode = mode

    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply median filter preprocessing

        Args:
            image: Tensor of shape (C, H, W) or (B, C, H, W)

        Returns:
            Processed image tensor
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        results = []

        for kernel_size in self.kernel_sizes:
            # Apply median filtering
            filtered = self._median_filter(image, kernel_size)

            if self.mode == "filtered":
                results.append(filtered)
            elif self.mode == "concat":
                results.append(image)
                results.append(filtered)
            elif self.mode == "difference":
                # Compute difference
                diff = image - filtered
                results.append(diff)

        # Concatenate all results
        result = torch.cat(results, dim=1)

        if squeeze_output:
            result = result.squeeze(0)

        return result

    def _median_filter(self, image: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """Apply median filtering to image

        Uses a simplified median filter implementation.
        For production use, consider optimized median filter implementations.
        """
        B, C, H, W = image.shape
        padding = kernel_size // 2

        # Use unfold to get local neighborhoods
        unfolded = F.unfold(image, kernel_size=kernel_size, padding=padding, stride=1)

        # Reshape to separate kernel elements
        unfolded = unfolded.view(B, C, kernel_size * kernel_size, H, W)

        # Compute median along kernel dimension
        median_values, _ = torch.median(unfolded, dim=2)

        return median_values

    def get_name(self) -> str:
        """Return strategy name for logging"""
        kernel_sizes_str = "_".join(map(str, self.kernel_sizes))
        return f"median_filter_{self.mode}_kernels_{kernel_sizes_str}"

    def get_output_channels(self) -> int:
        """Return number of output channels"""
        if self.mode == "filtered":
            return 3 * len(self.kernel_sizes)
        elif self.mode == "concat":
            return 3 * len(self.kernel_sizes) * 2  # original + filtered
        elif self.mode == "difference":
            return 3 * len(self.kernel_sizes)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def get_model_type(self) -> str:
        """Return model type - strategy is designed for scratch models"""
        return "scratch"


class MedianFilterDifferenceStrategy(BaseInputStrategy):
    """Median filter difference strategy for scratch models

    This strategy computes the difference between original and median-filtered
    images, which highlights artifacts and inconsistencies in AI-generated images.
    """

    def __init__(self,
                 kernel_sizes: List[int] = [3, 5, 7],
                 normalize: bool = True):
        """
        Initialize median filter difference strategy

        Args:
            kernel_sizes: List of kernel sizes for median filtering
            normalize: Whether to normalize the differences
        """
        # Ensure all kernel sizes are odd
        self.kernel_sizes = [k if k % 2 == 1 else k + 1 for k in kernel_sizes]
        self.normalize = normalize

    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply median filter difference preprocessing

        Args:
            image: Tensor of shape (C, H, W) or (B, C, H, W)

        Returns:
            Difference tensor
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        results = []

        for kernel_size in self.kernel_sizes:
            # Apply median filtering
            filtered = self._median_filter(image, kernel_size)

            # Compute difference
            diff = image - filtered

            if self.normalize:
                # Normalize to zero mean, unit variance
                diff = (diff - diff.mean()) / (diff.std() + 1e-8)

            results.append(diff)

        # Concatenate all results
        result = torch.cat(results, dim=1)

        if squeeze_output:
            result = result.squeeze(0)

        return result

    def _median_filter(self, image: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """Apply median filtering to image"""
        B, C, H, W = image.shape
        padding = kernel_size // 2

        # Use unfold to get local neighborhoods
        unfolded = F.unfold(image, kernel_size=kernel_size, padding=padding, stride=1)

        # Reshape to separate kernel elements
        unfolded = unfolded.view(B, C, kernel_size * kernel_size, H, W)

        # Compute median along kernel dimension
        median_values, _ = torch.median(unfolded, dim=2)

        return median_values

    def get_name(self) -> str:
        """Return strategy name for logging"""
        kernel_sizes_str = "_".join(map(str, self.kernel_sizes))
        norm_str = "_norm" if self.normalize else ""
        return f"median_filter_diff{kernel_sizes_str}{norm_str}"

    def get_output_channels(self) -> int:
        """Return number of output channels"""
        return 3 * len(self.kernel_sizes)

    def get_model_type(self) -> str:
        """Return model type - strategy is designed for scratch models"""
        return "scratch"
import torch
import torch.nn.functional as F
from typing import List

class LocalPixelDependencyStrategy(BaseInputStrategy):
    """
    Local Pixel Dependency (LPD) strategy based on FerretNet.
    
    Implements 'Algorithm 1: Local Dependency Feature Extraction via 
    Zero-Masked Median Deviation'. It computes the difference between 
    the original image and a median-reconstructed image where the 
    center pixel is masked (set to 0) to capture local anomalies.
    """

    def __init__(self,
                 kernel_sizes: List[int] = [3],
                 normalize: bool = False):
        """
        Initialize LPD strategy.

        Args:
            kernel_sizes: List of kernel sizes. The paper recommends [3].
                          Larger sizes (5, 7) were shown to decrease accuracy.
            normalize: Whether to normalize the resulting LPD map.
                       (Paper uses raw residuals, but normalization aids training).
        """
        # Ensure all kernel sizes are odd
        self.kernel_sizes = [k if k % 2 == 1 else k + 1 for k in kernel_sizes]
        self.normalize = normalize

    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply LPD (Zero-Masked Median Deviation) preprocessing.

        Args:
            image: Tensor of shape (C, H, W) or (B, C, H, W)

        Returns:
            LPD Feature Map tensor
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        results = []

        for kernel_size in self.kernel_sizes:
            # 1. Compute Zero-Masked Median Reconstruction (I_med)
            median_reconstruction = self._zero_masked_median_filter(image, kernel_size)

            # 2. Compute LPD: I - I_med
            lpd = image - median_reconstruction

            if self.normalize:
                # Normalize to zero mean, unit variance to stabilize training
                lpd = (lpd - lpd.mean()) / (lpd.std() + 1e-8)

            results.append(lpd)

        # Concatenate all results (if multiple kernel sizes used)
        result = torch.cat(results, dim=1)

        if squeeze_output:
            result = result.squeeze(0)

        return result

    def _zero_masked_median_filter(self, image: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """
        Computes the median of local neighborhoods with the center pixel set to 0.
        Corresponds to Algorithm 1 in the FerretNet paper.
        """
        B, C, H, W = image.shape
        padding = kernel_size // 2
        
        # Paper specifies constant padding with value 0 (Algorithm 1, Line 4)
        padded_image = F.pad(image, (padding, padding, padding, padding), mode='constant', value=0)

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

    def get_name(self) -> str:
        """Return strategy name for logging"""
        kernel_sizes_str = "_".join(map(str, self.kernel_sizes))
        norm_str = "_norm" if self.normalize else ""
        return f"lpd_ferretnet_{kernel_sizes_str}{norm_str}"

    def get_output_channels(self) -> int:
        """Return number of output channels"""
        return 3 * len(self.kernel_sizes)

    def get_model_type(self) -> str:
        """Return model type"""
        return "scratch"

class LaplacianPyramidDifferenceStrategy(BaseInputStrategy):
    """Laplacian pyramid difference strategy for scratch models

    This strategy creates a Laplacian pyramid and returns only the difference
    (detail) levels, which are effective for detecting AI-generated artifacts.
    """

    def __init__(self,
                 levels: List[int] = [0, 1, 2, 3],
                 normalize: bool = True,
                 smooth_sigma: float = 1.0):
        """
        Initialize Laplacian pyramid difference strategy

        Args:
            levels: List of pyramid levels to include (0 is original resolution)
            normalize: Whether to normalize each level
            smooth_sigma: Sigma for Gaussian smoothing before downsampling
        """
        self.levels = levels
        self.normalize = normalize
        self.smooth_sigma = smooth_sigma

    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply Laplacian pyramid difference preprocessing

        Args:
            image: Tensor of shape (C, H, W) or (B, C, H, W)

        Returns:
            Concatenated Laplacian pyramid difference levels
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Build Laplacian pyramid
        pyramid = self._build_laplacian_pyramid(image)

        # Select requested levels and resize to original size
        target_size = image.shape[-2:]
        selected_levels = []

        for level in self.levels:
            if level < len(pyramid):
                level_tensor = pyramid[level]
                # Resize to target size
                resized = F.interpolate(level_tensor, size=target_size,
                                      mode='bilinear', align_corners=False)
                selected_levels.append(resized)

        # Concatenate along channel dimension
        result = torch.cat(selected_levels, dim=1)

        if squeeze_output:
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

        return laplacian_pyramid

    def _build_gaussian_pyramid(self, image: torch.Tensor) -> List[torch.Tensor]:
        """Build Gaussian pyramid from input image with proper smoothing"""
        pyramid = [image]
        current = image

        # Create Gaussian kernel
        kernel_size = 5
        sigma = self.smooth_sigma
        k = (kernel_size - 1) // 2
        x = torch.arange(-k, k + 1, dtype=torch.float32, device=image.device)
        y = torch.arange(-k, k + 1, dtype=torch.float32, device=image.device)
        x, y = torch.meshgrid(x, y, indexing='xy')
        kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
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
        norm_str = "_norm" if self.normalize else ""
        return f"laplacian_diff_levels_{levels_str}_sigma{self.smooth_sigma:.1f}{norm_str}"

    def get_output_channels(self) -> int:
        """Return number of output channels"""
        return 3 * len(self.levels)

    def get_model_type(self) -> str:
        """Return model type - strategy is designed for scratch models"""
        return "scratch"
