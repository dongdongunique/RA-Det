"""
Smoothed difference input strategy for scratch models.

This strategy computes the difference between original images and smoothed versions
using different sigma values for Gaussian smoothing.
"""

import torch
import torch.nn.functional as F
from typing import List, Union, Tuple
from .base import BaseInputStrategy


class SmoothedDifferenceStrategy(BaseInputStrategy):
    """Generate difference between image and smoothed version for scratch training

    This strategy applies Gaussian smoothing with different sigma values and computes
    the difference between the original and smoothed images. This helps capture
    artifacts and inconsistencies in AI-generated images.
    """

    def __init__(self,
                 sigmas: List[float] = [0.5, 1.0, 2.0, 4.0],
                 kernel_size: int = 5,
                 normalize: bool = True,
                 mode: str = "difference"):  # "difference", "concat"
        """
        Initialize smoothed difference strategy

        Args:
            sigmas: List of sigma values for Gaussian smoothing
            kernel_size: Kernel size for Gaussian filter (should be odd)
            normalize: Whether to normalize the differences to [-1, 1] range
            mode: How to process the smoothed images
                - "difference": Return (original - smoothed) for each sigma
                - "concat": Return [original, smoothed, difference] for each sigma
        """
        self.sigmas = sigmas
        self.kernel_size = kernel_size
        self.normalize = normalize
        self.mode = mode

    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply smoothed difference preprocessing

        Args:
            image: Tensor of shape (C, H, W) or (B, C, H, W)

        Returns:
            differences: Tensor of shape (N*C, H, W) or (B, N*C, H, W)
                         where N depends on mode and number of sigmas
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)  # Add batch dimension
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size, channels, height, width = image.shape
        results = []

        for sigma in self.sigmas:
            # Apply Gaussian smoothing
            smoothed = self._gaussian_smooth(image, sigma)

            if self.mode == "difference":
                # Compute difference (original - smoothed)
                diff = image - smoothed
                if self.normalize:
                    diff = torch.tanh(diff * 2.0)  # Scale and normalize
                results.append(diff)
            elif self.mode == "concat":
                # Add original, smoothed, and difference
                results.append(image)
                results.append(smoothed)
                diff = image - smoothed
                if self.normalize:
                    diff = torch.tanh(diff * 2.0)
                results.append(diff)

        # Concatenate all results along channel dimension
        result = torch.cat(results, dim=1)  # (B, N*C, H, W)

        if squeeze_output:
            result = result.squeeze(0)  # Remove batch dimension

        return result

    def _gaussian_smooth(self, image: torch.Tensor, sigma: float) -> torch.Tensor:
        """Apply Gaussian smoothing to image"""
        # Use PyTorch's built-in Gaussian blur
        kernel_size = self.kernel_size

        # Ensure kernel_size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Apply Gaussian blur
        smoothed = F.gaussian_blur(image, kernel_size=kernel_size, sigma=sigma)

        return smoothed

    def get_name(self) -> str:
        """Return strategy name for logging"""
        sigmas_str = "_".join([f"{s:.1f}" for s in self.sigmas])
        return f"smoothed_diff_{self.mode}_sigmas_{sigmas_str}_k{self.kernel_size}"

    def get_output_channels(self) -> int:
        """Return number of output channels"""
        if self.mode == "difference":
            return 3 * len(self.sigmas)
        elif self.mode == "concat":
            return 3 * len(self.sigmas) * 3  # original + smoothed + diff for each sigma
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def get_model_type(self) -> str:
        """Return model type - strategy is compatible with both foundation and scratch models"""
        return "universal"


class AdaptiveSmoothedStrategy(SmoothedDifferenceStrategy):
    """Adaptive smoothed difference with content-aware sigma selection"""

    def __init__(self,
                 sigma_range: Tuple[float, float] = (0.5, 4.0),
                 num_sigmas: int = 3,
                 adaptation_method: str = "gradient",  # "gradient", "variance", "frequency"
                 **kwargs):
        """
        Initialize adaptive smoothed strategy

        Args:
            sigma_range: Range of sigma values to use
            num_sigmas: Number of sigma values to select
            adaptation_method: How to adapt sigma based on image content
            **kwargs: Additional arguments for SmoothedDifferenceStrategy
        """
        # Generate initial sigma values
        sigmas = torch.linspace(sigma_range[0], sigma_range[1], num_sigmas).tolist()
        super().__init__(sigmas=sigmas, **kwargs)
        self.sigma_range = sigma_range
        self.num_sigmas = num_sigmas
        self.adaptation_method = adaptation_method

    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive smoothed difference preprocessing

        Args:
            image: Input image tensor

        Returns:
            Adaptively processed image differences
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size, channels, height, width = image.shape

        # Compute adaptive sigmas for each image in batch
        adaptive_sigmas = self._compute_adaptive_sigmas(image)

        results = []

        for i, sigma in enumerate(adaptive_sigmas):
            # Apply smoothing with adaptive sigma
            smoothed = self._adaptive_gaussian_smooth(image, sigma)

            # Compute difference
            diff = image - smoothed
            if self.normalize:
                diff = torch.tanh(diff * 2.0)

            if self.mode == "concat":
                results.append(image)
                results.append(smoothed)

            results.append(diff)

        # Concatenate results
        result = torch.cat(results, dim=1)

        if squeeze_output:
            result = result.squeeze(0)

        return result

    def _compute_adaptive_sigmas(self, image: torch.Tensor) -> List[torch.Tensor]:
        """Compute adaptive sigma values based on image content"""
        batch_size = image.shape[0]
        adaptive_sigmas = []

        for i in range(batch_size):
            single_img = image[i:i+1]

            if self.adaptation_method == "gradient":
                # Use edge density to determine sigma
                grad_x = torch.diff(single_img, dim=-1)
                grad_y = torch.diff(single_img, dim=-2)
                edge_density = (torch.abs(grad_x).mean() + torch.abs(grad_y).mean()) / 2
                # More edges -> smaller sigma (preserve details)
                # Less edges -> larger sigma (more smoothing)
                sigma_factor = 1.0 - torch.tanh(edge_density * 10)

            elif self.adaptation_method == "variance":
                # Use local variance
                local_var = F.avg_pool2d(single_img ** 2, 7, stride=1, padding=3) - \
                           F.avg_pool2d(single_img, 7, stride=1, padding=3) ** 2
                var_mean = local_var.mean()
                # Higher variance -> smaller sigma
                sigma_factor = 1.0 - torch.tanh(var_mean * 5)

            elif self.adaptation_method == "frequency":
                # Use frequency domain analysis
                # Simple approximation using high-pass filter
                kernel = torch.tensor([[[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]]],
                                    dtype=image.dtype, device=image.device)
                kernel = kernel.repeat(channels, 1, 1, 1)
                high_freq = F.conv2d(single_img, kernel, padding=1, groups=channels)
                freq_energy = torch.abs(high_freq).mean()
                # Higher frequency energy -> smaller sigma
                sigma_factor = 1.0 - torch.tanh(freq_energy * 10)

            else:
                sigma_factor = 1.0

            # Apply sigma factor to base sigmas
            adapted_sigmas = [s * sigma_factor.item() for s in self.sigmas]
            adaptive_sigmas.append(adapted_sigmas)

        return adaptive_sigmas

    def _adaptive_gaussian_smooth(self, image: torch.Tensor, sigmas: Union[float, List[float]]) -> torch.Tensor:
        """Apply Gaussian smoothing with adaptive sigma values"""
        if isinstance(sigmas, list):
            # Use average sigma for batch processing
            sigma = sum(sigmas) / len(sigmas)
        else:
            sigma = sigmas

        return self._gaussian_smooth(image, sigma)

    def get_name(self) -> str:
        """Return strategy name for logging"""
        sigmas_str = "_".join([f"{s:.1f}" for s in self.sigmas])
        return (f"adaptive_smoothed_{self.adaptation_method}_{self.mode}"
                f"_sigmas_{sigmas_str}_range{self.sigma_range[0]:.1f}-{self.sigma_range[1]:.1f}")