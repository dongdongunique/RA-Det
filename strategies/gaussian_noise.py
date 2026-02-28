"""
Gaussian noise input strategy for AI detection models.

This strategy applies Gaussian noise with different sigma values to images,
which helps expose vulnerabilities and artifacts in AI-generated images.
"""

import torch
import torch.nn.functional as F
from typing import List, Union, Tuple
from .base import BaseInputStrategy


class GaussianNoiseStrategy(BaseInputStrategy):
    """Apply Gaussian noise with different sigma values for robustness testing

    This strategy adds Gaussian noise to input images and can work in two modes:
    1. For foundation models: Returns [original, noised] pairs
    2. For scratch models: Returns differences or concatenated versions
    """

    def __init__(self,
                 sigmas: List[float] = [1.0, 5.0, 10.0, 20.0],
                 mode: str = "concat",  # "concat", "difference", "pair"
                 normalize: bool = True,
                 clip_values: bool = True):
        """
        Initialize Gaussian noise strategy

        Args:
            sigmas: List of sigma values (in 255 scale, e.g., 1.0 means 1/255)
            mode: Mode of operation
                - "concat": Concatenate original + all noised versions
                - "difference": Compute difference between original and noised
                - "pair": Return [original, noised] pair (for foundation models)
            normalize: Whether to normalize values to [0, 1] or [-1, 1]
            clip_values: Whether to clip values to valid range [0, 1]
        """
        self.sigmas = sigmas
        self.mode = mode
        self.normalize = normalize
        self.clip_values = clip_values

    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply Gaussian noise preprocessing

        Args:
            image: Input image tensor of shape (C, H, W) or (B, C, H, W)
                   Values should be in [0, 1] range

        Returns:
            Processed image tensor based on mode
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Ensure image is in [0, 1] range
        if self.normalize and image.max() > 1.0:
            image = image / 255.0

        batch_size, channels, height, width = image.shape

        if self.mode == "pair":
            # Return [original, noised] pair
            # Use first sigma for pair mode
            noised = self._add_noise(image, self.sigmas[0])
            result = torch.cat([image, noised], dim=0)  # (2B, C, H, W)
        else:
            # Process all sigmas
            results = []

            if self.mode == "concat":
                # Start with original
                results.append(image)

            # Add noised versions for each sigma
            for sigma in self.sigmas:
                noised = self._add_noise(image, sigma)

                if self.mode == "difference":
                    # Compute difference (noised - original)
                    diff = noised - image
                    results.append(diff)
                elif self.mode == "concat":
                    # Add noised version
                    results.append(noised)

            # Concatenate all results
            result = torch.cat(results, dim=1)  # (B, C * N, H, W)

        if squeeze_output:
            result = result.squeeze(0)

        return result

    def _add_noise(self, image: torch.Tensor, sigma: float) -> torch.Tensor:
        """Add Gaussian noise to image"""
        # Convert sigma from 255-scale to [0, 1] scale
        sigma_scaled = sigma / 255.0

        # Generate Gaussian noise
        noise = torch.randn_like(image) * sigma_scaled

        # Add noise to image
        noised = image + noise

        # Clip to valid range
        if self.clip_values:
            noised = torch.clamp(noised, 0.0, 1.0)

        return noised

    def get_name(self) -> str:
        """Return strategy name for logging"""
        sigmas_str = "_".join([f"{s:.1f}" for s in self.sigmas])
        return f"gaussian_noise_{self.mode}_sigmas_{sigmas_str}"

    def get_output_channels(self) -> int:
        """Return number of output channels"""
        if self.mode == "pair":
            return 3  # Returns pair, not concatenated
        elif self.mode == "concat":
            return 3 * (1 + len(self.sigmas))  # Original + all noised
        elif self.mode == "difference":
            return 3 * len(self.sigmas)  # Only differences
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def get_model_type(self) -> str:
        """Return model type ('scratch' or 'foundation')"""
        if self.mode == "pair":
            return "foundation"  # Foundation models expect [original, noised] pairs
        else:
            return "scratch"  # Scratch models use concatenated or difference features


class MultiSigmaNoiseStrategy(GaussianNoiseStrategy):
    """Advanced Gaussian noise strategy with adaptive sigma selection"""

    def __init__(self,
                 sigma_range: Tuple[float, float] = (0.5, 50.0),
                 num_sigmas: int = 5,
                 distribution: str = "log_uniform",  # "uniform", "log_uniform"
                 **kwargs):
        """
        Initialize multi-sigma noise strategy

        Args:
            sigma_range: Min and max sigma values
            num_sigmas: Number of sigma values to generate
            distribution: How to distribute sigma values
            **kwargs: Additional arguments passed to GaussianNoiseStrategy
        """
        # Generate sigma values based on distribution
        if distribution == "uniform":
            sigmas = torch.linspace(sigma_range[0], sigma_range[1], num_sigmas).tolist()
        elif distribution == "log_uniform":
            log_sigmas = torch.linspace(torch.log(sigma_range[0]),
                                      torch.log(sigma_range[1]), num_sigmas)
            sigmas = torch.exp(log_sigmas).tolist()
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        super().__init__(sigmas=sigmas, **kwargs)
        self.sigma_range = sigma_range
        self.num_sigmas = num_sigmas
        self.distribution = distribution

    def get_name(self) -> str:
        """Return strategy name for logging"""
        return (f"multisigma_noise_{self.mode}_{self.distribution}"
                f"_range{self.sigma_range[0]:.1f}-{self.sigma_range[1]:.1f}"
                f"_n{self.num_sigmas}")


class AdaptiveNoiseStrategy(BaseInputStrategy):
    """Adaptive noise strategy based on image content"""

    def __init__(self,
                 base_sigma: float = 5.0,
                 adaptation_factor: float = 0.1,
                 measure: str = "variance",  # "variance", "gradient", "laplacian"
                 **kwargs):
        """
        Initialize adaptive noise strategy

        Args:
            base_sigma: Base sigma value
            adaptation_factor: How much to adapt based on image content
            measure: Image complexity measure to use for adaptation
            **kwargs: Additional arguments
        """
        self.base_sigma = base_sigma
        self.adaptation_factor = adaptation_factor
        self.measure = measure

    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive Gaussian noise based on image content

        Args:
            image: Input image tensor

        Returns:
            Noised image with adaptive sigma
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Compute image complexity measure
        complexity = self._compute_complexity(image)

        # Adapt sigma based on complexity
        # Higher complexity -> less noise, Lower complexity -> more noise
        adapted_sigma = self.base_sigma * (1.0 - self.adaptation_factor * complexity)

        # Apply noise with adapted sigma
        noise_strategy = GaussianNoiseStrategy(sigmas=[adapted_sigma])
        result = noise_strategy.preprocess(image)

        if squeeze_output:
            result = result.squeeze(0)

        return result

    def _compute_complexity(self, image: torch.Tensor) -> torch.Tensor:
        """Compute image complexity measure"""
        if self.measure == "variance":
            # Use local variance as complexity measure
            kernel_size = 3
            padding = kernel_size // 2
            # Compute local variance
            mean = F.avg_pool2d(image, kernel_size, stride=1, padding=padding)
            mean_sq = F.avg_pool2d(image ** 2, kernel_size, stride=1, padding=padding)
            variance = mean_sq - mean ** 2
            # Average over spatial dimensions and channels
            complexity = variance.mean(dim=(1, 2, 3))
            # Normalize to [0, 1]
            complexity = torch.tanh(complexity / 0.1)  # Normalize factor
            return complexity

        elif self.measure == "gradient":
            # Use gradient magnitude as complexity measure
            grad_x = torch.diff(image, dim=-1)
            grad_y = torch.diff(image, dim=-2)
            grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2)
            complexity = grad_mag.mean(dim=(1, 2, 3))
            complexity = torch.tanh(complexity / 0.1)
            return complexity

        elif self.measure == "laplacian":
            # Use Laplacian as complexity measure
            laplacian_kernel = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]],
                                           dtype=image.dtype, device=image.device)
            laplacian_kernel = laplacian_kernel.repeat(image.shape[1], 1, 1, 1)
            laplacian = F.conv2d(image, laplacian_kernel, padding=1, groups=image.shape[1])
            complexity = torch.abs(laplacian).mean(dim=(1, 2, 3))
            complexity = torch.tanh(complexity / 0.1)
            return complexity

        else:
            raise ValueError(f"Unknown complexity measure: {self.measure}")

    def get_name(self) -> str:
        """Return strategy name for logging"""
        return f"adaptive_noise_{self.measure}_base{self.base_sigma:.1f}_adapt{self.adaptation_factor:.2f}"

    def get_output_channels(self) -> int:
        """Return number of output channels"""
        return 3

    def get_model_type(self) -> str:
        """Return model type"""
        return "scratch"