"""
Low-level feature strategies for scratch branches.

Provides:
- GaussianDifferenceStrategy: subtract Gaussian-blurred image from original
- ResizeDifferenceStrategy: subtract downsampled-then-upsamped image from original
"""

from io import BytesIO
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .base import BaseInputStrategy
from .median_filter import LocalPixelDependencyStrategy


def _gaussian_kernel_2d(kernel_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_1d = g.view(1, 1, -1)
    kernel_2d = kernel_1d.transpose(1, 2) @ kernel_1d
    return kernel_2d


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
        buffer.close()
        jpeg_tensor = torch.from_numpy(jpeg_np).permute(2, 0, 1)
        compressed.append(jpeg_tensor)

    result = torch.stack(compressed, dim=0)
    if input_in_minus_one_one:
        result = result * 2.0 - 1.0

    return result.to(device)


class GaussianDifferenceStrategy(BaseInputStrategy):
    """
    Compute (image - gaussian_blur(image)) for one or more sigmas.
    """

    def __init__(self, sigmas: List[float] = [1.0], kernel_size: int = 5, normalize: bool = False):
        self.sigmas = sigmas
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.normalize = normalize

    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        if image.dim() == 3:
            image = image.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        B, C, H, W = image.shape
        results = []

        for sigma in self.sigmas:
            kernel = _gaussian_kernel_2d(self.kernel_size, sigma, image.device, image.dtype)
            kernel = kernel.view(1, 1, self.kernel_size, self.kernel_size)
            kernel = kernel.repeat(C, 1, 1, 1)
            padding = self.kernel_size // 2
            blurred = F.conv2d(image, kernel, padding=padding, groups=C)
            diff = image - blurred
            if self.normalize:
                diff = (diff - diff.mean()) / (diff.std() + 1e-8)
            results.append(diff)

        out = torch.cat(results, dim=1)
        if squeeze_output:
            out = out.squeeze(0)
        return out

    def get_name(self) -> str:
        sigma_str = "_".join([str(s).replace('.', 'p') for s in self.sigmas])
        norm_str = "_norm" if self.normalize else ""
        return f"gaussian_diff_k{self.kernel_size}_s{sigma_str}{norm_str}"

    def get_output_channels(self) -> int:
        return 3 * len(self.sigmas)

    def get_model_type(self) -> str:
        return "scratch"


class ResizeDifferenceStrategy(BaseInputStrategy):
    """
    Compute (image - downsample_upsample(image)) for one or more scales.
    """

    def __init__(self, scales: List[float] = [0.5], mode: str = "bilinear", normalize: bool = False):
        self.scales = scales
        self.mode = mode
        self.normalize = normalize

    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        if image.dim() == 3:
            image = image.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        results = []
        for scale in self.scales:
            down = F.interpolate(image, scale_factor=scale, mode=self.mode, align_corners=False, recompute_scale_factor=True)
            up = F.interpolate(down, size=image.shape[-2:], mode=self.mode, align_corners=False)
            diff = image - up
            if self.normalize:
                diff = (diff - diff.mean()) / (diff.std() + 1e-8)
            results.append(diff)

        out = torch.cat(results, dim=1)
        if squeeze_output:
            out = out.squeeze(0)
        return out

    def get_name(self) -> str:
        scale_str = "_".join([str(s).replace('.', 'p') for s in self.scales])
        norm_str = "_norm" if self.normalize else ""
        return f"resize_diff_s{scale_str}_{self.mode}{norm_str}"

    def get_output_channels(self) -> int:
        return 3 * len(self.scales)

    def get_model_type(self) -> str:
        return "scratch"


class JpegDifferenceStrategy(BaseInputStrategy):
    """
    Compute (image - jpeg_compress(image)) for one or more JPEG qualities.
    """

    def __init__(self, qualities: List[int] = [95, 75, 50], normalize: bool = False):
        self.qualities = [int(q) for q in qualities]
        self.normalize = normalize

    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        if image.dim() == 3:
            image = image.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        results = []
        for quality in self.qualities:
            compressed = _jpeg_compress_batch(image, quality)
            diff = image - compressed
            if self.normalize:
                diff = (diff - diff.mean()) / (diff.std() + 1e-8)
            results.append(diff)

        out = torch.cat(results, dim=1)
        if squeeze_output:
            out = out.squeeze(0)
        return out

    def get_name(self) -> str:
        quality_str = "_".join(str(q) for q in self.qualities)
        norm_str = "_norm" if self.normalize else ""
        return f"jpeg_diff_q{quality_str}{norm_str}"

    def get_output_channels(self) -> int:
        return 3 * len(self.qualities)

    def get_model_type(self) -> str:
        return "scratch"


class LpdJpegDifferenceStrategy(BaseInputStrategy):
    """
    Concatenate LPD (median-based) features with JPEG-compression differences.
    """

    def __init__(
        self,
        lpd_kernel_sizes: List[int] = [3],
        jpeg_qualities: List[int] = [95, 75, 50],
        lpd_normalize: bool = False,
        jpeg_normalize: bool = False,
    ):
        self.lpd_strategy = LocalPixelDependencyStrategy(
            kernel_sizes=lpd_kernel_sizes,
            normalize=lpd_normalize,
        )
        self.jpeg_strategy = JpegDifferenceStrategy(
            qualities=jpeg_qualities,
            normalize=jpeg_normalize,
        )
        self.lpd_kernel_sizes = [int(k) for k in lpd_kernel_sizes]
        self.jpeg_qualities = [int(q) for q in jpeg_qualities]
        self.lpd_normalize = lpd_normalize
        self.jpeg_normalize = jpeg_normalize

    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        if image.dim() == 3:
            image = image.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        lpd_features = self.lpd_strategy.preprocess(image)
        jpeg_features = self.jpeg_strategy.preprocess(image)
        out = torch.cat([lpd_features, jpeg_features], dim=1)

        if squeeze_output:
            out = out.squeeze(0)
        return out

    def get_name(self) -> str:
        kernel_str = "_".join(map(str, self.lpd_kernel_sizes))
        quality_str = "_".join(map(str, self.jpeg_qualities))
        norm_str = "_norm" if (self.lpd_normalize or self.jpeg_normalize) else ""
        return f"lpd_jpeg_diff_k{kernel_str}_q{quality_str}{norm_str}"

    def get_output_channels(self) -> int:
        return self.lpd_strategy.get_output_channels() + self.jpeg_strategy.get_output_channels()

    def get_model_type(self) -> str:
        return "scratch"
