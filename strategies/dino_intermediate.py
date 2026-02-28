"""
DINOv3 intermediate-layer input strategy.

This strategy converts CLIP-normalized inputs to ImageNet normalization
for DINOv3-style encoders.
"""

from typing import Tuple

import torch

from .base import BaseInputStrategy


def _as_tensor(values: Tuple[float, float, float], ref: torch.Tensor) -> torch.Tensor:
    return torch.tensor(values, device=ref.device, dtype=ref.dtype).view(1, 3, 1, 1)


class DinoIntermediateStrategy(BaseInputStrategy):
    """Convert CLIP-normalized inputs to ImageNet normalization."""

    CLIP_MEAN = (0.485, 0.456, 0.406)
    CLIP_STD = (0.229, 0.224, 0.225)
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, input_norm: str = "clip", output_norm: str = "imagenet", clamp: bool = True):
        self.input_norm = input_norm
        self.output_norm = output_norm
        self.clamp = clamp

    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        if image.dim() == 3:
            image = image.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        x = image

        if self.input_norm == "clip":
            clip_mean = _as_tensor(self.CLIP_MEAN, x)
            clip_std = _as_tensor(self.CLIP_STD, x)
            x = x * clip_std + clip_mean
        elif self.input_norm == "imagenet":
            img_mean = _as_tensor(self.IMAGENET_MEAN, x)
            img_std = _as_tensor(self.IMAGENET_STD, x)
            x = x * img_std + img_mean

        if self.clamp:
            x = x.clamp(0.0, 1.0)

        if self.output_norm == "imagenet":
            img_mean = _as_tensor(self.IMAGENET_MEAN, x)
            img_std = _as_tensor(self.IMAGENET_STD, x)
            x = (x - img_mean) / img_std
        elif self.output_norm == "clip":
            clip_mean = _as_tensor(self.CLIP_MEAN, x)
            clip_std = _as_tensor(self.CLIP_STD, x)
            x = (x - clip_mean) / clip_std

        if squeeze_output:
            x = x.squeeze(0)

        return x

    def get_name(self) -> str:
        return f"dino_intermediate_{self.input_norm}_to_{self.output_norm}"

    def get_output_channels(self) -> int:
        return 3

    def get_model_type(self) -> str:
        return "foundation"
