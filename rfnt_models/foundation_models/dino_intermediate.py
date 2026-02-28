"""
RINE-style detector using intermediate DINOv3 encoder blocks.
"""

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .intermediate_base import BaseIntermediateFoundationModel


class TrainableImportanceEstimator(nn.Module):
    """Trainable importance estimator over block tokens."""

    def __init__(self, num_layers: int):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_layers))

    def forward(self) -> torch.Tensor:
        return torch.softmax(self.logits, dim=0)


class DinoIntermediateRINEModel(BaseIntermediateFoundationModel):
    """
    RINE-style model that aggregates intermediate CLS tokens from DINOv3.

    Pipeline:
    1) Extract CLS tokens from selected encoder blocks.
    2) Project tokens with Q1.
    3) Apply trainable importance estimator (TIE) weights.
    4) Sum across blocks to obtain a single feature per image.
    5) Project with Q2 and classify.
    """

    def __init__(
        self,
        encoder_name: str,
        layer_indices: Optional[Sequence[int]] = None,
        num_layers: int = 4,
        proj_hidden_dim: int = 1024,
        proj_dropout: float = 0.1,
        classifier_hidden_dims: Optional[List[int]] = None,
        tie_temperature: float = 1.0,
        num_classes: int = 2,
        use_contrastive: bool = False,
        contrastive_temperature: float = 0.07,
        contrastive_weight: float = 0.1,
        **kwargs,
    ):
        super().__init__(encoder_name=encoder_name, num_classes=num_classes, **kwargs)
        self.layer_indices = list(layer_indices) if layer_indices is not None else None
        self.num_layers = num_layers
        self.proj_hidden_dim = proj_hidden_dim
        self.proj_dropout = proj_dropout
        self.tie_temperature = tie_temperature
        self.use_contrastive = use_contrastive
        self.contrastive_temperature = contrastive_temperature
        self.contrastive_weight = contrastive_weight

        if classifier_hidden_dims is None:
            classifier_hidden_dims = [512, 256]
        self.classifier_hidden_dims = classifier_hidden_dims

        self.model_type = "dino_intermediate_rine"

        self.load_encoder(config={'encoder_name': encoder_name})

        self.q1 = nn.Sequential(
            nn.Linear(self.feature_dim, proj_hidden_dim),
            nn.ReLU(),
            nn.Dropout(proj_dropout),
            nn.Linear(proj_hidden_dim, self.feature_dim),
            nn.ReLU(),
            nn.Dropout(proj_dropout),
        )

        num_blocks = len(self.layer_indices) if self.layer_indices is not None else self.num_layers
        self.tie = TrainableImportanceEstimator(num_blocks)

        self.q2 = nn.Sequential(
            nn.Linear(self.feature_dim, proj_hidden_dim),
            nn.ReLU(),
            nn.Dropout(proj_dropout),
            nn.Linear(proj_hidden_dim, self.feature_dim),
            nn.ReLU(),
            nn.Dropout(proj_dropout),
        )

        cls_layers = []
        prev_dim = self.feature_dim
        for hidden_dim in self.classifier_hidden_dims:
            cls_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(proj_dropout),
            ])
            prev_dim = hidden_dim
        cls_layers.append(nn.Linear(prev_dim, self.num_classes))
        self.classifier = nn.Sequential(*cls_layers)

        for param in self.encoder.parameters():
            param.requires_grad = False

    def _get_intermediate_cls_tokens(self, images: torch.Tensor) -> torch.Tensor:
        encoder = self.encoder
        if not hasattr(encoder, "get_intermediate_layers") and hasattr(encoder, "dinov2"):
            encoder = encoder.dinov2

        if not hasattr(encoder, "get_intermediate_layers"):
            raise RuntimeError("Encoder does not support get_intermediate_layers")

        if self.layer_indices is not None:
            outputs = encoder.get_intermediate_layers(
                images,
                n=self.layer_indices,
                return_class_token=True,
            )
        else:
            outputs = encoder.get_intermediate_layers(
                images,
                n=self.num_layers,
                return_class_token=True,
            )

        cls_tokens = [cls_token for (_, cls_token) in outputs]
        return torch.stack(cls_tokens, dim=1)

    def get_intermediate_tokens(self, images: torch.Tensor) -> torch.Tensor:
        return self._get_intermediate_cls_tokens(images)

    def aggregate_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        projected = self.q1(tokens)
        weights = self.tie() / self.tie_temperature
        weighted = projected * weights.view(1, -1, 1)
        fused = weighted.sum(dim=1)
        fused = self.q2(fused)
        return fused

    def forward(self, x: torch.Tensor):
        tokens = self.get_intermediate_tokens(x)
        fused = self.aggregate_tokens(tokens)
        logits = self.classifier(fused)

        if self.use_contrastive:
            features = F.normalize(fused, dim=1)
            return logits, features

        return logits

    def get_name(self) -> str:
        if self.layer_indices is not None:
            layer_str = "_".join(map(str, self.layer_indices))
        else:
            layer_str = f"last{self.num_layers}"
        return f"dino_intermediate_rine_{self.encoder_name}_{layer_str}"
