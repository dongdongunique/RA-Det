"""
Foundation models with frozen encoders and trainable classifiers.

These models use a frozen foundation encoder (DINO, CLIP, etc.) with a trainable
classifier that processes multi-scale features. The models support two patterns:
1. List of scale tensors (each scale as separate input)
2. Single concatenated tensor (all scales concatenated along channels)

Generic foundation models work with ANY strategy that provides multi-scale inputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Union
from .base import BaseFoundationModel


class SplitChannelFoundationModel(BaseFoundationModel):
    """
    Generic foundation model that processes multiple transformed versions of the same image.

    This model treats each transformed version (resized, smoothed, etc.) as a token
    in a sequence. All tokens from the same original image are processed by the encoder
    and then fused using attention.

    Input pattern:
    - Single concatenated tensor (B, C_total, H, W)
    - where C_total = channels_per_scale * num_scales * images_per_scale
    - Contains multiple transformed versions of the same original image

    Processing:
    1. Split into individual 3-channel images (tokens)
    2. Process each token with the frozen encoder
    3. Treat all tokens as a sequence for self-attention
    4. Fuse using concat, attention, or weighted_sum
    """

    def __init__(self,
                 encoder_name: str,
                 num_scales: int,
                 channels_per_scale: int,
                 images_per_scale: int = 1,
                 feature_fusion: str = "attention",  # "concat", "attention", "weighted_sum"
                 classifier_hidden_dims: List[int] = [1024, 512],
                 dropout: float = 0.1,
                 **kwargs):
        """
        Initialize split-channel foundation model

        Args:
            encoder_name: Name of the foundation encoder to use
            num_scales: Number of scales (levels) in the concatenated input
            channels_per_scale: Number of channels per 3-channel image (typically 3)
            images_per_scale: Number of individual 3-channel images per scale level
            feature_fusion: How to fuse multi-scale features
            classifier_hidden_dims: Hidden dimensions for classifier
            dropout: Dropout rate
        """
        super().__init__(encoder_name=encoder_name, **kwargs)
        self.num_scales = num_scales
        self.channels_per_scale = channels_per_scale
        self.images_per_scale = images_per_scale
        self.feature_fusion = feature_fusion
        self.classifier_hidden_dims = classifier_hidden_dims
        self.dropout = dropout
        self.model_type = "split_channel_foundation"

        # Total number of individual 3-channel images
        self.total_images = num_scales * images_per_scale

        # Load encoder and build classifier
        self.load_encoder(config={'encoder_name': encoder_name})

    def load_encoder(self, config: Dict[str, Any]):
        """Load encoder and build fusion/classifier layers"""
        super().load_encoder(config)
        self._build_fusion_layers()
        self._build_classifier()

        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

    def _build_fusion_layers(self):
        """Build layers for fusing multi-scale features"""
        if self.feature_fusion == "attention":
            # Self-attention over scales (treat each scale as a token)
            self.scale_attention = nn.MultiheadAttention(
                embed_dim=self.feature_dim,
                num_heads=8,
                dropout=self.dropout,
                batch_first=True
            )
            self.scale_norm = nn.LayerNorm(self.feature_dim)

        elif self.feature_fusion == "weighted_sum":
            # Learnable weights for each scale
            self.scale_weights = nn.Parameter(
                torch.ones(self.num_scales) / self.num_scales
            )

    def _build_classifier(self):
        """Build classifier based on fusion method"""
        if self.feature_fusion == "concat":
            input_dim = self.feature_dim * self.num_scales
        else:  # attention or weighted_sum
            input_dim = self.feature_dim

        layers = []
        prev_dim = input_dim

        for hidden_dim in self.classifier_hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout)
                
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, self.num_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """
        Forward pass - process multiple images from same original as token sequences

        Input: Single concatenated tensor (B, C_total, H, W) where:
               - C_total = channels_per_scale * num_scales * images_per_scale
               - Contains multiple transformed versions of the same original image
               - Split into individual 3-channel images (tokens)
               - Each token is processed by the encoder
               - All tokens from same image are treated as a sequence for attention

        Returns:
            logits: Classification logits
        """
        # Handle both list and tensor inputs
        if isinstance(x, list):
            # Input is already a list of scale tensors
            image_tensors = x
        else:
            # Input is a concatenated tensor - split into individual 3-channel images
            #print("input x shape: ",x.shape, "total images: ",self.total_images)
            image_tensors = torch.chunk(x, chunks=self.total_images, dim=1)
            # Remove extra dimension if created by chunk
            image_tensors = [t.squeeze(1) if t.shape[1] == 1 else t for t in image_tensors]

        # Verify each image has the expected number of channels (should be 3 for RGB)
        for i, image_tensor in enumerate(image_tensors):
            expected_channels = self.channels_per_scale
            actual_channels = image_tensor.shape[1]
            if actual_channels != expected_channels:
                raise ValueError(
                    f"Image {i} has {actual_channels} channels, but expected "
                    f"{expected_channels} channels. Each image must have exactly "
                    f"{expected_channels} channels (typically 3 for RGB images)."
                )

        # Extract features from each individual 3-channel image
        image_features = []
        for image_tensor in image_tensors:
            features = self.extract_batch_features(image_tensor)
            image_features.append(features)

        # Treat ALL images as a single sequence for attention
        # We have total_images individual encoded features, each (B, feature_dim)
        # Stack into sequence: (total_images, B, feature_dim)
        # Permute to: (B, total_images, feature_dim) for _fuse_scales
        sequence_features = torch.stack(image_features, dim=0)
        # Shape: (total_images, B, feature_dim)
        sequence_features = sequence_features.permute(1, 0, 2)
        # Shape: (B, total_images, feature_dim)

        # Fuse all image tokens using attention
        fused_features = self._fuse_scales(sequence_features)

        # Classify
        logits = self.classifier(fused_features)
        return logits

    def _fuse_scales(self, scale_features: Union[List[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """Fuse features from multiple scales or image tokens

        Args:
            scale_features: Either:
                - List of tensors, each (B, feature_dim) for multi-scale fusion
                - Single tensor (B, seq_len, feature_dim) for sequence fusion
                  where seq_len = total number of images/tokens

        Returns:
            Fused features: (B, fused_feature_dim)
        """
        # Handle both list and tensor inputs
        if isinstance(scale_features, torch.Tensor):
            # Input is already a stacked tensor: (B, seq_len, feature_dim)
            stacked = scale_features
        else:
            # Input is a list of tensors: [(B, feature_dim), ...]
            # Stack features: [batch, num_scales, feature_dim]
            stacked = torch.stack(scale_features, dim=1)

        if self.feature_fusion == "concat":
            # Concatenate all features along feature dimension
            # (B, seq_len, feature_dim) -> (B, seq_len * feature_dim)
            batch_size, seq_len, feature_dim = stacked.shape
            return stacked.reshape(batch_size, seq_len * feature_dim)

        elif self.feature_fusion == "attention":
            # Apply self-attention where sequence tokens are the dimension
            # Apply self-attention across sequence tokens
            attended, attention_weights = self.scale_attention(
                stacked, stacked, stacked
            )

            # Apply layer normalization
            attended = self.scale_norm(attended)

            # Aggregate across sequence tokens (mean pooling)
            fused = attended.mean(dim=1)

            return fused

        elif self.feature_fusion == "weighted_sum":
            # Weighted sum of features along sequence dimension
            # (B, seq_len, feature_dim) -> (B, feature_dim)
            normalized_weights = F.softmax(self.scale_weights[:stacked.shape[1]], dim=0)
            # Apply weights along seq_len dimension
            weighted = stacked * normalized_weights.view(1, -1, 1)
            return weighted.sum(dim=1)

        else:
            raise ValueError(f"Unknown fusion method: {self.feature_fusion}")

    def get_name(self) -> str:
        """Return model name for logging"""
        return (f"split_channel_{self.encoder_name}_{self.feature_fusion}_scales{self.num_scales}_"
                f"imgs{self.images_per_scale}_ch{self.channels_per_scale}")


class ScaleAwareFoundationModel(BaseFoundationModel):
    """Scale-aware foundation model with learnable scale embeddings"""

    def __init__(self,
                 encoder_name: str,
                 levels: List[int] = [0, 1, 2, 3],
                 scale_embed_dim: int = 64,
                 classifier_hidden_dims: List[int] = [1024, 512],
                 dropout: float = 0.1,
                 **kwargs):
        """
        Initialize scale-aware foundation model

        Args:
            encoder_name: Name of the foundation encoder
            levels: Pyramid levels to process
            scale_embed_dim: Dimension for scale embeddings
            classifier_hidden_dims: Hidden dimensions for classifier
        """
        super().__init__(encoder_name=encoder_name, **kwargs)
        self.levels = levels
        self.scale_embed_dim = scale_embed_dim
        self.classifier_hidden_dims = classifier_hidden_dims
        self.dropout = dropout
        self.model_type = "scale_aware_foundation"

        # Load encoder and build classifier
        self.load_encoder(config={'encoder_name': encoder_name})

    def load_encoder(self, config: Dict[str, Any]):
        """Load encoder and build scale-aware components"""
        super().load_encoder(config)
        self._build_scale_embeddings()
        self._build_classifier()

        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

    def _build_scale_embeddings(self):
        """Build learnable embeddings for each scale"""
        self.scale_embeddings = nn.Embedding(len(self.levels), self.scale_embed_dim)
        self.scale_projection = nn.Linear(self.feature_dim + self.scale_embed_dim,
                                        self.feature_dim)

    def _build_classifier(self):
        """Build classifier"""
        layers = []
        prev_dim = self.feature_dim

        for hidden_dim in self.classifier_hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, self.num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass with scale-aware processing

        Args:
            x: List of Gaussian pyramid levels

        Returns:
            logits: Classification logits
        """
        # Extract features from each scale
        scale_features = []
        for i, scale_tensor in enumerate(x):
            if scale_tensor.dim() == 3:
                scale_tensor = scale_tensor.unsqueeze(0)
            features = self.extract_batch_features(scale_tensor)

            # Get scale embedding
            scale_idx = torch.tensor(i, device=features.device)
            scale_emb = self.scale_embeddings(scale_idx).unsqueeze(0).expand(features.shape[0], -1)

            # Concatenate features and scale embedding
            combined = torch.cat([features, scale_emb], dim=1)
            projected = self.scale_projection(combined)

            scale_features.append(projected)

        # Aggregate across scales (mean pooling)
        fused = torch.stack(scale_features, dim=1).mean(dim=1)

        # Classify
        logits = self.classifier(fused)
        return logits

    def get_name(self) -> str:
        """Return model name"""
        levels_str = "_".join(map(str, self.levels))
        return f"scale_aware_{self.encoder_name}_embed{self.scale_embed_dim}_levels_{levels_str}"


class GaussianPyramidFoundationModel(BaseFoundationModel):
    """Foundation model that processes Gaussian pyramid features

    This model extracts features from multiple scales of the Gaussian pyramid
    and combines them for AI-generated image detection.
    """

    def __init__(self,
                 encoder_name: str,
                 levels: List[int] = [0, 1, 2, 3],
                 feature_fusion: str = "attention",  # "concat", "attention", "weighted_sum"
                 classifier_hidden_dims: List[int] = [1024, 512],
                 dropout: float = 0.1,
                 **kwargs):
        """
        Initialize Gaussian pyramid foundation model

        Args:
            encoder_name: Name of the foundation encoder to use
            levels: Pyramid levels to process
            feature_fusion: How to fuse multi-scale features
            classifier_hidden_dims: Hidden dimensions for classifier
            dropout: Dropout rate
        """
        super().__init__(encoder_name=encoder_name, **kwargs)
        self.levels = levels
        self.feature_fusion = feature_fusion
        self.classifier_hidden_dims = classifier_hidden_dims
        self.dropout = dropout
        self.model_type = "gaussian_pyramid_foundation"

        # Load encoder and build classifier
        self.load_encoder(config={'encoder_name': encoder_name})

    def load_encoder(self, config: Dict[str, Any]):
        """Load encoder and build classifier"""
        super().load_encoder(config)
        self._build_fusion_layers()
        self._build_classifier()

        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

    def _build_fusion_layers(self):
        """Build layers for fusing multi-scale features"""
        if self.feature_fusion == "attention":
            # Self-attention over scales (treat each scale as a token)
            self.scale_attention = nn.MultiheadAttention(
                embed_dim=self.feature_dim,
                num_heads=8,
                dropout=self.dropout,
                batch_first=True  # [batch, seq_len, feature_dim]
            )
            self.scale_norm = nn.LayerNorm(self.feature_dim)

        elif self.feature_fusion == "weighted_sum":
            # Learnable weights for each scale
            self.scale_weights = nn.Parameter(
                torch.ones(len(self.levels)) / len(self.levels)
            )

    def _build_classifier(self):
        """Build classifier based on fusion method"""
        if self.feature_fusion == "concat":
            input_dim = self.feature_dim * len(self.levels)
        else:  # attention or weighted_sum
            input_dim = self.feature_dim

        layers = []
        prev_dim = input_dim

        for hidden_dim in self.classifier_hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, self.num_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through Gaussian pyramid model

        Args:
            x: List of Gaussian pyramid levels (all resized to input_size)

        Returns:
            logits: Classification logits
        """
        # Extract features from each scale
        scale_features = []
        for scale_tensor in x:
            # Ensure batch dimension
            if scale_tensor.dim() == 3:
                scale_tensor = scale_tensor.unsqueeze(0)
            features = self.extract_batch_features(scale_tensor)
            scale_features.append(features)

        # Fuse multi-scale features
        fused_features = self._fuse_scales(scale_features)

        # Classify
        logits = self.classifier(fused_features)
        return logits

    def _fuse_scales(self, scale_features: List[torch.Tensor]) -> torch.Tensor:
        """Fuse features from multiple scales"""
        if self.feature_fusion == "concat":
            # Concatenate all scale features
            return torch.cat(scale_features, dim=1)

        elif self.feature_fusion == "attention":
            # Apply self-attention where scales are the sequence dimension

            # Stack features: [batch, num_scales, feature_dim]
            stacked = torch.stack(scale_features, dim=1)

            # Apply self-attention across scales
            attended, attention_weights = self.scale_attention(
                stacked, stacked, stacked
            )

            # Apply layer normalization
            attended = self.scale_norm(attended)

            # Aggregate across scales (mean pooling)
            fused = attended.mean(dim=1)

            return fused

        elif self.feature_fusion == "weighted_sum":
            # Weighted sum of scale features
            normalized_weights = F.softmax(self.scale_weights, dim=0)

            weighted_features = []
            for i, features in enumerate(scale_features):
                weight = normalized_weights[i]
                weighted_features.append(weight * features)

            return sum(weighted_features)

        else:
            raise ValueError(f"Unknown fusion method: {self.feature_fusion}")

    def get_name(self) -> str:
        """Return model name for logging"""
        levels_str = "_".join(map(str, self.levels))
        return f"gaussian_foundation_{self.encoder_name}_{self.feature_fusion}_levels_{levels_str}"