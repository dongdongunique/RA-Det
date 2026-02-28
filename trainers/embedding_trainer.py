"""
AnyAttack Embedding Trainer with Optional Classification Loss (DDP Compatible).

This module implements a trainer for the AnyAttack-style noise decoder that:
1. Primary: Minimizes cosine similarity between clean and noisy DINOv3 embeddings
2. Optional: Adds binary classification loss (real/fake detection) from RFNT
3. Supports Distributed Data Parallel (DDP) for multi-GPU training with torchrun
4. Tracks similarity separately for real and fake images
5. Supports cross-generator evaluation

Usage:
    # Single GPU
    python train.py --config anyattack_decoder_vitl16

    # Multi-GPU with torchrun
    torchrun --nproc_per_node=4 train.py --config anyattack_decoder_vitl16

Author: Implementation Plan
"""

import os
import sys
from datetime import datetime

# Get absolute path of this file to ensure correct path resolution
_FILE_DIR = os.path.dirname(os.path.abspath(__file__))

# Add release root to sys.path for local imports
RELEASE_ROOT = os.path.dirname(os.path.dirname(_FILE_DIR))
if RELEASE_ROOT not in sys.path:
    sys.path.insert(0, RELEASE_ROOT)

# Now we can import from local rfnt_models
import math
from typing import Dict, List, Optional, Tuple, Union
from rfnt_models.foundation_models.gaussian_pyramid import SplitChannelFoundationModel
from rfnt_models.scratch_models.multi_scale_difference_cnn import MultiScaleDifferenceCNN
# Import new flexible ensemble classifier
from rfnt_models.ensemble import BackwardCompatibleEnsemble as EnsembleClassifier
from rfnt_models.ensemble import FourBranchEnsemble, ThreeBranchEnsembleNoLPD
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset
from tqdm import tqdm
import numpy as np

# Temporarily add anyattack_training directory for local decoder import
anyattack_dir = os.path.dirname(_FILE_DIR)
if anyattack_dir not in sys.path:
    sys.path.insert(0, anyattack_dir)

# Import local modules first
from models.decoder import AnyAttackDecoder, UNetDecoder, create_decoder, DINOV3_CHANNELS

# Remove anyattack_dir from sys.path to prevent conflicts with training models
if anyattack_dir in sys.path:
    sys.path.remove(anyattack_dir)

# Add release root directory to path for local create_models
release_root = os.path.dirname(os.path.dirname(_FILE_DIR))
if release_root not in sys.path:
    sys.path.insert(0, release_root)

from create_models import load_models

# Import existing dataset classes
from datasets.progan import AdaptiveAIGCDataset, ProGANTrainingDataset
from datasets.aigctest import AIGCTestDataset, CrossGeneratorEvaluator as BaseCrossGeneratorEvaluator


class CrossAttentionClassifier(nn.Module):
    """
    Cross-attention classifier that takes both clean and noisy embeddings.

    The key insight: the DISCREPANCY between clean and noisy embeddings
    contains the signal for real vs fake classification.

    Architecture:
    1. Cross-attention: clean (query) attends to noisy (key/value)
    2. Self-attention on the concatenated features
    3. MLP head for classification
    """

    def __init__(self, feature_dim: int, num_heads: int = 8, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads

        # Cross-attention: clean (query) -> noisy (key, value)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(feature_dim)

        # Self-attention on combined features
        self.self_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(feature_dim)

        # Difference projection
        self.diff_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),  # concat: attn_out + self_attn + diff
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, 1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, clean_embeddings, noisy_embeddings):
        """
        Args:
            clean_embeddings: [B, feature_dim] Clean image embeddings
            noisy_embeddings: [B, feature_dim] Noisy image embeddings

        Returns:
            logits: [B, 1] Binary classification logits
        """
        B = clean_embeddings.size(0)

        # Add sequence dimension for attention: [B, 1, feature_dim]
        clean_query = clean_embeddings.unsqueeze(1)  # [B, 1, D]
        noisy_kv = noisy_embeddings.unsqueeze(1)     # [B, 1, D]

        # Cross-attention: clean attends to noisy
        # "What information from noisy is relevant to clean?"
        cross_attn_out, _ = self.cross_attn(
            query=clean_query,
            key=noisy_kv,
            value=noisy_kv
        )
        cross_attn_out = self.norm1(clean_query + cross_attn_out).squeeze(1)  # [B, D]

        # Self-attention on stacked features
        stacked = torch.stack([clean_embeddings, noisy_embeddings], dim=1)  # [B, 2, D]
        self_attn_out, _ = self.self_attn(stacked, stacked, stacked)
        self_attn_out = self.norm2(stacked + self_attn_out)  # [B, 2, D]

        # Pool over sequence dimension
        self_attn_out = self_attn_out.mean(dim=1)  # [B, D]

        # Difference feature (explicitly model discrepancy)
        diff = noisy_embeddings - clean_embeddings
        diff_feat = self.diff_proj(diff)

        # Concatenate all features
        combined = torch.cat([cross_attn_out, self_attn_out, diff_feat], dim=1)  # [B, 3*D]

        # Classify
        logits = self.classifier(combined)
        return logits


class NoiseEmbeddingClassifier(nn.Module):
    """
    Classifier that uses noise plus clean/noisy embeddings.

    Inputs:
      - noise: [B, 3, H, W]
      - clean_embeddings: [B, D]
      - noisy_embeddings: [B, D]
    """

    def __init__(
        self,
        feature_dim: int,
        noise_channels: int = 3,
        noise_feature_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.noise_channels = noise_channels
        self.noise_feature_dim = noise_feature_dim

        self.noise_encoder = nn.Sequential(
            nn.Conv2d(noise_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, noise_feature_dim),
            nn.ReLU(inplace=True),
        )

        self.embedding_proj = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        combined_dim = feature_dim + noise_feature_dim
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, 1),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, noise, clean_embeddings, noisy_embeddings):
        if noise.dim() != 4:
            raise ValueError(f"noise must be 4D [B, C, H, W], got {noise.dim()}D")

        diff = noisy_embeddings - clean_embeddings
        embedding_features = torch.cat([clean_embeddings, noisy_embeddings, diff], dim=1)
        embedding_features = self.embedding_proj(embedding_features)

        noise_features = self.noise_encoder(noise)
        combined = torch.cat([embedding_features, noise_features], dim=1)
        logits = self.classifier(combined)
        return logits


class NoiseEmbeddingClassifierWithL2(nn.Module):
    """
    Noise + embedding classifier with an additional L2-distance branch.

    Returns ensemble logits and per-branch logits.
    """

    def __init__(
        self,
        feature_dim: int,
        noise_channels: int = 3,
        noise_feature_dim: int = 128,
        dropout: float = 0.1,
        l2_hidden_dims: Optional[List[int]] = None,
    ):
        super().__init__()
        if l2_hidden_dims is None:
            l2_hidden_dims = [128, 64, 1]

        self.feature_dim = feature_dim
        self.noise_channels = noise_channels
        self.noise_feature_dim = noise_feature_dim

        self.noise_encoder = nn.Sequential(
            nn.Conv2d(noise_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, noise_feature_dim),
            nn.ReLU(inplace=True),
        )

        self.embedding_proj = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        combined_dim = feature_dim + noise_feature_dim
        self.noise_embedding_head = nn.Sequential(
            nn.Linear(combined_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, 1),
        )

        l2_layers = []
        prev_dim = 1
        for i, hidden_dim in enumerate(l2_hidden_dims):
            l2_layers.append(nn.Linear(prev_dim, hidden_dim))
            if i < len(l2_hidden_dims) - 1:
                l2_layers.append(nn.BatchNorm1d(hidden_dim))
                l2_layers.append(nn.ReLU(inplace=True))
                if dropout > 0:
                    l2_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        self.l2_head = nn.Sequential(*l2_layers)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        noise: torch.Tensor,
        clean_embeddings: torch.Tensor,
        noisy_embeddings: torch.Tensor,
        return_all_logits: bool = True,
        use_max_for_eval: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if noise.dim() != 4:
            raise ValueError(f"noise must be 4D [B, C, H, W], got {noise.dim()}D")

        diff = noisy_embeddings - clean_embeddings
        embedding_features = torch.cat([clean_embeddings, noisy_embeddings, diff], dim=1)
        embedding_features = self.embedding_proj(embedding_features)

        noise_features = self.noise_encoder(noise)
        combined = torch.cat([embedding_features, noise_features], dim=1)
        noise_emb_logits = self.noise_embedding_head(combined)

        l2_dist = torch.norm(clean_embeddings - noisy_embeddings, p=2, dim=1, keepdim=True)
        l2_logits = self.l2_head(l2_dist)

        if use_max_for_eval:
            probs = torch.stack([torch.sigmoid(noise_emb_logits), torch.sigmoid(l2_logits)], dim=-1)
            max_probs, _ = torch.max(probs, dim=-1)
            max_probs = torch.clamp(max_probs, min=1e-7, max=1 - 1e-7)
            ensemble_logits = torch.log(max_probs / (1 - max_probs)).unsqueeze(-1)
        else:
            ensemble_logits = (noise_emb_logits + l2_logits) / 2.0

        if return_all_logits:
            return ensemble_logits, {
                "noise_embedding": noise_emb_logits,
                "l2_distance": l2_logits,
            }
        return ensemble_logits


# Keep the old simple classifier as fallback
class EmbeddingClassifierHead(nn.Module):
    """
    Simple binary classification head for real/fake detection (fallback).
    Use CrossAttentionClassifier for better performance.
    """

    def __init__(self, feature_dim: int, num_classes: int = 1):
        super().__init__()
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.bn = nn.BatchNorm1d(feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        """
        Args:
            x: [B, feature_dim] DINOv3 embeddings
        Returns:
            logits: [B, 1] binary classification logits
        """
        x = x.unsqueeze(-1)
        x = self.pooling(x).squeeze(-1)
        x = self.bn(x)
        logits = self.classifier(x)
        return logits


class AblationEnsembleClassifier(nn.Module):
    """
    Ablation study classifier with embedding difference + noise+LPD branches.

    Architecture:
    1. Embedding difference branch (MLP):
       - Input: clean_embedding - noisy_embedding
       - Takes difference embedding [B, feature_dim] → MLP

    2. Scratch branch (MultiScaleDifferenceCNN):
       - Input: concat(noise, lpd_features) on channel dimension
       - Takes noise (3 channels) + LPD (3 channels) = 6 channels → multi-scale CNN

    3. Ensemble fusion: Logit-weighted softmax (2 branches)

    Args:
        feature_dim: DINOv3 embedding dimension (default: 1024)
        noise_channels: Number of channels for noise input (default: 3)
        lpd_channels: Number of channels for LPD features (default: 3)
        resnet_variant: ResNet variant for scratch branch
        dropout: Dropout rate
        temperature: Temperature for softmax weighting
    """

    def __init__(
        self,
        feature_dim: int = 1024,
        noise_channels: int = 3,
        lpd_channels: int = 3,
        resnet_variant: str = "resnet18",
        dropout: float = 0.1,
        temperature: float = 1.0
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.noise_channels = noise_channels
        self.lpd_channels = lpd_channels
        self.resnet_variant = resnet_variant
        self.temperature = temperature

        # ====================================================================
        # Embedding difference branch: MLP to process embedding difference
        # Input: clean_embedding - noisy_embedding [B, feature_dim]
        # ====================================================================
        self.embedding_diff_mlp = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

        # ====================================================================
        # Scratch branch: MultiScaleDifferenceCNN
        # Input: concat(noise, lpd_features) = noise_channels + lpd_channels
        # ====================================================================
        self.scratch_model = MultiScaleDifferenceCNN(
            input_channels=noise_channels + lpd_channels,  # 3 (noise) + 3 (LPD) = 6 channels
            architecture=resnet_variant,
            pretrained=False,
            num_classes=1
        )

    def forward(self, embedding_diff, noise, lpd_features, use_max_for_eval=False):
        """
        Forward pass with ensemble fusion.

        Args:
            embedding_diff: [B, feature_dim] clean_embedding - noisy_embedding
            noise: [B, 3, H, W] Decoder noise output
            lpd_features: [B, 3, H, W] LPD features from original image
            use_max_for_eval: If True, use max fusion (for evaluation only)

        Returns:
            ensemble_logits: [B, 1] Ensemble classification logits
            embedding_diff_logits: [B, 1] Embedding difference branch logits
            scratch_logits: [B, 1] Scratch branch logits
        """
        # ====================================================================
        # Embedding difference branch: Process embedding difference vector
        # ====================================================================
        embedding_diff_logits = self.embedding_diff_mlp(embedding_diff)

        # ====================================================================
        # Scratch branch: Process concat(noise, lpd_features)
        # ====================================================================
        # Concatenate noise and LPD features on channel dimension
        noise_lpd = torch.cat([noise, lpd_features], dim=1)  # [B, 6, H, W]
        scratch_logits = self.scratch_model(noise_lpd)

        # ====================================================================
        # Ensemble fusion (2 branches)
        # ====================================================================
        if use_max_for_eval:
            # Max fusion (for evaluation)
            diff_probs = torch.sigmoid(embedding_diff_logits)
            scratch_probs = torch.sigmoid(scratch_logits)
            ensemble_probs = torch.max(diff_probs, scratch_probs)
            ensemble_probs = torch.clamp(ensemble_probs, min=1e-7, max=1 - 1e-7)
            ensemble_logits = torch.log(ensemble_probs / (1 - ensemble_probs))
        else:
            # Logit-weighted softmax fusion (differentiable for training)
            stacked = torch.cat([embedding_diff_logits, scratch_logits], dim=-1)
            weights = F.softmax(stacked / self.temperature, dim=-1)
            ensemble_logits = weights[:, 0:1] * embedding_diff_logits + weights[:, 1:2] * scratch_logits

        return ensemble_logits, embedding_diff_logits, scratch_logits

    def get_name(self) -> str:
        """Return model name for logging"""
        return (f"ablation_ensemble_"
                f"embeddingdiff_mlp_"
                f"scratch_{self.resnet_variant}_noise{self.noise_channels}ch_lpd{self.lpd_channels}ch_temp{self.temperature}")


class CrossGeneratorEvaluator(BaseCrossGeneratorEvaluator):
    """
    Cross-generator evaluator for decoder performance.
    Extends AIGCTest's CrossGeneratorEvaluator to work with decoder.
    """

    @torch.no_grad()
    def evaluate_decoder(self, encoder, decoder, test_dataloader: DataLoader, eps: float, classifier=None, training_mode=None, strategy=None, lpd_strategy=None, use_multi_scale_decoder: bool = True) -> Dict[str, Dict[str, float]]:
        """
        Evaluate decoder across different generators with AUC and optimal accuracy using classification head.

        Args:
            encoder: DINOv3 encoder
            decoder: AnyAttack decoder
            test_dataloader: Test data loader (from AIGCTestDataset)
            eps: Epsilon budget
            classifier: Optional classification head for computing AUC/accuracy
            training_mode: Training mode to determine classifier type
                ("with_classification", "noise_embedding_classifier", "ensemble", or "ablation")
            strategy: Multi-scale strategy for ensemble foundation branch
            lpd_strategy: LPD strategy for ensemble scratch branch
            use_multi_scale_decoder: Whether to pass multi-scale images into the decoder

        Returns:
            Dictionary mapping generator names to metrics
        """
        from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, average_precision_score

        encoder.eval()
        decoder.eval()
        if classifier is not None:
            classifier.eval()

        # Create generator mapping
        real_dataset = test_dataloader.dataset
        dataset_for_gens = real_dataset.dataset if isinstance(real_dataset, Subset) else real_dataset
        unique_gens = sorted(list(set(dataset_for_gens.get_generator_list())))
        gen_to_idx = {name: i for i, name in enumerate(unique_gens)}
        idx_to_gen = {i: name for i, name in enumerate(unique_gens)}

        # Collect results per generator
        local_similarities = []
        local_labels = []
        local_gen_indices = []
        local_l2_distances = []
        local_probs = []  # Classifier probabilities (ensemble or main)
        local_foundation_probs = []  # Foundation branch probabilities (for ensemble)
        local_scratch_probs = []  # Scratch branch probabilities (for ensemble)
        local_l2_probs = []  # L2 distance branch probabilities (for 4-branch ensemble)
        local_diff_probs = []  # Embedding diff branch probabilities (for 4-branch ensemble)

        for batch in tqdm(test_dataloader, desc="Cross-generator eval", disable=self.rank != 0):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            generators = batch['generator']

            # Generate noise
            clean_embeddings = encoder.encode_image(images)

            # Handle different decoder types
            # AnyAttackDecoder: noise = decoder(embeddings)
            # UNetDecoder: noise = decoder(original_image, clean_embedding, multi_scale_images)
            decoder_model = decoder.module if hasattr(decoder, 'module') else decoder

            if hasattr(decoder_model, 'fc'):  # AnyAttackDecoder
                noise = decoder(clean_embeddings)
            else:  # UNetDecoder
                # Prepare multi-scale images if strategy is available
                multi_scale_images = None
                if strategy is not None and use_multi_scale_decoder:
                    # Denormalize images for strategy processing (ImageNet normalization)
                    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
                    denorm_images = images * std + mean
                    multi_scale_images = strategy.preprocess(denorm_images)
                    # Normalize from [0, 1] to [-1, 1] to match the original_image range
                    multi_scale_images = multi_scale_images * 2 - 1

                noise = decoder(original_image=images, clean_embedding=clean_embeddings, multi_scale_images=multi_scale_images)

            #noise = torch.clamp(noise, -eps, eps)

            noisy_images = images + noise
            noisy_embeddings = encoder.encode_image(noisy_images)

            # Compute similarity
            similarity = F.cosine_similarity(clean_embeddings, noisy_embeddings, dim=1)
            l2_dist = torch.norm(clean_embeddings - noisy_embeddings, dim=1)

            # Get classifier probabilities if available
            if classifier is not None:
                classifier_model = classifier.module if hasattr(classifier, 'module') else classifier
                has_branch_names = hasattr(classifier_model, 'get_branch_names')

                # Check classifier type
                is_ablation = (training_mode == "ablation")
                is_noise_embedding = (training_mode in ["noise_embedding_classifier", "noise_embedding_joint"])
                is_ensemble = (
                    training_mode == "ensemble"
                    or (has_branch_names and not is_ablation and not is_noise_embedding)
                )

                if is_ensemble:
                    # EnsembleClassifier needs multi_scale_raw_images and lpd_features
                    # Denormalize images for strategy processing (ImageNet normalization)
                    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
                    denorm_images = images * std + mean

                    # Generate multi-scale raw images for foundation branch (WITH decoder noise)
                    # The strategy uses external_noise for the "Original + Noise" image type
                    multi_scale_raw_images = None
                    if strategy is not None:
                        multi_scale_raw_images = strategy.preprocess(denorm_images, external_noise=noise)

                    # Generate LPD features for scratch branch
                    lpd_features = None
                    if lpd_strategy is not None:
                        lpd_features = lpd_strategy.preprocess(denorm_images)

                    # Get classifier if needed (handle DDP wrapper)
                    classifier_model = classifier.module if hasattr(classifier, 'module') else classifier
                    branch_names = classifier_model.get_branch_names() if hasattr(classifier_model, 'get_branch_names') else []
                    has_scratch = "scratch" in branch_names
                    has_l2 = "l2_distance" in branch_names
                    has_diff = "embedding_diff" in branch_names

                    if has_scratch and lpd_strategy is not None:
                        lpd_features = lpd_strategy.preprocess(denorm_images)

                    if multi_scale_raw_images is not None and (not has_scratch or lpd_features is not None):
                        inputs = {"multi_scale_raw_images": multi_scale_raw_images}
                        if has_scratch:
                            inputs["lpd_features"] = lpd_features
                        if has_l2:
                            inputs["l2_distance"] = l2_dist
                        if has_diff:
                            embedding_diff = clean_embeddings - noisy_embeddings
                            inputs["embedding_diff"] = embedding_diff

                        outputs = classifier_model(use_max_for_eval=True, **inputs)
                        if isinstance(outputs, tuple) and len(outputs) == 2 and isinstance(outputs[1], dict):
                            ensemble_logits, branch_logits = outputs
                        elif isinstance(outputs, tuple) and len(outputs) == 3:
                            ensemble_logits, foundation_logits, scratch_logits = outputs
                            branch_logits = {
                                "foundation": foundation_logits,
                                "scratch": scratch_logits,
                            }
                        else:
                            ensemble_logits = outputs
                            branch_logits = {"ensemble": outputs}

                        cls_probs = torch.sigmoid(ensemble_logits.squeeze(-1))
                        local_probs.extend(cls_probs.cpu().numpy())

                        for branch_name, branch_logits in branch_logits.items():
                            branch_probs = torch.sigmoid(branch_logits.squeeze(-1))
                            if branch_name == "foundation":
                                local_foundation_probs.extend(branch_probs.cpu().numpy())
                            elif branch_name == "scratch":
                                local_scratch_probs.extend(branch_probs.cpu().numpy())
                            elif branch_name == "l2_distance":
                                local_l2_probs.extend(branch_probs.cpu().numpy())
                            elif branch_name == "embedding_diff":
                                local_diff_probs.extend(branch_probs.cpu().numpy())
                    else:
                        local_probs.extend((-similarity).cpu().numpy())

                elif is_ablation:
                    # AblationEnsembleClassifier: embedding diff MLP + noise+LPD CNN

                    # Compute embedding difference
                    embedding_diff = clean_embeddings - noisy_embeddings  # [B, feature_dim]

                    # Noise is already 3 channels [B, 3, H, W] - use directly
                    noise_input = noise  # [B, 3, H, W]

                    # Generate LPD features for scratch branch
                    lpd_features = None
                    if lpd_strategy is not None:
                        # Denormalize images for LPD strategy processing (ImageNet normalization)
                        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
                        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
                        denorm_images = images * std + mean
                        lpd_features = lpd_strategy.preprocess(denorm_images)

                    # Get classifier if needed (handle DDP wrapper)
                    classifier_model = classifier.module if hasattr(classifier, 'module') else classifier

                    # Use max fusion for evaluation
                    ensemble_logits, _, _ = classifier_model(
                        embedding_diff, noise_input, lpd_features, use_max_for_eval=True
                    )
                    cls_probs = torch.sigmoid(ensemble_logits.squeeze(-1))
                    local_probs.extend(cls_probs.cpu().numpy())

                elif is_noise_embedding:
                    classifier_model = classifier.module if hasattr(classifier, 'module') else classifier
                    cls_out = classifier_model(noise, clean_embeddings, noisy_embeddings)
                    if isinstance(cls_out, tuple) and len(cls_out) == 2 and isinstance(cls_out[1], dict):
                        ensemble_logits, branch_logits = cls_out
                        cls_probs = torch.sigmoid(ensemble_logits.squeeze(-1))
                        local_probs.extend(cls_probs.cpu().numpy())
                        if "l2_distance" in branch_logits:
                            l2_probs = torch.sigmoid(branch_logits["l2_distance"].squeeze(-1))
                            local_l2_probs.extend(l2_probs.cpu().numpy())
                    else:
                        cls_logits = cls_out.squeeze(-1)
                        cls_probs = torch.sigmoid(cls_logits)
                        local_probs.extend(cls_probs.cpu().numpy())

                else:
                    # CrossAttentionClassifier takes both clean and noisy embeddings
                    cls_logits = classifier(clean_embeddings, noisy_embeddings).squeeze(-1)
                    cls_probs = torch.sigmoid(cls_logits)
                    local_probs.extend(cls_probs.cpu().numpy())
            else:
                # Use negative similarity as probability score
                local_probs.extend((-similarity).cpu().numpy())

            local_similarities.extend(similarity.cpu().numpy())
            local_l2_distances.extend(l2_dist.cpu().numpy())
            local_labels.extend(labels.cpu().numpy())
            local_gen_indices.extend([gen_to_idx[g] for g in generators])

        # Gather results from all GPUs
        if self.world_size > 1 and dist.is_available() and dist.is_initialized():
            similarities_tensor = torch.as_tensor(
                np.asarray(local_similarities, dtype=np.float32),
                device=self.device
            )
            l2_tensor = torch.as_tensor(
                np.asarray(local_l2_distances, dtype=np.float32),
                device=self.device
            )
            labels_tensor = torch.as_tensor(
                np.asarray(local_labels, dtype=np.int64),
                device=self.device
            )
            probs_tensor = torch.as_tensor(
                np.asarray(local_probs, dtype=np.float32),
                device=self.device
            )
            gen_indices_tensor = torch.as_tensor(
                np.asarray(local_gen_indices, dtype=np.int64),
                device=self.device
            )

            # Gather individual branch probabilities if available
            foundation_probs_tensor = torch.as_tensor(
                np.asarray(local_foundation_probs, dtype=np.float32),
                device=self.device
            ) if local_foundation_probs else None
            scratch_probs_tensor = torch.as_tensor(
                np.asarray(local_scratch_probs, dtype=np.float32),
                device=self.device
            ) if local_scratch_probs else None
            l2_probs_tensor = torch.as_tensor(
                np.asarray(local_l2_probs, dtype=np.float32),
                device=self.device
            ) if local_l2_probs else None
            diff_probs_tensor = torch.as_tensor(
                np.asarray(local_diff_probs, dtype=np.float32),
                device=self.device
            ) if local_diff_probs else None

            # Gather all tensors
            all_similarities = self.all_gather_tensor(similarities_tensor, self.world_size, self.rank)
            all_l2 = self.all_gather_tensor(l2_tensor, self.world_size, self.rank)
            all_labels = self.all_gather_tensor(labels_tensor, self.world_size, self.rank)
            all_probs = self.all_gather_tensor(probs_tensor, self.world_size, self.rank)
            all_gen_indices_tensor = self.all_gather_tensor(gen_indices_tensor, self.world_size, self.rank)

            # Gather individual branch probabilities
            all_foundation_probs = self.all_gather_tensor(foundation_probs_tensor, self.world_size, self.rank) if foundation_probs_tensor is not None else None
            all_scratch_probs = self.all_gather_tensor(scratch_probs_tensor, self.world_size, self.rank) if scratch_probs_tensor is not None else None
            all_l2_probs = self.all_gather_tensor(l2_probs_tensor, self.world_size, self.rank) if l2_probs_tensor is not None else None
            all_diff_probs = self.all_gather_tensor(diff_probs_tensor, self.world_size, self.rank) if diff_probs_tensor is not None else None

            # Convert to numpy on CPU
            all_similarities = all_similarities.cpu().numpy()
            all_l2 = all_l2.cpu().numpy()
            all_labels = all_labels.cpu().numpy()
            all_probs = all_probs.cpu().numpy()
            all_gen_indices = all_gen_indices_tensor.cpu().numpy()
            all_foundation_probs = all_foundation_probs.cpu().numpy() if all_foundation_probs is not None else None
            all_scratch_probs = all_scratch_probs.cpu().numpy() if all_scratch_probs is not None else None
            all_l2_probs = all_l2_probs.cpu().numpy() if all_l2_probs is not None else None
            all_diff_probs = all_diff_probs.cpu().numpy() if all_diff_probs is not None else None
            all_generators = [idx_to_gen[idx] for idx in all_gen_indices]
        else:
            all_similarities = np.array(local_similarities)
            all_l2 = np.array(local_l2_distances)
            all_labels = np.array(local_labels)
            all_probs = np.array(local_probs)
            all_foundation_probs = np.array(local_foundation_probs) if local_foundation_probs else None
            all_scratch_probs = np.array(local_scratch_probs) if local_scratch_probs else None
            all_l2_probs = np.array(local_l2_probs) if local_l2_probs else None
            all_diff_probs = np.array(local_diff_probs) if local_diff_probs else None
            all_generators = [idx_to_gen[idx] for idx in local_gen_indices]

        # Compute metrics (only on rank 0)
        generator_metrics = {}
        if self.rank == 0:
            unique_generators = list(set(all_generators))

            for generator in unique_generators:
                indices = [i for i, g in enumerate(all_generators) if g == generator]
                gen_sims = all_similarities[indices]
                gen_labels = all_labels[indices]
                gen_l2 = all_l2[indices]
                gen_probs = all_probs[indices]

                # Get individual branch probabilities if available
                gen_foundation_probs = all_foundation_probs[indices] if all_foundation_probs is not None else None
                gen_scratch_probs = all_scratch_probs[indices] if all_scratch_probs is not None else None
                gen_l2_probs = all_l2_probs[indices] if all_l2_probs is not None else None
                gen_diff_probs = all_diff_probs[indices] if all_diff_probs is not None else None

                # Separate by real/fake
                real_mask = gen_labels == 0
                fake_mask = gen_labels == 1

                # Calculate AUC, average precision, and optimal accuracy using classifier probabilities
                try:
                    auc = roc_auc_score(gen_labels, gen_probs)
                    ap = average_precision_score(gen_labels, gen_probs)

                    # Calculate optimal accuracy by finding best threshold
                    fpr, tpr, thresholds = roc_curve(gen_labels, gen_probs)
                    accuracies = []
                    for threshold in thresholds:
                        predictions = [(prob >= threshold) for prob in gen_probs]
                        acc = accuracy_score(gen_labels, predictions)
                        accuracies.append(acc)

                    if accuracies:
                        optimal_acc = max(accuracies)
                        optimal_threshold = thresholds[accuracies.index(optimal_acc)]
                    else:
                        optimal_acc = 0.0
                        optimal_threshold = 0.0
                except:
                    auc = 0.0
                    ap = 0.0
                    optimal_acc = 0.0
                    optimal_threshold = 0.0

                metrics = {
                    'mean_similarity': float(np.mean(gen_sims)),
                    'std_similarity': float(np.std(gen_sims)),
                    'mean_l2_distance': float(np.mean(gen_l2)),
                    'auc': float(auc),
                    'average_precision': float(ap),
                    'optimal_accuracy': float(optimal_acc),
                    'optimal_threshold': float(optimal_threshold),
                    'num_samples': len(indices)
                }

                if real_mask.any():
                    metrics['similarity_real'] = float(np.mean(gen_sims[real_mask]))
                    metrics['l2_distance_real'] = float(np.mean(gen_l2[real_mask]))
                    metrics['num_real'] = int(real_mask.sum())

                if fake_mask.any():
                    metrics['similarity_fake'] = float(np.mean(gen_sims[fake_mask]))
                    metrics['l2_distance_fake'] = float(np.mean(gen_l2[fake_mask]))
                    metrics['num_fake'] = int(fake_mask.sum())

                generator_metrics[generator] = metrics

        # Synchronize all ranks after metric computation
        if self.world_size > 1 and dist.is_available() and dist.is_initialized():
            dist.barrier()

        return generator_metrics

    def all_gather_tensor(self, tensor: torch.Tensor, world_size: int, rank: int) -> torch.Tensor:
        """Gather tensors from all GPUs in DDP training

        Uses all_gather_object to handle variable-sized tensors (when dataset
        size is not evenly divisible by world_size).
        """
        # Move to CPU and convert to numpy for pickling
        tensor_np = tensor.cpu().numpy()
        gathered_list = [None] * world_size
        dist.all_gather_object(gathered_list, tensor_np)
        # Concatenate all gathered tensors
        return torch.from_numpy(np.concatenate(gathered_list, axis=0)).to(self.device)


class EmbeddingTrainer:
    """
    Trainer for AnyAttack decoder with optional classification loss.
    Supports DDP for multi-GPU training.

    Training modes:
    1. "embedding_only": Only minimize embedding similarity (pure AnyAttack)
    2. "with_classification": Add BCE classification loss

    Args:
        model_name: DINOv3 model name (e.g., 'dinov3_vitl16')
        eps: Maximum perturbation budget (default: 16/255)
        lr: Learning rate (default: 0.0005)
        weight_decay: Weight decay (default: 0.01)
        device: torch device
        checkpoint_dir: Directory to save checkpoints
        rank: Process rank for DDP (default: 0)
        world_size: Total number of processes (default: 1)
        decoder_type: Type of decoder ('unet' or 'simple', default: 'unet')
        decoder_kwargs: Additional kwargs for decoder initialization
        strategy: RFNT strategy for multi-scale inputs (optional)
        lpd_strategy: LPD strategy for ensemble scratch branch
        loss_type: "similarity" or "discrepancy"
        margin: Margin for discrepancy loss
        fusion_method: Ensemble fusion method for EnsembleClassifier
        eps_randomization: Enable epsilon randomization for domain generalization
        eps_min: Minimum epsilon value for randomization
        eps_max: Maximum epsilon value for randomization
        eps_schedule: Schedule for epsilon randomization ('random', 'cycle', 'inverse_cycle', 'linear_increase')
        normalize_loss: Normalize loss by real similarity/L2 for scale-invariance
    """

    def __init__(
        self,
        model_name: str = "dinov3_vitl16",
        eps: float = 16/255,
        lr: float = 0.0005,
        weight_decay: float = 0.01,
        device: Optional[torch.device] = None,
        checkpoint_dir: str = "./checkpoints",
        rank: int = 0,
        world_size: int = 1,
        decoder_type: str = "unet",
        decoder_kwargs: Optional[dict] = None,
        strategy=None,
        use_multi_scale_decoder: Optional[bool] = None,
        lpd_strategy=None,  # LPD strategy for ensemble scratch branch
        loss_type: str = "similarity",  # "similarity" or "discrepancy"
        margin: float = 0.1,  # Margin for discrepancy loss
        fusion_method: str = "logit_weighted",  # Ensemble fusion method
        eps_randomization: bool = False,  # Enable epsilon randomization
        eps_min: float = 4/255,  # Minimum epsilon for randomization
        eps_max: float = 64/255,  # Maximum epsilon for randomization
        eps_schedule: str = "random",  # Schedule type
        normalize_loss: bool = False,  # Normalize loss for domain generalization
        use_four_branch_ensemble: bool = False,  # Use 4-branch ensemble (foundation, scratch, l2, embedding_diff)
        noise_embedding_use_l2: bool = False,  # Add L2-distance branch to noise_embedding_classifier
        embedding_loss_weight: float = 1.0,  # Weight for embedding loss
    ):
        self.model_name = model_name
        self.eps = eps
        self.lr = lr
        self.weight_decay = weight_decay
        self.checkpoint_dir = checkpoint_dir
        self.rank = rank
        self.world_size = world_size
        self.is_ddp = world_size > 1
        self.decoder_type = decoder_type
        self.decoder_kwargs = decoder_kwargs or {}
        self.strategy = strategy
        self.use_multi_scale_decoder = use_multi_scale_decoder if use_multi_scale_decoder is not None else bool(strategy)
        self.lpd_strategy = lpd_strategy
        self.loss_type = loss_type
        self.margin = margin
        self.fusion_method = fusion_method
        self.embedding_loss_weight = embedding_loss_weight

        # Epsilon randomization for domain generalization
        self.eps_randomization = eps_randomization
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.eps_schedule = eps_schedule
        self.current_eps = eps  # Will be updated per batch if randomization enabled
        self.base_eps = eps  # Store base epsilon for validation (always use fixed eps during eval)

        # Loss normalization for domain generalization
        self.normalize_loss = normalize_loss

        # Four-branch ensemble option
        self.use_four_branch_ensemble = use_four_branch_ensemble

        # Noise+embedding classifier L2 branch option
        self.noise_embedding_use_l2 = noise_embedding_use_l2

        # Ensure decoder multiscale inputs are disabled when requested
        if self.decoder_type == "unet" and not self.use_multi_scale_decoder:
            if self.decoder_kwargs.get("strategy_channels", 0) != 0:
                if self.rank == 0:
                    print("Disabling decoder multi-scale inputs: setting strategy_channels=0")
                self.decoder_kwargs["strategy_channels"] = 0

        # Setup device
        if device is None:
            if torch.cuda.is_available():
                device = torch.device(f'cuda:{rank}')
            else:
                device = torch.device('cpu')
        self.device = device

        # Create directories (only on rank 0)
        if self.rank == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)

        # Setup logging
        self._setup_logging()

        # Load DINOv3 encoder
        if self.rank == 0:
            print(f"Loading DINOv3 encoder: {model_name}")
        model_config = {
            'model_name': model_name,
            'device': self.device
        }
        self.encoder, self.tokenizer, self.preprocess = load_models(model_config)
        self.intermediate_encoder = None
        if hasattr(self.encoder, "get_intermediate_layers"):
            self.intermediate_encoder = self.encoder
        elif hasattr(self.encoder, "dinov2") and hasattr(self.encoder.dinov2, "get_intermediate_layers"):
            self.intermediate_encoder = self.encoder.dinov2
        if hasattr(self.encoder, "output_dim"):
            self.feature_dim = self.encoder.output_dim
        elif hasattr(self.encoder, "visual") and hasattr(self.encoder.visual, "output_dim"):
            self.feature_dim = self.encoder.visual.output_dim
        else:
            raise AttributeError("Encoder has no output_dim; expected model.output_dim or model.visual.output_dim")
        if self.rank == 0:
            print(f"Feature dimension: {self.feature_dim}")

        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

        # Initialize decoder using factory
        if self.rank == 0:
            print(f"Initializing {decoder_type} decoder with eps={eps}")
            if decoder_kwargs:
                print(f"  Decoder kwargs: {decoder_kwargs}")

        self.decoder = create_decoder(
            model_name=model_name,
            eps=eps,
            decoder_type=decoder_type,
            **decoder_kwargs
        ).to(self.device)

        # Wrap decoder with DDP if needed
        if self.is_ddp:
            self.decoder = DDP(self.decoder, device_ids=[self.rank], find_unused_parameters=False)
            if self.rank == 0:
                print(f"Wrapped decoder with DDP (world_size={world_size})")

        self.classifier = None
        self.decoder_optimizer = None
        self.classifier_optimizer = None
        self.optimizer = None  # For backward compatibility, will point to decoder_optimizer
        self.bce_loss_fn = nn.BCEWithLogitsLoss()

        # Cross-generator evaluator
        self.evaluator = CrossGeneratorEvaluator(
            model=None,  # Not used for decoder evaluation
            device=self.device,
            output_dir=self.checkpoint_dir,  # Use checkpoint_dir for output
            rank=rank,
            world_size=world_size
        )

        # Training state
        self.current_epoch = 0
        self.global_step = 0

    def _setup_logging(self):
        """
        Setup stdout redirection to log file.
        All print statements will be saved to the log file.
        Log file name format: train_{config_name}_eps{eps}_{timestamp}.log
        """
        # Create logs directory within checkpoint_dir for organized structure
        logs_dir = os.path.join(self.checkpoint_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)

        # Create log filename with config_name, eps value, and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Format epsilon value (e.g., "16_255" for 16/255)
        eps_str = str(self.eps).replace("/", "_")

        # Extract config name from checkpoint_dir for better organization
        config_name = os.path.basename(self.checkpoint_dir)

        log_filename = f"train_{config_name}_eps{eps_str}_{timestamp}.log"
        log_filepath = os.path.join(logs_dir, log_filename)

        # Only redirect stdout for rank 0
        if self.rank == 0:
            # Open log file
            self.log_file = open(log_filepath, 'a', buffering=1)  # Line buffering
            self.original_stdout = sys.stdout

            # Create custom stdout that writes to both console and file
            class TeeOutput:
                def __init__(self, original_stdout, log_file):
                    self.original_stdout = original_stdout
                    self.log_file = log_file

                def write(self, data):
                    self.original_stdout.write(data)
                    self.log_file.write(data)

                def flush(self):
                    self.original_stdout.flush()
                    self.log_file.flush()

            sys.stdout = TeeOutput(self.original_stdout, self.log_file)

            # Print header and all arguments
            print("=" * 80)
            print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Log file: {log_filepath}")
            print("=" * 80)
            print("\n=== Initialization Arguments ===")
            print(f"model_name: {self.model_name}")
            print(f"eps: {self.eps}")
            print(f"lr: {self.lr}")
            print(f"weight_decay: {self.weight_decay}")
            print(f"device: {self.device}")
            print(f"checkpoint_dir: {self.checkpoint_dir}")
            print(f"rank: {self.rank}")
            print(f"world_size: {self.world_size}")
            print(f"decoder_type: {self.decoder_type}")
            print(f"decoder_kwargs: {self.decoder_kwargs}")
            print(f"strategy: {self.strategy}")
            print(f"lpd_strategy: {self.lpd_strategy}")
            print(f"loss_type: {self.loss_type}")
            print(f"margin: {self.margin}")
            if self.eps_randomization:
                print(f"eps_randomization: ENABLED (schedule: {self.eps_schedule})")
                print(f"  eps_min: {self.eps_min:.6f} (~{self.eps_min*255:.1f}/255)")
                print(f"  eps_max: {self.eps_max:.6f} (~{self.eps_max*255:.1f}/255)")
            if self.normalize_loss:
                print(f"normalize_loss: ENABLED (loss scaled by real similarity for domain generalization)")
            print("=" * 80)
        else:
            self.log_file = None
            self.original_stdout = None

    def close_log(self):
        """Close log file and restore original stdout."""
        if self.rank == 0 and hasattr(self, 'log_file') and self.log_file is not None:
            print("=" * 80)
            print(f"Training ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 80)

            sys.stdout = self.original_stdout
            self.log_file.close()
            self.log_file = None

    def __del__(self):
        """Ensure log file is closed when trainer is destroyed."""
        self.close_log()

    def save_results(self, results_dict: dict):
        """
        Save training results to a JSON file.

        Args:
            results_dict: Dictionary containing training metrics and config
        """
        import json

        if self.rank != 0:
            return

        # Ensure checkpoint directory exists
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        results_file = os.path.join(self.checkpoint_dir, "results.json")

        # Add metadata
        results_dict.update({
            'config_name': os.path.basename(self.checkpoint_dir),
            'eps': self.eps,
            'model_name': self.model_name,
            'timestamp': datetime.now().isoformat()
        })

        try:
            with open(results_file, 'w') as f:
                json.dump(results_dict, f, indent=2)
            print(f"Saved results to: {results_file}")
        except Exception as e:
            print(f"Error saving results: {e}")

    def get_current_eps(self, epoch: int, batch_idx: int, total_batches: int) -> float:
        """
        Get current epsilon value based on randomization schedule.

        This enables domain generalization by varying epsilon during training,
        simulating different "domains" (generators) with different L2 characteristics.

        Args:
            epoch: Current epoch number
            batch_idx: Current batch index within epoch
            total_batches: Total number of batches in epoch

        Returns:
            Current epsilon value to use for this batch
        """
        if not self.eps_randomization:
            return self.eps

        import random

        if self.eps_schedule == "random":
            # Random epsilon per batch (most diverse)
            return random.uniform(self.eps_min, self.eps_max)

        elif self.eps_schedule == "cycle":
            # Cycle through epsilon range (sinusoidal)
            # Period = 10 epochs
            cycle_pos = (epoch % 10) / 10.0 * 2 * 3.14159
            scale = 0.5 * (1 + math.sin(cycle_pos))  # 0 to 1
            return self.eps_min + scale * (self.eps_max - self.eps_min)

        elif self.eps_schedule == "inverse_cycle":
            # Inverse cycle (start high, go low, then high again)
            cycle_pos = (epoch % 10) / 10.0 * 2 * 3.14159
            scale = 0.5 * (1 - math.sin(cycle_pos))  # 1 to 0 to 1
            return self.eps_min + scale * (self.eps_max - self.eps_min)

        elif self.eps_schedule == "linear_increase":
            # Linearly increase from min to max over epochs
            # Assume 30 epochs for full cycle
            max_epochs = 30
            progress = min(epoch / max_epochs, 1.0)
            return self.eps_min + progress * (self.eps_max - self.eps_min)

        else:
            return self.eps

    def get_decoder_eps(self) -> float:
        """
        Get the current epsilon value from the decoder.

        Returns:
            Current epsilon value being used by the decoder
        """
        decoder_model = self.decoder.module if hasattr(self.decoder, 'module') else self.decoder

        if hasattr(decoder_model, 'eps'):
            # AnyAttackDecoder has direct eps attribute
            return decoder_model.eps
        elif hasattr(decoder_model, 'output_head') and hasattr(decoder_model.output_head, 'eps'):
            # UNetDecoder has eps in output_head
            return decoder_model.output_head.eps
        else:
            return self.eps

    def update_decoder_eps(self, eps_value: float):
        """
        Update the decoder's epsilon value dynamically.

        This allows epsilon randomization during training for domain generalization.

        Args:
            eps_value: New epsilon value to use
        """
        self.current_eps = eps_value

        # Update decoder's epsilon based on decoder type
        decoder_model = self.decoder.module if hasattr(self.decoder, 'module') else self.decoder

        if hasattr(decoder_model, 'eps'):
            # AnyAttackDecoder has direct eps attribute
            decoder_model.eps = eps_value
        elif hasattr(decoder_model, 'output_head') and hasattr(decoder_model.output_head, 'eps'):
            # UNetDecoder has eps in output_head
            decoder_model.output_head.eps = eps_value

    def setup_training_mode(
        self,
        mode: str = "embedding_only",
        lambda_classification: float = 0.1
    ):
        """
        Setup training mode and optimizers.

        Modes:
        - "embedding_only": Only minimize embedding similarity (pure AnyAttack)
        - "with_classification": Add BCE classification loss with CrossAttentionClassifier
        - "noise_embedding_classifier": Classifier using noise + clean/noisy embeddings
        - "noise_embedding_joint": Same as noise_embedding_classifier, but joint optimizer
        - "ensemble": Use EnsembleClassifier with foundation + scratch branches
        """
        self.training_mode = mode
        self.lambda_classification = lambda_classification

        # Get decoder parameters
        if self.is_ddp:
            decoder_params = self.decoder.module.parameters()
        else:
            decoder_params = self.decoder.parameters()

        if mode == "with_classification":
            if self.classifier is None:
                # Use CrossAttentionClassifier that takes both clean and noisy embeddings
                classifier = CrossAttentionClassifier(
                    feature_dim=self.feature_dim,
                    num_heads=8,
                    num_layers=2,
                    dropout=0.1
                ).to(self.device)

                if self.is_ddp:
                    self.classifier = DDP(classifier, device_ids=[self.rank], find_unused_parameters=False)
                else:
                    self.classifier = classifier

            # Get classifier parameters
            if self.is_ddp:
                classifier_params = self.classifier.module.parameters()
            else:
                classifier_params = self.classifier.parameters()

            # Separate optimizers for decoder and classifier
            self.decoder_optimizer = torch.optim.AdamW(
                decoder_params,
                lr=self.lr,
                weight_decay=self.weight_decay
            )
            self.classifier_optimizer = torch.optim.AdamW(
                classifier_params,
                lr=self.lr,  # Can use different LR if needed
                weight_decay=self.weight_decay
            )

            if self.rank == 0:
                print(f"Training mode: {mode} (embedding + {lambda_classification} * classification)")
                print("  Classifier: CrossAttentionClassifier")
                print(f"  Decoder optimizer: AdamW(lr={self.lr}, wd={self.weight_decay})")
                print(f"  Classifier optimizer: AdamW(lr={self.lr}, wd={self.weight_decay})")

        elif mode in ["noise_embedding_classifier", "noise_embedding_joint"]:
            if self.classifier is None:
                if self.noise_embedding_use_l2:
                    classifier = NoiseEmbeddingClassifierWithL2(
                        feature_dim=self.feature_dim,
                        noise_channels=3,
                        noise_feature_dim=128,
                        dropout=0.1,
                    ).to(self.device)
                else:
                    classifier = NoiseEmbeddingClassifier(
                        feature_dim=self.feature_dim,
                        noise_channels=3,
                        noise_feature_dim=128,
                        dropout=0.1,
                    ).to(self.device)

                if self.is_ddp:
                    self.classifier = DDP(classifier, device_ids=[self.rank], find_unused_parameters=False)
                else:
                    self.classifier = classifier

            if self.is_ddp:
                classifier_params = self.classifier.module.parameters()
            else:
                classifier_params = self.classifier.parameters()

            if mode == "noise_embedding_joint":
                combined_params = list(decoder_params) + list(classifier_params)
                self.decoder_optimizer = torch.optim.AdamW(
                    combined_params,
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                )
                self.classifier_optimizer = None
            else:
                self.decoder_optimizer = torch.optim.AdamW(
                    decoder_params,
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                )
                self.classifier_optimizer = torch.optim.AdamW(
                    classifier_params,
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                )

            if self.rank == 0:
                print(f"Training mode: {mode} (embedding + {lambda_classification} * noise+embedding classification)")
                print(f"  Noise encoder: Conv2d -> Conv2d -> pool -> MLP")
                if self.noise_embedding_use_l2:
                    print(f"  Extra branch: L2 distance MLP (1 -> 128 -> 64 -> 1)")
                print(f"  Classifier input: clean + noisy + diff embeddings + noise features")
                if mode == "noise_embedding_joint":
                    print(f"  Optimizer: AdamW(decoder + classifier, lr={self.lr}, wd={self.weight_decay})")
                else:
                    print(f"  Decoder optimizer: AdamW(lr={self.lr}, wd={self.weight_decay})")
                    print(f"  Classifier optimizer: AdamW(lr={self.lr}, wd={self.weight_decay})")

        elif mode == "ensemble":
            if self.classifier is None:
                # Use EnsembleClassifier with foundation + scratch branches
                # Prefer explicit model_name (supports CLIP/DINOv3) over feature-dim mapping
                encoder_name = getattr(self, "ensemble_encoder_name", None) or self.model_name

                # Get strategy info
                num_scales = 5  # Default for MultiScaleRaw
                images_per_scale = 5
                if self.strategy is not None:
                    num_scales = len(getattr(self.strategy, 'levels', [0])) if hasattr(self.strategy, 'levels') else 5
                    if hasattr(self.strategy, 'get_images_per_scale'):
                        images_per_scale = self.strategy.get_images_per_scale()

                # Get LPD channels (only used if scratch branch is enabled)
                lpd_channels = 3  # Default for LPD
                if self.lpd_strategy is not None:
                    lpd_channels = self.lpd_strategy.get_output_channels() if hasattr(self.lpd_strategy, 'get_output_channels') else 3

                # Choose 2-branch or 4-branch ensemble
                if self.use_four_branch_ensemble:
                    if self.lpd_strategy is None:
                        classifier = ThreeBranchEnsembleNoLPD(
                            encoder_name=encoder_name,
                            num_scales=num_scales,
                            images_per_scale=images_per_scale,
                            feature_fusion="attention",
                            embedding_dim=self.feature_dim,
                            dropout=0.1,
                            temperature=1.0,
                            fusion_method=self.fusion_method
                        ).to(self.device)
                    else:
                        classifier = FourBranchEnsemble(
                            encoder_name=encoder_name,
                            num_scales=num_scales,
                            images_per_scale=images_per_scale,
                            lpd_channels=lpd_channels,
                            resnet_variant="resnet34",
                            feature_fusion="attention",
                            embedding_dim=self.feature_dim,
                            dropout=0.1,
                            temperature=1.0,
                            fusion_method=self.fusion_method
                        ).to(self.device)
                else:
                    classifier = EnsembleClassifier(
                        encoder_name=encoder_name,
                        num_scales=num_scales,
                        images_per_scale=images_per_scale,
                        lpd_channels=lpd_channels,
                        resnet_variant="resnet34",
                        feature_fusion="attention",
                        dropout=0.1,
                        temperature=1.0,
                        fusion_method=self.fusion_method
                    ).to(self.device)

                if self.is_ddp:
                    self.classifier = DDP(classifier, device_ids=[self.rank], find_unused_parameters=False)
                else:
                    self.classifier = classifier

            # Get classifier parameters
            if self.is_ddp:
                classifier_params = self.classifier.module.parameters()
            else:
                classifier_params = self.classifier.parameters()

            # Separate optimizers for decoder and classifier
            self.decoder_optimizer = torch.optim.AdamW(
                decoder_params,
                lr=self.lr,
                weight_decay=self.weight_decay
            )
            self.classifier_optimizer = torch.optim.AdamW(
                classifier_params,
                lr=self.lr,  # Can use different LR if needed
                weight_decay=self.weight_decay
            )

            if self.rank == 0:
                print(f"Training mode: {mode} (embedding + {lambda_classification} * ensemble classification)")
                print(f"  Fusion method: {self.fusion_method}")
                if self.use_four_branch_ensemble:
                    if self.lpd_strategy is None:
                        print(f"  Ensemble: 3-branch (foundation + l2_distance + embedding_diff)")
                    else:
                        print(f"  Ensemble: 4-branch (foundation + scratch + l2_distance + embedding_diff)")
                else:
                    print(f"  Ensemble: 2-branch (foundation + scratch)")
                print(f"  Foundation: SplitChannelFoundationModel ({encoder_name})")
                if self.lpd_strategy is not None:
                    print(f"  Scratch: MultiScaleDifferenceCNN (resnet34, lpd_channels={lpd_channels})")
                if self.use_four_branch_ensemble:
                    print(f"  L2 Distance: MLP (1 -> 128 -> 64 -> 1)")
                    print(f"  Embedding Diff: MLP ({self.feature_dim} -> 512 -> 256 -> 1)")
                print(f"  Decoder optimizer: AdamW(lr={self.lr}, wd={self.weight_decay})")
                print(f"  Classifier optimizer: AdamW(lr={self.lr}, wd={self.weight_decay})")

        elif mode == "ablation":
            # Ablation mode uses AblationEnsembleClassifier with embedding difference + noise+LPD branches
            if self.classifier is None:
                # Get LPD channels
                lpd_channels = 3  # Default for LPD
                if self.lpd_strategy is not None:
                    lpd_channels = self.lpd_strategy.get_output_channels() if hasattr(self.lpd_strategy, 'get_output_channels') else 3

                classifier = AblationEnsembleClassifier(
                    feature_dim=self.feature_dim,
                    noise_channels=3,
                    lpd_channels=lpd_channels,
                    resnet_variant="resnet18",
                    dropout=0.1,
                    temperature=1.0
                ).to(self.device)

                if self.is_ddp:
                    self.classifier = DDP(classifier, device_ids=[self.rank], find_unused_parameters=False)
                else:
                    self.classifier = classifier

            # Get classifier parameters
            if self.is_ddp:
                classifier_params = self.classifier.module.parameters()
            else:
                classifier_params = self.classifier.parameters()

            # Separate optimizers for decoder and classifier
            self.decoder_optimizer = torch.optim.AdamW(
                decoder_params,
                lr=self.lr,
                weight_decay=self.weight_decay
            )
            self.classifier_optimizer = torch.optim.AdamW(
                classifier_params,
                lr=self.lr,
                weight_decay=self.weight_decay
            )

            if self.rank == 0:
                print(f"Training mode: {mode} (embedding + {lambda_classification} * ablation classification)")
                print(f"  Using AblationEnsembleClassifier architecture")
                print(f"  Branch 1: Embedding Difference MLP (clean - noisy)")
                print(f"  Branch 2: MultiScaleDifferenceCNN (resnet18, noise+LPD input)")
                print(f"  Feature dim: {self.feature_dim}")
                print(f"  LPD channels: {lpd_channels}")
                print(f"  Decoder optimizer: AdamW(lr={self.lr}, wd={self.weight_decay})")
                print(f"  Classifier optimizer: AdamW(lr={self.lr}, wd={self.weight_decay})")

        else:  # "embedding_only"
            # Only decoder optimizer needed
            self.decoder_optimizer = torch.optim.AdamW(
                decoder_params,
                lr=self.lr,
                weight_decay=self.weight_decay
            )
            self.classifier_optimizer = None

            if self.rank == 0:
                print(f"Training mode: {mode} (embedding discrepancy only)")
                print(f"  Decoder optimizer: AdamW(lr={self.lr}, wd={self.weight_decay})")

        # For backward compatibility
        self.optimizer = self.decoder_optimizer

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch"""
        self.decoder.train()
        if self.classifier is not None:
            self.classifier.train()
        self.encoder.eval()

        # Set epoch for distributed sampler
        if self.is_ddp and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        total_loss = 0.0
        total_emb_loss = 0.0
        total_cls_loss = 0.0
        total_similarity = 0.0
        total_similarity_real = 0.0
        total_similarity_fake = 0.0
        total_correct = 0
        total_samples = 0
        num_batches = 0
        num_real = 0
        num_fake = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=self.rank != 0)

        # Get total number of batches for epsilon scheduling
        total_batches = len(train_loader)

        for batch_idx, batch in enumerate(pbar):
            # Update epsilon for this batch (for domain generalization)
            current_eps = self.get_current_eps(epoch, batch_idx, total_batches)
            self.update_decoder_eps(current_eps)

            # Handle both dict and tuple formats
            if isinstance(batch, dict):
                images = batch['image']
                labels = batch.get('label', None)
                if labels is not None:
                    labels = labels.to(self.device).float()
            else:
                images = batch[0]
                labels = batch[1].to(self.device).float() if len(batch) > 1 else None

            images = images.to(self.device)

            # Extract features from clean images
            with torch.no_grad():
                clean_embeddings = self.encoder.encode_image(images)

            # Denormalize images for strategy processing (ImageNet normalization)
            mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
            denorm_images = images * std + mean

            # Generate initial multi-scale images using strategy (without decoder noise for now)
            # This is used as input to the decoder
            multi_scale_images = None
            if self.strategy is not None and self.decoder_type == "unet" and self.use_multi_scale_decoder:
                # Apply strategy WITHOUT external noise to get features for decoder input
                multi_scale_images = self.strategy.preprocess(denorm_images, external_noise=None)
                # Normalize from [0, 1] to [-1, 1] to match the original_image range
                multi_scale_images = multi_scale_images * 2 - 1

            # Generate noise (different approach for UNet vs Simple decoder)
            if self.decoder_type == "unet":
                # UNet requires multi-modal inputs
                noise = self.decoder(
                    original_image=images,
                    clean_embedding=clean_embeddings,
                    multi_scale_images=multi_scale_images
                )
            else:
                # Simple decoder only needs embeddings
                noise = self.decoder(clean_embeddings)

            # Re-generate multi-scale images for ensemble classifier
            # - Ensemble mode: WITH decoder noise (uses decoder-generated noise for "Original + Noise" type)
            multi_scale_raw_images = None
            if self.strategy is not None and (self.training_mode == "ensemble" or self.decoder_type == "unet"):
                multi_scale_raw_images = self.strategy.preprocess(denorm_images, external_noise=noise)

            # Create noisy images
            noisy_images = images + noise

            # Extract features from noisy images
            # IMPORTANT: Don't use no_grad() here! We need gradients to flow through encoder
            # to reach the decoder. Encoder parameters are frozen (requires_grad=False),
            # so they won't be updated, but gradients will still flow through.
            noisy_embeddings = self.encoder.encode_image(noisy_images)

            # Compute cosine similarity between clean and noisy embeddings
            embedding_similarity = F.cosine_similarity(clean_embeddings, noisy_embeddings, dim=1)

            # === Compute embedding loss based on loss_type ===
            if self.loss_type == "similarity":
                # Simply minimize cosine similarity
                embedding_loss = embedding_similarity.mean()
            elif self.loss_type == "discrepancy":
                # Maximize discrepancy between real and fake with margin
                # Goal: (discrepancy_fake - discrepancy_real) + margin
                # Where discrepancy = 1 - similarity (higher = more different)
                # This becomes: (1 - fake_sim) - (1 - real_sim) + margin
                #               = real_sim - fake_sim + margin
                if labels is not None:
                    mask_real = (labels == 0)
                    mask_fake = (labels == 1)

                    # Compute real and fake discrepancies
                    real_similarity = embedding_similarity[mask_real].mean() if mask_real.any() else torch.tensor(0.0, device=self.device)
                    fake_similarity = embedding_similarity[mask_fake].mean() if mask_fake.any() else torch.tensor(0.0, device=self.device)

                    # Discrepancy loss: maximize (fake_discrepancy - real_discrepancy)
                    # = maximize ((1-fake_sim) - (1-real_sim))
                    # = maximize (real_sim - fake_sim)
                    # = minimize (fake_sim - real_sim)
                    # With margin: loss = (fake_sim - real_sim) + margin (clipped at 0)

                    if self.normalize_loss:
                        # Normalize loss by real similarity for scale-invariance
                        # This makes the loss focus on relative difference rather than absolute L2
                        # When real_similarity is high (near 1), we expect smaller absolute gap
                        # Normalized margin scales with the baseline (1 - real_similarity)
                        normalized_margin = self.margin * (1 - real_similarity + 1e-6)
                        embedding_loss = F.relu((fake_similarity - real_similarity) + normalized_margin) / (1 - real_similarity + 1e-6)
                    else:
                        embedding_loss = F.relu((fake_similarity - real_similarity) + self.margin)
                else:
                    # Fallback to simple similarity if no labels
                    embedding_loss = embedding_similarity.mean()
            else:
                raise ValueError(f"Unknown loss_type: {self.loss_type}. Must be 'similarity' or 'discrepancy'")

            # Track metrics
            total_similarity = embedding_similarity.mean().item()
            num_real = 0
            num_fake = 0
            if labels is not None:
                num_real = (labels == 0).sum().item()
                num_fake = (labels == 1).sum().item()

                # Also track real/fake similarity separately for monitoring
                mask_real = (labels == 0)
                mask_fake = (labels == 1)

                if mask_real.any():
                    real_sim = embedding_similarity[mask_real].mean().item()
                    # Store for debugging
                    self._last_real_similarity = real_sim

                if mask_fake.any():
                    fake_sim = embedding_similarity[mask_fake].mean().item()
                    # Store for debugging
                    self._last_fake_similarity = fake_sim
            else:
                # No labels available
                self._last_real_similarity = 0.0
                self._last_fake_similarity = 0.0

            # Optional classification loss
            classification_loss = torch.tensor(0.0, device=self.device)
            batch_accuracy = 0.0
            if labels is not None and self.training_mode in ["with_classification", "noise_embedding_classifier", "noise_embedding_joint", "ensemble", "ablation"]:
                if self.training_mode == "with_classification":
                    # CrossAttentionClassifier takes both clean and noisy embeddings
                    cls_logits = self.classifier(clean_embeddings, noisy_embeddings).squeeze(-1)
                    classification_loss = self.bce_loss_fn(cls_logits, labels)

                    # Compute accuracy
                    cls_probs = torch.sigmoid(cls_logits)
                    cls_preds = (cls_probs >= 0.5).float()
                    # Ensure shapes match for comparison
                    labels_flat = labels.squeeze(-1) if labels.dim() > 1 else labels
                    batch_correct = (cls_preds == labels_flat).sum().item()
                    batch_accuracy = batch_correct / labels.size(0)
                    total_correct += batch_correct
                    total_samples += labels.size(0)

                elif self.training_mode in ["noise_embedding_classifier", "noise_embedding_joint"]:
                    cls_out = self.classifier(noise, clean_embeddings, noisy_embeddings)
                    if isinstance(cls_out, tuple) and len(cls_out) == 2 and isinstance(cls_out[1], dict):
                        ensemble_logits, branch_logits = cls_out
                        labels_squeezed = labels.squeeze(-1) if labels.dim() > 1 else labels
                        classification_loss = torch.tensor(0.0, device=self.device)
                        for logits in branch_logits.values():
                            classification_loss = classification_loss + self.bce_loss_fn(logits.squeeze(-1), labels_squeezed)
                        cls_logits = ensemble_logits.squeeze(-1)
                    else:
                        cls_logits = cls_out.squeeze(-1)
                        classification_loss = self.bce_loss_fn(cls_logits, labels)

                    cls_probs = torch.sigmoid(cls_logits)
                    cls_preds = (cls_probs >= 0.5).float()
                    labels_flat = labels.squeeze(-1) if labels.dim() > 1 else labels
                    batch_correct = (cls_preds == labels_flat).sum().item()
                    batch_accuracy = batch_correct / labels.size(0)
                    total_correct += batch_correct
                    total_samples += labels.size(0)

                elif self.training_mode == "ensemble":
                    # Flexible ensemble: use required inputs per-branch
                    classifier_model = self.classifier.module if hasattr(self.classifier, 'module') else self.classifier
                    branch_names = classifier_model.get_branch_names() if hasattr(classifier_model, 'get_branch_names') else []
                    has_scratch = "scratch" in branch_names
                    has_l2 = "l2_distance" in branch_names
                    has_diff = "embedding_diff" in branch_names

                    lpd_features = None
                    if has_scratch and self.lpd_strategy is not None:
                        lpd_features = self.lpd_strategy.preprocess(denorm_images)

                    if multi_scale_raw_images is not None and (not has_scratch or lpd_features is not None):
                        inputs = {"multi_scale_raw_images": multi_scale_raw_images}
                        if has_scratch:
                            inputs["lpd_features"] = lpd_features
                        if has_l2:
                            l2_dist = torch.norm(clean_embeddings - noisy_embeddings, p=2, dim=1, keepdim=True)
                            inputs["l2_distance"] = l2_dist
                        if has_diff:
                            embedding_diff = clean_embeddings - noisy_embeddings
                            inputs["embedding_diff"] = embedding_diff

                        outputs = self.classifier(use_max_for_eval=False, **inputs)
                        if isinstance(outputs, tuple) and len(outputs) == 2 and isinstance(outputs[1], dict):
                            ensemble_logits, branch_logits = outputs
                        elif isinstance(outputs, tuple) and len(outputs) == 3:
                            ensemble_logits, foundation_logits, scratch_logits = outputs
                            branch_logits = {
                                "foundation": foundation_logits,
                                "scratch": scratch_logits,
                            }
                        else:
                            ensemble_logits = outputs
                            branch_logits = {"ensemble": outputs}

                        labels_squeezed = labels.squeeze(-1) if labels.dim() > 1 else labels
                        classification_loss = torch.tensor(0.0, device=self.device)
                        for logits in branch_logits.values():
                            classification_loss = classification_loss + self.bce_loss_fn(logits.squeeze(-1), labels_squeezed)

                        cls_logits = ensemble_logits.reshape(-1)
                        cls_probs = torch.sigmoid(cls_logits)
                        cls_preds = (cls_probs >= 0.5).float()
                        labels_flat = labels.reshape(-1)
                        batch_correct = (cls_preds == labels_flat).sum().item()
                        batch_accuracy = batch_correct / labels_flat.size(0)
                        total_correct += batch_correct
                        total_samples += labels_flat.size(0)

                elif self.training_mode == "ablation":
                    # AblationEnsembleClassifier: embedding diff MLP + noise+LPD CNN

                    # Compute embedding difference
                    embedding_diff = clean_embeddings - noisy_embeddings  # [B, feature_dim]

                    # Noise is already 3 channels [B, 3, H, W] - use directly
                    noise_input = noise  # [B, 3, H, W]

                    # Generate LPD features for scratch branch
                    lpd_features = None
                    if self.lpd_strategy is not None:
                        # LPD strategy expects denormalized images
                        lpd_features = self.lpd_strategy.preprocess(denorm_images)

                    # AblationEnsembleClassifier returns (ensemble_logits, embedding_diff_logits, scratch_logits)
                    ensemble_logits, embedding_diff_logits, scratch_logits = self.classifier(
                        embedding_diff, noise_input, lpd_features, use_max_for_eval=False
                    )

                    # === SUM individual losses from both branches ===
                    embedding_diff_loss = self.bce_loss_fn(embedding_diff_logits.squeeze(-1), labels)
                    scratch_loss = self.bce_loss_fn(scratch_logits.squeeze(-1), labels)
                    classification_loss = embedding_diff_loss + scratch_loss

                    # For accuracy, use ensemble logits (for monitoring)
                    # Squeeze all trailing dims to get [B] from [B, 1, 1]
                    cls_logits = ensemble_logits.reshape(-1)
                    cls_probs = torch.sigmoid(cls_logits)
                    cls_preds = (cls_probs >= 0.5).float()
                    # Also squeeze labels to [B] for consistent comparison
                    labels_flat = labels.reshape(-1)
                    batch_correct = (cls_preds == labels_flat).sum().item()
                    batch_accuracy = batch_correct / labels_flat.size(0)
                    total_correct += batch_correct
                    total_samples += labels_flat.size(0)
                    

            # Combined loss
            if self.training_mode in ["with_classification", "noise_embedding_classifier", "noise_embedding_joint", "ensemble", "ablation"] and labels is not None:
                loss = self.embedding_loss_weight * embedding_loss + self.lambda_classification * classification_loss
            else:
                loss = self.embedding_loss_weight * embedding_loss

            # Update
            # Zero gradients for both optimizers
            self.decoder_optimizer.zero_grad()
            if self.classifier_optimizer is not None:
                self.classifier_optimizer.zero_grad()

            loss.backward()

            # Check gradients (first batch only, for debugging)
            if num_batches == 0 and self.rank == 0:
                # Get decoder parameters
                decoder_model = self.decoder.module if self.is_ddp else self.decoder

                # Check gradients for key layers (handle different decoder types)
                # AnyAttackDecoder has fc and final_conv, UNetDecoder has output_head
                if hasattr(decoder_model, 'fc'):
                    fc_grad = decoder_model.fc.weight.grad
                    fc_grad_norm = fc_grad.norm().item() if fc_grad is not None else 0.0
                    fc_grad_mean = fc_grad.abs().mean().item() if fc_grad is not None else 0.0
                else:
                    fc_grad_norm = 0.0
                    fc_grad_mean = 0.0

                if hasattr(decoder_model, 'final_conv'):
                    final_conv_grad = decoder_model.final_conv.weight.grad
                    final_conv_grad_norm = final_conv_grad.norm().item() if final_conv_grad is not None else 0.0
                    final_conv_grad_mean = final_conv_grad.abs().mean().item() if final_conv_grad is not None else 0.0
                elif hasattr(decoder_model, 'output_head'):
                    # UNetDecoder uses output_head (Sequential with Conv2d layers)
                    # Get the last conv layer (index 3 in the Sequential)
                    final_conv_grad = decoder_model.output_head.output[3].weight.grad
                    final_conv_grad_norm = final_conv_grad.norm().item() if final_conv_grad is not None else 0.0
                    final_conv_grad_mean = final_conv_grad.abs().mean().item() if final_conv_grad is not None else 0.0
                else:
                    final_conv_grad_norm = 0.0
                    final_conv_grad_mean = 0.0

                grad_info = {
                    'fc_grad_norm': fc_grad_norm,
                    'final_conv_grad_norm': final_conv_grad_norm,
                    'fc_grad_mean': fc_grad_mean,
                    'final_conv_grad_mean': final_conv_grad_mean,
                }

                pbar_dict = {
                    'loss': f'{loss.item():.4f}',
                    'sim': f'{embedding_similarity.mean():.4f}',
                    'acc': f'{batch_accuracy:.4f}',
                    'grad': f"{grad_info['fc_grad_norm']:.2e}",
                }
                pbar.set_postfix(pbar_dict)
                print(f"\n[Gradient Check - Batch 1]")
                print(f"  FC grad norm: {grad_info['fc_grad_norm']:.6f}, mean: {grad_info['fc_grad_mean']:.6f}")
                print(f"  Final conv grad norm: {grad_info['final_conv_grad_norm']:.6f}, mean: {grad_info['final_conv_grad_mean']:.6f}")
                print(f"  Loss: {loss.item():.6f}")
                print(f"  Loss type: {self.loss_type}, margin: {self.margin}")
                print(f"  Cosine similarity: {embedding_similarity.mean():.6f}")
                print(f"  Real similarity: {self._last_real_similarity:.6f}")
                print(f"  Fake similarity: {self._last_fake_similarity:.6f}")
                print(f"  Batch accuracy: {batch_accuracy:.4f}")
                print(f"  Noise range: [{noise.min():.6f}, {noise.max():.6f}], std: {noise.std():.6f}")

            # Step both optimizers
            self.decoder_optimizer.step()
            if self.classifier_optimizer is not None:
                self.classifier_optimizer.step()

            # Track metrics
            total_loss += loss.item()
            total_emb_loss += embedding_loss.item()
            if self.training_mode in ["with_classification", "noise_embedding_classifier", "noise_embedding_joint", "ensemble", "ablation"]:
                total_cls_loss += classification_loss.item()
            num_batches += 1

            # Update progress bar
            # Get current epsilon from decoder for display
            decoder_eps = self.get_decoder_eps()
            eps_int = int(decoder_eps * 255)

            if self.rank == 0 and labels is not None and (labels == 0).any() and (labels == 1).any():
                # Show real and fake similarities separately when we have both
                pbar_dict = {
                    'loss': f'{loss.item():.4f}',
                    'sim_r': f'{self._last_real_similarity:.4f}',
                    'sim_f': f'{self._last_fake_similarity:.4f}',
                    'eps': eps_int if self.eps_randomization else int(self.eps * 255),
                }
                # Add classification metrics in classification/ensemble/ablation mode
                if self.training_mode in ["with_classification", "noise_embedding_classifier", "noise_embedding_joint", "ensemble", "ablation"]:
                    pbar_dict['cls_loss'] = f'{classification_loss.item():.4f}'
                    pbar_dict['acc'] = f'{batch_accuracy:.4f}'
                pbar.set_postfix(pbar_dict)
            elif self.rank == 0:
                pbar_dict = {
                    'loss': f'{loss.item():.4f}',
                    'eps': eps_int if self.eps_randomization else int(self.eps * 255),
                }
                # Add classification metrics in classification/ensemble/ablation mode
                if self.training_mode in ["with_classification", "noise_embedding_classifier", "noise_embedding_joint", "ensemble", "ablation"] and labels is not None:
                    pbar_dict['cls_loss'] = f'{classification_loss.item():.4f}'
                    pbar_dict['acc'] = f'{batch_accuracy:.4f}'
                pbar.set_postfix(pbar_dict)

            self.global_step += 1

        # Aggregate metrics
        metrics = {
            'loss': total_loss / num_batches,
            'embedding_loss': total_emb_loss / num_batches,
            'disc_r_minus_f': total_similarity / num_batches,  # Real discrepancy - Fake discrepancy
        }
        if self.training_mode in ["with_classification", "noise_embedding_classifier", "noise_embedding_joint", "ensemble", "ablation"]:
            metrics['classification_loss'] = total_cls_loss / num_batches
            if total_samples > 0:
                metrics['accuracy'] = total_correct / total_samples

        if self.is_ddp:
            for key in metrics:
                metric_tensor = torch.tensor(metrics[key], device=self.device)
                dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
                metrics[key] = (metric_tensor / self.world_size).item()

        return metrics

    @torch.no_grad()
    def validate(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Validate the model using cross-generator evaluation.

        Returns aggregated metrics across all generators.
        """
        # Reset decoder epsilon to base_eps for consistent validation
        # (epsilon randomization is only for training, not validation)
        if self.eps_randomization:
            prev_eps = self.get_decoder_eps()
            self.update_decoder_eps(self.base_eps)

        # Use cross-generator evaluator
        generator_metrics = self.cross_generator_evaluate(val_loader)

        # Restore epsilon (optional, since training will update to new random value)
        if self.eps_randomization:
            self.update_decoder_eps(prev_eps)

        # Aggregate metrics across all generators
        metrics = {}
        if generator_metrics:
            # Average across all generators
            all_similarities = [m['mean_similarity'] for m in generator_metrics.values()]
            all_l2_distances = [m['mean_l2_distance'] for m in generator_metrics.values()]
            all_aucs = [m['auc'] for m in generator_metrics.values()]
            all_aps = [m['average_precision'] for m in generator_metrics.values()]
            all_opt_accs = [m['optimal_accuracy'] for m in generator_metrics.values()]

            metrics['val/similarity'] = sum(all_similarities) / len(all_similarities)
            metrics['val/l2_distance'] = sum(all_l2_distances) / len(all_l2_distances)
            metrics['val/auc'] = sum(all_aucs) / len(all_aucs)
            metrics['val/average_precision'] = sum(all_aps) / len(all_aps)
            metrics['val/optimal_accuracy'] = sum(all_opt_accs) / len(all_opt_accs)

            # Aggregate real/fake similarities if available
            real_sims = [m['similarity_real'] for m in generator_metrics.values() if 'similarity_real' in m]
            fake_sims = [m['similarity_fake'] for m in generator_metrics.values() if 'similarity_fake' in m]
            real_l2s = [m['l2_distance_real'] for m in generator_metrics.values() if 'l2_distance_real' in m]
            fake_l2s = [m['l2_distance_fake'] for m in generator_metrics.values() if 'l2_distance_fake' in m]

            if real_sims:
                metrics['val/similarity_real'] = sum(real_sims) / len(real_sims)
                metrics['val/l2_distance_real'] = sum(real_l2s) / len(real_l2s)
            if fake_sims:
                metrics['val/similarity_fake'] = sum(fake_sims) / len(fake_sims)
                metrics['val/l2_distance_fake'] = sum(fake_l2s) / len(fake_l2s)

            # Print per-generator metrics on rank 0
            if self.rank == 0:
                print(f"\nPer-Generator Validation Metrics (Epoch {epoch}):")
                for gen, gen_metrics in sorted(generator_metrics.items()):
                    print(f"  {gen}:")
                    print(f"    similarity: {gen_metrics['mean_similarity']:.4f} (std: {gen_metrics['std_similarity']:.4f})")
                    print(f"    l2_distance: {gen_metrics['mean_l2_distance']:.4f}")
                    print(f"    auc: {gen_metrics['auc']:.4f}")
                    print(f"    average_precision: {gen_metrics['average_precision']:.4f}")
                    print(f"    optimal_accuracy: {gen_metrics['optimal_accuracy']:.4f}")

                    if 'similarity_real' in gen_metrics:
                        print(f"      real - similarity: {gen_metrics['similarity_real']:.4f}, l2: {gen_metrics['l2_distance_real']:.4f}")
                    if 'similarity_fake' in gen_metrics:
                        print(f"      fake - similarity: {gen_metrics['similarity_fake']:.4f}, l2: {gen_metrics['l2_distance_fake']:.4f}")
                    # Compute gap
                    if 'similarity_real' in gen_metrics and 'similarity_fake' in gen_metrics:
                        sim_gap = gen_metrics['similarity_fake'] - gen_metrics['similarity_real']
                        l2_gap = gen_metrics['l2_distance_fake'] - gen_metrics['l2_distance_real']
                        print(f"      gap - similarity: {sim_gap:.4f}, l2: {l2_gap:.4f}")

        return metrics

    def cross_generator_evaluate(self, test_loader: DataLoader) -> Dict[str, Dict[str, float]]:
        """Run cross-generator evaluation with classifier if available"""
        results = self.evaluator.evaluate_decoder(
            self.encoder,
            self.decoder,
            test_loader,
            self.eps,
            classifier=self.classifier,
            training_mode=getattr(self, 'training_mode', None),
            strategy=getattr(self, 'strategy', None),
            lpd_strategy=getattr(self, 'lpd_strategy', None),
            use_multi_scale_decoder=getattr(self, 'use_multi_scale_decoder', True)
        )

        # Synchronize all ranks to prevent hang at 100%
        if self.world_size > 1 and dist.is_available() and dist.is_initialized():
            dist.barrier()

        return results

    def save_checkpoint(self, epoch: int, filename: str = None):
        """Save training checkpoint (only on rank 0)"""
        if self.rank != 0:
            return

        if filename is None:
            filename = f"checkpoint_epoch_{epoch}.pt"

        # Ensure checkpoint directory exists
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'decoder_state_dict': self.decoder.module.state_dict() if self.is_ddp else self.decoder.state_dict(),
            'decoder_optimizer_state_dict': self.decoder_optimizer.state_dict(),
            'training_mode': getattr(self, 'training_mode', 'embedding_only'),
            'lambda_classification': getattr(self, 'lambda_classification', 0.0),
        }

        if self.classifier is not None:
            checkpoint['classifier_state_dict'] = self.classifier.module.state_dict() if self.is_ddp else self.classifier.state_dict()

        if self.classifier_optimizer is not None:
            checkpoint['classifier_optimizer_state_dict'] = self.classifier_optimizer.state_dict()

        filepath = os.path.join(self.checkpoint_dir, filename)
        try:
            torch.save(checkpoint, filepath)
            print(f"Saved checkpoint: {filepath}")
        except Exception as e:
            print(f"Error saving checkpoint: {e}")

    def load_checkpoint(self, checkpoint_path: str, load_optimizers: bool = True):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        decoder_state = checkpoint['decoder_state_dict']
        if self.is_ddp:
            self.decoder.module.load_state_dict(decoder_state)
        else:
            self.decoder.load_state_dict(decoder_state)

        # Load decoder optimizer state
        if load_optimizers and self.decoder_optimizer is not None and 'decoder_optimizer_state_dict' in checkpoint:
            self.decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']

        if 'classifier_state_dict' in checkpoint and self.classifier is not None:
            classifier_state = checkpoint['classifier_state_dict']
            if self.is_ddp:
                self.classifier.module.load_state_dict(classifier_state)
            else:
                self.classifier.load_state_dict(classifier_state)

        # Load classifier optimizer state if available
        if load_optimizers and 'classifier_optimizer_state_dict' in checkpoint and self.classifier_optimizer is not None:
            self.classifier_optimizer.load_state_dict(checkpoint['classifier_optimizer_state_dict'])

        if self.rank == 0:
            print(f"Loaded checkpoint from epoch {self.current_epoch}, global_step {self.global_step}")


def setup_ddp():
    """Setup distributed training environment"""
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        if rank == 0:
            print(f"Initialized DDP: world_size={world_size}, backend=nccl")

    return rank, world_size, local_rank


def cleanup_ddp():
    """Cleanup distributed training environment"""
    if dist.is_initialized():
        dist.destroy_process_group()
