"""
Base class for foundation model-based detectors.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List
from ..base import BaseDetectionModel


class BaseFoundationModel(BaseDetectionModel):
    """Base class for foundation model-based detectors using noise perturbation"""

    def __init__(self, encoder_name: str, metrics: List[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.encoder_name = encoder_name
        self.encoder = None
        self.classifier = None
        self.feature_dim = None
        # Use cosine similarity as default metric if not specified
        self.metrics = metrics if metrics is not None else ["cos_sim"]

        # Initialize distance metrics using existing registry
        self._init_distance_metrics()

    def _init_distance_metrics(self):
        """Initialize distance metrics using the existing registry"""
        try:
            # Import the existing distance registry
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            from distance_metrics import get_distance_registry

            registry = get_distance_registry()
            self.distance_metrics = []

            for metric_name in self.metrics:
                try:
                    metric = registry.get_metric(metric_name)
                    self.distance_metrics.append(metric)
                except ValueError as e:
                    print(f"Warning: Metric '{metric_name}' not found, skipping. {e}")

            # If no valid metrics were loaded, fall back to cosine similarity
            if not self.distance_metrics:
                print("Warning: No valid metrics specified, using cosine similarity as default")
                self.distance_metrics = [registry.get_metric("cos_sim")]

        except ImportError:
            print("Warning: Could not import distance metrics registry, using default cosine similarity")
            # Fallback implementation
            self.distance_metrics = [self._default_cosine_metric()]

    def _default_cosine_metric(self):
        """Fallback cosine similarity metric class"""
        class DefaultCosineMetric:
            def compute(self, original_features, noisy_features):
                cos_sims = torch.nn.functional.cosine_similarity(original_features, noisy_features, dim=1)
                return (1 - cos_sims).unsqueeze(1)  # Convert to difference
            def get_name(self):
                return "cos_sim"
        return DefaultCosineMetric()

    def load_encoder(self, config: Dict[str, Any]):
        """Load vision encoder using existing create_models.py"""
        import sys
        import os
        import torch
        # Add the parent directory to the path to find create_models.py
        # From: /mnt/shared-storage-user/chenyunhao/LeakyCLIP/ai_generated_image_detection/training/models/foundation_models/base.py
        # To: /mnt/shared-storage-user/chenyunhao/LeakyCLIP/
        parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
        sys.path.insert(0, parent_dir)
        from create_models import load_models

        # Convert encoder_name to model_name for load_models
        encoder_name = config.get('encoder_name', config.get('model_name', 'ViT-B-32'))
        model_config = {
            'model_name': encoder_name,
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        }

        model, _, _ = load_models(model_config)
        self.encoder = model
        if hasattr(model, "output_dim"):
            self.feature_dim = model.output_dim
        elif hasattr(model, "visual") and hasattr(model.visual, "output_dim"):
            self.feature_dim = model.visual.output_dim
        else:
            raise AttributeError("Loaded encoder has no output_dim; expected model.output_dim or model.visual.output_dim")

    def extract_batch_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features from batch of images (original + noised)
        Args:
            images: Tensor of shape (2B, C, H, W) where first B are original, second B are noised
        Returns:
            features: Tensor of shape (2B, feature_dim)
        """
        if self.encoder is None:
            raise RuntimeError("Encoder not loaded")
        return self.encoder.encode_image(images)

    def compute_feature_differences(self, original_features: torch.Tensor,
                                  noised_features: torch.Tensor) -> torch.Tensor:
        """
        Compute differences between original and noised features using specified metrics
        Args:
            original_features: Tensor of shape (B, feature_dim)
            noised_features: Tensor of shape (B, feature_dim)
        Returns:
            differences: Tensor of shape (B, feature_dim * len(metrics)) or (B, len(metrics))
                  depending on metric output format
        """
        diff_features = []

        for metric in self.distance_metrics:
            # Use the existing distance metric interface
            try:
                # Convert cosine similarity to difference if needed
                if metric.get_name() == "cos_sim":
                    # For cosine similarity, we need 1 - similarity to get difference
                    cos_sims = torch.nn.functional.cosine_similarity(original_features, noised_features, dim=1)
                    cos_diff = 1 - cos_sims
                    # Expand to match feature dimension
                    diff_features.append(cos_diff.unsqueeze(1).expand(-1, original_features.size(1)))
                else:
                    # For other metrics, use the compute method
                    distances = metric.compute(original_features, noised_features)
                    # Convert numpy to tensor
                    if isinstance(distances, np.ndarray):
                        distances = torch.from_numpy(distances).float()
                    # Expand to match feature dimension
                    diff_features.append(distances.unsqueeze(1).expand(-1, original_features.size(1)))
            except Exception as e:
                print(f"Warning: Failed to compute metric {metric.get_name()}: {e}")
                # Fallback: use zeros
                diff_features.append(torch.zeros(original_features.size(0), original_features.size(1)))

        # Concatenate all difference features
        return torch.cat(diff_features, dim=1)
