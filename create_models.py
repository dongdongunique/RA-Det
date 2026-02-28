"""
Model loading utilities with explicit paths for DINOv3 models.
This is a simplified version for the release package.
"""

import os
import sys
import torch
import torchvision.transforms as transforms
from typing import Tuple, Optional
from PIL import Image
import torch.nn as nn

# Add release root to path for importing paths.py
RELEASE_ROOT = os.path.dirname(os.path.abspath(__file__))
if RELEASE_ROOT not in sys.path:
    sys.path.insert(0, RELEASE_ROOT)

from paths import DINOV3_REPO_DIR, DINOV2_REPO_DIR, DINO_WEIGHTS_DIR

# Supported models
DINOV3_MODELS = {
    "dinov3_vits16": {"variant": "vits16", "dim": 384},
    "dinov3_vits16plus": {"variant": "vits16plus", "dim": 384},
    "dinov3_vitb16": {"variant": "vitb16", "dim": 768},
    "dinov3_vitl16": {"variant": "vitl16", "dim": 1024},
    "dinov3_vith16plus": {"variant": "vith16plus", "dim": 1536},
    "dinov3_vit7b16": {"variant": "vit7b16", "dim": 3072},
}

DINOV2_MODELS = {
    "dinov2_vits14": {"variant": "vit_s", "dim": 384},
    "dinov2_vitb14": {"variant": "vit_b", "dim": 768},
    "dinov2_vitl14": {"variant": "vit_l", "dim": 1024},
    "dinov2_vitg14": {"variant": "vit_g", "dim": 1536},
}


class DINOv2Wrapper(nn.Module):
    """Wrapper for DINOv2/DINOv3 models to provide CLIP-like interface."""

    def __init__(self, model, model_name: str):
        super().__init__()
        self.model = model
        self.model_name = model_name
        self.output_dim = model.embed_dim if hasattr(model, 'embed_dim') else 1024

    @torch.cuda.amp.autocast(dtype=torch.bfloat16)
    def encode_image(self, image):
        """Encode images to embeddings."""
        # Note: Not using torch.no_grad() to allow gradient flow for training
        requires_grad = image.requires_grad or any(p.requires_grad for p in self.model.parameters())

        if requires_grad:
            features = self.model(image)
        else:
            with torch.no_grad():
                features = self.model(image)
        return features.float()

    def forward(self, images):
        return self.encode_image(images)


def load_dinov3(model_name: str, device: str = "cuda") -> Tuple:
    """
    Load DINOv3 model with explicit paths.

    Args:
        model_name: Model name (e.g., 'dinov3_vitl16')
        device: Device to load model on

    Returns:
        Tuple of (model, tokenizer, preprocess)
    """
    if model_name not in DINOV3_MODELS:
        raise ValueError(f"Unknown DINOv3 model: {model_name}. Available: {list(DINOV3_MODELS.keys())}")

    config = DINOV3_MODELS[model_name]
    variant = config["variant"]

    # Add dinov3_repo to path
    if os.path.exists(DINOV3_REPO_DIR):
        sys.path.insert(0, DINOV3_REPO_DIR)
        print(f"Using DINOv3 from local repository: {DINOV3_REPO_DIR}")

    try:
        # Try local import
        from dinov3.hub.backbones import dinov3_vits16, dinov3_vits16plus, dinov3_vitb16, dinov3_vitl16, dinov3_vith16plus, dinov3_vit7b16

        model_map = {
            "vits16": dinov3_vits16,
            "vits16plus": dinov3_vits16plus,
            "vitb16": dinov3_vitb16,
            "vitl16": dinov3_vitl16,
            "vith16plus": dinov3_vith16plus,
            "vit7b16": dinov3_vit7b16,
        }
        model_fn = model_map[variant]
        model = model_fn(pretrained=False)
        print(f"✓ DINOv3 {variant} architecture loaded successfully")
    except Exception as e:
        print(f"Failed to load DINOv3 locally: {e}")
        raise ImportError(f"Cannot load DINOv3 from {DINOV3_REPO_DIR}")

    # Load weights from local checkpoint
    checkpoint_path = os.path.join(DINO_WEIGHTS_DIR, f"{model_name}_pretrain.pth")
    if os.path.exists(checkpoint_path):
        print(f"Loading DINOv3 weights from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
    else:
        print(f"Warning: No checkpoint found at {checkpoint_path}")

    model = model.to(device)
    model.eval()

    # Create wrapper
    wrapped_model = DINOv2Wrapper(model, model_name)

    # Create preprocess
    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dummy tokenizer (DINOv3 is vision-only)
    class DummyTokenizer:
        def tokenize(self, text):
            return torch.zeros(1, 77, dtype=torch.long)

    return wrapped_model, DummyTokenizer(), preprocess


def load_dinov2(model_name: str, device: str = "cuda") -> Tuple:
    """
    Load DINOv2 model with explicit paths.

    Args:
        model_name: Model name (e.g., 'dinov2_vitl14')
        device: Device to load model on

    Returns:
        Tuple of (model, tokenizer, preprocess)
    """
    if model_name not in DINOV2_MODELS:
        raise ValueError(f"Unknown DINOv2 model: {model_name}. Available: {list(DINOV2_MODELS.keys())}")

    config = DINOV2_MODELS[model_name]
    variant = config["variant"]

    # Add dinov2_repo to path
    if os.path.exists(DINOV2_REPO_DIR):
        sys.path.insert(0, DINOV2_REPO_DIR)
        print(f"Using DINOv2 from local repository: {DINOV2_REPO_DIR}")

    try:
        import dinov2
        model = dinov2.__dict__[f"dinov2_{variant}"](pretrained=False)
        print(f"✓ DINOv2 {variant} architecture loaded successfully")
    except Exception as e:
        print(f"Failed to load DINOv2 locally: {e}")
        # Fallback to torch.hub
        try:
            model = torch.hub.load('facebookresearch/dinov2', f'dinov2_{variant}', pretrained=False)
            print(f"✓ DINOv2 {variant} loaded from torch.hub")
        except Exception as hub_error:
            raise ImportError(f"Cannot load DINOv2: {hub_error}")

    # Load weights from local checkpoint
    checkpoint_path = os.path.join(DINO_WEIGHTS_DIR, f"{model_name}_pretrain.pth")
    if os.path.exists(checkpoint_path):
        print(f"Loading DINOv2 weights from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
    else:
        print(f"Warning: No checkpoint found at {checkpoint_path}")

    model = model.to(device)
    model.eval()

    # Create wrapper
    wrapped_model = DINOv2Wrapper(model, model_name)

    # Create preprocess
    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dummy tokenizer
    class DummyTokenizer:
        def tokenize(self, text):
            return torch.zeros(1, 77, dtype=torch.long)

    return wrapped_model, DummyTokenizer(), preprocess


def load_models(config: dict):
    """
    Main function to load models based on config.

    Args:
        config: Dictionary with 'model_name' and 'device' keys

    Returns:
        Tuple of (model, tokenizer, preprocess)
    """
    model_name = config.get("model_name", "dinov3_vitl16")
    device = config.get("device", "cuda")

    print(f"Loading model: {model_name}")

    if model_name.startswith("dinov3_"):
        return load_dinov3(model_name, device)
    elif model_name.startswith("dinov2_"):
        return load_dinov2(model_name, device)
    else:
        raise ValueError(f"Unsupported model: {model_name}. Use dinov3_* or dinov2_*")


# For testing
if __name__ == "__main__":
    # Test loading
    config = {"model_name": "dinov3_vitl16", "device": "cuda"}
    model, tokenizer, preprocess = load_models(config)
    print(f"Model output dim: {model.output_dim}")
    print("Model loaded successfully!")
