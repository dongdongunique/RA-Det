"""
AnyAttack-style Noise Decoder for Adversarial Training.

This module implements a decoder that inverts DINOv3 embeddings to adversarial noise.
Based on the AnyAttack paper architecture with EfficientAttention.

Author: Implementation Plan
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Channel dimensions for different DINOv3 variants
DINOV3_CHANNELS = {
    "dinov3_vits16": 384,
    "dinov3_vits16plus": 384,
    "dinov3_vitb16": 768,
    "dinov3_vitl16": 1024,
    "dinov3_vith16plus": 1536,
    "dinov3_vit7b16": 3072,
}

# Channel dimensions for different CLIP variants
CLIP_CHANNELS = {
    "ViT-B-32": 512,
    "ViT-B-16": 512,
    "ViT-L-14": 768,
    "ViT-H-14": 1024,
    "ViT-B-32_DataComp": 512,
    "ViT-B-16_DataComp": 512,
    "ViT-L-14_DataComp": 768,
    "RN101": 512,
    "ConNext-Base": 1024,
    "ConvNext-Large": 1024,
    "ConvNext-xxlarge": 1024,
    "Swin_Transformer-Base": 1024,
    "CLIPAG": 512,
    "longclip": 512,
    "ViT-B-32_robust_fair_4": 512,
    "ViT-B-16_robust_fair_4": 512,
    "ViT-L-14_robust_fair_4": 768,
    "ViT-L-14_robust_fair_8": 768,
    "ViT-L-14_robust_fair_16": 768,
    "ViT-L-14_robust_fair_32": 768,
    "ViT-B-16_robust_tecoa_4": 512,
    "ViT-L-14_robust_fare_16_1600": 768,
    "ViT-L-14_robust_fare_16_5600": 768,
    "ViT-L-14_robust_fare_16_9600": 768,
    "ViT-L-14_robust_fare_32_4000": 768,
    "ViT-L-14_robust_fare_32_10000": 768,
    "ViT-B-16_prompt-tuning_all_5": 512,
    "ViT-B-16_prompt-tuning_all_5_V": 512,
    "ViT-B-16_prompt-tuning_all_eps4_2_V": 512,
    "ViT-B-16_prompt-tuning_all_eps1_1_V": 512,
    "ViT-B-16_prompt-tuning_all_eps2_3_V": 512,
    "ViT-B-32_prompt-tuning_all_eps2_3_V": 512,
}

# Channel dimensions for DINOv2 variants
DINOV2_CHANNELS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}

# Combined model channels
MODEL_CHANNELS = {
    **DINOV3_CHANNELS,
    **CLIP_CHANNELS,
    **DINOV2_CHANNELS,
}


class EfficientAttention(nn.Module):
    """Efficient multi-head attention module."""

    def __init__(self, in_channels, key_channels, head_count, value_channels):
        super(EfficientAttention, self).__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)

    def forward(self, input_):
        n, _, h, w = input_.size()
        keys = self.keys(input_).reshape((n, self.key_channels, h * w))
        queries = self.queries(input_).reshape(n, self.key_channels, h * w)
        values = self.values(input_).reshape((n, self.value_channels, h * w))

        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=2)
            query = F.softmax(queries[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=1)
            value = values[:, i * head_value_channels: (i + 1) * head_value_channels, :]
            context = key @ value.transpose(1, 2)
            attended_value = (context.transpose(1, 2) @ query).reshape(n, head_value_channels, h, w)
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        reprojected_value = self.reprojection(aggregated_values)
        attention = reprojected_value + input_

        return attention


class ResBlock(nn.Module):
    """Residual block with efficient attention."""

    def __init__(self, in_channels, out_channels, key_channels, head_count, value_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.attention = EfficientAttention(out_channels, key_channels, head_count, value_channels)
        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        else:
            self.skip_conv = nn.Identity()

    def forward(self, x):
        residual = x
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.attention(out)
        residual = self.skip_conv(residual)
        out += residual
        return self.activation(out)


class UpBlock(nn.Module):
    """Upsampling block with nearest neighbor interpolation."""

    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.up(x)
        x = self.activation(self.bn(self.conv(x)))
        return x


class AnyAttackDecoder(nn.Module):
    """
    Decoder that inverts embeddings (DINOv3/DINOv2/CLIP) to adversarial noise.

    Takes model embeddings as input and outputs adversarial noise
    that, when added to images, maximizes the embedding discrepancy.

    Architecture (matching AnyAttack repo):
    1. Linear projection: embed_dim → 256 × (img_size/16)²
    2. Alternating ResBlock and UpBlock:
       - ResBlock(256→256) → UpBlock(256→128)
       - ResBlock(128→128) → UpBlock(128→64)
       - ResBlock(64→64) → UpBlock(64→32)
       - ResBlock(32→32) → UpBlock(32→16)
       - ResBlock(16→16)
    3. Final: Conv2d(16→3, 3×3) + clamp by eps

    Args:
        embed_dim (int): Model embedding dimension (supports DINOv3/DINOv2/CLIP)
        noise_channels (int): Number of noise channels (default: 3 for RGB)
        img_size (int): Output image size (default: 224)
        eps (float): Maximum perturbation budget (default: 16/255)

    Forward:
        Input: [B, embed_dim] Model embeddings
        Output: [B, 3, 224, 224] Adversarial noise clamped by eps
    """

    def __init__(self, embed_dim=1024, noise_channels=3, img_size=224, eps=16/255):
        super(AnyAttackDecoder, self).__init__()
        self.embed_dim = embed_dim
        self.noise_channels = noise_channels
        self.img_size = img_size
        self.eps = eps
        self.init_size = img_size // 16  # Initial spatial size before upsampling

        # Initial projection: embed_dim → 256 × init_size²
        self.fc = nn.Linear(embed_dim, 256 * self.init_size ** 2)

        # Alternating ResBlock and UpBlock
        self.upsample_blocks = nn.ModuleList([
            ResBlock(256, 256, 64, 8, 256),   # 14×14
            UpBlock(256, 128),                 # 14×14 → 28×28
            ResBlock(128, 128, 32, 8, 128),    # 28×28
            UpBlock(128, 64),                  # 28×28 → 56×56
            ResBlock(64, 64, 16, 8, 64),       # 56×56
            UpBlock(64, 32),                   # 56×56 → 112×112
            ResBlock(32, 32, 8, 8, 32),        # 112×112
            UpBlock(32, 16),                   # 112×112 → 224×224
            ResBlock(16, 16, 4, 8, 16),        # 224×224
        ])

        # Final output layer
        self.final_conv = nn.Conv2d(16, noise_channels, kernel_size=3, stride=1, padding=1)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights following best practices."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Linear layers: Xavier uniform
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                # Conv layers: Kaiming normal for LeakyReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # BatchNorm: standard initialization
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, embeddings):
        """
        Forward pass: embeddings → adversarial noise.

        Args:
            embeddings: [B, embed_dim] DINOv3 CLS token features

        Returns:
            noise: [B, 3, 224, 224] Adversarial noise scaled by eps
        """
        batch_size = embeddings.size(0)

        # Project and reshape to [B, 256, init_size, init_size]
        x = self.fc(embeddings.float())
        x = x.view(batch_size, 256, self.init_size, self.init_size)

        # Pass through upsample blocks
        for block in self.upsample_blocks:
            x = block(x)

        # Final output
        noise = torch.tanh(self.final_conv(x))
        # Scale by epsilon to bound the perturbation
        noise = noise * self.eps
        return noise


class UNetDecoder(nn.Module):
    """
    UNet-based decoder with multi-modal inputs.

    Takes original image, noised image, clean/noisy embeddings, and optional
    multi-scale features as input. Outputs adversarial noise.

    Architecture:
    1. InputProcessor: Concatenate multi-modal inputs
    2. ImageEncoder: 5-stage downsampling CNN
    3. EmbeddingEncoders: Project embeddings to spatial features
    4. CrossAttentionBottleneck: Fuse image and embedding features
    5. Decoder: 5-stage upsampling with skip connections
    6. OutputHead: Generate noise

    Args:
        embed_dim (int): Model embedding dimension (e.g., 1024 for ViT-L16)
        strategy_channels (int): Multi-scale strategy output channels (0 if none)
        base_channels (int): Base channel dimension for UNet (default: 64)
        num_levels (int): Number of UNet levels (default: 5, results in 224→7)
        num_heads (int): Number of attention heads (default: 8)
        use_attention (bool): Use attention in encoder/decoder stages (default: True)
        eps (float): Maximum perturbation budget (default: 16/255)
        bottleneck_size (int): Bottleneck spatial size (default: 7)
    """

    def __init__(
        self,
        embed_dim=1024,
        strategy_channels=0,
        base_channels=64,
        num_levels=5,
        num_heads=8,
        use_attention=True,
        eps=16/255,
        bottleneck_size=7
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.strategy_channels = strategy_channels
        self.base_channels = base_channels
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.use_attention = use_attention
        self.eps = eps
        self.bottleneck_size = bottleneck_size

        # Import here to avoid circular dependency
        from .unet_components import (
            InputProcessor,
            EncoderStage,
            DecoderStage,
            CrossAttentionBottleneck,
            EmbeddingEncoder,
            OutputHead
        )

        # Input processor
        self.input_processor = InputProcessor(
            base_channels=base_channels,
            strategy_channels=strategy_channels
        )

        # Image encoder (multi-scale downsampling)
        self.encoder_stages = nn.ModuleList()
        channels = base_channels

        for i in range(num_levels):
            out_channels = min(base_channels * (2 ** i), 512)
            self.encoder_stages.append(
                EncoderStage(
                    channels,
                    out_channels,
                    downsample=(i < num_levels - 1),  # Don't downsample at bottleneck
                    use_attention=use_attention and (i > 0)  # No attention at first stage
                )
            )
            channels = out_channels

        # Embedding encoders (clean and noisy)
        self.clean_embedding_encoder = EmbeddingEncoder(
            embed_dim=embed_dim,
            spatial_size=bottleneck_size,
            channels=channels
        )
        self.noisy_embedding_encoder = EmbeddingEncoder(
            embed_dim=embed_dim,
            spatial_size=bottleneck_size,
            channels=channels
        )

        # Bottleneck with cross-attention
        self.bottleneck = CrossAttentionBottleneck(
            image_channels=channels,
            embed_channels=channels,
            num_heads=num_heads
        )

        # Decoder (multi-scale upsampling with skip connections)
        self.decoder_stages = nn.ModuleList()
        # Create decoder stages in forward order (stage 0 processes bottleneck output first)
        # Skip connections are used in order: highest (14x14) → lowest (112x112)
        # We need to track which skip each decoder stage uses
        self.skip_connections = []  # Store which encoder stage index each decoder stage connects to

        # Create decoder stages from bottleneck (14x14) up to 224x224
        # The decoder outputs channels in reverse order of encoder:
        # - Stage 0: 512 → 256 (processes 512→512+512 concat → output 256)
        # - Stage 1: 256 → 128 (processes 256→256+256 concat → output 128)
        # - Stage 2: 128 → 64 (processes 128→128+128 concat → output 64)
        # - Stage 3: 64 → 64 (processes 64→64+64 concat → output 64)
        for decoder_stage_idx in range(num_levels - 1):  # 0, 1, 2, 3 (4 decoder stages)
            # Calculate which encoder stage this decoder stage connects to
            # decoder_stage_idx 0 → encoder stage 3 (14x14 skip)
            # decoder_stage_idx 1 → encoder stage 2 (28x28 skip)
            # decoder_stage_idx 2 → encoder stage 1 (56x56 skip)
            # decoder_stage_idx 3 → encoder stage 0 (112x112 skip)
            encoder_stage_idx = (num_levels - 2) - decoder_stage_idx

            skip_channels = min(base_channels * (2 ** encoder_stage_idx), 512)

            # Output channels decrease as we go up the decoder (mirroring encoder in reverse)
            # Stage 0: output 256, Stage 1: output 128, Stage 2: output 64, Stage 3: output 64
            if decoder_stage_idx < num_levels - 2:
                out_channels = min(base_channels * (2 ** ((num_levels - 2) - decoder_stage_idx - 1)), 512)
            else:
                out_channels = base_channels  # Final stage outputs base_channels

            # ALL stages: concat skip at same resolution, then upsample
            # The skip connections are from encoder outputs, so they're at the SAME
            # resolution as the decoder input (not 2x resolution)
            upsample_first = False

            self.decoder_stages.append(
                DecoderStage(
                    in_channels=channels,
                    skip_channels=skip_channels,
                    out_channels=out_channels,
                    use_attention=use_attention,
                    upsample_first=upsample_first
                )
            )
            self.skip_connections.append(encoder_stage_idx)
            channels = out_channels

        # Output head
        self.output_head = OutputHead(
            in_channels=channels,
            eps=eps
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, original_image, clean_embedding, multi_scale_images=None):
        """
        Forward pass: multi-modal inputs → adversarial noise.

        Args:
            original_image: [B, 3, 224, 224] Clean image
            clean_embedding: [B, embed_dim] Clean image CLS embedding
            multi_scale_images: [B, strategy_channels, 224, 224] Multi-scale/difference features

        Returns:
            noise: [B, 3, 224, 224] Adversarial noise scaled by eps
        """
        # Process inputs through input processor
        x = self.input_processor(original_image, multi_scale_images)

        # Encode (downsampling) with skip connections
        encoder_skips = []
        for i, stage in enumerate(self.encoder_stages):
            x = stage(x)
            if i < self.num_levels - 1:  # Store all skips except bottleneck
                encoder_skips.append(x)

        # Encode embeddings to spatial features
        # Use clean_embedding for both encoders (noisy_embedding not available yet)
        clean_embed_feat = self.clean_embedding_encoder(clean_embedding)
        noisy_embed_feat = self.noisy_embedding_encoder(clean_embedding)  # Placeholder, will use same

        # Fuse with cross-attention bottleneck
        x = self.bottleneck(x, clean_embed_feat, noisy_embed_feat)

        # Decode (upsampling) with skip connections
        # Each decoder stage uses the skip from its corresponding encoder stage
        for i, stage in enumerate(self.decoder_stages):
            encoder_stage_idx = self.skip_connections[i]
            skip = encoder_skips[encoder_stage_idx]
            x = stage(x, skip)

        # Generate noise
        noise = self.output_head(x)

        return noise


def create_decoder(model_name, eps=16/255, decoder_type='simple', **kwargs):
    """
    Factory function to create decoder with correct embedding dimension.

    Supports:
    - DINOv3: dinov3_vits16, dinov3_vitb16, dinov3_vitl16, dinov3_vith16plus, dinov3_vit7b16
    - DINOv2: dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14
    - CLIP: ViT-B-32, ViT-B-16, ViT-L-14, ViT-H-14, and many more

    Args:
        model_name: Model name (e.g., 'dinov3_vitl16', 'ViT-B-32', 'ViT-L-14')
        eps: Maximum perturbation budget
        decoder_type: Type of decoder ('simple' or 'unet')
        **kwargs: Additional arguments for UNetDecoder (strategy_channels, base_channels, etc.)

    Returns:
        Decoder instance (AnyAttackDecoder or UNetDecoder)
    """
    if model_name not in MODEL_CHANNELS:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: {list(MODEL_CHANNELS.keys())}"
        )

    embed_dim = MODEL_CHANNELS[model_name]

    if decoder_type == 'simple':
        return AnyAttackDecoder(embed_dim=embed_dim, eps=eps)
    elif decoder_type == 'unet':
        return UNetDecoder(embed_dim=embed_dim, eps=eps, **kwargs)
    else:
        raise ValueError(
            f"Unknown decoder type: {decoder_type}. "
            f"Available: 'simple', 'unet'"
        )


if __name__ == "__main__":
    # Test the decoder
    print("Testing AnyAttackDecoder...")

    # Test DINOv3 variants
    print("\n=== DINOv3 Models ===")
    for model_name in ["dinov3_vits16", "dinov3_vitb16", "dinov3_vitl16"]:
        decoder = create_decoder(model_name)
        print(f"\n{model_name}:")
        print(f"  Embed dim: {decoder.embed_dim}")
        print(f"  Init size: {decoder.init_size}")
        print(f"  EPS: {decoder.eps}")

        # Test forward pass
        batch_size = 4
        embed_dim = DINOV3_CHANNELS[model_name]
        embeddings = torch.randn(batch_size, embed_dim)
        noise = decoder(embeddings)
        print(f"  Input shape: {embeddings.shape}")
        print(f"  Output shape: {noise.shape}")
        print(f"  Noise range: [{noise.min():.4f}, {noise.max():.4f}]")
        print(f"  Noise mean: {noise.mean():.4f}, std: {noise.std():.4f}")

    # Test CLIP variants
    print("\n=== CLIP Models ===")
    for model_name in ["ViT-B-32", "ViT-B-16", "ViT-L-14"]:
        decoder = create_decoder(model_name)
        print(f"\n{model_name}:")
        print(f"  Embed dim: {decoder.embed_dim}")
        print(f"  Init size: {decoder.init_size}")
        print(f"  EPS: {decoder.eps}")

        # Test forward pass
        batch_size = 4
        embed_dim = CLIP_CHANNELS[model_name]
        embeddings = torch.randn(batch_size, embed_dim)
        noise = decoder(embeddings)
        print(f"  Input shape: {embeddings.shape}")
        print(f"  Output shape: {noise.shape}")
        print(f"  Noise range: [{noise.min():.4f}, {noise.max():.4f}]")
        print(f"  Noise mean: {noise.mean():.4f}, std: {noise.std():.4f}")

    print("\n✓ All tests passed!")
