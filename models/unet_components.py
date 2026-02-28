"""
UNet Building Blocks for AnyAttack Decoder.

This module implements the core components for a UNet-based decoder that
processes multi-modal inputs (images, embeddings, multi-scale features).

Author: Implementation Plan
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Handle both relative and direct imports
try:
    from .decoder import EfficientAttention
except ImportError:
    from decoder import EfficientAttention


class ConvBlock(nn.Module):
    """
    Basic convolutional block with optional attention.

    Architecture: Conv → BatchNorm → LeakyReLU → (optional) Attention → Conv → BatchNorm → LeakyReLU

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        use_attention (bool): Whether to include EfficientAttention
        key_channels (int): Key channels for attention (if used)
        head_count (int): Number of attention heads (if used)
    """

    def __init__(self, in_channels, out_channels, use_attention=False, key_channels=None, head_count=8):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.activation = nn.LeakyReLU(0.2, inplace=True)

        # Optional attention
        self.use_attention = use_attention
        if use_attention:
            if key_channels is None:
                key_channels = out_channels // 8
            self.attention = EfficientAttention(
                out_channels,
                key_channels=key_channels,
                head_count=head_count,
                value_channels=out_channels
            )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)

        if self.use_attention:
            x = self.attention(x)

        return x


class EncoderStage(nn.Module):
    """
    Encoder downsampling stage.

    Architecture: (optional) Downsample → ConvBlock

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        downsample (bool): Whether to downsample (stride=2 conv or maxpool)
        use_attention (bool): Whether to use attention in ConvBlock
    """

    def __init__(self, in_channels, out_channels, downsample=True, use_attention=False):
        super().__init__()

        self.downsample = downsample
        if downsample:
            # Use strided convolution for downsampling
            self.downsample_conv = nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=2,
                padding=1
            )
            self.downsample_bn = nn.BatchNorm2d(in_channels)

        self.conv_block = ConvBlock(
            in_channels if not downsample else in_channels,
            out_channels,
            use_attention=use_attention
        )

    def forward(self, x):
        if self.downsample:
            x = self.downsample_conv(x)
            x = self.downsample_bn(x)
            x = F.leaky_relu_(x)

        x = self.conv_block(x)
        return x


class DecoderStage(nn.Module):
    """
    Decoder upsampling stage with skip connection.

    Architecture: Upsample → Concat skip → ConvBlock

    Args:
        in_channels (int): Number of input channels (from previous decoder stage)
        skip_channels (int): Number of channels from encoder skip connection
        out_channels (int): Number of output channels
        use_attention (bool): Whether to use attention in ConvBlock
        upsample_first (bool): Whether to upsample before concat (True) or after (False)
    """

    def __init__(self, in_channels, skip_channels, out_channels, use_attention=False, upsample_first=True):
        super().__init__()

        # Nearest neighbor upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample_first = upsample_first

        # After concatenating skip connection: in_channels + skip_channels
        total_channels = in_channels + skip_channels

        self.conv_block = ConvBlock(
            total_channels,
            out_channels,
            use_attention=use_attention
        )

    def forward(self, x, skip):
        """
        Args:
            x: Features from previous decoder stage [B, in_channels, H, W]
            skip: Skip connection from encoder [B, skip_channels, 2*H, 2*W] or [B, skip_channels, H, W]

        Returns:
            Upsampled and refined features [B, out_channels, 2*H, 2*W]
        """
        if self.upsample_first:
            # Standard UNet: upsample first, then concat with skip (which is at 2x resolution)
            x = self.upsample(x)
            x = torch.cat([x, skip], dim=1)
        else:
            # For bottleneck stage: concat with skip at same resolution, then upsample
            x = torch.cat([x, skip], dim=1)
            x = self.upsample(x)

        x = self.conv_block(x)
        return x


class CrossAttentionBottleneck(nn.Module):
    """
    Cross-attention bottleneck for fusing image and embedding features.

    Uses image features as queries and embedding features as keys/values.
    Allows spatial features to attend to global embedding information.

    Args:
        image_channels (int): Image feature channels
        embed_channels (int): Embedding feature channels (per embedding)
        num_heads (int): Number of attention heads
    """

    def __init__(self, image_channels=512, embed_channels=512, num_heads=8):
        super().__init__()

        self.image_channels = image_channels
        self.embed_channels = embed_channels
        self.num_heads = num_heads

        # Cross-attention: image queries attend to embedding keys/values
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=image_channels,
            num_heads=num_heads,
            batch_first=True
        )

        # Self-attention for refinement
        self.self_attn = nn.MultiheadAttention(
            embed_dim=image_channels,
            num_heads=num_heads,
            batch_first=True
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(image_channels)
        self.norm2 = nn.LayerNorm(image_channels)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(image_channels, image_channels * 4),
            nn.GELU(),
            nn.Linear(image_channels * 4, image_channels)
        )
        self.norm3 = nn.LayerNorm(image_channels)

    def forward(self, image_feat, clean_embed_feat, noisy_embed_feat):
        """
        Args:
            image_feat: [B, C, H, W] Image features from encoder
            clean_embed_feat: [B, C, H, W] Clean embedding spatial features
            noisy_embed_feat: [B, C, H, W] Noisy embedding spatial features

        Returns:
            fused_feat: [B, C, H, W] Fused features
        """
        B, C, H, W = image_feat.shape

        # Flatten spatial dimensions for attention: [B, H*W, C]
        image_flat = image_feat.flatten(2).transpose(1, 2)  # [B, H*W, C]
        clean_flat = clean_embed_feat.flatten(2).transpose(1, 2)
        noisy_flat = noisy_embed_feat.flatten(2).transpose(1, 2)

        # Concatenate embedding features
        embed_feat = torch.cat([clean_flat, noisy_flat], dim=1)  # [B, 2*H*W, C]

        # Cross-attention: image (query) attends to embeddings (key/value)
        attn_out, _ = self.cross_attn(
            query=image_flat,
            key=embed_feat,
            value=embed_feat
        )
        image_flat = self.norm1(image_flat + attn_out)

        # Self-attention for refinement
        self_attn_out, _ = self.self_attn(image_flat, image_flat, image_flat)
        image_flat = self.norm2(image_flat + self_attn_out)

        # Feed-forward network
        ffn_out = self.ffn(image_flat)
        image_flat = self.norm3(image_flat + ffn_out)

        # Reshape back to spatial: [B, C, H, W]
        output = image_flat.transpose(1, 2).reshape(B, C, H, W)

        return output


class InputProcessor(nn.Module):
    """
    Process multi-modal inputs into initial feature map.

    Concatenates original image and optional multi-scale features,
    then projects to base channel dimension.

    Args:
        base_channels (int): Base channel dimension for UNet
        strategy_channels (int): Number of channels from multi-scale strategy (0 if none)
    """

    def __init__(self, base_channels=64, strategy_channels=0):
        super().__init__()

        # Total input: original(3) + strategy_channels
        total_input_channels = 3 + strategy_channels

        self.input_conv = nn.Sequential(
            nn.Conv2d(total_input_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, original_image, multi_scale_images=None):
        """
        Args:
            original_image: [B, 3, H, W]
            multi_scale_images: [B, strategy_channels, H, W] or None

        Returns:
            features: [B, base_channels, H, W]
        """
        inputs = [original_image]

        if multi_scale_images is not None:
            inputs.append(multi_scale_images)

        x = torch.cat(inputs, dim=1)
        x = self.input_conv(x)

        return x


class EmbeddingEncoder(nn.Module):
    """
    Convert CLS embeddings to spatial feature maps.

    Projects embedding vector to spatial features using linear projection
    and refinement convolutions.

    Args:
        embed_dim (int): Input embedding dimension (e.g., 1024 for ViT-L16)
        spatial_size (int): Target spatial size (e.g., 7 for 224/32)
        channels (int): Output channel dimension
    """

    def __init__(self, embed_dim=1024, spatial_size=7, channels=512):
        super().__init__()

        self.embed_dim = embed_dim
        self.spatial_size = spatial_size
        self.channels = channels

        # Linear projection to spatial features
        self.proj = nn.Linear(embed_dim, channels * spatial_size ** 2)

        # Refinement convolutions
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, embeddings):
        """
        Args:
            embeddings: [B, embed_dim] CLS token embeddings

        Returns:
            features: [B, channels, spatial_size, spatial_size]
        """
        B = embeddings.size(0)

        # Project and reshape to spatial
        x = self.proj(embeddings.float())  # [B, channels * spatial_size^2]
        x = x.view(B, self.channels, self.spatial_size, self.spatial_size)

        # Refine
        x = self.refine(x)

        return x


class OutputHead(nn.Module):
    """
    Generate final noise tensor from decoder features.

    Args:
        in_channels (int): Input feature channels
        eps (float): Maximum perturbation budget
    """

    def __init__(self, in_channels=64, eps=16/255):
        super().__init__()

        self.eps = eps

        self.output = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        """
        Args:
            x: [B, in_channels, H, W] Decoder features

        Returns:
            noise: [B, 3, H, W] Bounded noise tensor
        """
        noise = torch.tanh(self.output(x))
        noise = noise * self.eps
        return noise


if __name__ == "__main__":
    print("Testing UNet components...")

    # Test ConvBlock
    print("\n=== Testing ConvBlock ===")
    block = ConvBlock(64, 128, use_attention=True)
    x = torch.randn(2, 64, 56, 56)
    y = block(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
    assert y.shape == (2, 128, 56, 56)

    # Test EncoderStage
    print("\n=== Testing EncoderStage ===")
    encoder = EncoderStage(64, 128, downsample=True, use_attention=False)
    x = torch.randn(2, 64, 224, 224)
    y = encoder(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
    assert y.shape == (2, 128, 112, 112)

    # Test DecoderStage
    print("\n=== Testing DecoderStage ===")
    decoder = DecoderStage(128, 128, 64, use_attention=False)
    x = torch.randn(2, 128, 56, 56)
    skip = torch.randn(2, 128, 112, 112)
    y = decoder(x, skip)
    print(f"Input: {x.shape}, Skip: {skip.shape}, Output: {y.shape}")
    assert y.shape == (2, 64, 112, 112)

    # Test CrossAttentionBottleneck
    print("\n=== Testing CrossAttentionBottleneck ===")
    bottleneck = CrossAttentionBottleneck(image_channels=512, embed_channels=512, num_heads=8)
    image_feat = torch.randn(2, 512, 7, 7)
    clean_embed = torch.randn(2, 512, 7, 7)
    noisy_embed = torch.randn(2, 512, 7, 7)
    fused = bottleneck(image_feat, clean_embed, noisy_embed)
    print(f"Image feat: {image_feat.shape}, Output: {fused.shape}")
    assert fused.shape == (2, 512, 7, 7)

    # Test InputProcessor
    print("\n=== Testing InputProcessor ===")
    processor = InputProcessor(base_channels=64, strategy_channels=15)
    original = torch.randn(2, 3, 224, 224)
    multi_scale = torch.randn(2, 15, 224, 224)
    features = processor(original, multi_scale)
    print(f"Inputs: original={original.shape}, multi_scale={multi_scale.shape}")
    print(f"Output: {features.shape}")
    assert features.shape == (2, 64, 224, 224)

    # Test EmbeddingEncoder
    print("\n=== Testing EmbeddingEncoder ===")
    embed_encoder = EmbeddingEncoder(embed_dim=1024, spatial_size=7, channels=512)
    embeddings = torch.randn(2, 1024)
    spatial_feat = embed_encoder(embeddings)
    print(f"Input: {embeddings.shape}, Output: {spatial_feat.shape}")
    assert spatial_feat.shape == (2, 512, 7, 7)

    # Test OutputHead
    print("\n=== Testing OutputHead ===")
    output_head = OutputHead(in_channels=64, eps=16/255)
    x = torch.randn(2, 64, 224, 224)
    noise = output_head(x)
    print(f"Input: {x.shape}, Output: {noise.shape}")
    print(f"Noise range: [{noise.min():.6f}, {noise.max():.6f}]")
    assert noise.shape == (2, 3, 224, 224)
    assert noise.abs().max() <= 16/255 + 1e-6

    print("\n✓ All UNet components tests passed!")
