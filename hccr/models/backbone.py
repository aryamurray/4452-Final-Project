"""TinyCNN backbone for HCCR models.

Shared convolutional backbone that extracts 256-dimensional features from
grayscale images. Used by all TinyCNN variants. Supports optional CBAM
attention before global average pooling.
"""

import torch
import torch.nn as nn
from typing import Tuple


class ChannelAttention(nn.Module):
    """Channel attention from CBAM: squeeze-and-excite with avg+max pooling."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        scale = torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * scale


class SpatialAttention(nn.Module):
    """Spatial attention from CBAM: conv on channel-pooled feature maps."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * scale


class CBAM(nn.Module):
    """Convolutional Block Attention Module (channel + spatial)."""

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7) -> None:
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


class TinyCNNBackbone(nn.Module):
    """Lightweight CNN backbone for Chinese character feature extraction.

    Architecture (with 96x96 input):
        Conv(1->32) -> BN -> ReLU -> MaxPool  # 96->48
        Conv(32->64) -> BN -> ReLU -> MaxPool  # 48->24
        Conv(64->128) -> BN -> ReLU -> MaxPool  # 24->12
        Conv(128->256) -> BN -> ReLU -> MaxPool  # 12->6
        [CBAM (optional)]                        # 6x6 attention
        AdaptiveAvgPool(1x1) -> Flatten          # -> 256-dim

    Args:
        backbone_dim: Output feature dimension (default: 256)
        use_cbam: Whether to add CBAM attention before GAP (default: False)
    """

    def __init__(self, backbone_dim: int = 256, use_cbam: bool = False) -> None:
        super().__init__()
        self.backbone_dim = backbone_dim
        self.use_cbam = use_cbam

        # Feature extraction layers
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4
            nn.Conv2d(128, backbone_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(backbone_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Optional CBAM before pooling
        self.cbam = CBAM(backbone_dim) if use_cbam else None

        # Global pooling
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input images.

        Args:
            x: Input tensor of shape (B, 1, H, W)

        Returns:
            Feature tensor of shape (B, backbone_dim)
        """
        x = self.features(x)
        if self.cbam is not None:
            x = self.cbam(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x

    def get_num_params(self) -> Tuple[int, int]:
        """Calculate total and trainable parameters.

        Returns:
            Tuple of (total_params, trainable_params)
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
