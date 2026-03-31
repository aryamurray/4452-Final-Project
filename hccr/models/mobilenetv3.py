"""MobileNetV3 transfer learning model for HCCR.

Pretrained MobileNetV3-Small adapted for grayscale Chinese character recognition.
"""

import torch
import torch.nn as nn
from typing import Tuple
from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights


class MobileNetV3Wrapper(nn.Module):
    """MobileNetV3-Small wrapper for Chinese character recognition.

    Uses ImageNet-pretrained MobileNetV3-Small with modifications:
    - First conv layer adapted for 1-channel grayscale input
    - Final classifier layer replaced for num_classes output
    - Dropout added before final linear layer

    Architecture:
        MobileNetV3-Small backbone (pretrained on ImageNet)
        Classifier: Linear(576, 1024) -> Hardswish -> Dropout -> Linear(1024, num_classes)

    Args:
        num_classes: Number of character classes
        dropout: Dropout probability (default: 0.5)
        pretrained: Whether to use ImageNet pretrained weights (default: True)

    Attributes:
        model: Modified MobileNetV3-Small model
    """

    def __init__(
        self,
        num_classes: int,
        dropout: float = 0.5,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        # Load pretrained MobileNetV3-Small
        if pretrained:
            weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
            self.model = models.mobilenet_v3_small(weights=weights)
        else:
            self.model = models.mobilenet_v3_small(weights=None)

        # Modify first conv layer for grayscale input (1 channel -> 3 channel RGB)
        # Average pretrained RGB weights to get grayscale weights
        self._adapt_first_conv_for_grayscale()

        # Modify classifier for character recognition
        # Original: Sequential(Linear(576, 1024), Hardswish, Dropout(0.2), Linear(1024, 1000))
        # New: Replace last Linear and update Dropout
        in_features = self.model.classifier[0].in_features  # 576
        hidden_features = self.model.classifier[0].out_features  # 1024

        self.model.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(hidden_features, num_classes),
        )

    def _adapt_first_conv_for_grayscale(self) -> None:
        """Adapt first conv layer from 3-channel RGB to 1-channel grayscale.

        Averages pretrained RGB weights across channels to initialize grayscale weights.
        This preserves learned features while accepting single-channel input.
        """
        # First conv is at model.features[0][0]
        first_conv = self.model.features[0][0]

        if first_conv.in_channels == 3:
            # Get pretrained RGB weights: (out_channels, 3, kernel_h, kernel_w)
            pretrained_weights = first_conv.weight.data

            # Average across RGB channels: (out_channels, 1, kernel_h, kernel_w)
            grayscale_weights = pretrained_weights.mean(dim=1, keepdim=True)

            # Create new conv layer with 1 input channel
            new_conv = nn.Conv2d(
                in_channels=1,
                out_channels=first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=first_conv.bias is not None,
            )

            # Initialize with averaged weights
            new_conv.weight.data = grayscale_weights
            if first_conv.bias is not None:
                new_conv.bias.data = first_conv.bias.data

            # Replace first conv layer
            self.model.features[0][0] = new_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for character classification.

        Args:
            x: Input grayscale images of shape (B, 1, 64, 64)
               Note: Will be resized to 224x224 in preprocessing

        Returns:
            Character logits of shape (B, num_classes)
        """
        # Model now accepts 1-channel input directly
        return self.model(x)

    def get_num_params(self) -> Tuple[int, int]:
        """Calculate total and trainable parameters.

        Returns:
            Tuple of (total_params, trainable_params)
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
