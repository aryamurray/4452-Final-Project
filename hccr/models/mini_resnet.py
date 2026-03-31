"""MiniResNet for handwritten Chinese character recognition.

Custom residual CNN with 13 conv layers designed for fine-grained
stroke discrimination while staying edge-deployable (<3 MB INT8).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Basic residual block: two 3x3 convs with skip connection."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 1x1 projection shortcut when dimensions change
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class MiniResNetBackbone(nn.Module):
    """Custom residual backbone for HCCR.

    Architecture (96x96 input):
        Stem: Conv(1->32) -> BN -> ReLU -> MaxPool          # 48x48
        Stage 1: ResBlock(32->48) + ResBlock(48) -> MaxPool  # 24x24
        Stage 2: ResBlock(48->96) + ResBlock(96) -> MaxPool  # 12x12
        Stage 3: ResBlock(96->192) + ResBlock(192) -> MaxPool # 6x6
        AdaptiveAvgPool(1) -> Flatten                         # 192-dim

    6 residual blocks = 13 conv layers total.
    ~1.56M backbone params, ~8.8 MB FP32, ~3 MB INT8.
    """

    def __init__(self, out_dim: int = 192) -> None:
        super().__init__()
        self.out_dim = out_dim

        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.stage1 = nn.Sequential(
            ResBlock(32, 48),
            ResBlock(48, 48),
            nn.MaxPool2d(2),
        )

        self.stage2 = nn.Sequential(
            ResBlock(48, 96),
            ResBlock(96, 96),
            nn.MaxPool2d(2),
        )

        self.stage3 = nn.Sequential(
            ResBlock(96, out_dim),
            ResBlock(out_dim, out_dim),
            nn.MaxPool2d(2),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)
        return torch.flatten(x, 1)


class MiniResNet(nn.Module):
    """MiniResNet classifier for Chinese character recognition.

    Args:
        num_classes: Number of character classes
        backbone_dim: Output dimension of backbone (default: 192)
        dropout: Dropout probability (default: 0.5)
    """

    def __init__(
        self,
        num_classes: int,
        backbone_dim: int = 192,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.backbone_dim = backbone_dim
        self.backbone = MiniResNetBackbone(out_dim=backbone_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(backbone_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        features = self.dropout(features)
        return self.classifier(features)

    def get_num_params(self) -> Tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


class MiniResNetJoint(nn.Module):
    """MiniResNet with character + radical/structure/stroke heads.

    Supports differentiable symbolic constraint scoring when constraint
    tensors (master_table, radical_mask) are loaded.

    Args:
        num_classes: Number of character classes
        num_radicals: Number of radical classes
        num_structures: Number of structure types (default: 13)
        num_strokes: Number of stroke count bins (default: 30)
        backbone_dim: Output dimension of backbone (default: 192)
        dropout: Dropout probability (default: 0.5)
        symbolic_temperature: Exponent on symbolic prior (default: 1.0)
    """

    def __init__(
        self,
        num_classes: int,
        num_radicals: int,
        num_structures: int = 13,
        num_strokes: int = 30,
        backbone_dim: int = 192,
        dropout: float = 0.5,
        symbolic_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_radicals = num_radicals
        self.num_structures = num_structures
        self.num_strokes = num_strokes
        self.backbone_dim = backbone_dim
        self.symbolic_temperature = symbolic_temperature

        self.backbone = MiniResNetBackbone(out_dim=backbone_dim)
        self.dropout = nn.Dropout(p=dropout)

        self.char_head = nn.Linear(backbone_dim, num_classes)
        self.radical_head = nn.Linear(backbone_dim, num_radicals)
        self.structure_head = nn.Linear(backbone_dim, num_structures)
        self.stroke_head = nn.Linear(backbone_dim, num_strokes)

        # Constraint tensors for symbolic scoring
        self.register_buffer("master_table", None)
        self.register_buffer("radical_mask", None)
        self.register_buffer("_rad_counts", None)

    def load_constraint_tensors(
        self,
        master_table_path: str | Path,
        radical_mask_path: str | Path,
    ) -> None:
        """Load prebuilt constraint tensors for symbolic scoring."""
        master_table = torch.load(master_table_path, map_location="cpu", weights_only=True)
        radical_mask = torch.load(radical_mask_path, map_location="cpu", weights_only=True)

        if radical_mask.is_sparse:
            radical_mask = radical_mask.to_dense()
        radical_mask = radical_mask.half()

        assert master_table.shape == (self.num_classes, 3)
        assert radical_mask.shape == (self.num_classes, self.num_radicals)

        rad_counts = radical_mask.float().sum(dim=1).clamp(min=1)

        self.register_buffer("master_table", master_table)
        self.register_buffer("radical_mask", radical_mask)
        self.register_buffer("_rad_counts", rad_counts)

    def _radical_scores(self, radical_probs: torch.Tensor) -> torch.Tensor:
        """Compute radical match scores for all candidate characters."""
        match_sum = radical_probs.half() @ self.radical_mask.T
        return (match_sum / self._rad_counts.half().unsqueeze(0)).float()

    def forward(
        self, x: torch.Tensor, use_symbolic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with optional symbolic constraint scoring.

        Args:
            x: Input images of shape (B, 1, H, W)
            use_symbolic: If True, compute combined_probs via symbolic layer

        Returns:
            Dictionary with char_logits, radical_logits, structure_logits,
            stroke_logits, and optionally combined_probs.
        """
        features = self.dropout(self.backbone(x))

        char_logits = self.char_head(features)
        radical_logits = self.radical_head(features)
        structure_logits = self.structure_head(features)
        stroke_logits = self.stroke_head(features)

        result = {
            "char_logits": char_logits,
            "radical_logits": radical_logits,
            "structure_logits": structure_logits,
            "stroke_logits": stroke_logits,
        }

        if use_symbolic:
            assert self.master_table is not None, (
                "Constraint tensors not loaded. Call load_constraint_tensors() first."
            )

            char_probs = F.softmax(char_logits, dim=-1)
            structure_probs = F.softmax(structure_logits, dim=-1)
            stroke_probs = F.softmax(stroke_logits, dim=-1)
            radical_probs = torch.sigmoid(radical_logits)

            s_scores = structure_probs[:, self.master_table[:, 0]]
            st_scores = stroke_probs[:, self.master_table[:, 2]]
            r_scores = self._radical_scores(radical_probs)

            symbolic_prior = (s_scores * r_scores * st_scores).clamp(min=1e-10)
            if self.symbolic_temperature != 1.0:
                symbolic_prior = symbolic_prior ** self.symbolic_temperature

            combined = char_probs * symbolic_prior
            combined = combined / combined.sum(dim=-1, keepdim=True).clamp(min=1e-8)

            result["combined_probs"] = combined

        return result

    def freeze_backbone_and_char_head(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.char_head.parameters():
            p.requires_grad = False

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad = True

    def load_pretrained_backbone(self, checkpoint_path: str) -> None:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state = ckpt.get("model_state_dict", ckpt)
        backbone_state = {k: v for k, v in state.items() if k.startswith("backbone.")}
        char_head_state = {}
        for k, v in state.items():
            if k.startswith("classifier."):
                char_head_state[k.replace("classifier.", "char_head.")] = v
        self.load_state_dict({**backbone_state, **char_head_state}, strict=False)

    def load_joint_checkpoint(
        self, checkpoint_path: str | Path, device: Optional[torch.device] = None,
    ) -> None:
        """Load a joint checkpoint, handling stroke head shape mismatch."""
        map_location = device if device is not None else "cpu"
        ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
        state = ckpt.get("model_state_dict", ckpt)

        stroke_weight_key = "stroke_head.weight"
        reinit_stroke = False
        if stroke_weight_key in state:
            old_shape = state[stroke_weight_key].shape
            expected_shape = (self.num_strokes, self.backbone_dim)
            if old_shape != expected_shape:
                keys_to_remove = [k for k in state if k.startswith("stroke_head.")]
                for k in keys_to_remove:
                    del state[k]
                reinit_stroke = True

        buffer_keys = {"master_table", "radical_mask", "_rad_counts"}
        state = {k: v for k, v in state.items() if k not in buffer_keys}

        self.load_state_dict(state, strict=False)

        if reinit_stroke:
            nn.init.kaiming_normal_(self.stroke_head.weight)
            nn.init.zeros_(self.stroke_head.bias)

    def get_num_params(self) -> Tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
