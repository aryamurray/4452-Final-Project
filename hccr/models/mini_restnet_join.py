"""
MiniResNetJoint: Multi-head residual CNN with learned symbolic re-ranker
for Chinese character recognition.

Architecture:
  - MiniResNet backbone (stem + 3 ResBlocks, 64→128→256 channels, ~600K params)
  - 7 auxiliary heads for structural feature extraction (~500K params)
  - Top-K symbolic re-ranker: computes per-candidate symbolic match features
    from lookup tables, then scores via a small MLP (~258 learned params)

Two inference modes:
  'heads_only': raw character head output (Phase 1)
  'combined':   heads + top-K reranked logits (Phase 2+3)

Three training phases:
  Phase 1: Train all heads independently (standard multi-task)
  Phase 2: Add reranked combined loss (teaches backbone structural consistency)
  Phase 3: Freeze backbone, tune reranker MLP only

Targets <3MB INT8 quantized total (neural network + lookup tables).

Author: Arya
Project: RadicalLight
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Building blocks
# =============================================================================


class ResBlock(nn.Module):
    """Standard residual block with optional 1x1 channel projection."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)


# =============================================================================
# Backbone
# =============================================================================


class MiniResNetBackbone(nn.Module):
    """
    MiniResNet backbone: stem + 3 residual blocks.

    Input:  (B, 1, 64, 64) grayscale character image
    Output:
        features: (B, 256)       global feature vector (after GAP)
        spatial:  (B, 256, 8, 8) spatial feature map (before GAP)

    Architecture:
        Conv2d(1->64) + BN + ReLU     64x64   (stem)
        MaxPool(2)                     32x32
        ResBlock(64->64)               32x32
        MaxPool(2)                     16x16
        ResBlock(64->128)              16x16
        MaxPool(2)                     8x8
        ResBlock(128->256)             8x8
        GAP                            256-dim

    ~600K params
    """

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block1 = ResBlock(64, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.block2 = ResBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.block3 = ResBlock(128, 256)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.block3(x)
        spatial = x                          # (B, 256, 8, 8)
        features = self.gap(x).flatten(1)    # (B, 256)
        return features, spatial


# =============================================================================
# Differentiable symbolic constraint layer
# =============================================================================


def _build_radical_mask(
    radical_table: Dict[str, list], num_classes: int, num_radicals: int
) -> torch.Tensor:
    """Build (num_classes, num_radicals) binary mask from radical_table.

    Stored as uint8 (1 byte per entry) instead of float32 (4 bytes).
    Cast to float32 on-the-fly during forward pass.
    """
    mask = torch.zeros(num_classes, num_radicals, dtype=torch.uint8)
    for char_idx_str, radical_indices in radical_table.items():
        char_idx = int(char_idx_str)
        if char_idx < num_classes:
            for r in radical_indices:
                if r < num_radicals:
                    mask[char_idx, r] = 1
    return mask


def _build_int_tensor(
    table: Dict[str, int], num_classes: int, default: int = 0
) -> torch.Tensor:
    """Build (num_classes,) long tensor from char_idx -> int mapping."""
    t = torch.full((num_classes,), default, dtype=torch.long)
    for char_idx_str, val in table.items():
        idx = int(char_idx_str)
        if idx < num_classes:
            t[idx] = val
    return t


def _build_float_tensor(
    table: Dict[str, list], num_classes: int, dim: int
) -> torch.Tensor:
    """Build (num_classes, dim) float tensor from char_idx -> list mapping."""
    t = torch.zeros(num_classes, dim)
    for char_idx_str, vals in table.items():
        idx = int(char_idx_str)
        if idx < num_classes:
            for j, v in enumerate(vals[:dim]):
                t[idx, j] = float(v)
    return t


class SymbolicConstraintLayer(nn.Module):
    """
    Differentiable neurosymbolic constraint layer.

    Encodes structural knowledge about Chinese characters as fixed tensors
    (registered buffers -- on device, saved with model, NOT trainable).
    All operations are differentiable: gradients flow through the auxiliary
    head outputs back into the backbone.

    The key insight: we don't learn the constraints. We learn features
    that satisfy known constraints. The lookup tables act as bridges,
    connecting what the model learns about one character's radicals to
    all characters sharing those same radicals.

    Constraint scores computed (all differentiable, all shape (B, C)):
        radical_score:     avg P(radical) for each candidate's known radicals
        structure_score:   P(correct structure type) for each candidate
        stroke_score:      P(correct stroke count bin) for each candidate
        stroke_type_score: cosine similarity of stroke type profiles
    """

    def __init__(
        self,
        num_classes: int = 3755,
        num_radicals: int = 214,
        num_structure_types: int = 13,
        num_stroke_count_bins: int = 30,
        num_stroke_types: int = 6,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_radicals = num_radicals

        # Placeholders -- call load_tables() to populate
        self.register_buffer("radical_mask", torch.zeros(num_classes, num_radicals, dtype=torch.uint8))
        self.register_buffer("radical_counts", torch.ones(num_classes))
        self.register_buffer("structure_label", torch.zeros(num_classes, dtype=torch.long))
        self.register_buffer("stroke_count_label", torch.zeros(num_classes, dtype=torch.long))
        self.register_buffer("stroke_type_sig", torch.zeros(num_classes, num_stroke_types))
        self.register_buffer("tables_loaded", torch.tensor(False))

    def load_tables(
        self,
        radical_table: Dict[str, list],
        structure_table: Dict[str, int],
        stroke_count_table: Dict[str, int],
        stroke_type_table: Optional[Dict[str, list]] = None,
    ):
        """
        Populate constraint tensors from lookup table dictionaries.

        Args:
            radical_table:      char_idx (str) -> list of radical indices
            structure_table:    char_idx (str) -> structure type index (0-12)
            stroke_count_table: char_idx (str) -> stroke count bin index (0-29)
            stroke_type_table:  char_idx (str) -> [h, v, lf, rf, dot, turn] counts
        """
        device = self.radical_mask.device

        self.radical_mask = _build_radical_mask(
            radical_table, self.num_classes, self.num_radicals
        ).to(device)
        self.radical_counts = self.radical_mask.float().sum(dim=1).clamp(min=1).to(device)
        self.structure_label = _build_int_tensor(
            structure_table, self.num_classes
        ).to(device)
        self.stroke_count_label = _build_int_tensor(
            stroke_count_table, self.num_classes
        ).to(device)
        if stroke_type_table:
            self.stroke_type_sig = _build_float_tensor(
                stroke_type_table, self.num_classes, dim=6
            ).to(device)
        self.tables_loaded = torch.tensor(True, device=device)

    def load_tables_from_json(self, path: str):
        """
        Load from radical_table.json format:
        {
            "0": {"radicals": [45, 120], "structure": 0,
                  "stroke_count": 8, "stroke_types": [2,1,1,1,2,1]},
            ...
        }
        """
        with open(path, "r") as f:
            data = json.load(f)

        radical_table, structure_table = {}, {}
        stroke_count_table, stroke_type_table = {}, {}

        for idx_str, info in data.items():
            if isinstance(info, dict):
                if "radicals" in info:
                    radical_table[idx_str] = info["radicals"]
                if "structure" in info:
                    structure_table[idx_str] = info["structure"]
                if "stroke_count" in info:
                    stroke_count_table[idx_str] = min(info["stroke_count"] - 1, 29)
                if "stroke_types" in info:
                    stroke_type_table[idx_str] = info["stroke_types"]
            elif isinstance(info, list):
                radical_table[idx_str] = info

        self.load_tables(
            radical_table, structure_table, stroke_count_table, stroke_type_table
        )

    def forward(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Score every character candidate against auxiliary head predictions.

        All operations are differentiable -- gradients flow back through
        the auxiliary heads into the backbone.

        Args:
            outputs: model output dict with auxiliary head tensors

        Returns:
            Dict of (B, num_classes) constraint score tensors
        """
        B = outputs["char_logits"].shape[0]
        C = self.num_classes

        # --- Radical consistency ---
        # radical_probs: (B, 214), radical_mask: (C, 214)
        # Score = mean P(r) for each char's known radicals
        radical_probs = torch.sigmoid(outputs["radical_logits"])      # (B, R)
        radical_mask_f = self.radical_mask.float()                     # uint8 -> float32
        radical_score = radical_probs @ radical_mask_f.T               # (B, C)
        radical_score = radical_score / self.radical_counts.unsqueeze(0)

        # --- Structure consistency ---
        # For each candidate, gather P(its known structure type)
        structure_probs = F.softmax(outputs["structure"], dim=-1)     # (B, 13)
        # Index into structure_probs using each char's known structure label
        # structure_label: (C,) -> expand to (B, C)
        structure_score = structure_probs[:, self.structure_label]     # (B, C)

        # --- Stroke count consistency ---
        stroke_probs = F.softmax(outputs["stroke_count"], dim=-1)     # (B, 30)
        stroke_score = stroke_probs[:, self.stroke_count_label]       # (B, C)

        # --- Stroke type similarity ---
        # Cosine similarity between predicted and each char's known signature
        pred_norm = F.normalize(outputs["stroke_types"], dim=-1)      # (B, 6)
        sig_norm = F.normalize(self.stroke_type_sig, dim=-1)          # (C, 6)
        stroke_type_score = pred_norm @ sig_norm.T                    # (B, C)

        return {
            "radical_score": radical_score,
            "structure_score": structure_score,
            "stroke_score": stroke_score,
            "stroke_type_score": stroke_type_score,
        }


class SymbolicReranker(nn.Module):
    """Top-K symbolic re-ranker.

    Instead of scoring all classes with soft constraint sums, this module:
    1. Takes the top-K candidates from the character head
    2. Computes per-candidate symbolic match features using lookup tables
    3. Passes features through a small MLP to produce re-ranking scores
    4. Scatters re-ranked scores back into full logits

    The MLP can learn non-linear rules like "structure mismatch is a
    dealbreaker" or "1 missing radical is fine but 3 is fatal."
    """

    def __init__(
        self,
        num_classes: int = 3755,
        num_radicals: int = 214,
        num_structure_types: int = 13,
        num_stroke_count_bins: int = 30,
        num_stroke_types: int = 6,
        top_k: int = 5,
        hidden_dim: int = 32,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_radicals = num_radicals
        self.top_k = top_k

        # Lookup table buffers (populated by load_tables)
        self.register_buffer(
            "radical_mask",
            torch.zeros(num_classes, num_radicals, dtype=torch.uint8),
        )
        self.register_buffer("radical_counts", torch.ones(num_classes))
        self.register_buffer(
            "structure_label", torch.zeros(num_classes, dtype=torch.long)
        )
        self.register_buffer(
            "stroke_count_label", torch.zeros(num_classes, dtype=torch.long)
        )
        self.register_buffer(
            "stroke_type_sig", torch.zeros(num_classes, num_stroke_types)
        )
        self.register_buffer("tables_loaded", torch.tensor(False))

        # Re-ranking MLP: 5 symbolic features -> 1 score
        # (neural score excluded — it's already in the base logits)
        self.mlp = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.reranker_weight = nn.Parameter(torch.tensor(0.01))

    def load_tables(
        self,
        radical_table: Dict[str, list],
        structure_table: Dict[str, int],
        stroke_count_table: Dict[str, int],
        stroke_type_table: Optional[Dict[str, list]] = None,
    ):
        """Populate constraint tensors from lookup table dictionaries."""
        device = self.radical_mask.device

        self.radical_mask = _build_radical_mask(
            radical_table, self.num_classes, self.num_radicals
        ).to(device)
        self.radical_counts = (
            self.radical_mask.float().sum(dim=1).clamp(min=1).to(device)
        )
        self.structure_label = _build_int_tensor(
            structure_table, self.num_classes
        ).to(device)
        self.stroke_count_label = _build_int_tensor(
            stroke_count_table, self.num_classes
        ).to(device)
        if stroke_type_table:
            self.stroke_type_sig = _build_float_tensor(
                stroke_type_table, self.num_classes, dim=6
            ).to(device)
        self.tables_loaded = torch.tensor(True, device=device)

    def _compute_features(
        self,
        outputs: Dict[str, torch.Tensor],
        candidate_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute 6 symbolic match features for each candidate.

        Args:
            outputs: model output dict (auxiliary head predictions)
            candidate_indices: (B, K) indices of top-K candidates

        Returns:
            (B, K, 6) feature tensor
        """
        B, K = candidate_indices.shape

        # 1-2. Radical match/false-alarm ratios
        radical_preds = (torch.sigmoid(outputs["radical_logits"]) > 0.5)  # (B, R) bool
        # Get radical mask for each candidate: (B, K, R)
        cand_radical_mask = self.radical_mask[candidate_indices]  # (B, K, R) uint8
        cand_radical_mask_f = cand_radical_mask.float()
        radical_preds_expanded = radical_preds.unsqueeze(1).float()  # (B, 1, R)

        # Match ratio: fraction of candidate's radicals that were detected
        detected_and_expected = (radical_preds_expanded * cand_radical_mask_f).sum(-1)
        cand_radical_counts = cand_radical_mask_f.sum(-1).clamp(min=1)
        radical_match_ratio = detected_and_expected / cand_radical_counts  # (B, K)

        # False alarm ratio: fraction of detected radicals NOT in candidate
        num_detected = radical_preds.float().sum(-1, keepdim=True)  # (B, 1)
        false_alarms = (radical_preds_expanded * (1 - cand_radical_mask_f)).sum(-1)
        radical_false_ratio = false_alarms / num_detected.clamp(min=1)  # (B, K)

        # 3. Structure match probability
        structure_probs = F.softmax(outputs["structure"], dim=-1)  # (B, 13)
        cand_structure = self.structure_label[candidate_indices]  # (B, K)
        structure_match = structure_probs.gather(1, cand_structure)  # (B, K)

        # 4. Stroke count distance
        stroke_pred = outputs["stroke_count"].argmax(dim=-1)  # (B,)
        cand_stroke = self.stroke_count_label[candidate_indices]  # (B, K)
        stroke_distance = (
            (stroke_pred.unsqueeze(1) - cand_stroke).abs().float() / 29.0
        )  # normalized to [0, 1]

        # 5. Stroke type cosine similarity
        pred_norm = F.normalize(outputs["stroke_types"], dim=-1)  # (B, 6)
        cand_sig = self.stroke_type_sig[candidate_indices]  # (B, K, 6)
        cand_sig_norm = F.normalize(cand_sig, dim=-1)
        stroke_type_cos = (
            pred_norm.unsqueeze(1) * cand_sig_norm
        ).sum(-1)  # (B, K)

        # Stack: (B, K, 5) — purely symbolic features, no neural score
        features = torch.stack([
            radical_match_ratio,
            radical_false_ratio,
            structure_match,
            stroke_distance,
            stroke_type_cos,
        ], dim=-1)

        return features

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Re-rank top-K candidates using symbolic features.

        Args:
            outputs: model output dict with char_logits and auxiliary heads

        Returns:
            (B, C) combined logits with top-K re-ranked
        """
        char_logits = outputs["char_logits"]  # (B, C)
        B, C = char_logits.shape
        K = min(self.top_k, C)

        # Get top-K candidate indices (no gradient needed for selection)
        _, top_indices = char_logits.topk(K, dim=-1)  # (B, K)

        # Compute symbolic features (detached from aux heads)
        detached_outputs = {
            k: v.detach() if k != "char_logits" else v
            for k, v in outputs.items()
        }
        features = self._compute_features(detached_outputs, top_indices)  # (B, K, 6)

        # MLP re-ranking scores
        rerank_scores = self.mlp(features).squeeze(-1)  # (B, K)

        # Get original logits at top-K positions
        top_logits = char_logits.gather(1, top_indices)  # (B, K)

        # Combined scores for top-K
        combined_top = top_logits + self.reranker_weight * rerank_scores  # (B, K)

        # Scatter back into full logits
        combined_logits = char_logits.clone()
        combined_logits.scatter_(1, top_indices, combined_top)

        return combined_logits


# =============================================================================
# Full model
# =============================================================================


class MiniResNetJoint(nn.Module):
    """
    Multi-head model with differentiable neurosymbolic constraint layer.

    Heads (from 256-dim global features):
        A. Character classification  (256 -> 3755, softmax)       ~962K
        B. Radical detection          (256 -> 214, sigmoid)        ~55K
        C. Stroke count               (256 -> 30, classification)  ~8K
        D. Stroke types               (256 -> 6, regression)       ~2K
        E. Structure type             (256 -> 13, softmax)         ~3K
        F. Density profile            (256 -> 9, regression)       ~2K

    Heads (from 8x8 spatial features):
        G. Quadrant radicals          (shared 64->214, sigmoid)    ~30K

    Symbolic layer: 0 learned params (fixed lookup table buffers)

    Total: ~600K backbone + ~1060K heads = ~1.66M params
    INT8 quantized: ~1.66 MB
    """

    class SpatialRadicalHead(nn.Module):
        """Detect radicals from spatial features using per-radical 1x1 conv filters.

        Each output channel learns to fire when its radical's visual pattern
        appears anywhere in the 8x8 spatial feature map. Global max-pool
        selects the strongest activation per radical.
        """

        def __init__(self, in_channels: int = 256, mid_channels: int = 128,
                     num_radicals: int = 605):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, num_radicals, 1),
            )

        def forward(self, spatial: torch.Tensor) -> torch.Tensor:
            """spatial: (B, 256, 8, 8) -> (B, num_radicals)"""
            x = self.conv(spatial)       # (B, R, 8, 8)
            return x.amax(dim=(-2, -1))  # (B, R)

    def __init__(
        self,
        num_classes: int = 3755,
        num_radicals: int = 214,
        num_structure_types: int = 13,
        num_stroke_count_bins: int = 30,
        num_stroke_types: int = 6,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_radicals = num_radicals

        # --- Backbone ---
        self.backbone = MiniResNetBackbone()

        # --- Head A: Character classification ---
        self.head_char = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        # --- Head B: Radical detection (multi-label, spatial) ---
        self.head_radical = self.SpatialRadicalHead(256, 128, num_radicals)

        # --- Head C: Stroke count (30-bin classification) ---
        self.head_stroke_count = nn.Linear(256, num_stroke_count_bins)

        # --- Head D: Stroke types (6 basic types, count regression) ---
        self.head_stroke_types = nn.Linear(256, num_stroke_types)

        # --- Head E: Structure type (13-class) ---
        self.head_structure = nn.Linear(256, num_structure_types)

        # --- Head F: Density profile (3x3 grid) ---
        self.head_density = nn.Linear(256, 9)

        # --- Head G: Quadrant radical detection ---
        self.quadrant_pool = nn.AdaptiveAvgPool2d(2)        # 8x8 -> 2x2
        self.quadrant_proj = nn.Conv2d(256, 64, 1, bias=False)  # channel reduction
        self.head_quadrant_radical = nn.Linear(64, num_radicals)  # shared across 4 quadrants

        # --- Symbolic re-ranker (top-K MLP re-ranking with lookup tables) ---
        self.reranker = SymbolicReranker(
            num_classes=num_classes,
            num_radicals=num_radicals,
            num_structure_types=num_structure_types,
            num_stroke_count_bins=num_stroke_count_bins,
            num_stroke_types=num_stroke_types,
            top_k=5,
        )

    # -----------------------------------------------------------------
    # Table loading
    # -----------------------------------------------------------------

    def load_symbolic_tables(self, path: str):
        """Load symbolic constraint tables from JSON file."""
        # Parse JSON and call reranker.load_tables()
        with open(path, "r") as f:
            data = json.load(f)
        radical_table, structure_table = {}, {}
        stroke_count_table, stroke_type_table = {}, {}
        for idx_str, info in data.items():
            if isinstance(info, dict):
                if "radicals" in info:
                    radical_table[idx_str] = info["radicals"]
                if "structure" in info:
                    structure_table[idx_str] = info["structure"]
                if "stroke_count" in info:
                    stroke_count_table[idx_str] = min(info["stroke_count"] - 1, 29)
                if "stroke_types" in info:
                    stroke_type_table[idx_str] = info["stroke_types"]
            elif isinstance(info, list):
                radical_table[idx_str] = info
        self.reranker.load_tables(
            radical_table, structure_table, stroke_count_table, stroke_type_table
        )

    # -----------------------------------------------------------------
    # Pretrained backbone loading
    # -----------------------------------------------------------------

    def load_pretrained_backbone(self, checkpoint_path: str):
        """
        Load pretrained single-head MiniResNet weights into backbone.

        Supports checkpoints from:
          - Single-head MiniResNet (keys like stem.0.weight, block1.conv1.weight)
          - Wrapped checkpoint dicts (model_state_dict key)
        """
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        # Try backbone-prefixed keys first
        backbone_keys = {
            k: v for k, v in state_dict.items() if k.startswith("backbone.")
        }
        if not backbone_keys:
            # Map unprefixed keys
            backbone_keys = {
                f"backbone.{k}": v
                for k, v in state_dict.items()
                if any(k.startswith(p) for p in ("stem", "block", "pool", "gap"))
            }

        missing, unexpected = self.load_state_dict(backbone_keys, strict=False)
        loaded = len(backbone_keys) - len(unexpected)
        print(f"Loaded {loaded} backbone parameters from {checkpoint_path}")
        truly_missing = [k for k in missing if k.startswith("backbone.")]
        if truly_missing:
            print(f"  Warning: missing backbone keys: {truly_missing}")

    # -----------------------------------------------------------------
    # Forward pass
    # -----------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        mode: str = "heads_only",
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: (B, 1, 64, 64) grayscale character images
            mode:
                'heads_only' -- raw head outputs only (Phase 1)
                'combined'   -- heads + top-K reranked logits (Phase 2+3)

        Returns:
            Dict with head outputs and optionally combined_logits.
        """
        features, spatial = self.backbone(x)

        outputs = {
            "char_logits": self.head_char(features),
            "radical_logits": self.head_radical(spatial),
            "stroke_count": self.head_stroke_count(features),
            "stroke_types": self.head_stroke_types(features),
            "structure": self.head_structure(features),
            "density": torch.sigmoid(self.head_density(features)),
        }

        # Quadrant radical detection
        quad_feat = self.quadrant_proj(spatial)                            # (B,64,8,8)
        quad_feat = self.quadrant_pool(quad_feat)                          # (B,64,2,2)
        B = quad_feat.shape[0]
        quad_feat = quad_feat.flatten(2).permute(0, 2, 1)                  # (B,4,64)
        outputs["quad_radicals"] = self.head_quadrant_radical(quad_feat)   # (B,4,214)

        if mode == "heads_only":
            return outputs

        # Top-K symbolic re-ranking
        # The reranker internally detaches auxiliary outputs, computes per-candidate
        # symbolic match features, and uses an MLP to re-rank top-K candidates.
        outputs["combined_logits"] = self.reranker(outputs)
        return outputs

    # -----------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        use_symbolic: bool = True,
        top_k: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference convenience method.

        Returns:
            top_k_indices: (B, top_k)
            top_k_scores:  (B, top_k)
        """
        self.eval()
        mode = "combined" if use_symbolic else "heads_only"
        outputs = self.forward(x, mode=mode)

        logits = outputs.get("combined_logits", outputs["char_logits"])
        probs = F.softmax(logits, dim=-1)
        return probs.topk(top_k, dim=-1)


# =============================================================================
# Loss computation
# =============================================================================


class RadicalLightLoss(nn.Module):
    """
    Combined loss for all training phases.

    Phase 1 (heads_only):
        Multi-task loss on all 7 heads independently.

    Phase 2 (combined):
        Phase 1 losses + CE on combined (symbolically-reranked) logits.
        This teaches the backbone to produce features that are jointly
        consistent with structural knowledge.

    Phase 3 (tune_weights):
        Only the combined loss. Backbone frozen, only reranker MLP
        and reranker_weight are updated.
    """

    def __init__(
        self,
        w_char: float = 1.0,
        w_radical: float = 0.5,
        w_stroke_count: float = 0.3,
        w_stroke_types: float = 0.2,
        w_structure: float = 0.3,
        w_density: float = 0.1,
        w_quad_radical: float = 0.3,
        w_combined: float = 0.5,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.weights = {
            "char": w_char,
            "radical": w_radical,
            "stroke_count": w_stroke_count,
            "stroke_types": w_stroke_types,
            "structure": w_structure,
            "density": w_density,
            "quad_radical": w_quad_radical,
            "combined": w_combined,
        }

        self.ce_char = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.bce_radical = nn.BCEWithLogitsLoss()
        self.ce_stroke_count = nn.CrossEntropyLoss()
        self.mse_stroke_types = nn.MSELoss()
        self.ce_structure = nn.CrossEntropyLoss()
        self.mse_density = nn.MSELoss()
        self.bce_quad_radical = nn.BCEWithLogitsLoss()
        self.ce_combined = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        phase: str = "phase1",
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss.

        Args:
            outputs: model output dict
            targets: dict with keys:
                char_label      (B,)       character class index
                radical_label   (B, 214)   multi-hot radical vector
                stroke_count    (B,)       stroke count bin index
                stroke_types    (B, 6)     stroke type counts
                structure       (B,)       structure type index
                density         (B, 9)     density profile
                quad_radicals   (B, 4, 214) per-quadrant radical labels
            phase: 'phase1' | 'phase2' | 'phase3'

        Returns:
            total_loss, loss_dict
        """
        losses = {}
        w = self.weights

        # Individual head losses (Phase 1 and 2)
        if phase in ("phase1", "phase2"):
            losses["char"] = self.ce_char(
                outputs["char_logits"], targets["char_label"]
            )
            losses["radical"] = self.bce_radical(
                outputs["radical_logits"], targets["radical_label"]
            )
            losses["stroke_count"] = self.ce_stroke_count(
                outputs["stroke_count"], targets["stroke_count"]
            )
            losses["stroke_types"] = self.mse_stroke_types(
                outputs["stroke_types"], targets["stroke_types"]
            )
            losses["structure"] = self.ce_structure(
                outputs["structure"], targets["structure"]
            )
            losses["density"] = self.mse_density(
                outputs["density"], targets["density"]
            )
            losses["quad_radical"] = self.bce_quad_radical(
                outputs["quad_radicals"], targets["quad_radicals"]
            )

        # Combined constraint loss (Phase 2 and 3)
        if phase in ("phase2", "phase3") and "combined_logits" in outputs:
            losses["combined"] = self.ce_combined(
                outputs["combined_logits"], targets["char_label"]
            )

        # Weighted sum
        total = torch.tensor(0.0, device=outputs["char_logits"].device)
        if phase in ("phase1", "phase2"):
            for key in ("char", "radical", "stroke_count", "stroke_types",
                        "structure", "density", "quad_radical"):
                total = total + w[key] * losses[key]
        if phase in ("phase2", "phase3") and "combined" in losses:
            total = total + w["combined"] * losses["combined"]

        loss_dict = {k: v.item() for k, v in losses.items()}
        loss_dict["total"] = total.item()
        return total, loss_dict


# =============================================================================
# Hard symbolic cascade for inference (Mode B)
# =============================================================================


class SymbolicCascadeReranker:
    """
    Hard rule-based post-processing cascade.

    Mode B inference: after training with the differentiable symbolic layer
    (which shapes the backbone's features), apply hard rules for faster,
    more interpretable decisions.

    Rules (cascade order):
        1. Top-K extraction from character head
        2. Structure type filter (heavy penalty for mismatch)
        3. Stroke count filter (eliminate outside tolerance)
        4. Radical set intersection (score by overlap)
        5. Quadrant radical consistency (spatial verification)
        6. Stroke type signature similarity
    """

    def __init__(
        self,
        radical_table: Dict[int, list],
        structure_table: Dict[int, int],
        stroke_count_table: Dict[int, int],
        stroke_type_table: Optional[Dict[int, list]] = None,
        top_k: int = 20,
        stroke_tolerance: int = 2,
    ):
        self.radical_table = radical_table
        self.structure_table = structure_table
        self.stroke_count_table = stroke_count_table
        self.stroke_type_table = stroke_type_table
        self.top_k = top_k
        self.stroke_tolerance = stroke_tolerance

    @torch.no_grad()
    def rerank(
        self, outputs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply hard symbolic rules to rerank candidates."""
        char_probs = F.softmax(outputs["char_logits"].squeeze(0), dim=-1)
        radical_probs = torch.sigmoid(outputs["radical_logits"].squeeze(0))
        stroke_count_pred = outputs["stroke_count"].squeeze(0).argmax().item()
        structure_pred = outputs["structure"].squeeze(0).argmax().item()
        stroke_type_pred = outputs["stroke_types"].squeeze(0)
        quad_radical_probs = torch.sigmoid(outputs["quad_radicals"].squeeze(0))

        top_k_probs, top_k_indices = char_probs.topk(self.top_k)
        scores = torch.zeros(self.top_k)

        for i in range(self.top_k):
            idx = top_k_indices[i].item()
            score = top_k_probs[i].item()

            # Rule 1: Structure type
            known_structure = self.structure_table.get(idx, -1)
            if known_structure >= 0 and known_structure != structure_pred:
                score *= 0.3

            # Rule 2: Stroke count
            known_strokes = self.stroke_count_table.get(idx, -1)
            if known_strokes >= 0:
                diff = abs(known_strokes - stroke_count_pred)
                if diff > self.stroke_tolerance:
                    score *= 0.05
                elif diff > 0:
                    score *= (1.0 - 0.15 * diff)

            # Rule 3: Radical match
            known_radicals = self.radical_table.get(idx, [])
            if known_radicals:
                radical_match = sum(
                    radical_probs[r].item() for r in known_radicals
                ) / len(known_radicals)
                score *= (0.5 + radical_match)

            # Rule 4: Quadrant radical consistency
            if known_radicals and known_structure >= 0:
                spatial_match = sum(
                    quad_radical_probs[:, r].max().item() for r in known_radicals
                ) / len(known_radicals)
                score *= (0.7 + 0.3 * spatial_match)

            # Rule 5: Stroke type signature
            if self.stroke_type_table and idx in self.stroke_type_table:
                known_sig = torch.tensor(
                    self.stroke_type_table[idx], dtype=torch.float32
                )
                if known_sig.norm() > 0 and stroke_type_pred.norm() > 0:
                    sim = F.cosine_similarity(
                        stroke_type_pred.unsqueeze(0), known_sig.unsqueeze(0)
                    ).item()
                    score *= (0.7 + 0.3 * max(0, sim))

            scores[i] = score

        reranked_order = scores.argsort(descending=True)
        return top_k_indices[reranked_order], scores[reranked_order]


# =============================================================================
# Training phase manager
# =============================================================================


class PhaseManager:
    """
    Manages the three training phases.

    Phase 1: heads_only mode, all params trainable (except reranker MLP)
    Phase 2: combined mode, all params trainable, symbolic tables must be loaded
    Phase 3: combined mode, only reranker MLP + reranker_weight trainable
    """

    PHASES = {1: "phase1", 2: "phase2", 3: "phase3"}

    def __init__(self, model: MiniResNetJoint, loss_fn: RadicalLightLoss):
        self.model = model
        self.loss_fn = loss_fn
        self.current_phase = 0

    def setup_phase(
        self,
        phase: int,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        t_max: int = 50,
    ) -> Tuple:
        """Configure model and optimizer for the given phase."""
        assert phase in (1, 2, 3)
        self.current_phase = phase

        if phase == 1:
            for p in self.model.parameters():
                p.requires_grad = True
            # Exclude reranker MLP params in Phase 1 (no symbolic yet)
            params = [
                p for n, p in self.model.named_parameters()
                if not n.startswith("reranker.")
            ]
        elif phase == 2:
            assert self.model.reranker.tables_loaded.item(), \
                "Call model.reranker.load_tables() before Phase 2"
            for p in self.model.parameters():
                p.requires_grad = True
            params = list(self.model.parameters())
        elif phase == 3:
            for p in self.model.parameters():
                p.requires_grad = False
            for p in self.model.reranker.parameters():
                p.requires_grad = True
            params = list(self.model.reranker.parameters())

        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
        return optimizer, scheduler

    def get_forward_mode(self) -> str:
        return "heads_only" if self.current_phase == 1 else "combined"

    def get_loss_phase(self) -> str:
        return self.PHASES[self.current_phase]


# =============================================================================
# Utilities
# =============================================================================


def count_params(module: nn.Module, only_trainable: bool = False) -> int:
    if only_trainable:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def print_model_summary(model: MiniResNetJoint):
    sections = {
        "backbone": model.backbone,
        "head_char": model.head_char,
        "head_radical": model.head_radical,
        "head_stroke_count": model.head_stroke_count,
        "head_stroke_types": model.head_stroke_types,
        "head_structure": model.head_structure,
        "head_density": model.head_density,
        "quadrant_proj": model.quadrant_proj,
        "head_quadrant_radical": model.head_quadrant_radical,
        "reranker_mlp": model.reranker.mlp,
    }

    print("=" * 70)
    print("MiniResNetJoint -- Architecture Summary")
    print("=" * 70)

    total = 0
    for name, module in sections.items():
        n = count_params(module)
        total += n
        mb32 = n * 4 / 1024 / 1024
        mb8 = n / 1024 / 1024
        print(f"  {name:30s}  {n:>10,}  ({mb32:5.2f} MB fp32 | {mb8:5.2f} MB int8)")

    # +1 for reranker_weight scalar
    total += 1

    sym_bytes = sum(b.numel() * b.element_size() for b in model.reranker.buffers())
    sym_mb = sym_bytes / 1024 / 1024

    print("-" * 70)
    print(f"  {'Neural net total':30s}  {total:>10,}  "
          f"({total*4/1024/1024:5.2f} MB fp32 | {total/1024/1024:5.2f} MB int8)")
    print(f"  {'Symbolic buffers':30s}  {'':>10s}  ({sym_mb:5.2f} MB)")
    print(f"  {'GRAND TOTAL (int8+tables)':30s}  {'':>10s}  "
          f"({total/1024/1024 + sym_mb:5.2f} MB)")
    print("=" * 70)


# =============================================================================
# Quick test
# =============================================================================


if __name__ == "__main__":
    model = MiniResNetJoint()
    print_model_summary(model)

    B = 4
    x = torch.randn(B, 1, 64, 64)

    # Phase 1
    print("\n--- Phase 1: heads_only ---")
    out = model(x, mode="heads_only")
    for k, v in out.items():
        print(f"  {k:20s}  {tuple(v.shape)}")

    # Load dummy tables and test Phase 2
    dummy = {
        str(i): {
            "radicals": [i % 214, (i * 7) % 214],
            "structure": i % 13,
            "stroke_count": (i % 25) + 1,
            "stroke_types": [(i * j) % 5 for j in range(6)],
        }
        for i in range(3755)
    }
    model.reranker.load_tables(
        {k: v["radicals"] for k, v in dummy.items()},
        {k: v["structure"] for k, v in dummy.items()},
        {k: v["stroke_count"] - 1 for k, v in dummy.items()},
        {k: v["stroke_types"] for k, v in dummy.items()},
    )

    print("\n--- Phase 2: combined ---")
    out = model(x, mode="combined")
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:20s}  {tuple(v.shape)}")
        elif isinstance(v, dict):
            for k2, v2 in v.items():
                print(f"  {k}.{k2:20s}  {tuple(v2.shape)}")

    # Loss
    print("\n--- Loss (phase2) ---")
    loss_fn = RadicalLightLoss()
    targets = {
        "char_label": torch.randint(0, 3755, (B,)),
        "radical_label": torch.zeros(B, 214).bernoulli_(0.05),
        "stroke_count": torch.randint(0, 30, (B,)),
        "stroke_types": torch.rand(B, 6) * 5,
        "structure": torch.randint(0, 13, (B,)),
        "density": torch.rand(B, 9),
        "quad_radicals": torch.zeros(B, 4, 214).bernoulli_(0.05),
    }
    total_loss, loss_dict = loss_fn(out, targets, phase="phase2")
    for k, v in loss_dict.items():
        print(f"  {k:20s}  {v:.4f}")

    # Inference
    print("\n--- Predict ---")
    vals, idxs = model.predict(x[:1], use_symbolic=True, top_k=5)
    print(f"  Top-5: {vals[0].tolist()}")

    # Phase manager
    print("\n--- Phase Manager ---")
    mgr = PhaseManager(model, loss_fn)
    for p in (1, 2, 3):
        opt, sched = mgr.setup_phase(p, lr=1e-3)
        n = count_params(model, only_trainable=True)
        print(f"  Phase {p}: mode={mgr.get_forward_mode()}, "
              f"loss={mgr.get_loss_phase()}, trainable={n:,}")