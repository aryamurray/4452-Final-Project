"""MiniResNetJointV2 — improved symbolic reranker.

Changes from v1 (mini_restnet_join.py):

1. Soft radical features
   v1 binarised sigmoid outputs with a hard 0.5 threshold before computing
   match/false-alarm ratios.  At ~95% backbone accuracy the residual errors
   are near-confusable characters that share radicals — the hard threshold
   throws away probability mass that matters at this operating point.
   v2 uses raw sigmoid probabilities directly.

2. Zero-masked stroke type cosine
   stroke_type_sig is all-zeros for every character (the table was never
   populated).  F.normalize of a zero vector is undefined/zero, so feature 5
   was always 0.  v2 masks the feature to 0 only when the table entry is
   genuinely zero, and will start contributing signal once the table is filled.

3. Neural confidence as a 6th MLP feature
   The reranker MLP previously saw only symbolic features and had no way to
   know how confident the neural backbone already was.  Adding the softmax
   probability of each candidate lets the MLP learn "don't override a
   high-confidence neural prediction."  The char logits are detached for this
   feature so gradients only flow through the symbolic path.

4. top_k: 5 → 20
   With a 95% backbone the correct answer is usually in the top-5, but the
   reranker has the most impact on the ambiguous tail.  Widening to 20
   gives it room to rescue cases where a structurally valid candidate sits
   at rank 6–20.

5. reranker_weight init: 0.01 → 0.1
   The original near-zero init meant the combined loss signal was tiny
   throughout most of training.  0.1 lets the MLP see a meaningful gradient
   from the first batch.

6. MLP capacity: hidden_dim 32 → 64, input 5 → 6
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from hccr.models.mini_restnet_join import (
    MiniResNetJoint,
    PhaseManager,
    RadicalLightLoss,
    _build_float_tensor,
    _build_int_tensor,
    _build_radical_mask,
    count_params,
    print_model_summary,
)


class SymbolicRerankerV2(nn.Module):
    """Improved top-K symbolic re-ranker.

    Fixes vs v1:
      - Soft radical features (sigmoid probs, not hard binary)
      - Zero-masked stroke type cosine
      - Neural confidence as 6th feature
      - top_k=20, hidden_dim=64, reranker_weight init=0.1
    """

    def __init__(
        self,
        num_classes: int = 3755,
        num_radicals: int = 214,
        num_structure_types: int = 13,
        num_stroke_count_bins: int = 30,
        num_stroke_types: int = 6,
        top_k: int = 20,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_radicals = num_radicals
        self.top_k = top_k

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

        # 6 features: 5 symbolic + 1 neural confidence
        self.mlp = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        # Stronger init so the combined loss signal is meaningful early on
        self.reranker_weight = nn.Parameter(torch.tensor(0.1))

    def load_tables(
        self,
        radical_table: Dict[str, list],
        structure_table: Dict[str, int],
        stroke_count_table: Dict[str, int],
        stroke_type_table: Optional[Dict[str, list]] = None,
    ):
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
        """Compute 6 features per candidate: 5 symbolic + 1 neural confidence.

        Args:
            outputs: model output dict — char_logits detached by caller
            candidate_indices: (B, K) top-K indices

        Returns:
            (B, K, 6) feature tensor
        """
        B, K = candidate_indices.shape

        # ---- Feature 1-2: radical match / false-alarm (SOFT) ----
        # v1 used hard binary threshold; v2 uses raw sigmoid probabilities
        radical_probs = torch.sigmoid(outputs["radical_logits"])  # (B, R) ∈ [0,1]
        cand_radical_mask = self.radical_mask[candidate_indices].float()  # (B, K, R)
        radical_probs_exp = radical_probs.unsqueeze(1)                    # (B, 1, R)

        # Soft match: expected radical probability mass inside candidate's set
        detected_and_expected = (radical_probs_exp * cand_radical_mask).sum(-1)
        cand_radical_counts = cand_radical_mask.sum(-1).clamp(min=1)
        radical_match_ratio = detected_and_expected / cand_radical_counts  # (B, K)

        # Soft false-alarm: probability mass outside candidate's set
        total_prob = radical_probs.sum(-1, keepdim=True).clamp(min=1e-6)  # (B, 1)
        false_alarm_mass = (radical_probs_exp * (1 - cand_radical_mask)).sum(-1)
        radical_false_ratio = false_alarm_mass / total_prob               # (B, K)

        # ---- Feature 3: structure match probability ----
        structure_probs = F.softmax(outputs["structure"], dim=-1)         # (B, 13)
        cand_structure = self.structure_label[candidate_indices]          # (B, K)
        structure_match = structure_probs.gather(1, cand_structure)       # (B, K)

        # ---- Feature 4: stroke count distance (normalised) ----
        stroke_pred = outputs["stroke_count"].argmax(dim=-1)              # (B,)
        cand_stroke = self.stroke_count_label[candidate_indices]          # (B, K)
        stroke_distance = (
            (stroke_pred.unsqueeze(1) - cand_stroke).abs().float() / 29.0
        )

        # ---- Feature 5: stroke type cosine (zero-masked) ----
        pred_stroke = outputs["stroke_types"]                             # (B, 6)
        cand_sig = self.stroke_type_sig[candidate_indices]                # (B, K, 6)
        sig_norm_val = cand_sig.norm(dim=-1)                              # (B, K)
        has_sig = (sig_norm_val > 1e-6).float()                           # (B, K)
        # Only compute cosine where signatures exist; elsewhere output 0.0
        pred_norm = F.normalize(pred_stroke, dim=-1).unsqueeze(1)        # (B, 1, 6)
        cand_sig_norm = F.normalize(cand_sig + 1e-8 * ~has_sig.unsqueeze(-1).bool(), dim=-1)
        stroke_type_cos = (pred_norm * cand_sig_norm).sum(-1) * has_sig   # (B, K)

        # ---- Feature 6: neural confidence (detached) ----
        # Softmax probability the backbone assigns to each candidate.
        # Detached so gradients only flow through the symbolic path.
        neural_probs = F.softmax(outputs["char_logits"].detach(), dim=-1) # (B, C)
        neural_conf = neural_probs.gather(1, candidate_indices)           # (B, K)

        features = torch.stack([
            radical_match_ratio,
            radical_false_ratio,
            structure_match,
            stroke_distance,
            stroke_type_cos,
            neural_conf,
        ], dim=-1)  # (B, K, 6)

        return features

    def forward(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Re-rank top-K candidates using symbolic + neural-confidence features."""
        char_logits = outputs["char_logits"]
        B, C = char_logits.shape
        K = min(self.top_k, C)

        _, top_indices = char_logits.topk(K, dim=-1)  # (B, K)

        # Detach all aux outputs (char_logits handled inside _compute_features)
        detached = {
            k: v.detach() if k != "char_logits" else v
            for k, v in outputs.items()
        }
        features = self._compute_features(detached, top_indices)  # (B, K, 6)

        rerank_scores = self.mlp(features).squeeze(-1)            # (B, K)
        top_logits = char_logits.gather(1, top_indices)           # (B, K)
        combined_top = top_logits + self.reranker_weight * rerank_scores

        combined_logits = char_logits.clone()
        combined_logits.scatter_(1, top_indices, combined_top)
        return combined_logits


class MiniResNetJointV2(MiniResNetJoint):
    """MiniResNetJoint with SymbolicRerankerV2.

    Identical architecture and training protocol to v1; only the reranker
    is replaced.  All other methods (load_symbolic_tables, load_pretrained_
    backbone, forward, predict) are inherited unchanged.
    """

    def __init__(
        self,
        num_classes: int = 3755,
        num_radicals: int = 214,
        num_structure_types: int = 13,
        num_stroke_count_bins: int = 30,
        num_stroke_types: int = 6,
        dropout: float = 0.3,
    ):
        super().__init__(
            num_classes=num_classes,
            num_radicals=num_radicals,
            num_structure_types=num_structure_types,
            num_stroke_count_bins=num_stroke_count_bins,
            num_stroke_types=num_stroke_types,
            dropout=dropout,
        )
        # Replace v1 reranker (top_k=5, hidden=32, weight=0.01) with v2
        self.reranker = SymbolicRerankerV2(
            num_classes=num_classes,
            num_radicals=num_radicals,
            num_structure_types=num_structure_types,
            num_stroke_count_bins=num_stroke_count_bins,
            num_stroke_types=num_stroke_types,
            top_k=20,
            hidden_dim=64,
        )
