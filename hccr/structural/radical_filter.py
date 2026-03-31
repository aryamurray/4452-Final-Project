"""Radical-based candidate filtering and reweighting.

Combines classifier predictions with radical predictions to improve
top-K candidate selection.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from hccr.structural.radical_table import RadicalTable

logger = logging.getLogger(__name__)


class RadicalFilter:
    """Reweight classifier candidates using radical predictions.

    Combines character classification scores with radical prediction scores
    to improve candidate ranking. Optionally incorporates structure and
    stroke count constraints.
    """

    def __init__(self, radical_table: RadicalTable, alpha: float = 0.7) -> None:
        """Initialize radical filter.

        Args:
            radical_table: Radical decomposition table
            alpha: Weight for classifier score (vs radical score)
                   Final score = alpha * char_score + (1-alpha) * radical_score
        """
        self.radical_table = radical_table
        self.alpha = alpha

    def reweight(
        self,
        char_probs: np.ndarray,
        radical_probs: np.ndarray,
        label_map: dict[int, str],
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """Reweight top-K candidates using radical predictions.

        Args:
            char_probs: Character probability distribution, shape (num_classes,)
            radical_probs: Radical probability distribution, shape (num_radicals,)
            label_map: Maps class index to character
            top_k: Number of candidates to consider

        Returns:
            List of (class_index, combined_score) tuples, sorted descending
        """
        # Get top-K candidates from classifier
        top_k_indices = np.argsort(char_probs)[-top_k:][::-1]

        candidates: list[tuple[int, float]] = []

        for idx in top_k_indices:
            char_prob = char_probs[idx]
            char = label_map.get(idx)

            if char is None:
                # Unknown character, use classifier score only
                candidates.append((int(idx), float(char_prob)))
                continue

            # Get expected radicals for this character
            radicals = self.radical_table.char_to_radicals.get(char, [])

            if not radicals:
                # No radical info, use classifier score only
                candidates.append((int(idx), float(char_prob)))
                continue

            # Compute radical match score (mean probability of expected radicals)
            radical_scores = []
            for radical in radicals:
                if radical in self.radical_table.radical_to_index:
                    radical_idx = self.radical_table.radical_to_index[radical]
                    if radical_idx < len(radical_probs):
                        radical_scores.append(radical_probs[radical_idx])

            if radical_scores:
                radical_match_score = np.mean(radical_scores)
            else:
                radical_match_score = 0.0

            # Combined score
            combined_score = (
                self.alpha * char_prob + (1 - self.alpha) * radical_match_score
            )

            candidates.append((int(idx), float(combined_score)))

        # Sort by combined score descending
        candidates.sort(key=lambda x: x[1], reverse=True)

        return candidates

    def reweight_with_structure(
        self,
        char_probs: np.ndarray,
        radical_probs: np.ndarray,
        structure_probs: np.ndarray,
        stroke_pred: float,
        label_map: dict[int, str],
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """Reweight candidates using radicals, structure, and stroke count.

        Args:
            char_probs: Character probability distribution, shape (num_classes,)
            radical_probs: Radical probability distribution, shape (num_radicals,)
            structure_probs: Structure probability distribution, shape (13,)
            stroke_pred: Predicted stroke count (float)
            label_map: Maps class index to character
            top_k: Number of candidates to consider

        Returns:
            List of (class_index, combined_score) tuples, sorted descending
        """
        # Get base candidates with radical reweighting
        candidates = self.reweight(char_probs, radical_probs, label_map, top_k)

        # Get predicted structure (argmax)
        pred_structure = int(np.argmax(structure_probs))

        # Apply structure and stroke penalties/bonuses
        refined_candidates: list[tuple[int, float]] = []

        for idx, base_score in candidates:
            char = label_map.get(idx)

            if char is None:
                refined_candidates.append((idx, base_score))
                continue

            # Structure match bonus/penalty
            char_structure = self.radical_table.get_structure(char)
            if char_structure == pred_structure:
                structure_match = 1.0
            else:
                structure_match = 0.5  # Penalty for structure mismatch

            # Stroke count penalty (exponential decay based on distance)
            char_strokes = self.radical_table.get_strokes(char)
            stroke_diff = abs(stroke_pred - char_strokes)
            stroke_penalty = np.exp(-0.1 * stroke_diff)

            # Combined score with structure and stroke factors
            final_score = base_score * structure_match * stroke_penalty

            refined_candidates.append((idx, float(final_score)))

        # Re-sort by final score
        refined_candidates.sort(key=lambda x: x[1], reverse=True)

        return refined_candidates

    def filter_by_radicals(
        self,
        char_probs: np.ndarray,
        radical_probs: np.ndarray,
        label_map: dict[int, str],
        top_k: int = 10,
        radical_threshold: float = 0.3,
    ) -> list[tuple[int, float]]:
        """Filter candidates by radical presence threshold.

        More aggressive filtering that removes candidates whose expected
        radicals have low prediction scores.

        Args:
            char_probs: Character probability distribution
            radical_probs: Radical probability distribution
            label_map: Maps class index to character
            top_k: Initial number of candidates
            radical_threshold: Minimum mean radical score to keep candidate

        Returns:
            Filtered list of (class_index, score) tuples
        """
        # Get reweighted candidates
        candidates = self.reweight(char_probs, radical_probs, label_map, top_k)

        # Filter by radical threshold
        filtered: list[tuple[int, float]] = []

        for idx, score in candidates:
            char = label_map.get(idx)
            if char is None:
                filtered.append((idx, score))
                continue

            radicals = self.radical_table.char_to_radicals.get(char, [])
            if not radicals:
                filtered.append((idx, score))
                continue

            # Check mean radical score
            radical_scores = []
            for radical in radicals:
                if radical in self.radical_table.radical_to_index:
                    radical_idx = self.radical_table.radical_to_index[radical]
                    if radical_idx < len(radical_probs):
                        radical_scores.append(radical_probs[radical_idx])

            if radical_scores:
                mean_radical_score = np.mean(radical_scores)
                if mean_radical_score >= radical_threshold:
                    filtered.append((idx, score))
            else:
                # No radical info, keep candidate
                filtered.append((idx, score))

        return filtered

    def get_radical_coverage(
        self,
        char_idx: int,
        radical_probs: np.ndarray,
        label_map: dict[int, str],
        threshold: float = 0.5,
    ) -> float:
        """Compute fraction of character's radicals above threshold.

        Useful for diagnostics and confidence estimation.

        Args:
            char_idx: Character class index
            radical_probs: Radical probability distribution
            label_map: Maps class index to character
            threshold: Minimum probability to consider radical "detected"

        Returns:
            Fraction of radicals detected (0.0 to 1.0)
        """
        char = label_map.get(char_idx)
        if char is None:
            return 0.0

        radicals = self.radical_table.char_to_radicals.get(char, [])
        if not radicals:
            return 1.0  # No radicals expected, full coverage

        detected = 0
        for radical in radicals:
            if radical in self.radical_table.radical_to_index:
                radical_idx = self.radical_table.radical_to_index[radical]
                if radical_idx < len(radical_probs):
                    if radical_probs[radical_idx] >= threshold:
                        detected += 1

        return detected / len(radicals)
