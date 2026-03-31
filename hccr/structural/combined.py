"""Combined structural post-processing pipeline.

Integrates radical filtering, structure/stroke constraints, and bigram
language modeling for improved recognition accuracy.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from hccr.config import StructuralConfig

if TYPE_CHECKING:
    from hccr.structural.bigram import BigramModel
    from hccr.structural.radical_filter import RadicalFilter

logger = logging.getLogger(__name__)


class StructuralPipeline:
    """Full structural post-processing pipeline with beam search.

    Combines:
    1. Radical-based candidate reweighting
    2. Structure and stroke count constraints
    3. Bigram language model for sequence re-ranking
    4. Beam search for multi-character sequences
    """

    def __init__(
        self,
        radical_filter: RadicalFilter,
        bigram_model: BigramModel,
        label_map: dict[int, str],
        config: StructuralConfig,
    ) -> None:
        """Initialize structural pipeline.

        Args:
            radical_filter: Radical-based filter
            bigram_model: Character bigram language model
            label_map: Maps class index to character
            config: Structural configuration (alpha, beta, top_k, beam_width)
        """
        self.radical_filter = radical_filter
        self.bigram_model = bigram_model
        self.label_map = label_map
        self.config = config

        # Build reverse mapping (char -> index)
        self.char_to_index: dict[str, int] = {
            char: idx for idx, char in label_map.items()
        }

    def predict_single(
        self,
        char_probs: np.ndarray,
        radical_probs: np.ndarray,
        structure_probs: np.ndarray,
        stroke_pred: float,
        prev_char: str | None = None,
    ) -> list[tuple[int, float]]:
        """Predict single character with structural constraints.

        Args:
            char_probs: Character probabilities, shape (num_classes,)
            radical_probs: Radical probabilities, shape (num_radicals,)
            structure_probs: Structure probabilities, shape (13,)
            stroke_pred: Predicted stroke count
            prev_char: Previous character for bigram context (optional)

        Returns:
            Ranked list of (class_index, score) tuples
        """
        # Step 1: Radical + structure + stroke reweighting
        candidates = self.radical_filter.reweight_with_structure(
            char_probs=char_probs,
            radical_probs=radical_probs,
            structure_probs=structure_probs,
            stroke_pred=stroke_pred,
            label_map=self.label_map,
            top_k=self.config.top_k,
        )

        # Step 2: Bigram re-ranking (if context available)
        if prev_char is not None:
            candidates = self.bigram_model.rerank(
                candidates=candidates,
                prev_char=prev_char,
                label_map=self.label_map,
                beta=self.config.beta,
            )

        return candidates

    def predict_sequence_beam(
        self,
        char_probs_seq: list[np.ndarray],
        radical_probs_seq: list[np.ndarray],
        structure_probs_seq: list[np.ndarray],
        stroke_pred_seq: list[float],
        beam_width: int | None = None,
    ) -> list[int]:
        """Beam search over character sequence.

        Args:
            char_probs_seq: List of character probability arrays
            radical_probs_seq: List of radical probability arrays
            structure_probs_seq: List of structure probability arrays
            stroke_pred_seq: List of stroke count predictions
            beam_width: Beam width (defaults to config.beam_width)

        Returns:
            Best character sequence as list of class indices
        """
        if beam_width is None:
            beam_width = self.config.beam_width

        seq_len = len(char_probs_seq)
        if seq_len == 0:
            return []

        # Beam state: (sequence, cumulative_score)
        # Start with empty sequence
        beams: list[tuple[list[int], float]] = [([], 0.0)]

        for pos in range(seq_len):
            char_probs = char_probs_seq[pos]
            radical_probs = radical_probs_seq[pos]
            structure_probs = structure_probs_seq[pos]
            stroke_pred = stroke_pred_seq[pos]

            new_beams: list[tuple[list[int], float]] = []

            for sequence, beam_score in beams:
                # Get previous character for bigram context
                prev_char = None
                if len(sequence) > 0:
                    prev_idx = sequence[-1]
                    prev_char = self.label_map.get(prev_idx)

                # Get top-K candidates for this position
                candidates = self.predict_single(
                    char_probs=char_probs,
                    radical_probs=radical_probs,
                    structure_probs=structure_probs,
                    stroke_pred=stroke_pred,
                    prev_char=prev_char,
                )

                # Expand beam with each candidate
                for cand_idx, cand_score in candidates:
                    # Compute new cumulative score
                    # Use log-space for numerical stability
                    log_score = math.log(max(cand_score, 1e-10))
                    new_score = beam_score + log_score

                    # Add bigram bonus if context available
                    if prev_char is not None:
                        cand_char = self.label_map.get(cand_idx)
                        if cand_char is not None:
                            bigram_log_prob = self.bigram_model.log_prob(
                                prev_char, cand_char
                            )
                            # Scale bigram contribution
                            new_score += self.config.beta * bigram_log_prob

                    # Create new beam
                    new_sequence = sequence + [cand_idx]
                    new_beams.append((new_sequence, new_score))

            # Keep top beam_width beams
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam_width]

        # Return best sequence
        if beams:
            best_sequence, _ = beams[0]
            return best_sequence

        return []

    def predict_single_differentiable(
        self,
        model_output: dict[str, torch.Tensor],
    ) -> int:
        """Predict single character using differentiable symbolic output.

        Uses the combined_probs from the symbolic forward pass if available,
        otherwise falls back to standard top-K reranking.

        Args:
            model_output: Dictionary from model forward pass. Expected to
                contain 'combined_probs' (B=1, num_classes) when symbolic
                mode is active.

        Returns:
            Predicted class index.
        """
        combined_probs = model_output.get("combined_probs")
        if combined_probs is not None:
            return int(torch.argmax(combined_probs, dim=-1).item())

        # Fallback: use existing top-K reranking pipeline
        char_logits = model_output["char_logits"]
        char_probs = torch.softmax(char_logits, dim=-1).cpu().numpy()[0]

        radical_logits = model_output.get("radical_logits")
        if radical_logits is not None:
            radical_probs = torch.sigmoid(radical_logits).cpu().numpy()[0]
        else:
            num_radicals = len(self.radical_filter.radical_table.all_radicals)
            radical_probs = np.ones(num_radicals) * 0.5

        structure_logits = model_output.get("structure_logits")
        if structure_logits is not None:
            structure_probs = torch.softmax(structure_logits, dim=-1).cpu().numpy()[0]
        else:
            structure_probs = np.ones(13) / 13.0

        stroke_logits = model_output.get("stroke_logits")
        if stroke_logits is not None:
            stroke_pred = float(torch.argmax(stroke_logits, dim=-1).item())
        else:
            stroke_pred = 8.0

        candidates = self.predict_single(
            char_probs=char_probs,
            radical_probs=radical_probs,
            structure_probs=structure_probs,
            stroke_pred=stroke_pred,
        )

        if candidates:
            return candidates[0][0]
        return int(np.argmax(char_probs))

    def evaluate_on_loader(
        self,
        model: torch.nn.Module,
        test_loader: DataLoader,
        device: torch.device,
        mode: str = "joint",
    ) -> dict[str, float]:
        """Evaluate structural post-processing on test set.

        Args:
            model: Trained HCCR model
            test_loader: Test data loader
            device: Torch device
            mode: Evaluation mode ("joint" or "simple")
                  - "joint": Use full pipeline with structure/stroke
                  - "simple": Use only radical reweighting

        Returns:
            Dictionary with accuracy metrics before/after correction
        """
        model.eval()

        total = 0
        correct_before = 0
        correct_after = 0
        top5_before = 0
        top5_after = 0

        logger.info(f"Evaluating structural pipeline (mode={mode})...")

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                images = batch["image"].to(device)
                labels = batch["label"].to(device)

                # Forward pass
                outputs = model(images)

                # Extract outputs (depends on model architecture)
                if isinstance(outputs, dict):
                    char_logits = outputs["char_logits"]
                    radical_logits = outputs.get("radical_logits")
                    structure_logits = outputs.get("structure_logits")
                    stroke_pred = outputs.get("stroke_pred")
                else:
                    # Simple model, only char logits
                    char_logits = outputs
                    radical_logits = None
                    structure_logits = None
                    stroke_pred = None

                # Convert to probabilities
                char_probs = torch.softmax(char_logits, dim=1).cpu().numpy()

                if radical_logits is not None:
                    radical_probs = torch.sigmoid(radical_logits).cpu().numpy()
                else:
                    # No radical predictions, use uniform
                    num_radicals = len(self.radical_filter.radical_table.all_radicals)
                    radical_probs = np.ones((len(images), num_radicals)) * 0.5

                if structure_logits is not None:
                    structure_probs = torch.softmax(structure_logits, dim=1).cpu().numpy()
                else:
                    # No structure predictions, use uniform
                    structure_probs = np.ones((len(images), 13)) / 13.0

                if stroke_pred is not None:
                    stroke_preds = stroke_pred.cpu().numpy()
                else:
                    # No stroke predictions, use default
                    stroke_preds = np.ones(len(images)) * 8.0

                labels_np = labels.cpu().numpy()

                # Evaluate each sample
                for i in range(len(images)):
                    label = int(labels_np[i])

                    # Before correction
                    pred_before = int(np.argmax(char_probs[i]))
                    top5_before_indices = np.argsort(char_probs[i])[-5:][::-1]

                    if pred_before == label:
                        correct_before += 1

                    if label in top5_before_indices:
                        top5_before += 1

                    # After correction
                    if mode == "joint" and structure_logits is not None:
                        candidates = self.radical_filter.reweight_with_structure(
                            char_probs=char_probs[i],
                            radical_probs=radical_probs[i],
                            structure_probs=structure_probs[i],
                            stroke_pred=float(stroke_preds[i]),
                            label_map=self.label_map,
                            top_k=self.config.top_k,
                        )
                    else:
                        candidates = self.radical_filter.reweight(
                            char_probs=char_probs[i],
                            radical_probs=radical_probs[i],
                            label_map=self.label_map,
                            top_k=self.config.top_k,
                        )

                    if candidates:
                        pred_after = candidates[0][0]
                        top5_after_indices = [idx for idx, _ in candidates[:5]]

                        if pred_after == label:
                            correct_after += 1

                        if label in top5_after_indices:
                            top5_after += 1

                    total += 1

        # Compute metrics
        acc_before = correct_before / total if total > 0 else 0.0
        acc_after = correct_after / total if total > 0 else 0.0
        top5_acc_before = top5_before / total if total > 0 else 0.0
        top5_acc_after = top5_after / total if total > 0 else 0.0

        results = {
            "acc_before": acc_before,
            "acc_after": acc_after,
            "top5_before": top5_acc_before,
            "top5_after": top5_acc_after,
            "improvement": acc_after - acc_before,
            "total_samples": total,
        }

        logger.info(f"Results (n={total}):")
        logger.info(f"  Top-1 accuracy before: {acc_before:.4f}")
        logger.info(f"  Top-1 accuracy after:  {acc_after:.4f}")
        logger.info(f"  Improvement: {results['improvement']:.4f}")
        logger.info(f"  Top-5 accuracy before: {top5_acc_before:.4f}")
        logger.info(f"  Top-5 accuracy after:  {top5_acc_after:.4f}")

        return results

    def batch_predict(
        self,
        char_probs_batch: np.ndarray,
        radical_probs_batch: np.ndarray,
        structure_probs_batch: np.ndarray,
        stroke_pred_batch: np.ndarray,
    ) -> list[int]:
        """Predict batch of single characters (no sequence context).

        Args:
            char_probs_batch: Shape (batch_size, num_classes)
            radical_probs_batch: Shape (batch_size, num_radicals)
            structure_probs_batch: Shape (batch_size, 13)
            stroke_pred_batch: Shape (batch_size,)

        Returns:
            List of predicted class indices
        """
        predictions: list[int] = []

        for i in range(len(char_probs_batch)):
            candidates = self.radical_filter.reweight_with_structure(
                char_probs=char_probs_batch[i],
                radical_probs=radical_probs_batch[i],
                structure_probs=structure_probs_batch[i],
                stroke_pred=float(stroke_pred_batch[i]),
                label_map=self.label_map,
                top_k=self.config.top_k,
            )

            if candidates:
                predictions.append(candidates[0][0])
            else:
                # Fallback to argmax
                predictions.append(int(np.argmax(char_probs_batch[i])))

        return predictions

    def analyze_errors(
        self,
        model: torch.nn.Module,
        test_loader: DataLoader,
        device: torch.device,
        num_samples: int = 100,
    ) -> dict[str, Any]:
        """Analyze error cases and correction patterns.

        Args:
            model: Trained model
            test_loader: Test data loader
            device: Torch device
            num_samples: Number of samples to analyze

        Returns:
            Dictionary with error analysis statistics
        """
        model.eval()

        error_types = {
            "corrected": [],  # Wrong before, correct after
            "degraded": [],  # Correct before, wrong after
            "persistent": [],  # Wrong before and after
            "correct": [],  # Correct before and after
        }

        sample_count = 0

        with torch.no_grad():
            for batch in test_loader:
                if sample_count >= num_samples:
                    break

                images = batch["image"].to(device)
                labels = batch["label"].to(device)

                outputs = model(images)

                if isinstance(outputs, dict):
                    char_logits = outputs["char_logits"]
                    radical_logits = outputs.get("radical_logits")
                    structure_logits = outputs.get("structure_logits")
                    stroke_pred = outputs.get("stroke_pred")
                else:
                    char_logits = outputs
                    radical_logits = None
                    structure_logits = None
                    stroke_pred = None

                char_probs = torch.softmax(char_logits, dim=1).cpu().numpy()

                if radical_logits is not None:
                    radical_probs = torch.sigmoid(radical_logits).cpu().numpy()
                else:
                    num_radicals = len(self.radical_filter.radical_table.all_radicals)
                    radical_probs = np.ones((len(images), num_radicals)) * 0.5

                if structure_logits is not None:
                    structure_probs = torch.softmax(structure_logits, dim=1).cpu().numpy()
                else:
                    structure_probs = np.ones((len(images), 13)) / 13.0

                if stroke_pred is not None:
                    stroke_preds = stroke_pred.cpu().numpy()
                else:
                    stroke_preds = np.ones(len(images)) * 8.0

                labels_np = labels.cpu().numpy()

                for i in range(len(images)):
                    if sample_count >= num_samples:
                        break

                    label = int(labels_np[i])
                    pred_before = int(np.argmax(char_probs[i]))

                    candidates = self.radical_filter.reweight_with_structure(
                        char_probs=char_probs[i],
                        radical_probs=radical_probs[i],
                        structure_probs=structure_probs[i],
                        stroke_pred=float(stroke_preds[i]),
                        label_map=self.label_map,
                        top_k=self.config.top_k,
                    )

                    pred_after = candidates[0][0] if candidates else pred_before

                    # Categorize
                    correct_before = pred_before == label
                    correct_after = pred_after == label

                    if correct_before and correct_after:
                        error_types["correct"].append(label)
                    elif not correct_before and correct_after:
                        error_types["corrected"].append(label)
                    elif correct_before and not correct_after:
                        error_types["degraded"].append(label)
                    else:
                        error_types["persistent"].append(label)

                    sample_count += 1

        # Compute statistics
        total = sample_count
        stats = {
            "total": total,
            "correct": len(error_types["correct"]),
            "corrected": len(error_types["corrected"]),
            "degraded": len(error_types["degraded"]),
            "persistent": len(error_types["persistent"]),
            "correction_rate": len(error_types["corrected"]) / total if total > 0 else 0.0,
            "degradation_rate": len(error_types["degraded"]) / total if total > 0 else 0.0,
        }

        logger.info(f"Error analysis (n={total}):")
        logger.info(f"  Corrected: {stats['corrected']} ({stats['correction_rate']:.2%})")
        logger.info(f"  Degraded: {stats['degraded']} ({stats['degradation_rate']:.2%})")
        logger.info(f"  Persistent errors: {stats['persistent']}")
        logger.info(f"  Correct: {stats['correct']}")

        return stats
