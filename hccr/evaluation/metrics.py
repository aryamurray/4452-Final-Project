"""Metrics for evaluating HCCR model performance.

This module provides various evaluation metrics including:
- Top-k accuracy for multi-class classification
- Weighted F1 score for class imbalance
- Per-class precision and recall
- Character Error Rate (CER) for sequence evaluation
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support


def top_k_accuracy(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    k_values: List[int] = [1, 5, 10, 20]
) -> Dict[int, float]:
    """Compute top-k accuracy for multiple k values.

    Args:
        outputs: Model outputs of shape (N, C) where C is number of classes.
                Can be logits or probabilities.
        targets: Ground truth labels of shape (N,) with class indices.
        k_values: List of k values to compute accuracy for.

    Returns:
        Dictionary mapping each k to its corresponding accuracy (0-100%).

    Example:
        >>> outputs = torch.randn(100, 3755)
        >>> targets = torch.randint(0, 3755, (100,))
        >>> acc = top_k_accuracy(outputs, targets, k_values=[1, 5, 10])
        >>> print(f"Top-1: {acc[1]:.2f}%")
    """
    with torch.no_grad():
        max_k = max(k_values)
        batch_size = targets.size(0)

        # Get top-k predictions
        _, pred = outputs.topk(max_k, dim=1, largest=True, sorted=True)
        pred = pred.t()  # (max_k, N)
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        # Compute accuracy for each k
        accuracies = {}
        for k in k_values:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            accuracies[k] = (correct_k.mul_(100.0 / batch_size)).item()

    return accuracies


def weighted_f1(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int
) -> float:
    """Compute weighted F1 score across all classes.

    The weighted F1 score accounts for class imbalance by weighting
    each class's F1 score by its support (number of true instances).

    Args:
        predictions: Predicted class labels of shape (N,).
        targets: Ground truth class labels of shape (N,).
        num_classes: Total number of classes in the dataset.

    Returns:
        Weighted F1 score in range [0, 1].

    Example:
        >>> preds = np.array([0, 1, 2, 1, 0])
        >>> targs = np.array([0, 1, 1, 1, 0])
        >>> f1 = weighted_f1(preds, targs, num_classes=3)
        >>> print(f"Weighted F1: {f1:.4f}")
    """
    # Use sklearn's f1_score with weighted average
    # This automatically handles class imbalance
    return f1_score(
        targets,
        predictions,
        labels=np.arange(num_classes),
        average='weighted',
        zero_division=0
    )


def per_class_precision_recall(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute precision and recall for each class.

    Args:
        predictions: Predicted class labels of shape (N,).
        targets: Ground truth class labels of shape (N,).
        num_classes: Total number of classes in the dataset.

    Returns:
        Tuple of (precision_per_class, recall_per_class) where each
        is a numpy array of shape (num_classes,).

    Example:
        >>> preds = np.array([0, 1, 2, 1, 0])
        >>> targs = np.array([0, 1, 1, 1, 0])
        >>> prec, rec = per_class_precision_recall(preds, targs, num_classes=3)
        >>> for i in range(3):
        ...     print(f"Class {i}: P={prec[i]:.3f}, R={rec[i]:.3f}")
    """
    # Compute precision, recall, and f-score for all classes
    precision, recall, _, _ = precision_recall_fscore_support(
        targets,
        predictions,
        labels=np.arange(num_classes),
        average=None,
        zero_division=0
    )

    return precision, recall


def _edit_distance(seq1: List[int], seq2: List[int]) -> int:
    """Compute Levenshtein edit distance between two sequences.

    Args:
        seq1: First sequence of integers.
        seq2: Second sequence of integers.

    Returns:
        Minimum number of insertions, deletions, and substitutions
        needed to transform seq1 into seq2.
    """
    len1, len2 = len(seq1), len(seq2)

    # Create DP table
    dp = np.zeros((len1 + 1, len2 + 1), dtype=np.int32)

    # Initialize base cases
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    # Fill DP table
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # deletion
                    dp[i][j - 1],      # insertion
                    dp[i - 1][j - 1]   # substitution
                )

    return int(dp[len1][len2])


def character_error_rate(
    predicted_sequences: List[List[int]],
    target_sequences: List[List[int]]
) -> float:
    """Compute Character Error Rate (CER) for sequence predictions.

    CER is the average edit distance normalized by sequence length.
    It's commonly used in OCR/handwriting recognition evaluation.

    Args:
        predicted_sequences: List of predicted character sequences,
                           where each sequence is a list of character IDs.
        target_sequences: List of ground truth character sequences.

    Returns:
        Character Error Rate as a percentage (0-100+).
        Can exceed 100% if predictions are much longer than targets.

    Example:
        >>> pred_seqs = [[1, 2, 3], [4, 5]]
        >>> targ_seqs = [[1, 2, 4], [4, 5, 6]]
        >>> cer = character_error_rate(pred_seqs, targ_seqs)
        >>> print(f"CER: {cer:.2f}%")

    Note:
        CER = (sum of edit distances) / (sum of target lengths) * 100
    """
    if len(predicted_sequences) != len(target_sequences):
        raise ValueError(
            f"Number of predicted sequences ({len(predicted_sequences)}) "
            f"must match number of target sequences ({len(target_sequences)})"
        )

    if len(predicted_sequences) == 0:
        return 0.0

    total_edit_distance = 0
    total_target_length = 0

    for pred_seq, targ_seq in zip(predicted_sequences, target_sequences):
        edit_dist = _edit_distance(pred_seq, targ_seq)
        total_edit_distance += edit_dist
        total_target_length += len(targ_seq)

    # Avoid division by zero
    if total_target_length == 0:
        return 0.0

    # Return as percentage
    cer = (total_edit_distance / total_target_length) * 100.0
    return cer
