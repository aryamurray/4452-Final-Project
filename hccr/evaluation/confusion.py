"""Confusion analysis and error categorization for HCCR.

This module provides tools for analyzing model errors through:
- Error categorization based on character structure and radicals
- Resolution rate computation for structural corrections
- Confusion matrix visualization for most confused classes
"""

from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def categorize_errors(
    predictions: np.ndarray,
    targets: np.ndarray,
    radical_table: Dict
) -> Dict[str, int]:
    """Categorize prediction errors based on structural similarity.

    Analyzes misclassified samples to understand error patterns:
    - Single stroke difference: pred/target differ by 1 stroke count
    - Same radical, different structure: share radicals but different arrangement
    - Similar radical: share more than 50% of radicals
    - Unrelated: no structural similarity

    Args:
        predictions: Predicted class indices of shape (N,).
        targets: Ground truth class indices of shape (N,).
        radical_table: Dictionary mapping character index to properties:
            {
                char_idx: {
                    "radicals": [radical_ids],
                    "structure": structure_type_id,
                    "strokes": stroke_count
                }
            }

    Returns:
        Dictionary with error counts by category:
            {
                "single_stroke_diff": count,
                "same_radical_diff_structure": count,
                "similar_radical": count,
                "unrelated": count,
                "total_errors": count
            }

    Example:
        >>> radical_table = {
        ...     0: {"radicals": [1, 2], "structure": 0, "strokes": 5},
        ...     1: {"radicals": [1, 2], "structure": 1, "strokes": 5},
        ...     2: {"radicals": [3, 4], "structure": 0, "strokes": 8}
        ... }
        >>> preds = np.array([1, 2, 1])
        >>> targs = np.array([0, 0, 2])
        >>> cats = categorize_errors(preds, targs, radical_table)
        >>> print(cats)
    """
    categories = {
        "single_stroke_diff": 0,
        "same_radical_diff_structure": 0,
        "similar_radical": 0,
        "unrelated": 0,
        "total_errors": 0
    }

    # Find all errors
    error_mask = predictions != targets

    for pred_idx, targ_idx in zip(
        predictions[error_mask],
        targets[error_mask]
    ):
        categories["total_errors"] += 1

        # Get character properties (handle missing entries)
        pred_info = radical_table.get(
            int(pred_idx),
            {"radicals": [], "structure": -1, "strokes": 0}
        )
        targ_info = radical_table.get(
            int(targ_idx),
            {"radicals": [], "structure": -1, "strokes": 0}
        )

        pred_radicals = set(pred_info["radicals"])
        targ_radicals = set(targ_info["radicals"])
        pred_structure = pred_info["structure"]
        targ_structure = targ_info["structure"]
        pred_strokes = pred_info["strokes"]
        targ_strokes = targ_info["strokes"]

        # Categorize error
        stroke_diff = abs(pred_strokes - targ_strokes)

        if stroke_diff == 1:
            # Category 1: Single stroke difference
            categories["single_stroke_diff"] += 1
        elif (pred_radicals == targ_radicals and
              len(pred_radicals) > 0 and
              pred_structure != targ_structure):
            # Category 2: Same radicals, different structure
            categories["same_radical_diff_structure"] += 1
        elif len(pred_radicals) > 0 and len(targ_radicals) > 0:
            # Category 3: Similar radicals (>50% overlap)
            overlap = len(pred_radicals & targ_radicals)
            union = len(pred_radicals | targ_radicals)
            if overlap / union > 0.5:
                categories["similar_radical"] += 1
            else:
                categories["unrelated"] += 1
        else:
            # Category 4: Unrelated
            categories["unrelated"] += 1

    return categories


def compute_resolution_rates(
    predictions_before: np.ndarray,
    predictions_after: np.ndarray,
    targets: np.ndarray,
    radical_table: Dict
) -> Dict[str, float]:
    """Compute error resolution rates after structural correction.

    Compares errors before and after applying structural corrections
    to measure effectiveness for different error types.

    Args:
        predictions_before: Predictions before correction, shape (N,).
        predictions_after: Predictions after correction, shape (N,).
        targets: Ground truth labels, shape (N,).
        radical_table: Character structure information.

    Returns:
        Dictionary with resolution rates (0-100%) per category:
            {
                "single_stroke_diff_resolution": percentage,
                "same_radical_diff_structure_resolution": percentage,
                "similar_radical_resolution": percentage,
                "unrelated_resolution": percentage,
                "overall_resolution": percentage
            }

    Example:
        >>> before = np.array([1, 2, 3, 4])
        >>> after = np.array([0, 2, 2, 4])
        >>> targs = np.array([0, 1, 2, 3])
        >>> rates = compute_resolution_rates(before, after, targs, radical_table)
        >>> print(f"Overall: {rates['overall_resolution']:.1f}%")
    """
    # Get error categories before correction
    errors_before_mask = predictions_before != targets
    errors_after_mask = predictions_after != targets

    # Track resolutions by category
    category_errors = {
        "single_stroke_diff": [],
        "same_radical_diff_structure": [],
        "similar_radical": [],
        "unrelated": []
    }

    # Categorize errors and track which were resolved
    for idx, (pred_before, pred_after, target) in enumerate(
        zip(predictions_before, predictions_after, targets)
    ):
        if pred_before == target:
            continue  # Not an error before

        # Get character properties
        pred_info = radical_table.get(
            int(pred_before),
            {"radicals": [], "structure": -1, "strokes": 0}
        )
        targ_info = radical_table.get(
            int(target),
            {"radicals": [], "structure": -1, "strokes": 0}
        )

        pred_radicals = set(pred_info["radicals"])
        targ_radicals = set(targ_info["radicals"])
        pred_structure = pred_info["structure"]
        targ_structure = targ_info["structure"]
        pred_strokes = pred_info["strokes"]
        targ_strokes = targ_info["strokes"]

        # Categorize
        stroke_diff = abs(pred_strokes - targ_strokes)
        was_resolved = (pred_after == target)

        if stroke_diff == 1:
            category_errors["single_stroke_diff"].append(was_resolved)
        elif (pred_radicals == targ_radicals and
              len(pred_radicals) > 0 and
              pred_structure != targ_structure):
            category_errors["same_radical_diff_structure"].append(was_resolved)
        elif len(pred_radicals) > 0 and len(targ_radicals) > 0:
            overlap = len(pred_radicals & targ_radicals)
            union = len(pred_radicals | targ_radicals)
            if overlap / union > 0.5:
                category_errors["similar_radical"].append(was_resolved)
            else:
                category_errors["unrelated"].append(was_resolved)
        else:
            category_errors["unrelated"].append(was_resolved)

    # Compute resolution rates
    resolution_rates = {}
    for category, resolutions in category_errors.items():
        if len(resolutions) > 0:
            rate = (sum(resolutions) / len(resolutions)) * 100
            resolution_rates[f"{category}_resolution"] = rate
        else:
            resolution_rates[f"{category}_resolution"] = 0.0

    # Compute overall resolution rate
    total_errors_before = errors_before_mask.sum()
    total_errors_after = errors_after_mask.sum()
    if total_errors_before > 0:
        resolved_count = total_errors_before - total_errors_after
        overall_rate = (resolved_count / total_errors_before) * 100
        resolution_rates["overall_resolution"] = overall_rate
    else:
        resolution_rates["overall_resolution"] = 0.0

    return resolution_rates


def plot_confusion_matrix(
    predictions: np.ndarray,
    targets: np.ndarray,
    class_names: List[str],
    top_n: int = 50,
    save_path: Optional[Path] = None
) -> None:
    """Plot confusion matrix for the most confused classes.

    Creates a heatmap showing confusion patterns among the classes
    with the highest error rates.

    Args:
        predictions: Predicted class indices, shape (N,).
        targets: Ground truth class indices, shape (N,).
        class_names: List of class names indexed by class ID.
        top_n: Number of most-confused classes to include.
        save_path: Optional path to save the figure.

    Example:
        >>> preds = np.array([0, 1, 2, 1, 0])
        >>> targs = np.array([0, 1, 1, 1, 0])
        >>> names = ["A", "B", "C"]
        >>> plot_confusion_matrix(preds, targs, names, top_n=3,
        ...                      save_path=Path("confusion.png"))

    Note:
        The function identifies the top_n classes with the most errors
        and plots their confusion matrix for detailed analysis.
    """
    # Compute full confusion matrix
    n_classes = len(class_names)
    cm = confusion_matrix(targets, predictions, labels=np.arange(n_classes))

    # Find classes with most errors (excluding correct predictions)
    errors_per_class = cm.sum(axis=1) - np.diag(cm)
    top_confused_indices = np.argsort(errors_per_class)[-top_n:][::-1]

    # Extract submatrix for top confused classes
    cm_subset = cm[np.ix_(top_confused_indices, top_confused_indices)]

    # Normalize by row (true class)
    cm_normalized = cm_subset.astype('float')
    row_sums = cm_subset.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    cm_normalized = cm_normalized / row_sums

    # Get subset class names
    subset_names = [class_names[i] for i in top_confused_indices]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot heatmap using matplotlib imshow
    im = ax.imshow(cm_normalized, cmap='YlOrRd', aspect='auto')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Normalized Frequency')

    # Set tick labels
    ax.set_xticks(np.arange(len(subset_names)))
    ax.set_yticks(np.arange(len(subset_names)))
    ax.set_xticklabels(subset_names)
    ax.set_yticklabels(subset_names)

    ax.set_xlabel('Predicted Class', fontsize=12)
    ax.set_ylabel('True Class', fontsize=12)
    ax.set_title(
        f'Confusion Matrix - Top {top_n} Most Confused Classes',
        fontsize=14,
        pad=20
    )

    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()

    # Save or show
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def get_top_confused_pairs(
    predictions: np.ndarray,
    targets: np.ndarray,
    class_names: List[str],
    top_k: int = 20
) -> List[tuple]:
    """Get the most frequently confused class pairs.

    Args:
        predictions: Predicted class indices, shape (N,).
        targets: Ground truth class indices, shape (N,).
        class_names: List of class names indexed by class ID.
        top_k: Number of top confused pairs to return.

    Returns:
        List of tuples: [(true_class, pred_class, count, name_true, name_pred), ...]

    Example:
        >>> pairs = get_top_confused_pairs(preds, targs, names, top_k=10)
        >>> for true_cls, pred_cls, cnt, name_t, name_p in pairs[:5]:
        ...     print(f"{name_t} -> {name_p}: {cnt} times")
    """
    # Count confusion pairs (excluding correct predictions)
    confusion_pairs = []
    for true_idx, pred_idx in zip(targets, predictions):
        if true_idx != pred_idx:
            confusion_pairs.append((int(true_idx), int(pred_idx)))

    # Count frequencies
    pair_counts = Counter(confusion_pairs)

    # Get top-k most common
    top_pairs = pair_counts.most_common(top_k)

    # Add class names
    result = []
    for (true_idx, pred_idx), count in top_pairs:
        true_name = class_names[true_idx] if true_idx < len(class_names) else f"Class_{true_idx}"
        pred_name = class_names[pred_idx] if pred_idx < len(class_names) else f"Class_{pred_idx}"
        result.append((true_idx, pred_idx, count, true_name, pred_name))

    return result
