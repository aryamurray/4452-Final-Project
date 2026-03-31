"""Evaluation module for HCCR model assessment."""

from .metrics import (
    top_k_accuracy,
    weighted_f1,
    per_class_precision_recall,
    character_error_rate,
)
from .benchmark import (
    measure_model_size,
    measure_inference_latency,
    quantize_and_measure,
    run_full_benchmark,
)
from .confusion import (
    categorize_errors,
    compute_resolution_rates,
    plot_confusion_matrix,
)

__all__ = [
    "top_k_accuracy",
    "weighted_f1",
    "per_class_precision_recall",
    "character_error_rate",
    "measure_model_size",
    "measure_inference_latency",
    "quantize_and_measure",
    "run_full_benchmark",
    "categorize_errors",
    "compute_resolution_rates",
    "plot_confusion_matrix",
]
