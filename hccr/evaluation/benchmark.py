"""Benchmarking utilities for model performance analysis.

This module provides tools for measuring:
- Model size (parameters and buffers)
- Inference latency (with warmup and statistical analysis)
- Quantization effects (size reduction and accuracy impact)
- Full benchmark comparison across multiple models
"""

import tempfile
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic

from .metrics import top_k_accuracy


def measure_model_size(model: nn.Module) -> float:
    """Measure total model size including parameters and buffers.

    Args:
        model: PyTorch model to measure.

    Returns:
        Model size in megabytes (MB).

    Example:
        >>> from torch import nn
        >>> model = nn.Sequential(nn.Linear(100, 50), nn.Linear(50, 10))
        >>> size = measure_model_size(model)
        >>> print(f"Model size: {size:.2f} MB")
    """
    param_bytes = sum(
        p.numel() * p.element_size()
        for p in model.parameters()
    )
    buffer_bytes = sum(
        b.numel() * b.element_size()
        for b in model.buffers()
    )
    total_bytes = param_bytes + buffer_bytes
    return total_bytes / (1024 * 1024)


def measure_inference_latency(
    model: nn.Module,
    device: torch.device,
    input_size: Tuple[int, int, int, int] = (1, 1, 64, 64),
    n_runs: int = 1000,
    warmup: int = 100
) -> Dict[str, float]:
    """Measure model inference latency with statistical analysis.

    Performs warmup runs to stabilize GPU/CPU state, then measures
    inference time across multiple runs.

    Args:
        model: PyTorch model to benchmark.
        device: Device to run inference on (cuda or cpu).
        input_size: Input tensor shape (batch, channels, height, width).
        n_runs: Number of timed inference runs.
        warmup: Number of warmup runs before timing.

    Returns:
        Dictionary containing:
            - mean_ms: Mean inference time in milliseconds
            - std_ms: Standard deviation in milliseconds
            - median_ms: Median inference time in milliseconds

    Example:
        >>> model = MyModel().to(device)
        >>> stats = measure_inference_latency(model, device, n_runs=500)
        >>> print(f"Latency: {stats['mean_ms']:.2f} ± {stats['std_ms']:.2f} ms")
    """
    model.eval()
    model.to(device)

    # Create dummy input
    dummy_input = torch.randn(input_size, device=device)

    # Warmup runs to stabilize GPU state
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()

    # Timed runs
    timings = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()

            start_time = time.perf_counter()
            _ = model(dummy_input)

            if device.type == 'cuda':
                torch.cuda.synchronize()

            end_time = time.perf_counter()
            timings.append((end_time - start_time) * 1000)  # Convert to ms

    timings_array = np.array(timings)

    return {
        "mean_ms": float(np.mean(timings_array)),
        "std_ms": float(np.std(timings_array)),
        "median_ms": float(np.median(timings_array))
    }


def quantize_and_measure(
    model: nn.Module,
    test_loader,
    device: torch.device,
    label_map_size: int = 3755
) -> Dict[str, float]:
    """Apply dynamic quantization and measure size/accuracy impact.

    Quantizes Linear layers to int8, measures compression ratio,
    and evaluates accuracy degradation.

    Args:
        model: Original PyTorch model (will not be modified).
        test_loader: DataLoader for evaluation dataset.
        device: Device for inference (quantization requires CPU).
        label_map_size: Number of classes for top-1 accuracy.

    Returns:
        Dictionary containing:
            - original_size_mb: Original model size in MB
            - quantized_size_mb: Quantized model size in MB
            - original_acc: Original top-1 accuracy (%)
            - quantized_acc: Quantized top-1 accuracy (%)
            - compression_ratio: Size reduction factor

    Example:
        >>> results = quantize_and_measure(model, test_loader, device)
        >>> print(f"Compression: {results['compression_ratio']:.2f}x")
        >>> print(f"Accuracy drop: {results['original_acc'] - results['quantized_acc']:.2f}%")

    Note:
        Dynamic quantization is applied to torch.nn.Linear layers only.
        The model is temporarily moved to CPU for quantization.
    """
    # Measure original model
    original_size = measure_model_size(model)

    # Evaluate original accuracy
    model.eval()
    model.to(device)
    original_acc = _evaluate_top1_accuracy(model, test_loader, device)

    # Create quantized model (must be on CPU)
    model_cpu = model.cpu()
    quantized_model = quantize_dynamic(
        model_cpu,
        {nn.Linear},  # Quantize Linear layers
        dtype=torch.qint8
    )

    # Measure quantized size by saving to temp file
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
        tmp_path = Path(tmp.name)
        torch.save(quantized_model.state_dict(), tmp_path)
        quantized_size = tmp_path.stat().st_size / (1024 * 1024)
        tmp_path.unlink()  # Clean up

    # Evaluate quantized accuracy (on CPU)
    quantized_acc = _evaluate_top1_accuracy(
        quantized_model,
        test_loader,
        torch.device('cpu')
    )

    # Move original model back to specified device
    model.to(device)

    compression_ratio = original_size / quantized_size if quantized_size > 0 else 0.0

    return {
        "original_size_mb": original_size,
        "quantized_size_mb": quantized_size,
        "original_acc": original_acc,
        "quantized_acc": quantized_acc,
        "compression_ratio": compression_ratio
    }


def _evaluate_top1_accuracy(
    model: nn.Module,
    test_loader,
    device: torch.device
) -> float:
    """Helper function to evaluate top-1 accuracy on test set.

    Args:
        model: Model to evaluate.
        test_loader: DataLoader for test dataset.
        device: Device to run evaluation on.

    Returns:
        Top-1 accuracy as a percentage (0-100).
    """
    model.eval()
    model.to(device)

    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            # Handle different batch formats
            if isinstance(batch, (list, tuple)):
                images, targets = batch[0], batch[1]
            else:
                images = batch['image']
                targets = batch['label']

            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)

            # Handle multi-output models (e.g., multi-task)
            if isinstance(outputs, dict):
                if "char_logits" not in outputs:
                    # Model has no character classification head (e.g., TinyCNNRadical)
                    return -1.0
                outputs = outputs["char_logits"]
            elif isinstance(outputs, (list, tuple)):
                outputs = outputs[0]  # Use first output (main classifier)

            all_outputs.append(outputs.cpu())
            all_targets.append(targets.cpu())

    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    acc = top_k_accuracy(all_outputs, all_targets, k_values=[1])
    return acc[1]


def run_full_benchmark(
    models: Dict[str, nn.Module],
    device: torch.device,
    test_loader=None
) -> pd.DataFrame:
    """Run comprehensive benchmark across multiple models.

    Measures size, parameter count, and inference latency for each model.
    Optionally includes accuracy if test_loader is provided.

    Args:
        models: Dictionary mapping model names to model instances.
        device: Device for benchmarking.
        test_loader: Optional DataLoader for accuracy evaluation.

    Returns:
        pandas DataFrame with columns:
            - model: Model name
            - size_mb: Model size in MB
            - params: Number of trainable parameters
            - latency_mean_ms: Mean inference latency
            - latency_std_ms: Latency standard deviation
            - top1_acc: Top-1 accuracy (if test_loader provided)

    Example:
        >>> models = {
        ...     "ResNet18": resnet18_model,
        ...     "MobileNet": mobilenet_model
        ... }
        >>> df = run_full_benchmark(models, device, test_loader)
        >>> print(df.to_string(index=False))
    """
    results = []

    for model_name, model in models.items():
        print(f"Benchmarking {model_name}...")

        # Measure size
        size_mb = measure_model_size(model)

        # Count parameters
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Measure latency
        latency_stats = measure_inference_latency(
            model,
            device,
            input_size=(1, 1, 64, 64),
            n_runs=1000,
            warmup=100
        )

        result = {
            "model": model_name,
            "size_mb": size_mb,
            "params": params,
            "latency_mean_ms": latency_stats["mean_ms"],
            "latency_std_ms": latency_stats["std_ms"]
        }

        # Optional: measure accuracy if test_loader provided
        if test_loader is not None:
            top1_acc = _evaluate_top1_accuracy(model, test_loader, device)
            if top1_acc >= 0:  # -1.0 means model has no char head
                result["top1_acc"] = top1_acc

        results.append(result)

    df = pd.DataFrame(results)
    return df
