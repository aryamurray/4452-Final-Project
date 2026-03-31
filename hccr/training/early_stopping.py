"""Early stopping implementation for training loops.

Monitors a metric and stops training when it stops improving.
Supports both minimization (e.g., loss) and maximization (e.g., accuracy) modes.
"""

from typing import Literal, Optional


class EarlyStopping:
    """Early stopping to halt training when monitored metric stops improving.

    Args:
        patience: Number of epochs to wait for improvement before stopping
        mode: "min" for metrics to minimize (loss), "max" for metrics to maximize (accuracy)
        min_delta: Minimum change in metric to qualify as improvement

    Example:
        >>> early_stop = EarlyStopping(patience=5, mode="max")
        >>> for epoch in range(100):
        ...     val_acc = train_and_validate()
        ...     if early_stop(val_acc):
        ...         print(f"Early stopping at epoch {epoch}")
        ...         print(f"Best score: {early_stop.best_score}")
        ...         break
    """

    def __init__(
        self,
        patience: int,
        mode: Literal["min", "max"] = "min",
        min_delta: float = 0.0,
    ) -> None:
        """Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement
            mode: "min" to minimize metric, "max" to maximize metric
            min_delta: Minimum absolute change to count as improvement

        Raises:
            ValueError: If mode is not "min" or "max"
        """
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")

        self.patience = patience
        self.mode = mode
        self.min_delta = abs(min_delta)

        self._best_score: Optional[float] = None
        self._counter = 0
        self._improved = False

    def __call__(self, metric: float) -> bool:
        """Check if training should stop based on current metric.

        Args:
            metric: Current metric value to monitor

        Returns:
            True if training should stop (patience exhausted), False otherwise
        """
        # Initialize best score on first call
        if self._best_score is None:
            self._best_score = metric
            self._improved = True
            return False

        # Check if current metric improved over best score
        if self._is_improvement(metric, self._best_score):
            self._best_score = metric
            self._counter = 0
            self._improved = True
            return False
        else:
            self._counter += 1
            self._improved = False

            # Check if patience exhausted
            if self._counter >= self.patience:
                return True

            return False

    def _is_improvement(self, current: float, best: float) -> bool:
        """Check if current metric is an improvement over best.

        Args:
            current: Current metric value
            best: Best metric value so far

        Returns:
            True if current is better than best by at least min_delta
        """
        if self.mode == "min":
            return current < best - self.min_delta
        else:  # mode == "max"
            return current > best + self.min_delta

    @property
    def best_score(self) -> Optional[float]:
        """Get the best score observed so far.

        Returns:
            Best score, or None if no metrics have been recorded yet
        """
        return self._best_score

    @property
    def improved(self) -> bool:
        """Check if last call resulted in improvement.

        Returns:
            True if the last call to __call__ improved the metric
        """
        return self._improved

    def reset(self) -> None:
        """Reset early stopping state.

        Clears best score and counter. Useful for restarting training
        or switching to a different metric.
        """
        self._best_score = None
        self._counter = 0
        self._improved = False

    def __repr__(self) -> str:
        """String representation of early stopping state."""
        return (
            f"EarlyStopping(patience={self.patience}, mode='{self.mode}', "
            f"min_delta={self.min_delta}, best_score={self._best_score}, "
            f"counter={self._counter})"
        )
