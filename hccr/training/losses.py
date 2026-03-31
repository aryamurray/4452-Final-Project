"""Loss functions for HCCR training.

This module provides four loss classes for different training modes:
- ClassificationLoss: Standard cross-entropy for character classification
- RadicalMultiTaskLoss: Multi-task loss for radical/structure/stroke prediction
- JointLoss: Combined loss for character classification + radical features
- SymbolicLoss: Loss for differentiable symbolic layer training
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationLoss(nn.Module):
    """Cross-entropy loss for character classification with optional label smoothing.

    Args:
        label_smoothing: Label smoothing factor (0.0 = no smoothing, 0.1 = 10% smoothing)

    Example:
        >>> loss_fn = ClassificationLoss(label_smoothing=0.1)
        >>> logits = torch.randn(32, 3755)
        >>> targets = torch.randint(0, 3755, (32,))
        >>> loss = loss_fn(logits, targets)
    """

    def __init__(self, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.label_smoothing = label_smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss.

        Args:
            logits: Model outputs of shape (batch_size, num_classes)
            targets: Ground truth labels of shape (batch_size,)

        Returns:
            Scalar loss tensor
        """
        return self.criterion(logits, targets)


class RadicalMultiTaskLoss(nn.Module):
    """Multi-task loss for radical decomposition learning.

    Combines three loss components:
    - Binary cross-entropy for radical multi-label classification
    - Cross-entropy for structure type classification
    - MSE for stroke count regression

    Args:
        lambda_radical: Weight for radical loss component
        lambda_structure: Weight for structure loss component
        lambda_stroke: Weight for stroke count loss component

    Example:
        >>> loss_fn = RadicalMultiTaskLoss(lambda_radical=1.0, lambda_structure=0.5, lambda_stroke=0.1)
        >>> radical_logits = torch.randn(32, 500)
        >>> structure_logits = torch.randn(32, 13)
        >>> stroke_pred = torch.randn(32, 1)
        >>> radical_targets = torch.randint(0, 2, (32, 500)).float()
        >>> structure_targets = torch.randint(0, 13, (32,))
        >>> stroke_targets = torch.randint(1, 30, (32, 1)).float()
        >>> total_loss, components = loss_fn(
        ...     radical_logits, structure_logits, stroke_pred,
        ...     radical_targets, structure_targets, stroke_targets
        ... )
    """

    def __init__(
        self,
        lambda_radical: float = 1.0,
        lambda_structure: float = 0.5,
        lambda_stroke: float = 0.1,
    ) -> None:
        super().__init__()
        self.lambda_radical = lambda_radical
        self.lambda_structure = lambda_structure
        self.lambda_stroke = lambda_stroke

        # BCE with logits is more numerically stable than sigmoid + BCE
        self.radical_criterion = nn.BCEWithLogitsLoss()
        self.structure_criterion = nn.CrossEntropyLoss()
        self.stroke_criterion = nn.MSELoss()

    def forward(
        self,
        radical_logits: torch.Tensor,
        structure_logits: torch.Tensor,
        stroke_pred: torch.Tensor,
        radical_targets: torch.Tensor,
        structure_targets: torch.Tensor,
        stroke_targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute weighted multi-task loss.

        Args:
            radical_logits: Radical predictions of shape (batch_size, num_radicals)
            structure_logits: Structure predictions of shape (batch_size, num_structures)
            stroke_pred: Stroke count predictions of shape (batch_size, 1)
            radical_targets: Binary radical labels of shape (batch_size, num_radicals)
            structure_targets: Structure class labels of shape (batch_size,)
            stroke_targets: Ground truth stroke counts of shape (batch_size, 1)

        Returns:
            total_loss: Weighted sum of all loss components
            components: Dictionary mapping loss names to their values (detached)
        """
        # Compute individual losses
        radical_loss = self.radical_criterion(radical_logits, radical_targets)
        structure_loss = self.structure_criterion(structure_logits, structure_targets)
        stroke_loss = self.stroke_criterion(stroke_pred, stroke_targets)

        # Weighted combination
        total_loss = (
            self.lambda_radical * radical_loss
            + self.lambda_structure * structure_loss
            + self.lambda_stroke * stroke_loss
        )

        # Return components as dict for logging
        components = {
            "radical_loss": radical_loss.item(),
            "structure_loss": structure_loss.item(),
            "stroke_loss": stroke_loss.item(),
        }

        return total_loss, components


class JointLoss(nn.Module):
    """Joint loss for character classification and radical feature learning.

    Combines character classification loss with radical decomposition losses.
    Useful for training models that predict both character identity and
    structural features simultaneously.

    Args:
        label_smoothing: Label smoothing for character classification
        lambda_radical: Weight for radical loss component
        lambda_structure: Weight for structure loss component
        lambda_stroke: Weight for stroke count loss component
        stroke_as_class: If True, use CrossEntropyLoss for stroke count
            (expects logits of shape (B, num_strokes) and int targets).
            If False, use MSELoss for regression (existing behavior).
    """

    def __init__(
        self,
        label_smoothing: float = 0.0,
        lambda_radical: float = 1.0,
        lambda_structure: float = 0.5,
        lambda_stroke: float = 0.1,
        stroke_as_class: bool = False,
    ) -> None:
        super().__init__()
        self.label_smoothing = label_smoothing
        self.lambda_radical = lambda_radical
        self.lambda_structure = lambda_structure
        self.lambda_stroke = lambda_stroke
        self.stroke_as_class = stroke_as_class

        # Character classification loss
        self.char_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # Radical feature losses
        self.radical_criterion = nn.BCEWithLogitsLoss()
        self.structure_criterion = nn.CrossEntropyLoss()
        if stroke_as_class:
            self.stroke_criterion = nn.CrossEntropyLoss()
        else:
            self.stroke_criterion = nn.MSELoss()

    def forward(
        self,
        char_logits: torch.Tensor,
        radical_logits: torch.Tensor,
        structure_logits: torch.Tensor,
        stroke_pred: torch.Tensor,
        char_targets: torch.Tensor,
        radical_targets: torch.Tensor,
        structure_targets: torch.Tensor,
        stroke_targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute joint loss for character and radical prediction.

        Args:
            char_logits: Character predictions of shape (batch_size, num_classes)
            radical_logits: Radical predictions of shape (batch_size, num_radicals)
            structure_logits: Structure predictions of shape (batch_size, num_structures)
            stroke_pred: Stroke count predictions of shape (batch_size, 1) or (batch_size, num_strokes)
            char_targets: Character labels of shape (batch_size,)
            radical_targets: Binary radical labels of shape (batch_size, num_radicals)
            structure_targets: Structure class labels of shape (batch_size,)
            stroke_targets: Stroke counts — (batch_size, 1) float or (batch_size,) int

        Returns:
            total_loss: Weighted sum of all loss components
            components: Dictionary mapping loss names to their values (detached)
        """
        # Character classification loss
        char_loss = self.char_criterion(char_logits, char_targets)

        # Radical feature losses
        radical_loss = self.radical_criterion(radical_logits, radical_targets)
        structure_loss = self.structure_criterion(structure_logits, structure_targets)
        stroke_loss = self.stroke_criterion(stroke_pred, stroke_targets)

        # Weighted combination
        total_loss = (
            char_loss
            + self.lambda_radical * radical_loss
            + self.lambda_structure * structure_loss
            + self.lambda_stroke * stroke_loss
        )

        # Return components as dict for logging
        components = {
            "char_loss": char_loss.item(),
            "radical_loss": radical_loss.item(),
            "structure_loss": structure_loss.item(),
            "stroke_loss": stroke_loss.item(),
        }

        return total_loss, components


class SymbolicLoss(nn.Module):
    """Loss for differentiable symbolic layer training.

    Combines:
    - NLL loss on symbolically-reranked combined distribution (primary)
    - Direct CE on raw character logits (auxiliary)
    - BCE on radical multi-label predictions
    - CE on structure classification
    - CE on stroke count classification

    The key innovation: gradients from loss_combined flow through the symbolic
    layer into ALL four heads. Character-level errors improve auxiliary
    predictions via the symbolic coupling.

    Args:
        label_smoothing: Label smoothing for character CE losses
        lambda_combined: Weight for symbolic combined loss (primary)
        lambda_char: Weight for direct character CE (auxiliary)
        lambda_radical: Weight for radical BCE loss
        lambda_structure: Weight for structure CE loss
        lambda_stroke: Weight for stroke CE loss
    """

    def __init__(
        self,
        label_smoothing: float = 0.0,
        lambda_combined: float = 1.0,
        lambda_char: float = 0.5,
        lambda_radical: float = 0.5,
        lambda_structure: float = 0.3,
        lambda_stroke: float = 0.1,
    ) -> None:
        super().__init__()
        self.lambda_combined = lambda_combined
        self.lambda_char = lambda_char
        self.lambda_radical = lambda_radical
        self.lambda_structure = lambda_structure
        self.lambda_stroke = lambda_stroke

        self.char_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.radical_criterion = nn.BCEWithLogitsLoss()
        self.structure_criterion = nn.CrossEntropyLoss()
        self.stroke_criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        combined_probs: torch.Tensor,
        char_logits: torch.Tensor,
        radical_logits: torch.Tensor,
        structure_logits: torch.Tensor,
        stroke_logits: torch.Tensor,
        char_targets: torch.Tensor,
        radical_targets: torch.Tensor,
        structure_targets: torch.Tensor,
        stroke_targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute symbolic training loss.

        Args:
            combined_probs: Symbolically-reranked distribution (B, num_classes)
            char_logits: Raw character logits (B, num_classes)
            radical_logits: Radical logits (B, num_radicals)
            structure_logits: Structure logits (B, num_structures)
            stroke_logits: Stroke count logits (B, num_strokes)
            char_targets: Character class labels (B,)
            radical_targets: Binary radical labels (B, num_radicals)
            structure_targets: Structure class labels (B,)
            stroke_targets: Stroke count class labels (B,) int

        Returns:
            total_loss: Weighted sum of all loss components
            components: Dictionary mapping loss names to their values (detached)
        """
        # Primary: NLL on combined (symbolically-reranked) distribution
        loss_combined = F.nll_loss(
            torch.log(combined_probs + 1e-8), char_targets
        )

        # Auxiliary: direct supervision on each head
        loss_char = self.char_criterion(char_logits, char_targets)
        loss_radical = self.radical_criterion(radical_logits, radical_targets)
        loss_structure = self.structure_criterion(structure_logits, structure_targets)
        loss_stroke = self.stroke_criterion(stroke_logits, stroke_targets)

        # Weighted combination
        total_loss = (
            self.lambda_combined * loss_combined
            + self.lambda_char * loss_char
            + self.lambda_radical * loss_radical
            + self.lambda_structure * loss_structure
            + self.lambda_stroke * loss_stroke
        )

        components = {
            "combined_loss": loss_combined.item(),
            "char_loss": loss_char.item(),
            "radical_loss": loss_radical.item(),
            "structure_loss": loss_structure.item(),
            "stroke_loss": loss_stroke.item(),
        }

        return total_loss, components
