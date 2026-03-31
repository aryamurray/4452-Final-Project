"""Training loop implementation for HCCR models.

Supports three training modes:
- classification: Standard character classification
- radical: Multi-task radical/structure/stroke prediction
- joint: Character classification + radical features
"""

import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from hccr.config import TrainConfig
from hccr.training.early_stopping import EarlyStopping
from hccr.training.losses import ClassificationLoss, JointLoss, RadicalMultiTaskLoss, SymbolicLoss
from hccr.utils import get_logger, save_checkpoint

logger = get_logger(__name__)


class Trainer:
    """Training loop manager for HCCR models.

    Handles training, validation, checkpointing, logging, and early stopping
    for three different training modes.

    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        config: Training configuration
        mode: Training mode - "classification", "radical", or "joint"
        device: Device to train on (CPU or CUDA)
        log_dir: Directory for logs and checkpoints

    Example:
        >>> from hccr.config import TrainConfig
        >>> config = TrainConfig(epochs=30, batch_size=64)
        >>> trainer = Trainer(
        ...     model=my_model,
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     config=config,
        ...     mode="classification",
        ...     device=torch.device("cuda"),
        ...     log_dir=Path("outputs/logs")
        ... )
        >>> history = trainer.train()
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainConfig,
        mode: Literal["classification", "radical", "joint"],
        device: torch.device,
        log_dir: Path,
        resume_from: Optional[Path] = None,
    ) -> None:
        """Initialize trainer with model, data, and configuration."""
        if mode not in ("classification", "radical", "joint", "symbolic"):
            raise ValueError(
                f"mode must be 'classification', 'radical', 'joint', or 'symbolic', got '{mode}'"
            )

        # cuDNN autotuner — picks fastest convolution algorithms for fixed input sizes
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.mode = mode
        self.device = device
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.start_epoch = 1

        # Initialize logger
        self.logger = get_logger(
            "Trainer", log_file=self.log_dir / f"train_{mode}.log"
        )

        # Initialize optimizer and scheduler (only trainable params)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = Adam(
            trainable_params,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=config.epochs, eta_min=1e-6
        )

        # Resume from checkpoint if provided
        if resume_from is not None:
            self.logger.info(f"Resuming from checkpoint: {resume_from}")
            ckpt = torch.load(resume_from, map_location=device, weights_only=False)
            if "optimizer_state_dict" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                # Reset LR to config value for warm restart
                for pg in self.optimizer.param_groups:
                    pg["lr"] = config.lr
            # Fresh cosine schedule over the new epochs (warm restart)
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=config.epochs, eta_min=1e-6
            )
            self.start_epoch = ckpt.get("epoch", 0) + 1
            self.logger.info(f"Resuming from epoch {self.start_epoch} with warm restart (LR={config.lr})")

        # Initialize loss function based on mode
        self.criterion = self._create_loss_function()

        # Initialize early stopping
        # For classification/joint/symbolic, maximize accuracy; for radical, minimize loss
        if mode in ("classification", "joint", "symbolic"):
            self.early_stopping = EarlyStopping(
                patience=config.early_stopping_patience,
                mode="max",
                min_delta=1e-4,
            )
            self.monitor_metric = "val_acc"
        else:  # radical mode
            self.early_stopping = EarlyStopping(
                patience=config.early_stopping_patience,
                mode="min",
                min_delta=1e-4,
            )
            self.monitor_metric = "val_loss"

        self.best_metric = None

        # AMP (automatic mixed precision)
        self.use_amp = device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # TensorBoard writer (single directory per log_dir so restarts append)
        self.writer = SummaryWriter(log_dir=str(self.log_dir / "tb"))

        self.logger.info(f"Trainer initialized in '{mode}' mode")
        self.logger.info(f"Device: {device}")
        self.logger.info(f"Monitoring metric: {self.monitor_metric}")

    def _create_loss_function(self) -> nn.Module:
        """Create loss function based on training mode."""
        if self.mode == "classification":
            return ClassificationLoss(label_smoothing=self.config.label_smoothing)
        elif self.mode == "radical":
            return RadicalMultiTaskLoss(
                lambda_radical=self.config.lambda_radical,
                lambda_structure=self.config.lambda_structure,
                lambda_stroke=self.config.lambda_stroke,
            )
        elif self.mode == "symbolic":
            return SymbolicLoss(
                label_smoothing=self.config.label_smoothing,
                lambda_combined=self.config.lambda_combined,
                lambda_char=self.config.lambda_char,
                lambda_radical=self.config.lambda_radical,
                lambda_structure=self.config.lambda_structure,
                lambda_stroke=self.config.lambda_stroke,
            )
        else:  # joint
            return JointLoss(
                label_smoothing=self.config.label_smoothing,
                lambda_radical=self.config.lambda_radical,
                lambda_structure=self.config.lambda_structure,
                lambda_stroke=self.config.lambda_stroke,
                stroke_as_class=(self.config.num_strokes > 1),
            )

    def train(self) -> Dict[str, List[float]]:
        """Run full training loop.

        Returns:
            Dictionary containing training history with keys:
            - train_loss: List of training losses per epoch
            - val_loss: List of validation losses per epoch
            - val_acc: List of validation accuracies per epoch (if applicable)
            - lr: List of learning rates per epoch
            - Additional component losses for multi-task modes
        """
        self.logger.info("Starting training...")
        self.logger.info(f"Epochs: {self.config.epochs}")
        self.logger.info(f"Batch size: {self.config.batch_size}")
        self.logger.info(f"Learning rate: {self.config.lr}")
        self.logger.info(f"Gradient accumulation steps: {self.config.grad_accum_steps}")

        # Initialize history tracking
        history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "lr": [],
        }
        if self.mode in ("classification", "joint", "symbolic"):
            history["val_acc"] = []
            history["val_top5"] = []
            history["val_top10"] = []
        if self.mode == "symbolic":
            history["val_top20"] = []

        # Initialize CSV logging
        csv_path = self.log_dir / f"metrics_{self.mode}.csv"
        csv_header = self._get_csv_header()
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_header)
            writer.writeheader()

        # Training loop
        end_epoch = self.start_epoch + self.config.epochs - 1
        for epoch in range(self.start_epoch, end_epoch + 1):
            self.logger.info(f"\nEpoch {epoch}/{end_epoch}")

            # Train one epoch
            train_metrics = self.train_one_epoch(epoch)
            history["train_loss"].append(train_metrics["loss"])

            # Validate
            val_metrics = self.validate()
            history["val_loss"].append(val_metrics["loss"])
            if "accuracy" in val_metrics:
                history["val_acc"].append(val_metrics["accuracy"])
            if "top5_acc" in val_metrics:
                history["val_top5"].append(val_metrics["top5_acc"])
            if "top10_acc" in val_metrics:
                history["val_top10"].append(val_metrics["top10_acc"])
            if "top20_acc" in val_metrics:
                history["val_top20"].append(val_metrics["top20_acc"])

            # Track learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            history["lr"].append(current_lr)

            # Log metrics
            log_msg = (
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f}"
            )
            if "accuracy" in val_metrics:
                log_msg += f" | Val Acc: {val_metrics['accuracy']:.4f}"
            if "top5_acc" in val_metrics:
                log_msg += f" | Top5: {val_metrics['top5_acc']:.4f}"
            if "top10_acc" in val_metrics:
                log_msg += f" | Top10: {val_metrics['top10_acc']:.4f}"
            if "top20_acc" in val_metrics:
                log_msg += f" | Top20: {val_metrics['top20_acc']:.4f}"
            log_msg += f" | LR: {current_lr:.6f}"
            # Add component losses to console output
            for key in ("char_loss", "radical_loss", "structure_loss", "stroke_loss", "combined_loss"):
                if key in val_metrics:
                    short = key.replace("_loss", "")
                    log_msg += f" | {short}: {val_metrics[key]:.4f}"
            self.logger.info(log_msg)

            # Write to CSV
            csv_row = self._prepare_csv_row(epoch, train_metrics, val_metrics, current_lr)
            with open(csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=csv_header)
                writer.writerow(csv_row)

            # TensorBoard logging
            self.writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
            self.writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
            self.writer.add_scalar("LR", current_lr, epoch)
            if "accuracy" in val_metrics:
                self.writer.add_scalar("Accuracy/val", val_metrics["accuracy"], epoch)
            if "top5_acc" in val_metrics:
                self.writer.add_scalar("Accuracy/val_top5", val_metrics["top5_acc"], epoch)
            if "top10_acc" in val_metrics:
                self.writer.add_scalar("Accuracy/val_top10", val_metrics["top10_acc"], epoch)
            if "top20_acc" in val_metrics:
                self.writer.add_scalar("Accuracy/val_top20", val_metrics["top20_acc"], epoch)
            for key, value in train_metrics.items():
                if key != "loss":
                    self.writer.add_scalar(f"Components/train_{key}", value, epoch)
            for key, value in val_metrics.items():
                if key not in ("loss", "accuracy"):
                    self.writer.add_scalar(f"Components/val_{key}", value, epoch)
            self.writer.flush()

            # Update scheduler
            self.scheduler.step()

            # Early stopping check
            monitor_value = (
                val_metrics["accuracy"]
                if self.monitor_metric == "val_acc"
                else val_metrics["loss"]
            )

            if self.early_stopping(monitor_value):
                self.logger.info(
                    f"Early stopping triggered at epoch {epoch}. "
                    f"Best {self.monitor_metric}: {self.early_stopping.best_score:.4f}"
                )
                break

            # Save best checkpoint
            if self.early_stopping.improved:
                self.best_metric = monitor_value
                checkpoint_path = self.log_dir / f"best_model_{self.mode}.pt"
                save_checkpoint(
                    self.model,
                    checkpoint_path,
                    epoch=epoch,
                    optimizer_state_dict=self.optimizer.state_dict(),
                    scheduler_state_dict=self.scheduler.state_dict(),
                    best_metric=self.best_metric,
                    config=self.config.__dict__,
                )
                self.logger.info(
                    f"Saved best checkpoint with {self.monitor_metric}: {self.best_metric:.4f}"
                )

        self.writer.close()
        self.logger.info("Training completed!")
        self.logger.info(f"Best {self.monitor_metric}: {self.best_metric:.4f}")

        return history

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch with gradient accumulation.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of training metrics including loss and component losses
        """
        self.model.train()
        total_loss = torch.tensor(0.0, device=self.device)
        num_batches = 0

        # Component loss tracking for multi-task modes
        component_losses: Dict[str, float] = {}

        # Create progress bar
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch} [Train]",
            leave=False,
            dynamic_ncols=True,
        )

        # Zero gradients at start
        self.optimizer.zero_grad()

        for batch_idx, batch_data in enumerate(pbar):
            # Move data to device
            batch_data = self._move_to_device(batch_data)

            # Forward pass with AMP autocast
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                if self.mode == "classification":
                    images, labels = batch_data
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
                    components = {}
                elif self.mode == "radical":
                    images, radical_targets, structure_targets, stroke_targets = batch_data
                    outputs = self.model(images)
                    loss, components = self.criterion(
                        outputs["radical_logits"],
                        outputs["structure_logits"],
                        outputs["stroke_pred"],
                        radical_targets,
                        structure_targets,
                        stroke_targets,
                    )
                elif self.mode == "symbolic":
                    (
                        images,
                        char_targets,
                        radical_targets,
                        structure_targets,
                        stroke_targets,
                    ) = batch_data
                    outputs = self.model(images, use_symbolic=True)
                    loss, components = self.criterion(
                        outputs["combined_probs"],
                        outputs["char_logits"],
                        outputs["radical_logits"],
                        outputs["structure_logits"],
                        outputs["stroke_logits"],
                        char_targets,
                        radical_targets,
                        structure_targets,
                        stroke_targets,
                    )
                else:  # joint
                    (
                        images,
                        char_targets,
                        radical_targets,
                        structure_targets,
                        stroke_targets,
                    ) = batch_data
                    outputs = self.model(images)
                    loss, components = self.criterion(
                        outputs["char_logits"],
                        outputs["radical_logits"],
                        outputs["structure_logits"],
                        outputs.get("stroke_logits", outputs.get("stroke_pred")),
                        char_targets,
                        radical_targets,
                        structure_targets,
                        stroke_targets,
                    )

            # Scale loss by gradient accumulation steps
            loss = loss / self.config.grad_accum_steps

            # Backward pass with AMP scaler
            self.scaler.scale(loss).backward()

            # Optimizer step every grad_accum_steps batches
            if (batch_idx + 1) % self.config.grad_accum_steps == 0:
                # Gradient clipping for symbolic mode (prevents instability)
                if self.mode == "symbolic":
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip_norm
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            # Track loss without sync every step — detach and accumulate on GPU
            total_loss += loss.detach() * self.config.grad_accum_steps
            num_batches += 1

            # Accumulate component losses
            for key, value in components.items():
                if key not in component_losses:
                    component_losses[key] = 0.0
                component_losses[key] += value

            # Update progress bar every 50 steps (avoid CPU-GPU sync on .item())
            if batch_idx % 50 == 0:
                pbar.set_postfix({"loss": f"{loss.item() * self.config.grad_accum_steps:.3f}"})

        # Handle remaining gradients if batch count not divisible by grad_accum_steps
        if num_batches % self.config.grad_accum_steps != 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        # Compute averages — single .item() sync per epoch
        avg_loss = (total_loss / num_batches).item()
        metrics = {"loss": avg_loss}

        # Add component losses
        for key in component_losses:
            metrics[key] = component_losses[key] / num_batches

        return metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model on validation set.

        Returns:
            Dictionary of validation metrics including loss, accuracy (if applicable),
            and component losses for multi-task modes
        """
        self.model.eval()
        total_loss = torch.tensor(0.0, device=self.device)
        num_batches = 0

        # Accuracy tracking for classification/joint/symbolic modes (keep on GPU)
        total_correct = torch.tensor(0, device=self.device)
        total_top5 = torch.tensor(0, device=self.device)
        total_top10 = torch.tensor(0, device=self.device)
        total_top20 = torch.tensor(0, device=self.device)
        total_samples = 0

        # Component loss tracking
        component_losses: Dict[str, float] = {}

        # Create progress bar
        pbar = tqdm(
            self.val_loader,
            desc="Validating",
            leave=False,
            dynamic_ncols=True,
        )

        for batch_data in pbar:
            # Move data to device
            batch_data = self._move_to_device(batch_data)

            # Forward pass with AMP autocast
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                if self.mode == "classification":
                    images, labels = batch_data
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
                    components = {}

                    # Compute top-1, top-5, top-10 accuracy
                    preds = torch.argmax(logits, dim=1)
                    total_correct += (preds == labels).sum()
                    _, top5_preds = logits.topk(5, dim=1)
                    _, top10_preds = logits.topk(10, dim=1)
                    total_top5 += (top5_preds == labels.unsqueeze(1)).any(dim=1).sum()
                    total_top10 += (top10_preds == labels.unsqueeze(1)).any(dim=1).sum()
                    total_samples += labels.size(0)

                elif self.mode == "radical":
                    images, radical_targets, structure_targets, stroke_targets = batch_data
                    outputs = self.model(images)
                    loss, components = self.criterion(
                        outputs["radical_logits"],
                        outputs["structure_logits"],
                        outputs["stroke_pred"],
                        radical_targets,
                        structure_targets,
                        stroke_targets,
                    )

                elif self.mode == "symbolic":
                    (
                        images,
                        char_targets,
                        radical_targets,
                        structure_targets,
                        stroke_targets,
                    ) = batch_data
                    outputs = self.model(images, use_symbolic=True)
                    loss, components = self.criterion(
                        outputs["combined_probs"],
                        outputs["char_logits"],
                        outputs["radical_logits"],
                        outputs["structure_logits"],
                        outputs["stroke_logits"],
                        char_targets,
                        radical_targets,
                        structure_targets,
                        stroke_targets,
                    )

                    # Accuracy from combined_probs (symbolically-reranked)
                    combined = outputs["combined_probs"]
                    preds = torch.argmax(combined, dim=1)
                    total_correct += (preds == char_targets).sum()
                    _, top5_preds = combined.topk(5, dim=1)
                    _, top10_preds = combined.topk(10, dim=1)
                    _, top20_preds = combined.topk(20, dim=1)
                    total_top5 += (top5_preds == char_targets.unsqueeze(1)).any(dim=1).sum()
                    total_top10 += (top10_preds == char_targets.unsqueeze(1)).any(dim=1).sum()
                    total_top20 += (top20_preds == char_targets.unsqueeze(1)).any(dim=1).sum()
                    total_samples += char_targets.size(0)

                else:  # joint
                    (
                        images,
                        char_targets,
                        radical_targets,
                        structure_targets,
                        stroke_targets,
                    ) = batch_data
                    outputs = self.model(images)
                    loss, components = self.criterion(
                        outputs["char_logits"],
                        outputs["radical_logits"],
                        outputs["structure_logits"],
                        outputs.get("stroke_logits", outputs.get("stroke_pred")),
                        char_targets,
                        radical_targets,
                        structure_targets,
                        stroke_targets,
                    )

                    # Compute top-1, top-5, top-10, top-20 character accuracy
                    preds = torch.argmax(outputs["char_logits"], dim=1)
                    total_correct += (preds == char_targets).sum()
                    _, top5_preds = outputs["char_logits"].topk(5, dim=1)
                    _, top10_preds = outputs["char_logits"].topk(10, dim=1)
                    _, top20_preds = outputs["char_logits"].topk(20, dim=1)
                    total_top5 += (top5_preds == char_targets.unsqueeze(1)).any(dim=1).sum()
                    total_top10 += (top10_preds == char_targets.unsqueeze(1)).any(dim=1).sum()
                    total_top20 += (top20_preds == char_targets.unsqueeze(1)).any(dim=1).sum()
                    total_samples += char_targets.size(0)

            # Track metrics on GPU
            total_loss += loss.detach()
            num_batches += 1

            # Accumulate component losses
            for key, value in components.items():
                if key not in component_losses:
                    component_losses[key] = 0.0
                component_losses[key] += value

        # Compute averages — single .item() sync
        avg_loss = (total_loss / num_batches).item()
        metrics = {"loss": avg_loss}

        # Add accuracy for classification/joint/symbolic modes
        if self.mode in ("classification", "joint", "symbolic"):
            accuracy = total_correct.item() / total_samples
            metrics["accuracy"] = accuracy
            metrics["top5_acc"] = total_top5.item() / total_samples
            metrics["top10_acc"] = total_top10.item() / total_samples
        if self.mode in ("joint", "symbolic"):
            metrics["top20_acc"] = total_top20.item() / total_samples

        # Add component losses
        for key in component_losses:
            metrics[key] = component_losses[key] / num_batches

        return metrics

    def _move_to_device(self, batch_data: tuple) -> tuple:
        """Move batch data to device.

        Args:
            batch_data: Tuple of tensors from DataLoader

        Returns:
            Tuple with all tensors moved to self.device
        """
        return tuple(x.to(self.device, non_blocking=True) for x in batch_data)

    def _get_csv_header(self) -> List[str]:
        """Get CSV header based on training mode."""
        header = ["epoch", "train_loss", "val_loss"]

        if self.mode in ("classification", "joint", "symbolic"):
            header.extend(["val_acc", "val_top5", "val_top10"])
        if self.mode == "symbolic":
            header.append("val_top20")

        if self.mode == "radical":
            header.extend(["train_radical_loss", "train_structure_loss", "train_stroke_loss"])
            header.extend(["val_radical_loss", "val_structure_loss", "val_stroke_loss"])
        elif self.mode == "joint":
            header.extend([
                "train_char_loss",
                "train_radical_loss",
                "train_structure_loss",
                "train_stroke_loss",
            ])
            header.extend([
                "val_char_loss",
                "val_radical_loss",
                "val_structure_loss",
                "val_stroke_loss",
            ])
        elif self.mode == "symbolic":
            header.extend([
                "train_combined_loss",
                "train_char_loss",
                "train_radical_loss",
                "train_structure_loss",
                "train_stroke_loss",
            ])
            header.extend([
                "val_combined_loss",
                "val_char_loss",
                "val_radical_loss",
                "val_structure_loss",
                "val_stroke_loss",
            ])

        header.append("lr")

        return header

    def _prepare_csv_row(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        lr: float,
    ) -> Dict[str, Any]:
        """Prepare CSV row from metrics."""
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
        }

        if "accuracy" in val_metrics:
            row["val_acc"] = val_metrics["accuracy"]
        if "top5_acc" in val_metrics:
            row["val_top5"] = val_metrics["top5_acc"]
        if "top10_acc" in val_metrics:
            row["val_top10"] = val_metrics["top10_acc"]
        if "top20_acc" in val_metrics:
            row["val_top20"] = val_metrics["top20_acc"]

        # Add component losses
        for key, value in train_metrics.items():
            if key != "loss":
                row[f"train_{key}"] = value

        for key, value in val_metrics.items():
            if key not in ("loss", "accuracy", "top5_acc", "top10_acc"):
                row[f"val_{key}"] = value

        row["lr"] = lr

        return row
