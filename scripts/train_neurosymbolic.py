"""Train MiniResNetJoint with 3-phase neurosymbolic training.

Phase 1: Multi-task head training (heads_only mode)
  - All heads trained independently with standard losses
  - Backbone + all heads are trainable (reranker MLP excluded)

Phase 2: Combined symbolic + head losses (combined mode)
  - Symbolic lookup tables loaded; top-K re-ranking via MLP
  - Combined loss teaches backbone structural consistency
  - All parameters trainable including reranker

Phase 3: Freeze backbone, tune reranker only
  - Only reranker MLP + reranker_weight are updated
  - Fine-tunes the symbolic re-ranking scores

Usage:
    python scripts/train_neurosymbolic.py \\
        --train-dir data/HWDB1.1/train \\
        --epochs-phase1 50 --epochs-phase2 30 --epochs-phase3 10

    # With pretrained backbone and wandb:
    python scripts/train_neurosymbolic.py \\
        --pretrained-backbone outputs/mini_resnet/best_model_classification.pt \\
        --wandb --wandb-project hccr-neurosymbolic
"""

import argparse
import json
import random
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from hccr.data.dataset import HWDBNeurosymbolicDataset
from hccr.data.label_map import LabelMap
from hccr.data.transforms import (
    get_eval_transform,
    get_train_tensor_augment,
    get_train_transform,
)
from hccr.models.mini_restnet_join import (
    MiniResNetJoint,
    PhaseManager,
    RadicalLightLoss,
    count_params,
    print_model_summary,
)
from hccr.structural.radical_table import RadicalTable
from hccr.training.early_stopping import EarlyStopping
from hccr.utils import get_device, get_logger, save_checkpoint, set_seed

logger = get_logger(__name__)


# =============================================================================
# Data preparation utilities
# =============================================================================


def prepare_radical_table_dict(
    radical_table: RadicalTable,
    stroke_types: Dict[str, list] | None = None,
) -> dict:
    """Convert RadicalTable to dict format expected by HWDBNeurosymbolicDataset.

    Args:
        radical_table: RadicalTable with char_to_radicals, structure, strokes
        stroke_types: Optional dict mapping char -> [h,v,lf,rf,dot,turn] counts

    Returns:
        Dict mapping char_str -> {"radicals": [int], "structure": int,
                                   "strokes": int, "stroke_types": [float]*6}
    """
    result = {}
    for char in radical_table.char_to_radicals:
        radicals = radical_table.char_to_radicals.get(char, [])
        radical_indices = [
            radical_table.radical_to_index[r]
            for r in radicals
            if r in radical_table.radical_to_index
        ]
        result[char] = {
            "radicals": radical_indices,
            "structure": radical_table.char_to_structure.get(char, 12),
            "strokes": radical_table.char_to_strokes.get(char, 0),
            "stroke_types": (
                [float(v) for v in stroke_types[char]]
                if stroke_types and char in stroke_types
                else [0.0] * 6
            ),
        }
    return result


def build_symbolic_tables(
    radical_table: RadicalTable,
    label_map: LabelMap,
    stroke_types: Dict[str, list] | None = None,
) -> Tuple[Dict[str, list], Dict[str, int], Dict[str, int], Dict[str, list] | None]:
    """Build index-keyed tables for SymbolicConstraintLayer.load_tables().

    Converts from character-keyed RadicalTable to char-index-keyed dicts
    (keys are str(index) for JSON compatibility).

    Returns:
        (radical_tab, structure_tab, stroke_count_tab, stroke_type_tab)
    """
    radical_tab: Dict[str, list] = {}
    structure_tab: Dict[str, int] = {}
    stroke_count_tab: Dict[str, int] = {}
    stroke_type_tab: Dict[str, list] = {} if stroke_types else None

    for char in radical_table.char_to_radicals:
        if char not in label_map._char_to_idx:
            continue
        idx = label_map.encode(char)
        idx_str = str(idx)

        radical_indices = [
            radical_table.radical_to_index[r]
            for r in radical_table.char_to_radicals[char]
            if r in radical_table.radical_to_index
        ]
        radical_tab[idx_str] = radical_indices
        structure_tab[idx_str] = radical_table.char_to_structure.get(char, 12)
        strokes = radical_table.char_to_strokes.get(char, 8)
        stroke_count_tab[idx_str] = min(max(strokes - 1, 0), 29)

        if stroke_types and char in stroke_types:
            stroke_type_tab[idx_str] = [float(v) for v in stroke_types[char]]

    return radical_tab, structure_tab, stroke_count_tab, stroke_type_tab


# =============================================================================
# Training loop
# =============================================================================


def train_one_epoch(
    model: MiniResNetJoint,
    loss_fn: RadicalLightLoss,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    device: torch.device,
    forward_mode: str,
    loss_phase: str,
    scaler: torch.amp.GradScaler,
    grad_clip_norm: float = 1.0,
    grad_accum_steps: int = 1,
    epoch: int = 0,
    wandb_enabled: bool = False,
    phase_num: int = 1,
    global_step: int = 0,
) -> Tuple[float, Dict[str, float]]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    component_sums: Dict[str, float] = {}

    optimizer.zero_grad()
    use_amp = device.type == "cuda"
    total_batches = len(train_loader)
    log_interval = max(total_batches // 10, 1)

    pbar = tqdm(
        train_loader,
        desc=f"Epoch {epoch} [Train]",
        leave=False,
        dynamic_ncols=True,
    )

    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}

        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(images, mode=forward_mode)
            loss, loss_dict = loss_fn(outputs, targets, phase=loss_phase)

        loss_scaled = loss / grad_accum_steps
        scaler.scale(loss_scaled).backward()

        if (batch_idx + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item()
        num_batches += 1
        for k, v in loss_dict.items():
            component_sums[k] = component_sums.get(k, 0.0) + v

        if batch_idx % 50 == 0:
            pbar.set_postfix({"loss": f"{loss.item():.3f}"})

        # Log to wandb every ~10% of epoch
        if wandb_enabled and HAS_WANDB and (batch_idx + 1) % log_interval == 0:
            avg_so_far = total_loss / num_batches
            log_dict = {
                f"phase{phase_num}/train_loss_live": avg_so_far,
                f"phase{phase_num}/batch_loss": loss.item(),
            }
            for k, v in loss_dict.items():
                log_dict[f"phase{phase_num}/train_{k}_live"] = v
            wandb.log(log_dict)

    # Flush remaining gradients
    if num_batches % grad_accum_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    avg_loss = total_loss / max(num_batches, 1)
    avg_components = {k: v / max(num_batches, 1) for k, v in component_sums.items()}
    return avg_loss, avg_components


@torch.no_grad()
def validate(
    model: MiniResNetJoint,
    loss_fn: RadicalLightLoss,
    val_loader: DataLoader,
    device: torch.device,
    forward_mode: str,
    loss_phase: str,
    label_map: "LabelMap | None" = None,
    wandb_enabled: bool = False,
    phase_num: int = 1,
) -> Dict[str, float]:
    """Validate and compute accuracy metrics."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    correct = 0
    top5_correct = 0
    top10_correct = 0
    top20_correct = 0
    total_samples = 0
    component_sums: Dict[str, float] = {}

    # Per-head metric accumulators
    radical_tp = 0
    radical_fp = 0
    radical_fn = 0
    structure_correct = 0
    stroke_count_correct = 0
    stroke_count_within1 = 0
    stroke_type_cosine_sum = 0.0
    density_mse_sum = 0.0
    quad_radical_tp = 0
    quad_radical_fp = 0
    quad_radical_fn = 0

    # Symbolic boost accumulators (Phase 2+3 only)
    has_symbolic = forward_mode == "combined"
    symbolic_helped = 0
    symbolic_hurt = 0
    both_correct = 0
    both_wrong = 0

    # Sample predictions collector (first 16 samples)
    sample_rows = []
    max_samples = 16

    use_amp = device.type == "cuda"

    for images, targets in tqdm(
        val_loader, desc="Validating", leave=False, dynamic_ncols=True,
    ):
        images = images.to(device, non_blocking=True)
        targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}

        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(images, mode=forward_mode)
            loss, loss_dict = loss_fn(outputs, targets, phase=loss_phase)

        total_loss += loss.item()
        num_batches += 1
        for k, v in loss_dict.items():
            component_sums[k] = component_sums.get(k, 0.0) + v

        # Accuracy from combined_logits (Phase 2+3) or char_logits (Phase 1)
        logits = outputs.get("combined_logits", outputs["char_logits"])
        labels = targets["char_label"]
        B = labels.size(0)
        total_samples += B

        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()

        _, top5 = logits.topk(min(5, logits.size(-1)), dim=-1)
        _, top10 = logits.topk(min(10, logits.size(-1)), dim=-1)
        _, top20 = logits.topk(min(20, logits.size(-1)), dim=-1)
        top5_correct += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()
        top10_correct += (top10 == labels.unsqueeze(1)).any(dim=1).sum().item()
        top20_correct += (top20 == labels.unsqueeze(1)).any(dim=1).sum().item()

        # --- Per-head metrics ---
        # Radical F1 (micro): threshold sigmoid at 0.5 vs multi-hot labels
        radical_preds = (torch.sigmoid(outputs["radical_logits"]) > 0.5).float()
        radical_targets = targets["radical_label"].float()
        radical_tp += (radical_preds * radical_targets).sum().item()
        radical_fp += (radical_preds * (1 - radical_targets)).sum().item()
        radical_fn += ((1 - radical_preds) * radical_targets).sum().item()

        # Structure accuracy: argmax vs target
        structure_preds = outputs["structure"].argmax(dim=-1)
        structure_correct += (structure_preds == targets["structure"]).sum().item()

        # Stroke count accuracy: argmax vs target (+ within-1)
        stroke_preds = outputs["stroke_count"].argmax(dim=-1)
        stroke_targets = targets["stroke_count"]
        stroke_count_correct += (stroke_preds == stroke_targets).sum().item()
        stroke_count_within1 += ((stroke_preds - stroke_targets).abs() <= 1).sum().item()

        # Stroke type cosine similarity
        pred_norm = F.normalize(outputs["stroke_types"], dim=-1)
        tgt_norm = F.normalize(targets["stroke_types"], dim=-1)
        stroke_type_cosine_sum += (pred_norm * tgt_norm).sum(dim=-1).mean().item()

        # Density MSE
        density_mse_sum += F.mse_loss(
            outputs["density"], targets["density"]
        ).item()

        # Quad radical F1 (micro across all 4 quadrants)
        quad_preds = (torch.sigmoid(outputs["quad_radicals"]) > 0.5).float()
        quad_targets = targets["quad_radicals"].float()
        quad_radical_tp += (quad_preds * quad_targets).sum().item()
        quad_radical_fp += (quad_preds * (1 - quad_targets)).sum().item()
        quad_radical_fn += ((1 - quad_preds) * quad_targets).sum().item()

        # --- Symbolic boost (Phase 2+3 only) ---
        if has_symbolic:
            neural_preds = outputs["char_logits"].argmax(dim=-1)
            combined_preds = outputs["combined_logits"].argmax(dim=-1)
            neural_right = neural_preds == labels
            combined_right = combined_preds == labels
            symbolic_helped += (~neural_right & combined_right).sum().item()
            symbolic_hurt += (neural_right & ~combined_right).sum().item()
            both_correct += (neural_right & combined_right).sum().item()
            both_wrong += (~neural_right & ~combined_right).sum().item()

        # --- Sample predictions (first 16) ---
        if label_map is not None and len(sample_rows) < max_samples:
            remaining = max_samples - len(sample_rows)
            count = min(B, remaining)
            _, top5_indices = logits[:count].topk(
                min(5, logits.size(-1)), dim=-1,
            )
            for i in range(count):
                gt_char = label_map.decode(labels[i].item())
                top5_chars = [
                    label_map.decode(idx.item()) for idx in top5_indices[i]
                ]
                row = {
                    "ground_truth": gt_char,
                    "top5_combined": " ".join(top5_chars),
                }
                if has_symbolic:
                    _, neural_top5 = outputs["char_logits"][i].unsqueeze(0).topk(
                        min(5, logits.size(-1)), dim=-1,
                    )
                    neural_chars = [
                        label_map.decode(idx.item()) for idx in neural_top5[0]
                    ]
                    row["top5_neural"] = " ".join(neural_chars)
                    n_right = (outputs["char_logits"][i].argmax() == labels[i]).item()
                    c_right = (outputs["combined_logits"][i].argmax() == labels[i]).item()
                    if not n_right and c_right:
                        row["symbolic_effect"] = "helped"
                    elif n_right and not c_right:
                        row["symbolic_effect"] = "hurt"
                    elif n_right and c_right:
                        row["symbolic_effect"] = "both_correct"
                    else:
                        row["symbolic_effect"] = "both_wrong"
                sample_rows.append(row)

    n = max(total_samples, 1)
    nb = max(num_batches, 1)
    avg_loss = total_loss / nb
    avg_components = {k: v / nb for k, v in component_sums.items()}

    # Compute per-head metrics
    radical_precision = radical_tp / max(radical_tp + radical_fp, 1)
    radical_recall = radical_tp / max(radical_tp + radical_fn, 1)
    radical_f1 = (
        2 * radical_precision * radical_recall
        / max(radical_precision + radical_recall, 1e-8)
    )
    quad_precision = quad_radical_tp / max(quad_radical_tp + quad_radical_fp, 1)
    quad_recall = quad_radical_tp / max(quad_radical_tp + quad_radical_fn, 1)
    quad_f1 = (
        2 * quad_precision * quad_recall
        / max(quad_precision + quad_recall, 1e-8)
    )

    metrics = {
        "val_loss": avg_loss,
        "val_acc": correct / n,
        "val_top5": top5_correct / n,
        "val_top10": top10_correct / n,
        "val_top20": top20_correct / n,
        # Per-head metrics
        "val_radical_f1": radical_f1,
        "val_structure_acc": structure_correct / n,
        "val_stroke_count_acc": stroke_count_correct / n,
        "val_stroke_count_within1": stroke_count_within1 / n,
        "val_stroke_type_cosine": stroke_type_cosine_sum / nb,
        "val_density_mse": density_mse_sum / nb,
        "val_quad_radical_f1": quad_f1,
    }
    metrics.update({f"val_{k}": v for k, v in avg_components.items()})

    # Symbolic boost metrics
    if has_symbolic:
        total_compared = symbolic_helped + symbolic_hurt
        metrics["val_symbolic_helped"] = symbolic_helped
        metrics["val_symbolic_hurt"] = symbolic_hurt
        metrics["val_symbolic_both_correct"] = both_correct
        metrics["val_symbolic_both_wrong"] = both_wrong
        metrics["val_symbolic_boost_rate"] = (
            symbolic_helped / max(total_compared, 1)
        )

    # Sample predictions table (wandb)
    if wandb_enabled and HAS_WANDB and sample_rows:
        columns = ["ground_truth", "top5_combined"]
        if has_symbolic:
            columns += ["top5_neural", "symbolic_effect"]
        table = wandb.Table(columns=columns)
        for row in sample_rows:
            table.add_data(*[row.get(c, "") for c in columns])
        metrics["_sample_predictions_table"] = table

    return metrics


def train_phase(
    model: MiniResNetJoint,
    loss_fn: RadicalLightLoss,
    phase_manager: PhaseManager,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int,
    phase_num: int,
    lr: float,
    weight_decay: float,
    output_dir: Path,
    grad_clip_norm: float = 1.0,
    grad_accum_steps: int = 1,
    patience: int = 7,
    wandb_enabled: bool = False,
    global_step: int = 0,
    label_map: "LabelMap | None" = None,
) -> Tuple[float, int]:
    """Train one phase with early stopping and checkpointing.

    Returns:
        (best_accuracy, updated_global_step)
    """
    optimizer, scheduler = phase_manager.setup_phase(
        phase_num, lr=lr, weight_decay=weight_decay, t_max=num_epochs,
    )
    forward_mode = phase_manager.get_forward_mode()
    loss_phase = phase_manager.get_loss_phase()

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    early_stopping = EarlyStopping(patience=patience, mode="max", min_delta=1e-4)
    best_acc = 0.0

    trainable = count_params(model, only_trainable=True)
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Phase {phase_num}: mode={forward_mode}, loss={loss_phase}")
    logger.info(f"  LR: {lr}, Epochs: {num_epochs}, Trainable params: {trainable:,}")
    logger.info(f"{'=' * 60}")

    for epoch in range(1, num_epochs + 1):
        train_loss, train_components = train_one_epoch(
            model, loss_fn, optimizer, train_loader, device,
            forward_mode, loss_phase, scaler,
            grad_clip_norm=grad_clip_norm,
            grad_accum_steps=grad_accum_steps,
            epoch=epoch,
            wandb_enabled=wandb_enabled,
            phase_num=phase_num,
            global_step=global_step,
        )

        val_metrics = validate(
            model, loss_fn, val_loader, device, forward_mode, loss_phase,
            label_map=label_map,
            wandb_enabled=wandb_enabled,
            phase_num=phase_num,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        global_step += 1

        # Console logging
        log_msg = (
            f"P{phase_num} E{epoch}/{num_epochs} | "
            f"Train: {train_loss:.4f} | Val: {val_metrics['val_loss']:.4f} | "
            f"Acc: {val_metrics['val_acc']:.4f} | "
            f"Top5: {val_metrics['val_top5']:.4f} | "
            f"Top10: {val_metrics['val_top10']:.4f} | "
            f"Top20: {val_metrics['val_top20']:.4f} | "
            f"LR: {current_lr:.2e}"
        )
        logger.info(log_msg)
        head_msg = (
            f"  Heads | RadF1: {val_metrics['val_radical_f1']:.3f} | "
            f"Struct: {val_metrics['val_structure_acc']:.3f} | "
            f"Stroke: {val_metrics['val_stroke_count_acc']:.3f} "
            f"(±1: {val_metrics['val_stroke_count_within1']:.3f}) | "
            f"StTypeCos: {val_metrics['val_stroke_type_cosine']:.3f} | "
            f"DenMSE: {val_metrics['val_density_mse']:.4f} | "
            f"QuadF1: {val_metrics['val_quad_radical_f1']:.3f}"
        )
        logger.info(head_msg)
        if "val_symbolic_boost_rate" in val_metrics:
            sym_msg = (
                f"  Symbolic | boost_rate: {val_metrics['val_symbolic_boost_rate']:.3f} "
                f"(helped: {val_metrics['val_symbolic_helped']:.0f}, "
                f"hurt: {val_metrics['val_symbolic_hurt']:.0f}, "
                f"both_ok: {val_metrics['val_symbolic_both_correct']:.0f}, "
                f"both_wrong: {val_metrics['val_symbolic_both_wrong']:.0f})"
            )
            logger.info(sym_msg)

        # WandB logging
        if wandb_enabled and HAS_WANDB:
            log_dict = {
                "epoch": global_step,
                "phase": phase_num,
                f"phase{phase_num}/train_loss": train_loss,
                f"phase{phase_num}/lr": current_lr,
            }
            for k, v in train_components.items():
                log_dict[f"phase{phase_num}/train_{k}"] = v
            # Separate table from scalar metrics
            sample_table = val_metrics.pop("_sample_predictions_table", None)
            for k, v in val_metrics.items():
                log_dict[f"phase{phase_num}/{k}"] = v
            # Reranker weight trajectory
            log_dict["reranker/weight"] = model.reranker.reranker_weight.item()
            # Log sample predictions table
            if sample_table is not None:
                log_dict[f"phase{phase_num}/sample_predictions"] = sample_table
            wandb.log(log_dict, step=global_step)

        # Early stopping + checkpointing
        acc = val_metrics["val_acc"]
        if early_stopping(acc):
            logger.info(
                f"Early stopping at epoch {epoch}. Best acc: {best_acc:.4f}"
            )
            break

        if early_stopping.improved:
            best_acc = acc
            ckpt_path = output_dir / f"best_model_phase{phase_num}.pt"
            save_checkpoint(
                model, ckpt_path,
                epoch=epoch, phase=phase_num, val_acc=acc,
                optimizer_state_dict=optimizer.state_dict(),
            )
            logger.info(f"  -> Saved checkpoint (acc={acc:.4f})")

    return best_acc, global_step


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train MiniResNetJoint (3-phase neurosymbolic training)",
    )

    # Data paths
    parser.add_argument("--train-dir", type=Path, default=Path("data/HWDB1.1/train"))
    parser.add_argument("--label-map", type=Path, default=Path("resources/label_map.json"))
    parser.add_argument("--radical-table", type=Path, default=Path("resources/radical_table.json"))
    parser.add_argument("--stroke-types", type=Path, default=Path("resources/stroke_types.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/neurosymbolic"))

    # Model
    parser.add_argument(
        "--pretrained-backbone", type=Path, default=None,
        help="Pretrained MiniResNet backbone checkpoint (same architecture)",
    )
    parser.add_argument("--dropout", type=float, default=0.3)

    # Phase epochs
    parser.add_argument("--epochs-phase1", type=int, default=50)
    parser.add_argument("--epochs-phase2", type=int, default=30)
    parser.add_argument("--epochs-phase3", type=int, default=10)

    # Phase learning rates
    parser.add_argument("--lr-phase1", type=float, default=1e-3)
    parser.add_argument("--lr-phase2", type=float, default=5e-4)
    parser.add_argument("--lr-phase3", type=float, default=1e-3)

    # Training config
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    # Loss weights
    parser.add_argument("--w-char", type=float, default=1.0)
    parser.add_argument("--w-radical", type=float, default=0.5)
    parser.add_argument("--w-stroke-count", type=float, default=0.3)
    parser.add_argument("--w-stroke-types", type=float, default=0.2)
    parser.add_argument("--w-structure", type=float, default=0.3)
    parser.add_argument("--w-density", type=float, default=0.1)
    parser.add_argument("--w-quad-radical", type=float, default=0.3)
    parser.add_argument("--w-combined", type=float, default=0.5)
    parser.add_argument("--label-smoothing", type=float, default=0.1)

    # WandB
    parser.add_argument("--wandb", action="store_true", default=False,
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="hccr-neurosymbolic")
    parser.add_argument("--wandb-name", type=str, default=None)

    # Resume
    parser.add_argument(
        "--resume-phase", type=int, default=None, choices=[1, 2, 3],
        help="Resume from a previous run: skip phases before this number "
             "and load the best checkpoint from the prior phase",
    )

    args = parser.parse_args()

    # ---- Setup ----
    set_seed(args.seed)
    device = get_device()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- WandB ----
    wandb_enabled = args.wandb and HAS_WANDB
    if wandb_enabled:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args),
        )
    elif args.wandb and not HAS_WANDB:
        logger.warning("wandb requested but not installed. pip install wandb")

    logger.info("=" * 60)
    logger.info("Neurosymbolic Training: MiniResNetJoint (3-phase)")
    logger.info("=" * 60)

    # ---- Load label map and radical table ----
    label_map = LabelMap.load(args.label_map)
    num_classes = len(label_map)

    radical_table = RadicalTable.load(args.radical_table)
    num_radicals = len(radical_table.all_radicals)
    logger.info(f"Classes: {num_classes}, Radicals: {num_radicals}")

    # Load stroke type signatures
    stroke_types_data = None
    if args.stroke_types.exists():
        with open(args.stroke_types, "r", encoding="utf-8") as f:
            stroke_types_data = json.load(f)
        logger.info(f"Loaded stroke types for {len(stroke_types_data)} characters")
    else:
        logger.warning(f"Stroke types file not found: {args.stroke_types}")

    radical_dict = prepare_radical_table_dict(radical_table, stroke_types_data)

    # ---- Split GNT files ----
    all_files = list(args.train_dir.glob("*.gnt"))
    if not all_files:
        raise FileNotFoundError(f"No .gnt files found in {args.train_dir}")
    num_val = int(len(all_files) * args.val_ratio)

    random.seed(args.seed)
    shuffled = all_files.copy()
    random.shuffle(shuffled)
    train_files = shuffled[num_val:]
    val_files = shuffled[:num_val]

    # Create temp directories with symlinks
    temp_dir = Path(tempfile.mkdtemp())
    train_temp = temp_dir / "train"
    val_temp = temp_dir / "val"
    train_temp.mkdir()
    val_temp.mkdir()

    for f in train_files:
        dst = train_temp / f.name
        try:
            dst.symlink_to(f.resolve())
        except OSError:
            shutil.copy(f, dst)
    for f in val_files:
        dst = val_temp / f.name
        try:
            dst.symlink_to(f.resolve())
        except OSError:
            shutil.copy(f, dst)

    # ---- Create datasets ----
    train_transform = get_train_transform(args.image_size)
    val_transform = get_eval_transform(args.image_size)
    tensor_augment = get_train_tensor_augment()

    train_dataset = HWDBNeurosymbolicDataset(
        train_temp, label_map, radical_dict,
        num_radicals=num_radicals,
        num_structures=13,
        max_strokes=30,
        transform=train_transform,
        tensor_augment=tensor_augment,
        index_cache_path=args.output_dir / "train_index.pkl",
        preload=True,
        image_size=args.image_size,
    )
    val_dataset = HWDBNeurosymbolicDataset(
        val_temp, label_map, radical_dict,
        num_radicals=num_radicals,
        num_structures=13,
        max_strokes=30,
        transform=val_transform,
        index_cache_path=args.output_dir / "val_index.pkl",
        preload=True,
        image_size=args.image_size,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=0, pin_memory=True,
    )

    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # ---- Create model ----
    model = MiniResNetJoint(
        num_classes=num_classes,
        num_radicals=num_radicals,
        num_structure_types=13,
        num_stroke_count_bins=30,
        num_stroke_types=6,
        dropout=args.dropout,
    )
    print_model_summary(model)

    # Load pretrained backbone if available
    if args.pretrained_backbone and args.pretrained_backbone.exists():
        model.load_pretrained_backbone(str(args.pretrained_backbone))

    model = model.to(device)

    # ---- Create loss and phase manager ----
    loss_fn = RadicalLightLoss(
        w_char=args.w_char,
        w_radical=args.w_radical,
        w_stroke_count=args.w_stroke_count,
        w_stroke_types=args.w_stroke_types,
        w_structure=args.w_structure,
        w_density=args.w_density,
        w_quad_radical=args.w_quad_radical,
        w_combined=args.w_combined,
        label_smoothing=args.label_smoothing,
    )
    phase_manager = PhaseManager(model, loss_fn)
    global_step = 0
    resume_phase = args.resume_phase
    best_acc_p1 = 0.0
    best_acc_p2 = 0.0
    best_acc_p3 = 0.0

    # ---- Handle resume ----
    if resume_phase and resume_phase >= 2:
        # Load best checkpoint from the phase before the resume point
        ckpt_path = args.output_dir / f"best_model_phase{resume_phase - 1}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Cannot resume at Phase {resume_phase}: "
                f"{ckpt_path} not found"
            )
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        # Reset reranker weight to small init (let it learn up from near-zero)
        model.reranker.reranker_weight.data.fill_(0.01)
        global_step = ckpt.get("epoch", 0)
        if resume_phase == 2:
            best_acc_p1 = ckpt.get("val_acc", 0.0)
        elif resume_phase == 3:
            best_acc_p2 = ckpt.get("val_acc", 0.0)
        logger.info(
            f"Resumed from {ckpt_path} (acc={ckpt.get('val_acc', 0):.4f}, "
            f"step={global_step})"
        )

    # ===== Phase 1: Train all heads independently =====
    if not resume_phase or resume_phase <= 1:
        best_acc_p1, global_step = train_phase(
            model, loss_fn, phase_manager, train_loader, val_loader,
            device, args.epochs_phase1, phase_num=1,
            lr=args.lr_phase1, weight_decay=args.weight_decay,
            output_dir=args.output_dir,
            grad_clip_norm=args.grad_clip_norm,
            grad_accum_steps=args.grad_accum_steps,
            patience=args.patience,
            wandb_enabled=wandb_enabled,
            global_step=global_step,
            label_map=label_map,
        )
        logger.info(f"Phase 1 complete. Best acc: {best_acc_p1:.4f}")

        # Reload best Phase 1 checkpoint
        p1_ckpt = args.output_dir / "best_model_phase1.pt"
        if p1_ckpt.exists():
            ckpt = torch.load(p1_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            logger.info(f"Loaded best Phase 1 checkpoint")
    else:
        logger.info(f"Skipping Phase 1 (resumed at Phase {resume_phase})")

    # ===== Load symbolic constraint tables for Phase 2+ =====
    radical_tab, structure_tab, stroke_count_tab, stroke_type_tab = build_symbolic_tables(
        radical_table, label_map, stroke_types_data,
    )
    model.reranker.load_tables(
        radical_tab, structure_tab, stroke_count_tab, stroke_type_tab,
    )
    st_count = len(stroke_type_tab) if stroke_type_tab else 0
    logger.info(
        f"Loaded symbolic tables: "
        f"{len(radical_tab)} chars, {num_radicals} radicals, "
        f"13 structures, 30 stroke bins, {st_count} stroke type sigs"
    )

    # ===== Phase 2: Combined symbolic + head losses =====
    if not resume_phase or resume_phase <= 2:
        best_acc_p2, global_step = train_phase(
            model, loss_fn, phase_manager, train_loader, val_loader,
            device, args.epochs_phase2, phase_num=2,
            lr=args.lr_phase2, weight_decay=args.weight_decay,
            output_dir=args.output_dir,
            grad_clip_norm=args.grad_clip_norm,
            grad_accum_steps=args.grad_accum_steps,
            patience=args.patience,
            wandb_enabled=wandb_enabled,
            global_step=global_step,
            label_map=label_map,
        )
        logger.info(f"Phase 2 complete. Best acc: {best_acc_p2:.4f}")

        # Reload best Phase 2 checkpoint
        p2_ckpt = args.output_dir / "best_model_phase2.pt"
        if p2_ckpt.exists():
            ckpt = torch.load(p2_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            logger.info(f"Loaded best Phase 2 checkpoint")
    else:
        logger.info(f"Skipping Phase 2 (resumed at Phase {resume_phase})")

    # ===== Phase 3: Tune combination weights =====
    best_acc_p3, global_step = train_phase(
        model, loss_fn, phase_manager, train_loader, val_loader,
        device, args.epochs_phase3, phase_num=3,
        lr=args.lr_phase3, weight_decay=0.0,  # no weight decay for combo weights
        output_dir=args.output_dir,
        grad_clip_norm=args.grad_clip_norm,
        grad_accum_steps=args.grad_accum_steps,
        patience=args.patience,
        wandb_enabled=wandb_enabled,
        global_step=global_step,
        label_map=label_map,
    )
    logger.info(f"Phase 3 complete. Best acc: {best_acc_p3:.4f}")

    # ---- Summary ----
    logger.info(f"\n{'=' * 60}")
    logger.info("Training Summary")
    logger.info(f"  Phase 1 best acc: {best_acc_p1:.4f}")
    logger.info(f"  Phase 2 best acc: {best_acc_p2:.4f}")
    logger.info(f"  Phase 3 best acc: {best_acc_p3:.4f}")
    logger.info(f"  Checkpoints: {args.output_dir}")
    logger.info(f"{'=' * 60}")

    logger.info(f"Final reranker weight: {model.reranker.reranker_weight.item():.4f}")

    if wandb_enabled:
        wandb.log({
            "final/phase1_best_acc": best_acc_p1,
            "final/phase2_best_acc": best_acc_p2,
            "final/phase3_best_acc": best_acc_p3,
            "final/reranker_weight": model.reranker.reranker_weight.item(),
        })
        wandb.finish()

    # Cleanup
    shutil.rmtree(temp_dir)
    logger.info("Done!")


if __name__ == "__main__":
    main()
