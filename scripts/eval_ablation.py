"""Component ablation study for the symbolic layer.

Evaluates the contribution of each symbolic constraint by toggling
them on/off and measuring top-1/5/10/20 accuracy on the test set.

Configurations:
1. CNN only (no symbolic reranking)
2. + structure constraint only
3. + structure + radical
4. + structure + radical + stroke
5. + all constraints (full symbolic)
"""

import argparse
import csv
import random
import shutil
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from hccr.data.dataset import HWDBRadicalDataset
from hccr.data.label_map import LabelMap
from hccr.data.transforms import get_eval_transform
from hccr.models.tiny_cnn_joint import TinyCNNJoint
from hccr.structural.radical_table import RadicalTable
from hccr.utils import get_device, get_logger, set_seed

logger = get_logger(__name__)


def prepare_radical_table_dict(radical_table: RadicalTable) -> dict:
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
            "structure": radical_table.char_to_structure.get(char, 0),
            "strokes": radical_table.char_to_strokes.get(char, 0),
        }
    return result


def evaluate_with_constraints(
    model: TinyCNNJoint,
    val_loader: DataLoader,
    device: torch.device,
    use_structure: bool = True,
    use_radical: bool = True,
    use_stroke: bool = True,
) -> dict[str, float]:
    """Evaluate model with selective symbolic constraints.

    When a constraint is disabled, its score contribution is set to 1.0
    (neutral in the multiplicative combination).
    """
    model.eval()
    total_correct = 0
    total_top5 = 0
    total_top10 = 0
    total_top20 = 0
    total_samples = 0

    with torch.no_grad():
        for batch_data in tqdm(val_loader, desc="Evaluating", leave=False):
            batch_data = tuple(x.to(device, non_blocking=True) for x in batch_data)
            images, char_targets = batch_data[0], batch_data[1]

            outputs = model(images, use_symbolic=False)
            char_logits = outputs["char_logits"]
            radical_logits = outputs["radical_logits"]
            structure_logits = outputs["structure_logits"]
            stroke_logits = outputs["stroke_logits"]

            char_probs = F.softmax(char_logits, dim=-1)

            if use_structure or use_radical or use_stroke:
                # Compute individual constraint scores
                structure_probs = F.softmax(structure_logits, dim=-1)
                stroke_probs = F.softmax(stroke_logits, dim=-1)
                radical_probs = torch.sigmoid(radical_logits)

                s_scores = structure_probs[:, model.master_table[:, 0]] if use_structure else torch.ones_like(char_probs)
                st_scores = stroke_probs[:, model.master_table[:, 2]] if use_stroke else torch.ones_like(char_probs)

                if use_radical:
                    r_scores = model._radical_scores(radical_probs)
                else:
                    r_scores = torch.ones_like(char_probs)

                symbolic_prior = (s_scores * r_scores * st_scores).clamp(min=1e-10)
                combined = char_probs * symbolic_prior
                combined = combined / combined.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            else:
                combined = char_probs

            # Compute accuracy metrics
            preds = torch.argmax(combined, dim=1)
            total_correct += (preds == char_targets).sum().item()

            _, top5 = combined.topk(5, dim=1)
            _, top10 = combined.topk(10, dim=1)
            _, top20 = combined.topk(20, dim=1)
            total_top5 += (top5 == char_targets.unsqueeze(1)).any(dim=1).sum().item()
            total_top10 += (top10 == char_targets.unsqueeze(1)).any(dim=1).sum().item()
            total_top20 += (top20 == char_targets.unsqueeze(1)).any(dim=1).sum().item()
            total_samples += char_targets.size(0)

    return {
        "top1": total_correct / total_samples,
        "top5": total_top5 / total_samples,
        "top10": total_top10 / total_samples,
        "top20": total_top20 / total_samples,
        "total": total_samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Symbolic layer ablation study")
    parser.add_argument("--train-dir", type=Path, default=Path("data/HWDB1.1/train"))
    parser.add_argument("--label-map", type=Path, default=Path("resources/label_map.json"))
    parser.add_argument("--radical-table", type=Path, default=Path("resources/radical_table.json"))
    parser.add_argument("--master-table", type=Path, default=Path("resources/master_table.pt"))
    parser.add_argument("--radical-mask", type=Path, default=Path("resources/radical_mask.pt"))
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("outputs/symbolic/best_model_symbolic.pt"),
    )
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/ablation_results.csv"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-strokes", type=int, default=30)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    # Load resources
    label_map = LabelMap.load(args.label_map)
    num_classes = len(label_map)
    radical_table = RadicalTable.load(args.radical_table)
    num_radicals = len(radical_table.all_radicals)
    radical_dict = prepare_radical_table_dict(radical_table)

    # Create validation set
    all_files = list(args.train_dir.glob("*.gnt"))
    num_val = int(len(all_files) * args.val_ratio)
    random.seed(args.seed)
    shuffled = all_files.copy()
    random.shuffle(shuffled)
    val_files = shuffled[:num_val]

    temp_dir = Path(tempfile.mkdtemp())
    val_temp = temp_dir / "val"
    val_temp.mkdir()
    for f in val_files:
        dst = val_temp / f.name
        try:
            dst.symlink_to(f.resolve())
        except OSError:
            shutil.copy(f, dst)

    val_dataset = HWDBRadicalDataset(
        val_temp, label_map, radical_dict,
        num_radicals=num_radicals, num_structures=13,
        transform=get_eval_transform(args.image_size),
        preload=True, image_size=args.image_size,
        stroke_as_class=True, max_strokes=args.num_strokes,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=0, pin_memory=True,
    )

    # Load model
    model = TinyCNNJoint(
        num_classes=num_classes,
        num_radicals=num_radicals,
        num_structures=13,
        num_strokes=args.num_strokes,
    )
    model.load_joint_checkpoint(args.checkpoint, device=device)
    model.load_constraint_tensors(args.master_table, args.radical_mask)
    model = model.to(device)

    logger.info(f"Loaded model from {args.checkpoint}")
    logger.info(f"Evaluating on {len(val_dataset)} samples")

    # Run ablation configurations
    configs = [
        ("CNN only", False, False, False),
        ("+ structure", True, False, False),
        ("+ structure + radical", True, True, False),
        ("+ structure + radical + stroke", True, True, True),
    ]

    results = []
    for name, use_struct, use_rad, use_stroke in configs:
        logger.info(f"\nEvaluating: {name}")
        metrics = evaluate_with_constraints(
            model, val_loader, device,
            use_structure=use_struct,
            use_radical=use_rad,
            use_stroke=use_stroke,
        )
        results.append({"config": name, **metrics})
        logger.info(
            f"  Top-1: {metrics['top1']:.4f}  Top-5: {metrics['top5']:.4f}  "
            f"Top-10: {metrics['top10']:.4f}  Top-20: {metrics['top20']:.4f}"
        )

    # Print markdown table
    print("\n## Ablation Results\n")
    print("| Configuration | Top-1 | Top-5 | Top-10 | Top-20 |")
    print("|---|---|---|---|---|")
    for r in results:
        print(
            f"| {r['config']} | {r['top1']:.4f} | {r['top5']:.4f} | "
            f"{r['top10']:.4f} | {r['top20']:.4f} |"
        )

    # Save CSV
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["config", "top1", "top5", "top10", "top20", "total"])
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"\nSaved results to {args.output_csv}")

    # Cleanup
    shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
