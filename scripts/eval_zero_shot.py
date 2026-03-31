"""Zero-shot character recognition via symbolic layer.

Holds out N characters from the label map. The CNN has never seen these
character classes during training, but the symbolic layer knows their
radicals/structure/strokes from radical_table.json. Auxiliary heads
(trained on other characters) still detect radicals → symbolic layer
maps predictions to unseen characters.

Reports accuracy on held-out vs seen characters, with breakdown by
collision status (characters sharing identical symbolic signatures).
"""

import argparse
import json
import random
import shutil
import tempfile
from collections import defaultdict
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


def build_collision_sets(
    label_map: LabelMap,
    radical_table: RadicalTable,
) -> dict[int, set[int]]:
    """Group class indices by identical symbolic signatures.

    Returns a mapping from class index to the set of all class indices
    that share the same (structure, radical_set, stroke_count) signature.
    """
    sig_to_classes: dict[tuple, list[int]] = defaultdict(list)

    for idx in range(len(label_map)):
        char = label_map.index_to_char.get(idx, "")
        radicals = radical_table.char_to_radicals.get(char, [])
        rad_indices = frozenset(
            radical_table.radical_to_index[r]
            for r in radicals
            if r in radical_table.radical_to_index
        )
        structure = radical_table.char_to_structure.get(char, 0)
        strokes = radical_table.char_to_strokes.get(char, 0)
        sig = (structure, rad_indices, strokes)
        sig_to_classes[sig].append(idx)

    collision_map: dict[int, set[int]] = {}
    for group in sig_to_classes.values():
        group_set = set(group)
        for idx in group:
            collision_map[idx] = group_set

    return collision_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-shot character recognition eval")
    parser.add_argument("--train-dir", type=Path, default=Path("data/HWDB1.1/train"))
    parser.add_argument("--label-map", type=Path, default=Path("resources/label_map.json"))
    parser.add_argument("--radical-table", type=Path, default=Path("resources/radical_table.json"))
    parser.add_argument("--master-table", type=Path, default=Path("resources/master_table.pt"))
    parser.add_argument("--radical-mask", type=Path, default=Path("resources/radical_mask.pt"))
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("outputs/symbolic/best_model_symbolic.pt"),
    )
    parser.add_argument("--output-json", type=Path, default=Path("outputs/zeroshot_results.json"))
    parser.add_argument("--num-holdout", type=int, default=200,
                        help="Number of characters to hold out as 'unseen'")
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

    # Build collision map
    collision_map = build_collision_sets(label_map, radical_table)

    # Select held-out characters (seed-controlled)
    all_indices = list(range(num_classes))
    random.seed(args.seed + 1000)  # different seed from train/val split
    random.shuffle(all_indices)
    holdout_set = set(all_indices[: args.num_holdout])
    seen_set = set(all_indices[args.num_holdout :])

    logger.info(f"Holding out {len(holdout_set)} characters as unseen")
    logger.info(f"Seen characters: {len(seen_set)}")

    # Determine which held-out chars have unique vs colliding signatures
    holdout_unique = set()
    holdout_colliding = set()
    for idx in holdout_set:
        group = collision_map.get(idx, {idx})
        if len(group) == 1:
            holdout_unique.add(idx)
        else:
            holdout_colliding.add(idx)

    logger.info(f"Held-out unique (no collision): {len(holdout_unique)}")
    logger.info(f"Held-out colliding: {len(holdout_colliding)}")

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
    model.eval()

    logger.info(f"Loaded model from {args.checkpoint}")
    logger.info(f"Evaluating on {len(val_dataset)} samples")

    # Evaluate
    # Track per-group metrics
    metrics = {
        "seen": {"correct": 0, "top5": 0, "top10": 0, "top20": 0, "total": 0},
        "holdout_unique": {"correct": 0, "top5": 0, "top10": 0, "top20": 0, "total": 0},
        "holdout_colliding": {"correct": 0, "top5": 0, "top10": 0, "top20": 0, "total": 0},
    }

    with torch.no_grad():
        for batch_data in tqdm(val_loader, desc="Zero-shot eval", leave=False):
            batch_data = tuple(x.to(device, non_blocking=True) for x in batch_data)
            images, char_targets = batch_data[0], batch_data[1]

            outputs = model(images, use_symbolic=True)
            combined = outputs["combined_probs"]

            preds = torch.argmax(combined, dim=1)
            _, top5 = combined.topk(5, dim=1)
            _, top10 = combined.topk(10, dim=1)
            _, top20 = combined.topk(20, dim=1)

            for i in range(char_targets.size(0)):
                target = char_targets[i].item()

                if target in holdout_unique:
                    group = "holdout_unique"
                elif target in holdout_colliding:
                    group = "holdout_colliding"
                else:
                    group = "seen"

                metrics[group]["total"] += 1
                if preds[i].item() == target:
                    metrics[group]["correct"] += 1
                if target in top5[i].tolist():
                    metrics[group]["top5"] += 1
                if target in top10[i].tolist():
                    metrics[group]["top10"] += 1
                if target in top20[i].tolist():
                    metrics[group]["top20"] += 1

    # Compute rates
    results = {}
    for group_name, m in metrics.items():
        total = m["total"]
        if total > 0:
            results[group_name] = {
                "total": total,
                "top1": m["correct"] / total,
                "top5": m["top5"] / total,
                "top10": m["top10"] / total,
                "top20": m["top20"] / total,
            }
        else:
            results[group_name] = {
                "total": 0, "top1": 0.0, "top5": 0.0, "top10": 0.0, "top20": 0.0,
            }

    results["config"] = {
        "num_holdout": args.num_holdout,
        "holdout_unique_count": len(holdout_unique),
        "holdout_colliding_count": len(holdout_colliding),
        "seed": args.seed,
    }

    # Print results
    print("\n## Zero-Shot Recognition Results\n")
    print("| Group | N | Top-1 | Top-5 | Top-10 | Top-20 |")
    print("|---|---|---|---|---|---|")
    for group_name in ["seen", "holdout_unique", "holdout_colliding"]:
        r = results[group_name]
        print(
            f"| {group_name} | {r['total']} | {r['top1']:.4f} | {r['top5']:.4f} | "
            f"{r['top10']:.4f} | {r['top20']:.4f} |"
        )

    # Save JSON
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nSaved results to {args.output_json}")

    # Cleanup
    shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
