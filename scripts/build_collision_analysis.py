"""Analyze symbolic signature collisions across the character set.

Groups characters by identical (structure, radical_set, stroke_count) tuples
to determine the theoretical accuracy ceiling of the symbolic layer.

Characters that share an identical symbolic signature cannot be distinguished
by the symbolic layer — the CNN must do it alone.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from hccr.data.label_map import LabelMap
from hccr.structural.radical_table import RadicalTable
from hccr.utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze symbolic signature collisions")
    parser.add_argument(
        "--label-map", type=Path, default=Path("resources/label_map.json"),
    )
    parser.add_argument(
        "--radical-table", type=Path, default=Path("resources/radical_table.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/collision_analysis.json"),
    )
    args = parser.parse_args()

    label_map = LabelMap.load(args.label_map)
    radical_table = RadicalTable.load(args.radical_table)

    num_classes = len(label_map)
    logger.info(f"Analyzing {num_classes} characters")

    # Build signature for each character
    sig_to_chars: dict[tuple, list[str]] = defaultdict(list)
    chars_with_info = 0
    chars_without_info = 0

    for idx in range(num_classes):
        char = label_map.decode(idx)

        radicals = radical_table.char_to_radicals.get(char, [])
        structure = radical_table.char_to_structure.get(char, 12)
        strokes = radical_table.char_to_strokes.get(char, 8)

        if char not in radical_table.char_to_radicals:
            chars_without_info += 1

        # Convert radical list to frozenset of indices for hashable comparison
        radical_indices = frozenset(
            radical_table.radical_to_index[r]
            for r in radicals
            if r in radical_table.radical_to_index
        )

        sig = (structure, radical_indices, strokes)
        sig_to_chars[sig].append(char)
        chars_with_info += 1

    # Analyze collisions
    collision_groups = {
        sig: chars for sig, chars in sig_to_chars.items() if len(chars) > 1
    }
    unique_sigs = {
        sig: chars for sig, chars in sig_to_chars.items() if len(chars) == 1
    }

    total_colliding_chars = sum(len(chars) for chars in collision_groups.values())
    num_collision_groups = len(collision_groups)

    # Ceiling: within each collision group, the symbolic layer can pick any
    # of the group members but only 1 is correct. So it can resolve
    # num_collision_groups out of total_colliding_chars.
    # Chars with unique sigs: always resolvable.
    resolvable = len(unique_sigs) + num_collision_groups
    ceiling = resolvable / num_classes if num_classes > 0 else 0.0

    # Group size distribution
    size_dist: dict[int, int] = defaultdict(int)
    for chars in collision_groups.values():
        size_dist[len(chars)] += 1

    # Top collision groups (largest first)
    sorted_groups = sorted(collision_groups.items(), key=lambda x: -len(x[1]))
    top_groups = []
    for (structure, radical_set, strokes), chars in sorted_groups[:20]:
        top_groups.append({
            "structure": structure,
            "strokes": strokes,
            "num_radicals": len(radical_set),
            "characters": chars,
            "group_size": len(chars),
        })

    # Summary
    logger.info(f"Total characters: {num_classes}")
    logger.info(f"Characters with radical info: {chars_with_info}")
    logger.info(f"Characters without radical info: {chars_without_info}")
    logger.info(f"Unique signatures: {len(unique_sigs)}")
    logger.info(f"Collision groups: {num_collision_groups}")
    logger.info(f"Total colliding characters: {total_colliding_chars}")
    logger.info(f"Theoretical ceiling: {ceiling:.4f} ({ceiling*100:.2f}%)")
    logger.info(f"Group size distribution: {dict(sorted(size_dist.items()))}")

    if top_groups:
        logger.info(f"\nTop {len(top_groups)} largest collision groups:")
        for i, g in enumerate(top_groups):
            chars_preview = "".join(g["characters"][:10])
            if len(g["characters"]) > 10:
                chars_preview += "..."
            logger.info(
                f"  {i+1}. size={g['group_size']}, "
                f"struct={g['structure']}, strokes={g['strokes']}, "
                f"chars={chars_preview}"
            )

    # Save results
    results = {
        "num_classes": num_classes,
        "chars_with_radical_info": chars_with_info,
        "chars_without_radical_info": chars_without_info,
        "unique_signatures": len(unique_sigs),
        "collision_groups": num_collision_groups,
        "total_colliding_chars": total_colliding_chars,
        "theoretical_ceiling": ceiling,
        "group_size_distribution": {str(k): v for k, v in sorted(size_dist.items())},
        "top_collision_groups": top_groups,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(f"\nSaved analysis to {args.output}")


if __name__ == "__main__":
    main()
