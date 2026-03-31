"""Build radical decomposition table from IDS data.

Parses CJKVI Ideographic Description Sequences and constructs a table
mapping characters to their atomic radicals, structural composition, and stroke counts.
"""

import argparse
from collections import Counter
from pathlib import Path

from hccr.data.label_map import LabelMap
from hccr.structural.radical_table import RadicalTable
from hccr.utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build radical decomposition table from IDS file"
    )
    parser.add_argument(
        "--ids-file",
        type=Path,
        default=Path("resources/ids.txt"),
        help="Path to CJKVI IDS file",
    )
    parser.add_argument(
        "--label-map",
        type=Path,
        default=Path("resources/label_map.json"),
        help="Path to label map JSON (defines character set)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("resources/radical_table.json"),
        help="Output path for radical table JSON file",
    )
    args = parser.parse_args()

    logger.info(f"Building radical table from {args.ids_file}")

    if not args.ids_file.exists():
        raise FileNotFoundError(f"IDS file not found: {args.ids_file}")

    # Load label map to get character set
    char_set = None
    if args.label_map.exists():
        logger.info(f"Loading label map from {args.label_map}")
        label_map = LabelMap.load(args.label_map)
        char_set = set(label_map.chars)
        logger.info(f"Filtering to {len(char_set)} characters from label map")

    # Build radical table
    radical_table = RadicalTable.build_from_ids_file(args.ids_file, char_set)

    # Save to JSON
    args.output.parent.mkdir(parents=True, exist_ok=True)
    radical_table.save(args.output)

    # Print statistics
    logger.info("\n" + "=" * 60)
    logger.info("Radical Table Statistics")
    logger.info("=" * 60)
    logger.info(f"Total unique radicals: {len(radical_table.all_radicals)}")
    logger.info(f"Characters with decompositions: {len(radical_table.char_to_radicals)}")
    logger.info(f"Coverage: {len(radical_table.char_to_radicals) / len(char_set) * 100:.1f}%")

    # Structure distribution
    structure_counts = Counter(radical_table.char_to_structure.values())
    logger.info("\nStructure Distribution:")
    structure_names = [
        "⿰ left-right",
        "⿱ top-bottom",
        "⿲ left-middle-right",
        "⿳ top-middle-bottom",
        "⿴ surround",
        "⿵ surround-above",
        "⿶ surround-below",
        "⿷ surround-left",
        "⿸ surround-upper-left",
        "⿹ surround-upper-right",
        "⿺ surround-lower-left",
        "⿻ overlaid",
        "atomic/unknown",
    ]
    for idx in range(13):
        count = structure_counts.get(idx, 0)
        name = structure_names[idx] if idx < len(structure_names) else f"type-{idx}"
        logger.info(f"  {name}: {count}")

    # Stroke distribution
    stroke_counts = list(radical_table.char_to_strokes.values())
    if stroke_counts:
        logger.info(f"\nStroke Count Range: {min(stroke_counts)} - {max(stroke_counts)}")
        logger.info(f"Mean stroke count: {sum(stroke_counts) / len(stroke_counts):.1f}")

    logger.info(f"\nRadical table saved to {args.output}")


if __name__ == "__main__":
    main()
