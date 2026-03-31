"""Build stroke type signatures from Make-Me-a-Hanzi median data.

Classifies each stroke into 6 basic types by analyzing median point direction:
  0: heng  (horizontal)    — predominantly rightward
  1: shu   (vertical)      — predominantly downward
  2: pie   (left-falling)  — down-left diagonal
  3: na    (right-falling)  — down-right diagonal
  4: dian  (dot)           — very short stroke
  5: zhe   (turning/hook)  — significant direction change

Reads:  resources/makemeahanzi_graphics.txt  (9574 chars with median coords)
Writes: resources/stroke_types.json          (char -> [h, v, lf, rf, dot, turn])

Usage:
    python scripts/build_stroke_types.py
    python scripts/build_stroke_types.py --graphics resources/makemeahanzi_graphics.txt
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import List, Tuple

from hccr.utils import get_logger

logger = get_logger(__name__)

# Stroke type indices
HENG = 0   # horizontal
SHU = 1    # vertical
PIE = 2    # left-falling
NA = 3     # right-falling
DIAN = 4   # dot
ZHE = 5    # turning/hook

STROKE_NAMES = ["heng", "shu", "pie", "na", "dian", "zhe"]


def classify_stroke(median: List[List[int]]) -> int:
    """Classify a single stroke from its median point sequence.

    Make-Me-a-Hanzi uses math-style coordinates on a ~1024x1024 canvas:
      x: 0=left,  increases rightward
      y: 0=bottom, increases UPWARD  (NOT screen coords)

    So in this system, writing downward = negative dy.

    Args:
        median: List of [x, y] points tracing the stroke center

    Returns:
        Stroke type index (0-5)
    """
    if len(median) < 2:
        return DIAN

    # Total stroke path length
    total_len = 0.0
    for i in range(1, len(median)):
        dx = median[i][0] - median[i - 1][0]
        dy = median[i][1] - median[i - 1][1]
        total_len += math.sqrt(dx * dx + dy * dy)

    # Very short stroke = dot
    if total_len < 120:
        return DIAN

    # Net displacement from start to end
    dx_net = median[-1][0] - median[0][0]
    dy_net = median[-1][1] - median[0][1]
    net_len = math.sqrt(dx_net * dx_net + dy_net * dy_net)

    # Check for turning/hook: path much longer than net displacement
    if net_len > 1e-6 and total_len / net_len > 2.0:
        return ZHE

    # Check for sharp direction changes between segments
    if len(median) >= 3:
        max_angle_change = 0.0
        for i in range(1, len(median) - 1):
            v1x = median[i][0] - median[i - 1][0]
            v1y = median[i][1] - median[i - 1][1]
            v2x = median[i + 1][0] - median[i][0]
            v2y = median[i + 1][1] - median[i][1]

            len1 = math.sqrt(v1x * v1x + v1y * v1y)
            len2 = math.sqrt(v2x * v2x + v2y * v2y)
            if len1 < 1e-6 or len2 < 1e-6:
                continue

            cos_angle = (v1x * v2x + v1y * v2y) / (len1 * len2)
            cos_angle = max(-1.0, min(1.0, cos_angle))
            angle_change = math.acos(cos_angle)
            max_angle_change = max(max_angle_change, angle_change)

        # Sharp turn > ~80 degrees = turning/hook stroke
        if max_angle_change > 1.4:
            return ZHE

    if net_len < 1e-6:
        return DIAN

    # Classify by net angle (math-coords: y-up)
    # atan2 returns: 0=right, π/2=up, π=left, -π/2=down
    angle = math.atan2(dy_net, dx_net)
    if angle < 0:
        angle += 2 * math.pi

    # In math-coords, writing directions map to:
    #   Rightward (横 heng):     angle ≈ 0/2π
    #   Upward (提 ti=heng):     angle ≈ π/2
    #   Leftward (rare):         angle ≈ π
    #   Down-left (撇 pie):      angle ≈ 5π/4 ≈ 3.93
    #   Downward (竖 shu):       angle ≈ 3π/2 ≈ 4.71
    #   Down-right (捺 na):      angle ≈ 7π/4 ≈ 5.50
    #
    # Boundaries (in radians):
    #   [0, 2.62)     → HENG  (right, up-right, upward = heng/ti variants)
    #   [2.62, 3.67)  → HENG  (leftward, rare)
    #   [3.67, 4.36)  → PIE   (down-left, ~210°-250°)
    #   [4.36, 5.06)  → SHU   (downward, ~250°-290°)
    #   [5.06, 5.76)  → NA    (down-right, ~290°-330°)
    #   [5.76, 6.28)  → HENG  (near-rightward from below)

    if angle < 3.67:
        return HENG
    elif angle < 4.36:
        return PIE
    elif angle < 5.06:
        return SHU
    elif angle < 5.76:
        return NA
    else:
        return HENG


def build_stroke_types(graphics_path: Path) -> dict:
    """Parse graphics.txt and build stroke type counts per character.

    Returns:
        Dict mapping character -> [heng, shu, pie, na, dian, zhe] counts
    """
    result = {}

    with open(graphics_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            char = data["character"]
            medians = data.get("medians", [])

            counts = [0] * 6
            for median in medians:
                stroke_type = classify_stroke(median)
                counts[stroke_type] += 1

            result[char] = counts

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build stroke type signatures")
    parser.add_argument(
        "--graphics", type=Path,
        default=Path("resources/makemeahanzi_graphics.txt"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("resources/stroke_types.json"),
    )
    parser.add_argument(
        "--label-map", type=Path,
        default=Path("resources/label_map.json"),
    )
    args = parser.parse_args()

    logger.info(f"Building stroke types from {args.graphics}")

    stroke_types = build_stroke_types(args.graphics)
    logger.info(f"Processed {len(stroke_types)} characters")

    # Check coverage against label map if available
    if args.label_map.exists():
        from hccr.data.label_map import LabelMap
        label_map = LabelMap.load(args.label_map)
        covered = sum(1 for c in label_map._char_to_idx if c in stroke_types)
        logger.info(
            f"Coverage: {covered}/{len(label_map)} "
            f"({100 * covered / len(label_map):.1f}%) of label map characters"
        )

    # Show distribution
    all_counts = [0] * 6
    for counts in stroke_types.values():
        for i in range(6):
            all_counts[i] += counts[i]
    total = sum(all_counts)
    logger.info("Stroke type distribution:")
    for i, name in enumerate(STROKE_NAMES):
        pct = 100 * all_counts[i] / total if total > 0 else 0
        logger.info(f"  {name:10s}: {all_counts[i]:>7d} ({pct:.1f}%)")

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(stroke_types, f, ensure_ascii=False, indent=0)

    logger.info(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
