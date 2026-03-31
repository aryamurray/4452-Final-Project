"""Build constraint tensors for the differentiable symbolic layer.

Reads radical_table.json and label_map.json, produces:
- master_table.pt: (num_classes, 3) int64 — [structure_id, primary_radical_id, stroke_class]
- radical_mask.pt: (num_classes, num_radicals) sparse COO float32 — binary radical presence
"""

import argparse
from pathlib import Path

import torch

from hccr.data.label_map import LabelMap
from hccr.structural.radical_table import RadicalTable
from hccr.utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build constraint tensors for symbolic layer")
    parser.add_argument("--label-map", type=Path, default=Path("resources/label_map.json"))
    parser.add_argument("--radical-table", type=Path, default=Path("resources/radical_table.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("resources"))
    parser.add_argument("--max-strokes", type=int, default=30)
    args = parser.parse_args()

    label_map = LabelMap.load(args.label_map)
    radical_table = RadicalTable.load(args.radical_table)

    num_classes = len(label_map)
    num_radicals = len(radical_table.all_radicals)
    max_strokes = args.max_strokes

    logger.info(f"Classes: {num_classes}, Radicals: {num_radicals}, Max strokes: {max_strokes}")

    # Build master_table: (num_classes, 3) — [structure_id, primary_radical_id, stroke_class]
    master_table = torch.zeros(num_classes, 3, dtype=torch.long)

    # Build radical_mask as sparse: collect (row, col) indices
    row_indices = []
    col_indices = []

    chars_matched = 0

    for idx in range(num_classes):
        char = label_map.decode(idx)

        # Structure
        structure = radical_table.char_to_structure.get(char, 12)  # 12 = atomic/unknown
        master_table[idx, 0] = structure

        # Primary radical (first radical in decomposition)
        radicals = radical_table.char_to_radicals.get(char, [])
        if radicals and radicals[0] in radical_table.radical_to_index:
            master_table[idx, 1] = radical_table.radical_to_index[radicals[0]]
        else:
            master_table[idx, 1] = 0  # default

        # Stroke count class (clamped to [0, max_strokes-1])
        strokes = radical_table.char_to_strokes.get(char, 8)
        master_table[idx, 2] = min(strokes, max_strokes - 1)

        # Radical mask entries
        for radical in radicals:
            if radical in radical_table.radical_to_index:
                rad_idx = radical_table.radical_to_index[radical]
                row_indices.append(idx)
                col_indices.append(rad_idx)

        if char in radical_table.char_to_radicals:
            chars_matched += 1

    # Build sparse COO tensor
    if row_indices:
        indices = torch.tensor([row_indices, col_indices], dtype=torch.long)
        values = torch.ones(len(row_indices), dtype=torch.float32)
        radical_mask = torch.sparse_coo_tensor(
            indices, values, size=(num_classes, num_radicals)
        ).coalesce()
    else:
        radical_mask = torch.sparse_coo_tensor(
            torch.zeros(2, 0, dtype=torch.long),
            torch.zeros(0, dtype=torch.float32),
            size=(num_classes, num_radicals),
        )

    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)

    master_table_path = args.output_dir / "master_table.pt"
    radical_mask_path = args.output_dir / "radical_mask.pt"

    torch.save(master_table, master_table_path)
    torch.save(radical_mask, radical_mask_path)

    # Stats
    master_bytes = master_table.element_size() * master_table.nelement()
    nnz = radical_mask._nnz()
    # Sparse size: indices (2 * nnz * 8 bytes) + values (nnz * 4 bytes)
    sparse_bytes = 2 * nnz * 8 + nnz * 4
    dense_bytes = num_classes * num_radicals * 4  # float32

    logger.info(f"Characters matched to radical table: {chars_matched}/{num_classes}")
    logger.info(f"master_table shape: {tuple(master_table.shape)}, size: {master_bytes / 1024:.1f} KB")
    logger.info(f"radical_mask shape: ({num_classes}, {num_radicals})")
    logger.info(f"  nonzero entries: {nnz}")
    logger.info(f"  sparsity: {1 - nnz / (num_classes * num_radicals):.4f}")
    logger.info(f"  sparse size: {sparse_bytes / 1024:.1f} KB")
    logger.info(f"  dense equivalent: {dense_bytes / 1024:.1f} KB ({dense_bytes / 1024 / 1024:.1f} MB)")
    logger.info(f"  compression ratio: {dense_bytes / max(sparse_bytes, 1):.1f}x")
    logger.info(f"Saved to {master_table_path} and {radical_mask_path}")


if __name__ == "__main__":
    main()
