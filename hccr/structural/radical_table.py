"""Radical decomposition table from CJKVI IDS data.

Parses Ideographic Description Sequences (IDS) to extract:
- Atomic radicals for each character
- Structural composition patterns (13 IDC types)
- Stroke count estimates
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar

from hccr.utils import load_json, save_json

logger = logging.getLogger(__name__)


# Ideographic Description Characters (IDC) U+2FF0-U+2FFB
IDC_MAP: dict[str, int] = {
    "⿰": 0,  # left-right
    "⿱": 1,  # top-bottom
    "⿲": 2,  # left-middle-right
    "⿳": 3,  # top-middle-bottom
    "⿴": 4,  # surround
    "⿵": 5,  # surround from above
    "⿶": 6,  # surround from below
    "⿷": 7,  # surround from left
    "⿸": 8,  # surround from upper-left
    "⿹": 9,  # surround from upper-right
    "⿺": 10,  # surround from lower-left
    "⿻": 11,  # overlaid
}

# Common radical stroke counts (heuristic for ~100 common radicals)
RADICAL_STROKES: dict[str, int] = {
    "一": 1, "丨": 1, "丶": 1, "丿": 1, "乙": 1, "亅": 1,
    "二": 2, "亠": 2, "人": 2, "亻": 2, "儿": 2, "入": 2, "八": 2, "冂": 2,
    "冖": 2, "冫": 2, "几": 2, "凵": 2, "刀": 2, "刂": 2, "力": 2, "勹": 2,
    "匕": 2, "匚": 2, "匸": 2, "十": 2, "卜": 2, "卩": 2, "厂": 2, "厶": 2,
    "又": 2, "九": 2, "丁": 2,
    "口": 3, "囗": 3, "土": 3, "士": 3, "夂": 3, "夊": 3, "夕": 3, "大": 3,
    "女": 3, "子": 3, "宀": 3, "寸": 3, "小": 3, "尢": 3, "尸": 3, "屮": 3,
    "山": 3, "川": 3, "工": 3, "己": 3, "巾": 3, "干": 3, "幺": 3, "广": 3,
    "廴": 3, "廾": 3, "弋": 3, "弓": 3, "彐": 3, "彡": 3, "彳": 3, "三": 3,
    "千": 3, "万": 3, "丈": 3, "才": 3, "寸": 3,
    "心": 4, "忄": 4, "戈": 4, "戶": 4, "手": 4, "扌": 4, "支": 4, "攴": 4,
    "文": 4, "斗": 4, "斤": 4, "方": 4, "无": 4, "日": 4, "曰": 4, "月": 4,
    "木": 4, "欠": 4, "止": 4, "歹": 4, "殳": 4, "毋": 4, "比": 4, "毛": 4,
    "氏": 4, "气": 4, "水": 4, "氵": 4, "火": 4, "灬": 4, "爪": 4, "父": 4,
    "爻": 4, "爿": 4, "片": 4, "牙": 4, "牛": 4, "犬": 4, "犭": 4, "王": 4,
    "元": 4, "井": 4, "勿": 4, "五": 4, "屯": 4, "巴": 4, "车": 4,
    "玄": 5, "玉": 5, "瓜": 5, "瓦": 5, "甘": 5, "生": 5, "用": 5, "田": 5,
    "疋": 5, "疒": 5, "癶": 5, "白": 5, "皮": 5, "皿": 5, "目": 5, "矛": 5,
    "矢": 5, "石": 5, "示": 5, "禸": 5, "禾": 5, "穴": 5, "立": 5, "世": 5,
    "业": 5, "丝": 5, "礻": 5, "衤": 5, "正": 5, "母": 5,
    "竹": 6, "米": 6, "糸": 6, "缶": 6, "网": 6, "羊": 6, "羽": 6, "老": 6,
    "而": 6, "耒": 6, "耳": 6, "聿": 6, "肉": 6, "臣": 6, "自": 6, "至": 6,
    "臼": 6, "舌": 6, "舛": 6, "舟": 6, "艮": 6, "色": 6, "艸": 6, "虍": 6,
    "虫": 6, "血": 6, "行": 6, "衣": 6, "西": 6, "覀": 6, "糹": 6, "纟": 6,
    "言": 7, "訁": 7, "谷": 7, "豆": 7, "豕": 7, "豸": 7, "貝": 7, "贝": 7,
    "赤": 7, "走": 7, "足": 7, "身": 7, "車": 7, "辛": 7, "辰": 7, "辵": 7,
    "邑": 7, "酉": 7, "釆": 7, "里": 7, "見": 7, "角": 7, "青": 8,
    "金": 8, "釒": 8, "钅": 8, "長": 8, "門": 8, "阜": 8, "隶": 8, "隹": 8,
    "雨": 8, "非": 8, "奄": 8, "齿": 8, "齊": 8,
    "面": 9, "革": 9, "韋": 9, "韭": 9, "音": 9, "頁": 9, "页": 9, "風": 9,
    "飛": 9, "食": 9, "飠": 9, "首": 9, "香": 9,
    "馬": 10, "马": 10, "骨": 10, "高": 10, "髟": 10, "鬥": 10, "鬯": 10,
    "鬲": 10, "鬼": 10, "魚": 11, "鸟": 11, "鹵": 11, "鹿": 11, "麥": 11,
    "麻": 11, "黃": 12, "黍": 12, "黑": 12, "黹": 12, "黽": 13, "鼎": 13,
    "鼓": 13, "鼠": 13, "鼻": 14, "齊": 14, "齒": 15, "龍": 16, "龜": 16,
    "龠": 17,
}

DEFAULT_RADICAL_STROKES = 8


class RadicalTable:
    """Radical decomposition and structure table for Chinese characters.

    Attributes:
        char_to_radicals: Maps character to list of atomic radicals
        char_to_structure: Maps character to structure class (0-12)
        char_to_strokes: Maps character to stroke count
        radical_to_chars: Maps radical to list of containing characters
        all_radicals: Sorted list of unique radicals
        radical_to_index: Maps radical to index in all_radicals
    """

    def __init__(self) -> None:
        self.char_to_radicals: dict[str, list[str]] = {}
        self.char_to_structure: dict[str, int] = {}
        self.char_to_strokes: dict[str, int] = {}
        self.radical_to_chars: dict[str, list[str]] = {}
        self.all_radicals: list[str] = []
        self.radical_to_index: dict[str, int] = {}

    @classmethod
    def build_from_ids_file(
        cls,
        ids_path: Path,
        char_set: set[str] | None = None,
    ) -> RadicalTable:
        """Build radical table from CJKVI IDS file.

        Args:
            ids_path: Path to ids.txt file
            char_set: Optional set of characters to include (e.g., 3755 class set)

        Returns:
            Populated RadicalTable instance
        """
        table = cls()
        ids_path = Path(ids_path)

        if not ids_path.exists():
            logger.warning(f"IDS file not found: {ids_path}")
            return table

        logger.info(f"Building radical table from {ids_path}")

        # Parse IDS file
        char_to_ids: dict[str, str] = {}
        with open(ids_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split("\t")
                if len(parts) < 3:
                    continue

                # Format: codepoint\tchar\tIDS_sequence
                char = parts[1]
                ids_seq = parts[2]

                # Filter to char_set if provided
                if char_set and char not in char_set:
                    continue

                char_to_ids[char] = ids_seq

        logger.info(f"Loaded {len(char_to_ids)} character IDS entries")

        # Process each character
        for char, ids_seq in char_to_ids.items():
            # Extract structure (top-level IDC)
            structure = cls._extract_structure(ids_seq)
            table.char_to_structure[char] = structure

            # Recursively decompose to atomic radicals
            radicals = cls._decompose_to_radicals(ids_seq, char_to_ids)
            table.char_to_radicals[char] = radicals

            # Compute stroke count from radicals
            strokes = cls._estimate_strokes(radicals)
            table.char_to_strokes[char] = strokes

        # Build reverse mapping and indices
        table._build_radical_indices()

        logger.info(f"Built table with {len(table.all_radicals)} unique radicals")
        logger.info(f"Covered {len(table.char_to_structure)} characters")

        return table

    @staticmethod
    def _extract_structure(ids_seq: str) -> int:
        """Extract top-level structure class from IDS sequence.

        Returns structure index 0-12 (12 = atomic/unknown).
        """
        if not ids_seq:
            return 12

        # Check first character for IDC
        first_char = ids_seq[0]
        if first_char in IDC_MAP:
            return IDC_MAP[first_char]

        # No IDC found -> atomic character
        return 12

    @classmethod
    def _decompose_to_radicals(
        cls,
        ids_seq: str,
        char_to_ids: dict[str, str],
        visited: set[str] | None = None,
    ) -> list[str]:
        """Recursively decompose IDS to atomic radicals.

        Returns list of radical characters (atomic components).
        """
        if visited is None:
            visited = set()

        radicals: list[str] = []

        # Parse IDS sequence
        i = 0
        while i < len(ids_seq):
            char = ids_seq[i]

            # Skip IDC markers
            if char in IDC_MAP:
                i += 1
                continue

            # Check if this component can be further decomposed
            if (char in char_to_ids
                    and char not in visited
                    and char_to_ids[char] != char):
                # Recurse to decompose this component
                # Use a copy of visited for each branch to allow
                # the same radical to appear in multiple branches
                branch_visited = visited | {char}
                sub_radicals = cls._decompose_to_radicals(
                    char_to_ids[char], char_to_ids, branch_visited
                )
                radicals.extend(sub_radicals)
            else:
                # Atomic radical - cannot decompose further
                radicals.append(char)

            i += 1

        return radicals

    @staticmethod
    def _estimate_strokes(radicals: list[str]) -> int:
        """Estimate stroke count from radical components."""
        total = 0
        for radical in radicals:
            total += RADICAL_STROKES.get(radical, DEFAULT_RADICAL_STROKES)
        return max(1, total)  # At least 1 stroke

    def _build_radical_indices(self) -> None:
        """Build radical-to-character mappings and indices."""
        # Collect all unique radicals
        radical_set: set[str] = set()
        for radicals in self.char_to_radicals.values():
            radical_set.update(radicals)

        # Sort for consistent ordering
        self.all_radicals = sorted(radical_set)

        # Build index mapping
        self.radical_to_index = {
            radical: idx for idx, radical in enumerate(self.all_radicals)
        }

        # Build reverse mapping: radical -> characters
        self.radical_to_chars = {radical: [] for radical in self.all_radicals}
        for char, radicals in self.char_to_radicals.items():
            for radical in radicals:
                if radical in self.radical_to_chars:
                    self.radical_to_chars[radical].append(char)

    def get_radical_vector(self, char: str) -> list[int]:
        """Get multi-hot radical vector for a character.

        Args:
            char: Input character

        Returns:
            Binary vector of length len(all_radicals), 1 if radical present
        """
        vector = [0] * len(self.all_radicals)

        if char not in self.char_to_radicals:
            return vector

        for radical in self.char_to_radicals[char]:
            if radical in self.radical_to_index:
                idx = self.radical_to_index[radical]
                vector[idx] = 1

        return vector

    def get_structure(self, char: str) -> int:
        """Get structure class for a character.

        Returns structure index 0-12 (12 = unknown).
        """
        return self.char_to_structure.get(char, 12)

    def get_strokes(self, char: str) -> int:
        """Get estimated stroke count for a character."""
        return self.char_to_strokes.get(char, DEFAULT_RADICAL_STROKES)

    def save(self, path: Path) -> None:
        """Save radical table to JSON file."""
        data = {
            "char_to_radicals": self.char_to_radicals,
            "char_to_structure": self.char_to_structure,
            "char_to_strokes": self.char_to_strokes,
            "radical_to_chars": self.radical_to_chars,
            "all_radicals": self.all_radicals,
            "radical_to_index": self.radical_to_index,
        }
        save_json(data, path)
        logger.info(f"Saved radical table to {path}")

    @classmethod
    def load(cls, path: Path) -> RadicalTable:
        """Load radical table from JSON file."""
        data = load_json(path)
        table = cls()
        table.char_to_radicals = data["char_to_radicals"]
        table.char_to_structure = {
            k: int(v) for k, v in data["char_to_structure"].items()
        }
        table.char_to_strokes = {
            k: int(v) for k, v in data["char_to_strokes"].items()
        }
        table.radical_to_chars = data["radical_to_chars"]
        table.all_radicals = data["all_radicals"]
        table.radical_to_index = {
            k: int(v) for k, v in data["radical_to_index"].items()
        }
        logger.info(f"Loaded radical table from {path}")
        return table
