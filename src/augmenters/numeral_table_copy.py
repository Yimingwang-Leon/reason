"""Numeral value-table-copy drill (R8 sub-skill).

Drills the value<->symbol copy primitive in the IDENTICAL surface format used by
the numeral reasoner's Value table block. Two prompt shapes, both deterministic:
  ``XL = ?``  ->  ``XL = 40``
  ``? = 40``  ->  ``XL = 40``
empty-<think> (enable_thinking=False at tokenize time), category-tagged.
"""

from __future__ import annotations

import hashlib

from src.reasoners.numeral import ROMAN_VALUES


def generate() -> list[dict[str, str]]:
    problems: list[dict[str, str]] = []
    i = 0
    for val, sym in ROMAN_VALUES:
        # value-from-symbol direction
        problems.append({
            "id": hashlib.sha256(f"numtable_v_{i}".encode()).hexdigest()[:8],
            "prompt": f"{sym} = ?",
            "completion": f"{sym} = {val}",
            "category": "numeral_table_copy",
        })
        i += 1
        # symbol-from-value direction
        problems.append({
            "id": hashlib.sha256(f"numtable_s_{i}".encode()).hexdigest()[:8],
            "prompt": f"? = {val}",
            "completion": f"{sym} = {val}",
            "category": "numeral_table_copy",
        })
        i += 1
    return problems
