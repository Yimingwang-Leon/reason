"""Numeral single-subtraction drill (R8 sub-skill).

Drills the one local subtraction ``{r} - {v} = {new}`` in the EXACT surface used
by the numeral reasoner's TAKE line. r ranges 1..100, v is one of the Roman
values that can actually be subtracted from r (v <= r), so the (r, v) pairs are
exactly the ones the long table-walk emits. Deterministic, empty-<think>.
"""

from __future__ import annotations

import hashlib

from src.reasoners.numeral import ROMAN_VALUES

# values that ever appear as a subtractend for n in 1..100
_VALUES = [v for v, _ in ROMAN_VALUES if v <= 100]


def generate() -> list[dict[str, str]]:
    problems: list[dict[str, str]] = []
    i = 0
    for r in range(1, 101):
        for v in _VALUES:
            if v > r:
                continue
            new = r - v
            problems.append({
                "id": hashlib.sha256(f"numsub_{i}".encode()).hexdigest()[:8],
                "prompt": f"{r} - {v} = ?",
                "completion": f"{r} - {v} = {new}",
                "category": "numeral_subtract",
            })
            i += 1
    return problems
