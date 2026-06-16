"""Numeral symbol-concatenate drill (R8 sub-skill).

Drills the space-strip/concat copy that produces the boxed answer string, in the
EXACT surface used by the numeral reasoner:
  ``Symbols taken in order: L X X X V I I I``  ->  ``Concatenate: LXXXVIII``
Generated from the real n=1..100 outputs so the layouts match the reasoner's own
distribution. Deterministic, empty-<think>.
"""

from __future__ import annotations

import hashlib

from src.reasoners.numeral import ROMAN_VALUES


def _taken_symbols(n: int) -> list[str]:
    parts: list[str] = []
    remaining = n
    for val, sym in ROMAN_VALUES:
        while remaining >= val:
            parts.append(sym)
            remaining -= val
    return parts


def generate() -> list[dict[str, str]]:
    problems: list[dict[str, str]] = []
    seen: set[str] = set()
    i = 0
    for n in range(1, 101):
        taken = _taken_symbols(n)
        if not taken:
            continue
        spaced = " ".join(taken)
        joined = "".join(taken)
        if spaced in seen:
            continue
        seen.add(spaced)
        problems.append({
            "id": hashlib.sha256(f"numconcat_{i}".encode()).hexdigest()[:8],
            "prompt": f"Symbols taken in order: {spaced}",
            "completion": f"Concatenate: {joined}",
            "category": "numeral_concat",
        })
        i += 1
    return problems
