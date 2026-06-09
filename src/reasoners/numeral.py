"""Numeral: Arabic to Roman reasoning generator.

max-R redesign (design_numeral.json): the CoT is a uniform fixed table-walk.
Every committed token is a COPY from a value table printed verbatim above, and
the next reduction row is POSITIONALLY DETERMINISTIC (always the next of the 13
fixed table entries, in fixed order). The legacy take-only/skip-rows trace let
the row after ``{n} >= `` be one of many table rows -> the multimodal regime-3
row-jump that collapses greedy R. Here every row always emits at least a SKIP
line, so position alone keys the next (value, symbol) pair. The final boxed
answer is a character-for-character copy of the assembled ``Concatenate:`` line
and is byte-identical to the legacy ``_to_roman(n)`` for every n.
"""

from __future__ import annotations

from .store_types import Problem

ROMAN_VALUES: list[tuple[int, str]] = [
    (1000, "M"),
    (900, "CM"),
    (500, "D"),
    (400, "CD"),
    (100, "C"),
    (90, "XC"),
    (50, "L"),
    (40, "XL"),
    (10, "X"),
    (9, "IX"),
    (5, "V"),
    (4, "IV"),
    (1, "I"),
]


def _to_roman(n: int) -> str:
    parts: list[str] = []
    remaining = n
    for val, sym in ROMAN_VALUES:
        while remaining >= val:
            parts.append(sym)
            remaining -= val
    return "".join(parts)


def reasoning_numeral(problem: Problem) -> str:
    n = int(problem.question)
    computed = _to_roman(n)  # box computation: UNCHANGED from legacy.

    lines: list[str] = []
    lines.append("This is an Arabic to Roman numeral conversion.")
    lines.append("")
    # R2 anchor: restate the input verbatim, consume the remainder from here.
    lines.append(f"Input number: {n}")
    lines.append("")
    # R5 fixed block: the value table is byte-identical in EVERY trace, all rows
    # always, in this order. Every (value, symbol) consumed below is a copy of a
    # line in this table.
    lines.append("Value table (largest to smallest):")
    for val, sym in ROMAN_VALUES:
        lines.append(f"  {sym} = {val}")
    lines.append("")

    # R3 one micro-step per line, R1 uniform full table walk. Iterate the 13
    # table rows IN FIXED ORDER; for each row emit one TAKE line per subtraction
    # while r >= v, then exactly one SKIP line when the row is exhausted. Every
    # row always contributes at least its SKIP line, so the position
    # deterministically keys the next (value, symbol) pair.
    lines.append(f"Greedy subtraction from {n}:")
    remaining = n
    taken: list[str] = []
    for val, sym in ROMAN_VALUES:
        while remaining >= val:
            new = remaining - val
            lines.append(f"  {remaining} >= {val} : take {sym}, {remaining} - {val} = {new}")
            taken.append(sym)
            remaining = new
        # row exhausted (remaining < val): one uniform SKIP line.
        lines.append(f"  {remaining} < {val} : skip {sym}")

    lines.append("")
    # R6 incremental assembly: print the taken symbols (copies of the take lines'
    # symbols, in order), then the no-space concatenation (the literal answer).
    if taken:
        lines.append("Symbols taken in order: " + " ".join(taken))
    else:
        lines.append("Symbols taken in order: (none)")
    joined = "".join(taken)
    lines.append(f"Concatenate: {joined}")
    lines.append("")
    # R6 terminal box: character-for-character copy of the Concatenate line above.
    lines.append(f"The answer is \\boxed{{{joined}}}")
    return "\n".join(lines)
