"""Equation sub-skill augmenter: number-string reversal.

Drills the single mechanic behind ~73% of equation_numeric wins: reversing a
digit string (reverse operands / reverse result). The model must execute this
flawlessly at greedy decode or the equation CoT derails.

Input:  a list of numbers
Output: each number reversed (leading zeros preserved as written)
"""
from __future__ import annotations

import hashlib
import random

ROWS_PER_PROBLEM = 8
N_PROBLEMS = 300
DEMO_ROWS = 3


def _num(rng: random.Random) -> str:
    return "".join(str(rng.randint(0, 9)) for _ in range(rng.randint(2, 4)))


def generate() -> list[dict[str, str]]:
    rng = random.Random(1701)
    out: list[dict[str, str]] = []
    for i in range(N_PROBLEMS):
        demo = [_num(rng) for _ in range(DEMO_ROWS)]
        demo_lines = [f"{n} -> {n[::-1]}" for n in demo]
        tests = [_num(rng) for _ in range(ROWS_PER_PROBLEM)]
        prompt = (
            "In Alice's Wonderland, a secret rule rewrites each number.\n\n"
            "Sample:\n" + "\n".join(demo_lines) + "\n\n"
            "Now apply the same rule to each number:\n" + "\n".join(tests)
        )
        completion = "\n".join(f"{n} -> {n[::-1]}" for n in tests)
        pid = hashlib.sha256(f"equation_reversal_{i}".encode()).hexdigest()[:8]
        out.append({"id": pid, "prompt": prompt, "completion": completion,
                    "category": "equation_reversal"})
    print(f"[equation_reversal] Generated {len(out)} problems")
    return out
