"""Equation sub-skill augmenter: digit-wise primitives.

Drills the two-digit decompositions the equation CoT relies on for the
digit-wise rule family: concatenation (a || b), per-digit absolute difference,
and cross-multiply. These are the operations most likely to be mis-executed at
greedy decode once the rule is identified (an R-lever).
"""
from __future__ import annotations

import hashlib
import random

ROWS_PER_PROBLEM = 6
N_PROBLEMS = 250
DEMO_ROWS = 3

_KINDS = ["concat", "digabsdiff", "crossmul"]


def _apply(kind: str, a: str, b: str) -> str:
    d1, d2, d3, d4 = int(a[0]), int(a[1]), int(b[0]), int(b[1])
    if kind == "concat":
        return a + b
    if kind == "digabsdiff":
        return f"{abs(d1 - d3)}{abs(d2 - d4)}"
    if kind == "crossmul":
        return str(d1 * d3 + d2 * d4)
    raise ValueError(kind)


_DESC = {
    "concat": "concatenate the two numbers (a || b)",
    "digabsdiff": "per-digit absolute difference: |a1-b1| || |a2-b2|",
    "crossmul": "cross multiply: a1*b1 + a2*b2",
}


def _num2(rng: random.Random) -> str:
    return f"{rng.randint(0, 9)}{rng.randint(0, 9)}"


def generate() -> list[dict[str, str]]:
    rng = random.Random(3141)
    out: list[dict[str, str]] = []
    for i in range(N_PROBLEMS):
        kind = _KINDS[i % len(_KINDS)]
        demo = [(_num2(rng), _num2(rng)) for _ in range(DEMO_ROWS)]
        demo_lines = [f"{a}, {b} -> {_apply(kind, a, b)}" for a, b in demo]
        tests = [(_num2(rng), _num2(rng)) for _ in range(ROWS_PER_PROBLEM)]
        test_q = [f"{a}, {b} ->" for a, b in tests]
        prompt = (
            f"Rule: {_DESC[kind]}.\n\n"
            "Sample:\n" + "\n".join(demo_lines) + "\n\n"
            "Now apply the rule:\n" + "\n".join(test_q)
        )
        completion = "\n".join(f"{a}, {b} -> {_apply(kind, a, b)}" for a, b in tests)
        pid = hashlib.sha256(f"equation_digitwise_{i}".encode()).hexdigest()[:8]
        out.append({"id": pid, "prompt": prompt, "completion": completion,
                    "category": "equation_digitwise"})
    print(f"[equation_digitwise] Generated {len(out)} problems")
    return out
