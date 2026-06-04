"""Equation sub-skill augmenter: signed two-operand arithmetic.

Drills the raw computations the equation CoT emits — a+b, a-b (signed), b-a,
a*b — in the exact "expr = value" form the reasoner writes. Reliable execution
of these is what turns a correctly-identified rule into a correct boxed answer
at greedy decode (an R-lever).
"""
from __future__ import annotations

import hashlib
import random

ROWS_PER_PROBLEM = 6
N_PROBLEMS = 350
DEMO_ROWS = 3

_OPS = [
    ("+", lambda a, b: a + b),
    ("-", lambda a, b: a - b),
    ("*", lambda a, b: a * b),
]


def _line(a: int, b: int, sym: str, fn) -> str:
    return f"{a} {sym} {b} = {fn(a, b)}"


def generate() -> list[dict[str, str]]:
    rng = random.Random(2718)
    out: list[dict[str, str]] = []
    for i in range(N_PROBLEMS):
        sym, fn = _OPS[i % len(_OPS)]
        demo = [(rng.randint(0, 99), rng.randint(0, 99)) for _ in range(DEMO_ROWS)]
        demo_lines = [_line(a, b, sym, fn) for a, b in demo]
        tests = [(rng.randint(0, 99), rng.randint(0, 99)) for _ in range(ROWS_PER_PROBLEM)]
        test_q = [f"{a} {sym} {b} =" for a, b in tests]
        prompt = (
            "Compute each expression. Show 'a op b = result'.\n\n"
            "Sample:\n" + "\n".join(demo_lines) + "\n\n"
            "Now compute:\n" + "\n".join(test_q)
        )
        completion = "\n".join(_line(a, b, sym, fn) for a, b in tests)
        pid = hashlib.sha256(f"equation_arith_{i}".encode()).hexdigest()[:8]
        out.append({"id": pid, "prompt": prompt, "completion": completion,
                    "category": "equation_arith"})
    print(f"[equation_arith] Generated {len(out)} problems")
    return out
