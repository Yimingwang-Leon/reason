"""Run reasoners on parsed problems, save CoT to reasoning/<id>.txt for SFT.

Only saves traces whose extracted answer matches ground truth (per metric).
Reports per-category accuracy.
"""
from __future__ import annotations

import math
import re
import shutil
from collections import Counter
from pathlib import Path

from src.problems import load_problems
from src.reasoners.bit_manipulation import reasoning_bit_manipulation
from src.reasoners.cipher import reasoning_cipher
from src.reasoners.cryptarithm_deduce import reasoning_cryptarithm_deduce
from src.reasoners.equation_numeric_deduce import reasoning_equation_numeric_deduce
from src.reasoners.gravity import reasoning_gravity
from src.reasoners.numeral import reasoning_numeral
from src.reasoners.store_types import Problem
from src.reasoners.unit_conversion import reasoning_unit_conversion

# Phase 5: all 7 categories wired.
REASONERS = {
    "numeral": reasoning_numeral,
    "cipher": reasoning_cipher,
    "gravity": reasoning_gravity,
    "unit_conversion": reasoning_unit_conversion,
    "bit_manipulation": reasoning_bit_manipulation,
    "equation_numeric_deduce": reasoning_equation_numeric_deduce,
    "cryptarithm_deduce": reasoning_cryptarithm_deduce,
}

REASONING_DIR = Path(__file__).resolve().parent.parent / "reasoning"


def extract_answer(text: str) -> str:
    matches = re.findall(r"\\boxed\{([^}]*)(?:\}|$)", text)
    if not matches:
        return ""
    non_empty = [m.strip() for m in matches if m.strip()]
    return non_empty[-1] if non_empty else matches[-1].strip()


def metric_correct(truth: str, pred: str) -> bool:
    truth, pred = truth.strip(), pred.strip()
    if re.fullmatch(r"[01]+", truth):
        return pred.lower() == truth.lower()
    try:
        # abs_tol=0.0: strictly isomorphic to the grader's rel-tol-only rule
        # (abs_tol=1e-5 was a liberal superset near zero; 0 rows affected today,
        # hardened so selection can never admit a trace the grader would fail).
        return math.isclose(float(truth), float(pred), rel_tol=1e-2, abs_tol=0.0)
    except Exception:
        return pred.lower() == truth.lower()


def run_one(problem: Problem) -> tuple[str | None, bool]:
    fn = REASONERS.get(problem.category)
    if fn is None:
        return None, False
    text = fn(problem)
    if text is None:
        return None, False
    pred = extract_answer(text)
    return text, metric_correct(problem.answer, pred)


def main() -> None:
    problems = load_problems()
    if REASONING_DIR.exists():
        shutil.rmtree(REASONING_DIR)
    REASONING_DIR.mkdir(parents=True)

    cat_correct: Counter[str] = Counter()
    cat_total: Counter[str] = Counter()
    for p in problems:
        cat_total[p.category] += 1
        text, ok = run_one(p)
        if text is None:
            continue
        if ok:
            cat_correct[p.category] += 1
        # Per-category inclusion policy (run-011, dual-audit ruling):
        # - bit_manipulation: include wrong-but-self-consistent traces (+238). The
        #   legacy column-matching procedure ran faithfully and boxed its own result
        #   on underdetermined hard-tail layouts; dropping them is survivorship bias
        #   that leaves the hardest layouts with zero training coverage.
        # - every other category: correct-only. Wrong traces there are assertion-
        #   style (crypt concat-fallback shows a FAILED verification then boxes
        #   anyway; the old equation stub asserts an unverified rule) = the same
        #   poison family that collapsed R in runs 006/007/008.
        if not ok and p.category != "bit_manipulation":
            continue
        (REASONING_DIR / f"{p.id}.txt").write_text(text)

    width = 60
    print("=" * width)
    print(f"{'category':<25} {'found':>7} {'total':>7} {'%':>8}")
    print("-" * width)
    for cat in sorted(cat_total):
        n, t = cat_correct[cat], cat_total[cat]
        marker = "  <-- no reasoner" if cat not in REASONERS else ""
        print(f"{cat:<25} {n:>7} {t:>7} {n / t:>8.1%}{marker}")
    print("-" * width)
    total = sum(cat_total.values())
    found = sum(cat_correct.values())
    print(f"{'TOTAL':<25} {found:>7} {total:>7} {found / total:>8.1%}")
    print(f"\nSaved {found} CoT traces to {REASONING_DIR}/")


if __name__ == "__main__":
    main()
