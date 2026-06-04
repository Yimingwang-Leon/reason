"""P5 corpus validation gate + honest LB projection.

Runs the checks the spend-GATE requires before any paid training:
  1. mask/token invariants on every corpus entry
  2. PRE-rewrite truth gate: every saved reasoning/<id>.txt extracts to the truth
  3. count invariants: equation gain vs frozen baseline; augmenter rows accounted for
  4. honest LB projection: oracle = real_solved/9500 (augmenters EXCLUDED from the
     denominator); project LB at R=0.912 (pessimistic, gates go/no-go) and 0.978.

Exit non-zero on any hard-gate failure.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from src.problems import load_problems
from src.reasoning import extract_answer, metric_correct

ROOT = Path(__file__).resolve().parent.parent
CORPUS_INDEX = ROOT / "corpus.jsonl"
CORPUS_DIR = ROOT / "corpus"
REASONING_DIR = ROOT / "reasoning"
BASELINE = ROOT / "baseline.json"

R_PESSIMISTIC = 0.912   # gates go/no-go
R_OPTIMISTIC = 0.978    # reference only
TOTAL_PROBLEMS = 9500
TARGET_LB = 0.86
GEN_LIMIT = 7680        # grader generation cap; completion (sum(mask)) must fit


def main() -> int:
    fails: list[str] = []
    rows = [json.loads(l) for l in open(CORPUS_INDEX)]
    real = [r for r in rows if not r.get("augmenter")]
    aug = [r for r in rows if r.get("augmenter")]

    # --- 1. mask/token invariants (incl. completion <= GEN_LIMIT so the box is
    #        always reachable within the grader's generation budget) ---
    bad_inv = bad_gen = 0
    for r in rows:
        seg = json.loads((CORPUS_DIR / r["problem_id"] / "synthetic.jsonl").read_text())
        t, m = seg["tokens"], seg["mask"]
        if not (len(t) == len(m) and set(m) <= {0, 1} and sum(m) > 0 and len(t) <= 8192):
            bad_inv += 1
        if sum(m) > GEN_LIMIT:
            bad_gen += 1
    print(f"[1] mask/token invariants: {len(rows)} entries, {bad_inv} violations, "
          f"{bad_gen} completions>{GEN_LIMIT} -> {'PASS' if bad_inv == 0 and bad_gen == 0 else 'FAIL'}")
    if bad_inv or bad_gen:
        fails.append("mask/token/gen-length invariants")

    # --- 2. PRE-rewrite truth gate (raw reasoning/*.txt, not the rewritten corpus) ---
    problems = {p.id: p for p in load_problems()}
    bad_truth = 0
    for r in real:
        p = problems[r["problem_id"]]
        cot = (REASONING_DIR / f"{p.id}.txt").read_text()
        if not metric_correct(p.answer, extract_answer(cot)):
            bad_truth += 1
    print(f"[2] pre-rewrite truth gate: {len(real)} traces, {bad_truth} fail metric_correct "
          f"-> {'PASS' if bad_truth == 0 else 'FAIL'}")
    if bad_truth:
        fails.append("pre-rewrite truth gate")

    # --- 3. count invariants vs frozen baseline ---
    base = json.loads(BASELINE.read_text())
    per_cat = defaultdict(int)
    for r in real:
        per_cat[r["category"]] += 1
    eq_now = per_cat["equation_numeric_deduce"]
    eq_base = base["categories"]["equation_numeric_deduce"]["n_correct"]
    print(f"[3] counts: real_traces={len(real)} augmenter_rows={len(aug)} "
          f"(excluded from oracle)")
    print(f"    equation: baseline {eq_base} -> corpus {eq_now} (+{eq_now - eq_base})")
    if eq_now < eq_base:
        fails.append("equation regressed vs baseline")

    # --- 4. honest LB projection ---
    oracle = len(real) / TOTAL_PROBLEMS
    lb_pess = oracle * R_PESSIMISTIC
    lb_opt = oracle * R_OPTIMISTIC
    print(f"[4] oracle = {len(real)}/{TOTAL_PROBLEMS} = {oracle:.4f}  "
          f"(baseline oracle {base['total']['pct']:.4f})")
    print(f"    LB @ R={R_PESSIMISTIC} (pessimistic, GATES): {lb_pess:.4f}  "
          f"(gap to {TARGET_LB}: {TARGET_LB - lb_pess:+.4f})")
    print(f"    LB @ R={R_OPTIMISTIC} (optimistic, ref):    {lb_opt:.4f}  "
          f"(gap to {TARGET_LB}: {TARGET_LB - lb_opt:+.4f})")
    print(f"    NOTE: R is UNVALIDATED offline; the {R_PESSIMISTIC} figure is the prior "
          f"single-point estimate. Real R needs the R-harness on a checkpoint.")

    print()
    if fails:
        print(f"P5 GATE: FAIL -> {fails}")
        return 1
    print("P5 GATE: PASS (corpus valid; projection is oracle-based, R unvalidated)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
