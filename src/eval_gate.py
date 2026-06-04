"""Canonical accuracy gate for the reasoning corpus.

Single source of truth for "how good are the reasoners right now". Every phase
acceptance check imports from here so nobody hand-rolls a comparison that
mis-grades floats (metric_correct uses rel_tol=1e-2) or 8-bit binary strings.

Emits, per category: n_correct, n_total, pct, and the *set of solved problem ids*.
Regression checks are SET-based (baseline solved-ids must stay a subset of the
new solved-ids) because an aggregate count can rise while individual ids silently
swap out — exactly what reversal-rule shadowing would do.

CLI:
    python -m src.eval_gate            # run + print table
    python -m src.eval_gate --freeze   # run + write baseline.json
    python -m src.eval_gate --check    # run + diff against baseline.json (regression gate)
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from src.problems import load_problems
from src.reasoning import run_one

ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = ROOT / "baseline.json"

# Categories whose answers are exact (no float tolerance): a corpus.py box-rewrite
# for any of these means the reasoner produced a genuinely wrong CoT.
EXACT_CATEGORIES = {"cipher", "equation_numeric_deduce", "bit_manipulation",
                    "numeral", "cryptarithm_deduce"}


def evaluate() -> dict:
    """Run every reasoner over all problems; score with the canonical metric.

    Returns {"categories": {cat: {n_correct, n_total, pct, solved_ids:[...]}},
             "total": {n_correct, n_total, pct}}.
    """
    problems = load_problems()
    solved: dict[str, list[str]] = defaultdict(list)
    total: dict[str, int] = defaultdict(int)

    for p in problems:
        total[p.category] += 1
        _text, ok = run_one(p)
        if ok:
            solved[p.category].append(p.id)

    categories = {}
    for cat in sorted(total):
        ids = sorted(solved[cat])
        n, t = len(ids), total[cat]
        categories[cat] = {
            "n_correct": n,
            "n_total": t,
            "pct": n / t if t else 0.0,
            "solved_ids": ids,
        }
    n_correct = sum(c["n_correct"] for c in categories.values())
    n_total = sum(c["n_total"] for c in categories.values())
    return {
        "categories": categories,
        "total": {"n_correct": n_correct, "n_total": n_total,
                  "pct": n_correct / n_total if n_total else 0.0},
    }


def print_table(result: dict) -> None:
    w = 64
    print("=" * w)
    print(f"{'category':<26} {'found':>7} {'total':>7} {'pct':>8}")
    print("-" * w)
    for cat, c in result["categories"].items():
        print(f"{cat:<26} {c['n_correct']:>7} {c['n_total']:>7} {c['pct']:>8.1%}")
    print("-" * w)
    t = result["total"]
    print(f"{'TOTAL (oracle)':<26} {t['n_correct']:>7} {t['n_total']:>7} {t['pct']:>8.1%}")
    print("=" * w)


def freeze_baseline(result: dict, path: Path = BASELINE_PATH) -> None:
    path.write_text(json.dumps(result, indent=2))
    print(f"Froze baseline -> {path} (oracle={result['total']['pct']:.4f})")


def compare_to_baseline(result: dict, path: Path = BASELINE_PATH) -> dict:
    """Set-based regression diff. Returns a report; prints PASS/FAIL per category."""
    if not path.exists():
        raise FileNotFoundError(f"No baseline at {path}; run --freeze first.")
    base = json.loads(path.read_text())
    report = {"regressions": {}, "gains": {}, "ok": True}
    w = 64
    print("=" * w)
    print(f"{'category':<22} {'base':>6} {'new':>6} {'lost':>6} {'gained':>7}  {'status'}")
    print("-" * w)
    all_cats = sorted(set(base["categories"]) | set(result["categories"]))
    for cat in all_cats:
        b_ids = set(base["categories"].get(cat, {}).get("solved_ids", []))
        n_ids = set(result["categories"].get(cat, {}).get("solved_ids", []))
        lost = b_ids - n_ids       # regressions: were solved, now not
        gained = n_ids - b_ids
        status = "OK"
        if lost:
            status = f"REGRESSION ({len(lost)})"
            report["ok"] = False
            report["regressions"][cat] = sorted(lost)
        if gained:
            report["gains"][cat] = sorted(gained)
        print(f"{cat:<22} {len(b_ids):>6} {len(n_ids):>6} {len(lost):>6} "
              f"{len(gained):>7}  {status}")
    print("-" * w)
    bt = base["total"]["pct"]
    nt = result["total"]["pct"]
    print(f"oracle: {bt:.4f} -> {nt:.4f}  ({'+' if nt >= bt else ''}{(nt - bt) * 100:.2f}pp)")
    print(f"OVERALL: {'PASS (no regressions)' if report['ok'] else 'FAIL (regressions present)'}")
    print("=" * w)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true", help="write baseline.json")
    ap.add_argument("--check", action="store_true", help="diff against baseline.json")
    args = ap.parse_args()

    result = evaluate()
    print_table(result)

    if args.freeze:
        freeze_baseline(result)
    if args.check:
        compare_to_baseline(result)


if __name__ == "__main__":
    main()
