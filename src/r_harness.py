"""Offline realization (R) harness.

R = realization factor = (trained model's greedy accuracy) / (offline reasoner's
oracle accuracy) on the SAME holdout. R<1 is the gap between "the reasoner can
solve it" and "the trained model reproduces the solution at greedy decode" — the
quantity that, together with oracle coverage, determines LB.

Two modes:
  --oracle-only : free, offline. Computes the oracle accuracy denominator per
                  category on data/holdout.csv (validates the harness end-to-end).
  --adapter PATH: greedy-decode the holdout through the Tinker adapter and compute
                  model accuracy + R per category. Costs Tinker sampling $ — DEFERRED
                  to the first paid checkpoint; requires explicit invocation.

Usage:
    python -m src.r_harness --oracle-only
    python -m src.r_harness --adapter tinker://...        # paid; run on a checkpoint
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.problems import parse_prompt
from src.reasoners.store_types import Problem
from src.reasoning import extract_answer, metric_correct, run_one

ROOT = Path(__file__).resolve().parent.parent
HOLDOUT = ROOT / "data" / "holdout.csv"


def load_holdout() -> list[Problem]:
    df = pd.read_csv(HOLDOUT)
    probs: list[Problem] = []
    for _, r in df.iterrows():
        cat = r["cat"]
        examples, question = parse_prompt(cat, r["prompt"])
        probs.append(Problem(id=r["id"], category=cat, examples=examples,
                             question=question, answer=str(r["answer"])))
    return probs


def oracle_accuracy(probs: list[Problem]) -> dict:
    """Per-category offline-reasoner accuracy (the R denominator)."""
    correct: dict[str, int] = defaultdict(int)
    total: dict[str, int] = defaultdict(int)
    for p in probs:
        total[p.category] += 1
        _t, ok = run_one(p)
        if ok:
            correct[p.category] += 1
    return {c: {"correct": correct[c], "total": total[c],
                "acc": correct[c] / total[c] if total[c] else 0.0}
            for c in sorted(total)}


def score_model_outputs(probs: list[Problem], outputs: dict[str, str]) -> dict:
    """Given {problem_id: model_completion}, per-category model accuracy."""
    correct: dict[str, int] = defaultdict(int)
    total: dict[str, int] = defaultdict(int)
    for p in probs:
        total[p.category] += 1
        out = outputs.get(p.id, "")
        if metric_correct(p.answer, extract_answer(out)):
            correct[p.category] += 1
    return {c: {"correct": correct[c], "total": total[c],
                "acc": correct[c] / total[c] if total[c] else 0.0}
            for c in sorted(total)}


def compute_R(oracle: dict, model: dict) -> dict:
    out = {}
    for c in oracle:
        o, m = oracle[c]["acc"], model.get(c, {}).get("acc", 0.0)
        out[c] = {"oracle": round(o, 4), "model": round(m, 4),
                  "R": round(m / o, 4) if o > 0 else None}
    return out


def sample_adapter(probs: list[Problem], adapter_path: str) -> dict[str, str]:
    """Greedy-decode each holdout prompt through the Tinker adapter. Costs sampling $;
    lazy tinker import keeps the rest of the module free of any Tinker call.

    Prompts are tokenized with OUR training format (corpus.tokenize_prompt: chat
    template + boxed suffix + enable_thinking) so this measures the SFT realization
    R (model reproduces the reasoner CoT it was trained on). max_tokens=7680 matches
    the grader's generation cap exactly."""
    import tinker
    from transformers import AutoTokenizer
    from src.train_tinker import _load_env
    from src.corpus import tokenize_prompt, BASE_MODEL
    _load_env()
    chat_tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    sc = tinker.ServiceClient()
    sampler = sc.create_sampling_client(base_model=BASE_MODEL, model_path=adapter_path)
    sp = tinker.SamplingParams(max_tokens=7680, temperature=0.0, top_p=1.0)
    df = pd.read_csv(HOLDOUT).set_index("id")
    outputs: dict[str, str] = {}
    for i, p in enumerate(probs):
        ids = tokenize_prompt(str(df.loc[p.id, "prompt"]), chat_tok)
        mi = tinker.ModelInput.from_ints(ids)
        resp = sampler.sample(prompt=mi, num_samples=1, sampling_params=sp).result()
        outputs[p.id] = chat_tok.decode(resp.sequences[0].tokens)
        if (i + 1) % 20 == 0:
            print(f"  sampled {i + 1}/{len(probs)}")
    return outputs


def stratified_subset(probs: list[Problem], per_cat: int) -> list[Problem]:
    from collections import defaultdict
    by = defaultdict(list)
    for p in probs:
        by[p.category].append(p)
    return [p for c in sorted(by) for p in by[c][:per_cat]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle-only", action="store_true",
                    help="free: compute oracle accuracy denominator only")
    ap.add_argument("--adapter", default=None,
                    help="tinker:// path; greedy-decode holdout (COSTS $)")
    ap.add_argument("--per-cat", type=int, default=None,
                    help="stratified subset: this many problems per category (cheap diagnostic)")
    ap.add_argument("--out", default=str(ROOT / "training" / "r_report.json"))
    args = ap.parse_args()

    probs = load_holdout()
    if args.per_cat:
        probs = stratified_subset(probs, args.per_cat)
        print(f"(diagnostic subset: {args.per_cat}/category = {len(probs)} problems)")
    oracle = oracle_accuracy(probs)
    print(f"Holdout: {len(probs)} problems")
    print(f"{'category':<26} {'oracle_acc':>10}")
    for c, d in oracle.items():
        print(f"{c:<26} {d['acc']:>10.1%}  ({d['correct']}/{d['total']})")
    tot_c = sum(d["correct"] for d in oracle.values())
    tot_n = sum(d["total"] for d in oracle.values())
    print(f"{'OVERALL oracle':<26} {tot_c / tot_n:>10.1%}")

    if args.adapter:
        print(f"\nSampling adapter (COSTS $): {args.adapter}")
        outputs = sample_adapter(probs, args.adapter)
        model = score_model_outputs(probs, outputs)
        R = compute_R(oracle, model)
        Path(args.out).write_text(json.dumps({"oracle": oracle, "model": model,
                                              "R": R}, indent=2))
        print(f"\n{'category':<26} {'oracle':>8} {'model':>8} {'R':>8}")
        for c, d in R.items():
            print(f"{c:<26} {d['oracle']:>8.3f} {d['model']:>8.3f} {str(d['R']):>8}")
        print(f"\nR report -> {args.out}")
    else:
        print("\n(model R not computed: pass --adapter <tinker path> on a paid "
              "checkpoint. Oracle denominator above is the offline baseline.)")


if __name__ == "__main__":
    main()
