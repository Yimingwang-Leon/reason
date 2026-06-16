"""RFT 选样过滤(PLAN §1.2-② C5 选样硬化,防 luck-correct 蒸馏毒)。

输入 rft_harvest.py 的 rollout jsonl(+ .prompts.jsonl 侧车),输出 train-ready jsonl
(tokens=prompt+completion, mask=0/1,可直接喂 src.train_tinker._build_datum)+ 统计表。

闸门(顺序执行,逐级计数):
  1. stop_reason == "stop"(截断一律不收);
  2. C5 硬化判分:精确串等 OR 数值 rel_tol≤1e-4(grader 的 1e-2 只记分不选样);
  3. crypt 生成长 ≤3000 tok;
  4. 复读:word 12-gram 重复率 >0.3 弃(实测 12 条真实 trace ≤0.069,合成复读 0.993);
  5. 毒类:crypt 含"验证失败仍然 box / concat 兜底"标记不收;
     eq 无任何复核证据(re-check/verify/reproduced/match)= 错断言族,不收;
  6. 每题 ≤2 条(短者优先:短 trace 更干净且利 crypt 收口)。

Run:
    python rl/rft_filter.py --in /tmp/rft_dryrun/harvest.jsonl --out /tmp/rft_dryrun/train_rft.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.reasoning import extract_answer  # eval_gate 同源提取

CRYPT_MAX_TOK = 3000
NGRAM_N = 12
NGRAM_THRESH = 0.3
MAX_PER_PROBLEM = 2

# crypt 毒标记:验证失败仍 box / concat 兜底族(run-006/7/8 R 崩的断言毒同族)
CRYPT_POISON_RE = re.compile(
    r"(?i)(verification failed|failed verification|no consistent (?:mapping|cipher|rule)|"
    r"fall(?:s|ing)? ?back|as a last resort|cannot determine|unable to determine|"
    r"give up|concatenat\w* (?:fallback|as the answer|anyway))")
# eq 复核证据标记:缺失 = 断言未验证(老 equation stub 毒族)
EQ_VERIFY_RE = re.compile(r"(?i)(re-?check|verif|reproduced|consistent|match)")


def hardened_correct(truth: str, pred: str) -> bool:
    """C5 选样判分:精确串等 OR rel_tol≤1e-4(严格子集 of grader 1e-2)。"""
    truth, pred = truth.strip(), pred.strip()
    if pred == truth:
        return True
    if pred.lower() == truth.lower() and not re.fullmatch(r"[01]+", truth):
        return True  # 非二进制串允许大小写不敏感(与 metric_correct 同口径)
    try:
        return math.isclose(float(truth), float(pred), rel_tol=1e-4, abs_tol=0.0)
    except Exception:
        return False


def ngram_dup_ratio(text: str, n: int = NGRAM_N) -> float:
    w = text.split()
    if len(w) < n + 1:
        return 0.0
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def is_crypt(cat: str) -> bool:
    return cat.startswith("cryptarithm")


def is_eq(cat: str) -> bool:
    return cat.startswith("equation_numeric")


def poison(cat: str, text: str) -> str | None:
    """返回毒类标签或 None(通过)。"""
    if is_crypt(cat) and CRYPT_POISON_RE.search(text):
        return "crypt_fallback"
    if is_eq(cat) and not EQ_VERIFY_RE.search(text):
        return "eq_unverified_assertion"
    return None


def load_truth(csv_path: Path) -> dict[str, str]:
    import pandas as pd
    df = pd.read_csv(csv_path)
    truth = {str(r["id"]): str(r["answer"]) for _, r in df.iterrows()}
    extra = ROOT / "data" / "train.csv"
    if extra.exists() and extra != csv_path:
        df2 = pd.read_csv(extra)
        for _, r in df2.iterrows():
            truth.setdefault(str(r["id"]), str(r["answer"]))
    return truth


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--in", dest="inp", required=True, help="harvest rollout jsonl")
    ap.add_argument("--out", required=True, help="train-ready jsonl")
    ap.add_argument("--prompts", default=None, help="侧车,默认 <in>.prompts.jsonl")
    ap.add_argument("--csv", default=str(ROOT / "data" / "train_split.csv"))
    ap.add_argument("--max-per-problem", type=int, default=MAX_PER_PROBLEM)
    ap.add_argument("--crypt-max-tok", type=int, default=CRYPT_MAX_TOK)
    ap.add_argument("--ngram-thresh", type=float, default=NGRAM_THRESH)
    args = ap.parse_args()

    truth = load_truth(Path(args.csv))
    prompts_path = Path(args.prompts or (args.inp + ".prompts.jsonl"))
    prompt_tok: dict[str, list[int]] = {}
    if prompts_path.exists():
        for line in open(prompts_path):
            d = json.loads(line)
            prompt_tok[d["id"]] = d["prompt_tokens"]
    else:
        print(f"[warn] no prompt sidecar at {prompts_path}; train-ready 输出将缺 prompt 段")

    S = lambda: defaultdict(int)  # noqa: E731
    stats: dict[str, defaultdict] = {k: S() for k in (
        "n_rollouts", "n_reward1", "drop_stop", "drop_hardened", "drop_crypt_len",
        "drop_ngram", "drop_poison", "drop_cap", "selected")}
    survivors: dict[str, list[dict]] = defaultdict(list)

    with open(args.inp) as f:
        for line in f:
            r = json.loads(line)
            cat = r["cat"]
            stats["n_rollouts"][cat] += 1
            if r.get("reward"):
                stats["n_reward1"][cat] += 1
            if r.get("stop_reason") != "stop":
                stats["drop_stop"][cat] += 1
                continue
            t = truth.get(r["id"])
            if t is None or not hardened_correct(t, extract_answer(r["text"])):
                stats["drop_hardened"][cat] += 1
                continue
            n_tok = r.get("gen_len", len(r.get("tokens", [])))
            if is_crypt(cat) and n_tok > args.crypt_max_tok:
                stats["drop_crypt_len"][cat] += 1
                continue
            if ngram_dup_ratio(r["text"]) > args.ngram_thresh:
                stats["drop_ngram"][cat] += 1
                continue
            if poison(cat, r["text"]):
                stats["drop_poison"][cat] += 1
                continue
            survivors[r["id"]].append(r)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with open(out_path, "w") as f:
        for pid in sorted(survivors):
            rolls = sorted(survivors[pid], key=lambda r: (r.get("gen_len", 0), r.get("sample_idx", 0)))
            keep, spill = rolls[:args.max_per_problem], rolls[args.max_per_problem:]
            for r in spill:
                stats["drop_cap"][r["cat"]] += 1
            for r in keep:
                p_tok = prompt_tok.get(pid, [])
                tokens = list(p_tok) + list(r["tokens"])
                mask = [0] * len(p_tok) + [1] * len(r["tokens"])
                f.write(json.dumps({
                    "problem_id": pid, "category": r["cat"],
                    "sample_idx": r.get("sample_idx"),
                    "tokens": tokens, "mask": mask, "text": r["text"],
                    "n_prompt_tok": len(p_tok), "n_gen_tok": len(r["tokens"]),
                }, ensure_ascii=False) + "\n")
                stats["selected"][r["cat"]] += 1
                n_written += 1

    cats = sorted({c for d in stats.values() for c in d})
    cols = list(stats)
    w = 26
    print("=" * (w + 12 * len(cols)))
    print(f"{'category':<{w}}" + "".join(f"{c:>12}" for c in cols))
    print("-" * (w + 12 * len(cols)))
    for cat in cats:
        print(f"{cat:<{w}}" + "".join(f"{stats[c][cat]:>12}" for c in cols))
    tot = {c: sum(stats[c].values()) for c in cols}
    print("-" * (w + 12 * len(cols)))
    print(f"{'TOTAL':<{w}}" + "".join(f"{tot[c]:>12}" for c in cols))
    n_prob = len(survivors)
    n_prob_sel = sum(1 for pid in survivors if survivors[pid][:args.max_per_problem])
    print(f"\nselected {n_written} traces / {n_prob_sel} problems (had survivors: {n_prob}) -> {out_path}")

    stats_path = Path(str(out_path) + ".stats.json")
    stats_path.write_text(json.dumps(
        {k: dict(v) for k, v in stats.items()} |
        {"n_selected_traces": n_written, "n_problems_covered": n_prob_sel}, indent=2))
    print(f"stats -> {stats_path}")


if __name__ == "__main__":
    main()
