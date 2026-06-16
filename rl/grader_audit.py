"""C1 官方判分语义审计(rl/PLAN.md §4 条目2)。

官方语义 = huikang notebook 的提取 fallback 链 + 容差判分 + 二进制守卫:
  - 提取:external/nemotron-huikang/notebook_tinker.py:434-481 `extract_final_answer`
    (boxed → "final answer" 文案 → 文中最后一个数 → 最后一行)。
  - 判分:notebook_tinker.py:484-502 `verify`(float rel_tol=1e-2, abs_tol=1e-5,
    否则大小写不敏感串等)。
  - 二进制守卫 **钉死为 ON**:notebook 版 verify 本体没有守卫,但 huikang
    reasoning.py:91-93 `compare_answer` 有 `re.fullmatch(r"[01]+", stored)` 严格串等,
    且其 docstring 给出 `verify("10011000","10011001") -> False`、
    `verify("11011","00011011") -> False` —— 无守卫时 float 路径会把这两例判 True,
    docstring 断言 False 即官方 grader 含守卫的直接证据。脚本同时输出守卫 OFF 的
    敏感性计数作风险量化。

输出:
  - stdout:770 行重打的逐行差异(official_ok vs 原 ok)+ G0 闸数字
  - data/prey_holdout.json      三层猎物名单(官方语义修正后)
  - data/prey_train_candidates.json  train_split 侧候选池(--train 才跑 crypt solver,
    ~20min/8 进程;R-loss 子集一律标注"待 M0 直测")

用法:
  python -m rl.grader_audit            # 重打 770 行 + prey_holdout.json
  python -m rl.grader_audit --train    # 追加 train_split 候选池(慢)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DECODE = ROOT / "data/run012_holdout_decode.jsonl"
HOLDOUT = ROOT / "data/holdout.csv"
ORACLE_FAILS = ROOT / "data/probes/oracle_fast_fails.json"      # bit 23 + eq 29
CRYPT_FAILS = ROOT / "data/probes/crypt_holdout_fails.json"     # crypt 139
PREY_HOLDOUT = ROOT / "data/prey_holdout.json"
PREY_TRAIN = ROOT / "data/prey_train_candidates.json"


# ---------------------------------------------------------------- 官方语义
def extract_final_answer(text: str | None) -> str:
    """逐字搬运 notebook_tinker.py:434-481(官方提取 fallback 链)。"""
    if text is None:
        return "NOT_FOUND"
    matches = re.findall(r"\\boxed\{([^}]*)(?:\}|$)", text)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        if non_empty:
            return non_empty[-1]
        return matches[-1].strip()
    patterns = [
        r"The final answer is:\s*([^\n]+)",
        r"Final answer is:\s*([^\n]+)",
        r"Final answer\s*[:：]\s*([^\n]+)",
        r"final answer\s*[:：]\s*([^\n]+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].strip()
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    if matches:
        return matches[-1]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else "NOT_FOUND"


def verify_official(stored_answer: str, predicted: str, binary_guard: bool = True) -> bool:
    """notebook_tinker.py:484-502 verify + 钉死的二进制守卫(reasoning.py:91-93)。"""
    stored_answer = stored_answer.strip()
    predicted = predicted.strip()
    if binary_guard and re.fullmatch(r"[01]+", stored_answer):
        return predicted.lower() == stored_answer.lower()
    try:
        stored_num = float(stored_answer)
        predicted_num = float(predicted)
        return math.isclose(stored_num, predicted_num, rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        return predicted.lower() == stored_answer.lower()


# ---------------------------------------------------------------- 重打 770 行
def regrade() -> tuple[list[dict], list[dict]]:
    rows = [json.loads(l) for l in open(DECODE)]
    assert len(rows) == 770, f"decode 行数 {len(rows)} != 770"
    diffs = []
    for r in rows:
        pred_off = extract_final_answer(r["text"])
        r["official_pred"] = pred_off
        r["official_ok"] = verify_official(r["truth"], pred_off, binary_guard=True)
        r["official_ok_noguard"] = verify_official(r["truth"], pred_off, binary_guard=False)
        if r["official_ok"] != r["ok"]:
            diffs.append(r)
    return rows, diffs


def report(rows: list[dict], diffs: list[dict]) -> dict:
    oracle_fails = json.load(open(ORACLE_FAILS))
    bit_dead = set(oracle_fails["bit"])
    eq_dead = set(x[0] for x in oracle_fails["equation_numeric_deduce"])
    crypt_dead = set(x[0] for x in json.load(open(CRYPT_FAILS)))

    print("=" * 78)
    print("逐行差异(official_ok != 原 ok):")
    print(f"{'id':<10} {'cat':<24} {'原ok':<5} {'官方':<5} {'cap':<4} 原pred -> 官方pred (truth)")
    print("-" * 78)
    for r in diffs:
        print(f"{r['id']:<10} {r['cat']:<24} {str(r['ok']):<5} {str(r['official_ok']):<5} "
              f"{str(r['cap_hit']):<4} {r['pred'][:18]!r} -> {r['official_pred'][:18]!r} "
              f"({r['truth'][:18]!r})")
    if not diffs:
        print("(无差异)")
    print("-" * 78)
    print(f"差异行数: {len(diffs)} / 770")

    per_cat = defaultdict(Counter)
    for r in rows:
        per_cat[r["cat"]]["ours"] += r["ok"]
        per_cat[r["cat"]]["official"] += r["official_ok"]
        per_cat[r["cat"]]["official_noguard"] += r["official_ok_noguard"]
        per_cat[r["cat"]]["n"] += 1
    print(f"\n{'cat':<24} {'我们':>5} {'官方(守卫ON)':>12} {'官方(守卫OFF)':>13}   n")
    for c in sorted(per_cat):
        d = per_cat[c]
        print(f"{c:<24} {d['ours']:>5} {d['official']:>12} {d['official_noguard']:>13} {d['n']:>5}")
    tot = Counter()
    for d in per_cat.values():
        tot.update(d)
    print(f"{'TOTAL':<24} {tot['ours']:>5} {tot['official']:>12} {tot['official_noguard']:>13} {tot['n']:>5}")

    # ---- G0 闸:bit 猎物存活 ----
    bit = [r for r in rows if r["cat"] == "bit_manipulation"]
    bit_err_ours = [r for r in bit if not r["ok"]]
    bit_err_off = [r for r in bit if not r["official_ok"]]
    bit_rloss_ours = [r for r in bit_err_ours if r["id"] not in bit_dead]
    bit_rloss_off = [r for r in bit_err_off if r["id"] not in bit_dead]
    guard_flips = [r for r in bit_err_off if r["official_ok_noguard"]]
    print("\n---- G0:bit 猎物 ----")
    print(f"bit 错题(我们 19 口径): {len(bit_err_ours)} -> 官方重打仍错: {len(bit_err_off)}"
          f"  存活率 {len(bit_err_off)/max(len(bit_err_ours),1):.0%}")
    print(f"bit R-loss(错 ∧ oracle-ok): 我们 {len(bit_rloss_ours)} -> 官方 {len(bit_rloss_off)}")
    n_bit_holdout = 320
    extrap = len(bit_rloss_off) / len(bit) * n_bit_holdout
    print(f"外推 320 题分母: {len(bit_rloss_off)}/110 × 320 ≈ {extrap:.0f}(PLAN 旧口径 32)")
    print(f"守卫敏感性: 若官方 grader 无二进制守卫,{len(guard_flips)}/{len(bit_err_off)} "
          f"道官方错题会被 float 容差判对(猎物即蒸发)")

    # ---- eq ----
    eq = [r for r in rows if r["cat"] == "equation_numeric_deduce"]
    eq_err_off = [r for r in eq if not r["official_ok"]]
    eq_rloss_off = [r for r in eq_err_off if r["id"] not in eq_dead]
    print("\n---- eq ----")
    print(f"eq 错题: 我们 {sum(not r['ok'] for r in eq)} -> 官方 {len(eq_err_off)};"
          f"其中 R-loss(oracle-ok)= {len(eq_rloss_off)}")

    # ---- crypt ----
    crypt = [r for r in rows if r["cat"] == "cryptarithm_deduce"]
    crypt_off_ok = [r for r in crypt if r["official_ok"]]
    print("\n---- crypt ----")
    print(f"crypt 110 行官方重打判对: {len(crypt_off_ok)}(我们口径 0)"
          + (f" -> {[r['id'] for r in crypt_off_ok]}" if crypt_off_ok else ""))

    return {
        "bit_dead": bit_dead, "eq_dead": eq_dead, "crypt_dead": crypt_dead,
        "bit_rloss_off": bit_rloss_off, "eq_rloss_off": eq_rloss_off,
        "rows": rows,
    }


# ---------------------------------------------------------------- 猎物名单
def build_prey_holdout(ctx: dict) -> None:
    holdout = {r["id"]: r for r in csv.DictReader(open(HOLDOUT))}
    rows_by_id = {r["id"]: r for r in ctx["rows"]}

    def item(pid: str, cat: str) -> dict:
        return {"id": pid, "cat": cat, "truth": holdout[pid]["answer"]}

    bit_prey = [item(r["id"], r["cat"]) for r in ctx["bit_rloss_off"]]
    eq_prey = [item(r["id"], r["cat"]) for r in ctx["eq_rloss_off"]]

    crypt_all = [pid for pid, r in holdout.items()
                 if r["cat"] == "cryptarithm_deduce" and pid not in ctx["crypt_dead"]]
    crypt_prey = []
    for pid in sorted(crypt_all):
        dec = rows_by_id.get(pid)
        if dec is not None and dec["official_ok"]:
            continue  # 官方重打已对 -> 不再是猎物
        it = item(pid, "cryptarithm_deduce")
        it["decoded"] = dec is not None  # False = 未解码(110/165 抽样外),贪心表现未测
        crypt_prey.append(it)

    out = {
        "meta": {
            "source": "rl/grader_audit.py (C1 官方判分语义重打 run012_holdout_decode 770 行)",
            "semantics": "notebook_tinker.py:434-503 fallback 链 + 二进制守卫 ON (reasoning.py:91)",
            "counts": {"bit_rloss": len(bit_prey), "crypt_solver_ok": len(crypt_prey),
                       "eq_rloss": len(eq_prey)},
            "note": "bit/eq 来自已解码 110/类 抽样(官方语义判错 ∧ oracle-ok);"
                    "crypt = holdout 全部 solver-ok(26)剔官方已对,decoded=False 的 11 题贪心未测",
        },
        "bit_rloss": bit_prey,
        "crypt_solver_ok": crypt_prey,
        "eq_rloss": eq_prey,
    }
    PREY_HOLDOUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nwrote {PREY_HOLDOUT} (bit {len(bit_prey)} / crypt {len(crypt_prey)} / eq {len(eq_prey)})")


# ---------------------------------------------------------------- train_split 候选池
def _solve_one(p):  # 顶层函数,可 pickle
    from src.reasoning import run_one
    _text, ok = run_one(p)
    return p.id, ok


def build_prey_train(jobs: int) -> None:
    import multiprocessing as mp

    from src.problems import load_problems

    probs = load_problems(ROOT / "data/train_split.csv")
    by_cat = defaultdict(list)
    for p in probs:
        by_cat[p.category].append(p)

    pools: dict[str, list] = {}
    # bit / eq:秒级,串行
    for cat in ("bit_manipulation", "equation_numeric_deduce"):
        from src.reasoning import run_one
        ok_ids = [p.id for p in by_cat[cat] if run_one(p)[1]]
        pools[cat] = ok_ids
        print(f"train_split {cat}: oracle-ok {len(ok_ids)}/{len(by_cat[cat])}")

    # crypt:~14s/题,多进程
    crypt = by_cat["cryptarithm_deduce"]
    with mp.Pool(jobs) as pool:
        res = pool.map(_solve_one, crypt)
    crypt_ok = [pid for pid, ok in res if ok]
    print(f"train_split cryptarithm_deduce: solver-ok {len(crypt_ok)}/{len(crypt)}")

    truth = {p.id: p.answer for p in probs}
    cat_of = {p.id: p.category for p in probs}
    crypt_keep = [i for i in crypt_ok if "}" not in truth[i]]
    print(f"crypt 剔 '}}'-truth: {len(crypt_ok)} -> {len(crypt_keep)}")

    def items(ids):
        return [{"id": i, "cat": cat_of[i], "truth": truth[i]} for i in sorted(ids)]

    out = {
        "meta": {
            "source": "rl/grader_audit.py --train(repo production solver 直跑 train_split)",
            "status": "候选母池 = oracle-ok;R-loss 子集(贪心错)未测,待 M0 直测",
            "口径": "rl/recon_data.md §3.2:bit R-loss 预期 100-150;crypt solver-ok 剔 }truth;"
                  "eq 签名类默认不入池(M0 ≥25% 覆盖才许进)",
        },
        "bit_rloss": {
            "status": "待 M0 直测(母池 = oracle-ok,R-loss 为其贪心错子集)",
            "expected_rloss": "100-150",
            "n_pool": len(pools["bit_manipulation"]),
            "items": items(pools["bit_manipulation"]),
        },
        "crypt_solver_ok": {
            "status": "待 M0 直测(已训题贪心≈1/30,营救 pass@8≈17%)",
            "n_pool": len(crypt_keep),
            "items": items(crypt_keep),
        },
        "eq_rloss": {
            "status": "待 M0 直测(已训题贪心微探针 30/30,R-loss 预期少)",
            "n_pool": len(pools["equation_numeric_deduce"]),
            "items": items(pools["equation_numeric_deduce"]),
        },
    }
    PREY_TRAIN.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote {PREY_TRAIN}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="追加 train_split 候选池(慢)")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()

    rows, diffs = regrade()
    ctx = report(rows, diffs)
    build_prey_holdout(ctx)
    if args.train:
        build_prey_train(args.jobs)


if __name__ == "__main__":
    main()
