"""路A — 判分口径对齐(全 7 类)。

把官方评测判分语义提取出来,对 data/run012_holdout_decode.jsonl 的 770 行
全部重打,找出"我们判分算对、官方判分算错"的题(= 离线 oracle 虚高的来源)。

官方语义有 3 个并行版本(各队公开 notebook / huikang 双文件),逐字搬运:

  EXTRACT (答案提取):
    A. ours        — src/reasoning.extract_answer:仅 \boxed{},无 boxed 返回 ""
    B. official    — huikang notebook_tinker.py:434-481 / agi087:
                     boxed → "final answer" 文案 → 文中最后一个数 → 最后一行
                     (有 fallback 链,无 boxed 也能取到答案)

  VERIFY (判分):
    V1 ours        — src/reasoning.metric_correct:二进制守卫 ON + rel_tol=1e-2(abs_tol=0)
    V2 nb_noguard  — notebook_tinker.py:484-502 / agi087.verify:
                     无守卫,float rel_tol=1e-2 abs_tol=1e-5,否则 lower() 串等
    V3 rs_guard    — reasoning.py:91-93:二进制守卫 ON + float rel_tol=1e-2 abs_tol=1e-5
    V4 ult_norm    — ultimate-sft-grpo-v3 _answers_match:
                     normalize(int-cast) → lower() 串等 → float rel_tol=1e-2

关键产物:
  - 每类 (我们 ok 数) vs (各官方变体 ok 数) 差异表
  - "我们 ok ∧ 官方某变体 fail" 的逐题清单(= 离线虚高铁证)
  - 反向:"我们 fail ∧ 官方 ok"(= 官方更宽松,我们低估,LB 反而更高)
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DECODE = ROOT / "data/run012_holdout_decode.jsonl"
HOLDOUT = ROOT / "data/holdout.csv"


# ============================================================ 提取器
def extract_ours(text: str | None) -> str:
    """src/reasoning.extract_answer 逐字。仅 boxed,无 boxed -> ''。"""
    if text is None:
        return ""
    matches = re.findall(r"\\boxed\{([^}]*)(?:\}|$)", text)
    if not matches:
        return ""
    non_empty = [m.strip() for m in matches if m.strip()]
    return non_empty[-1] if non_empty else matches[-1].strip()


def extract_official(text: str | None) -> str:
    """huikang notebook_tinker.py:434-481 / agi087 fallback 链逐字。"""
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
        m = re.findall(pattern, text, re.IGNORECASE)
        if m:
            return m[-1].strip()
    m = re.findall(r"-?\d+(?:\.\d+)?", text)
    if m:
        return m[-1]
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else "NOT_FOUND"


# ============================================================ 判分器
def v1_ours(truth: str, pred: str) -> bool:
    """src/reasoning.metric_correct:守卫 ON,abs_tol=0。"""
    truth, pred = truth.strip(), pred.strip()
    if re.fullmatch(r"[01]+", truth):
        return pred.lower() == truth.lower()
    try:
        return math.isclose(float(truth), float(pred), rel_tol=1e-2, abs_tol=0.0)
    except Exception:
        return pred.lower() == truth.lower()


def v2_nb_noguard(truth: str, pred: str) -> bool:
    """notebook_tinker.py:484-502 / agi087.verify:无守卫。"""
    truth, pred = truth.strip(), pred.strip()
    try:
        return math.isclose(float(truth), float(pred), rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        return pred.lower() == truth.lower()


def v3_rs_guard(truth: str, pred: str) -> bool:
    """reasoning.py:91-93:守卫 ON,abs_tol=1e-5。"""
    truth, pred = truth.strip(), pred.strip()
    if re.fullmatch(r"[01]+", truth):
        return pred.lower() == truth.lower()
    try:
        return math.isclose(float(truth), float(pred), rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        return pred.lower() == truth.lower()


def _normalize_answer(s: str) -> str:
    s = s.strip()
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
        return str(f)
    except (ValueError, OverflowError):
        return s


def v4_ult_norm(truth: str, pred: str) -> bool:
    """ultimate-sft-grpo-v3 _answers_match:int-normalize + lower 串等 + rel_tol。"""
    pred_n, gt_n = _normalize_answer(pred), _normalize_answer(truth)
    if pred_n == gt_n:
        return True
    if pred_n.lower() == gt_n.lower():
        return True
    try:
        p, g = float(pred), float(truth)
        if g == 0:
            return abs(p) < 1e-6
        return abs(p - g) / abs(g) <= 1e-2
    except (ValueError, OverflowError):
        return False


VERIFIERS = {"v1_ours": v1_ours, "v2_nb_noguard": v2_nb_noguard,
             "v3_rs_guard": v3_rs_guard, "v4_ult_norm": v4_ult_norm}


# ============================================================ 主流程
def main() -> None:
    rows = [json.loads(l) for l in open(DECODE)]
    assert len(rows) == 770

    # 用 holdout.csv 的 answer 校验 decode 的 truth 字段是否一致
    import csv
    truth_csv = {r["id"]: r["answer"] for r in csv.DictReader(open(HOLDOUT))}
    mismatch = [r["id"] for r in rows
                if r["id"] in truth_csv and str(truth_csv[r["id"]]).strip() != str(r["truth"]).strip()]
    print(f"[完整性] decode.truth vs holdout.answer 不一致: {len(mismatch)} / 770")
    if mismatch[:5]:
        for i in mismatch[:5]:
            d = next(r for r in rows if r["id"] == i)
            print(f"   {i}: decode={d['truth']!r}  holdout={truth_csv[i]!r}")

    # 校验 decode 的 ok 字段确实 = v1_ours(truth, extract_ours(text))
    recompute_mismatch = 0
    for r in rows:
        ok_re = v1_ours(str(r["truth"]), extract_ours(r["text"]))
        if ok_re != r["ok"]:
            recompute_mismatch += 1
    print(f"[完整性] decode.ok vs 重算 v1_ours(extract_ours(text)) 不一致: "
          f"{recompute_mismatch} / 770 (应为 0,证明 ok 用的就是我们带守卫的判分)")

    # ---- 为每行计算: ours_ok(=v1+extract_ours) 与 4 个官方变体(用 official 提取) ----
    for r in rows:
        r["pred_ours"] = extract_ours(r["text"])
        r["pred_off"] = extract_official(r["text"])
        r["ours_ok"] = v1_ours(str(r["truth"]), r["pred_ours"])
        for name, fn in VERIFIERS.items():
            # 官方提取 + 官方判分
            r[f"off_{name}"] = fn(str(r["truth"]), r["pred_off"])

    cats = sorted({r["cat"] for r in rows})

    # ===== 表 1:每类 ours_ok vs 各官方变体 =====
    print("\n" + "=" * 96)
    print("表1 — 每类 ok 数:我们(v1+box-only)  vs  官方提取(fallback链)×4 判分变体")
    print("-" * 96)
    hdr = f"{'cat':<26}{'ours':>6}{'v1_ours':>9}{'v2_noguard':>11}{'v3_guard':>10}{'v4_ult':>8}{'n':>5}"
    print(hdr)
    print("-" * 96)
    agg = {k: Counter() for k in ["ours"] + list(VERIFIERS)}
    nT = Counter()
    for c in cats:
        sub = [r for r in rows if r["cat"] == c]
        nT[c] = len(sub)
        ours = sum(r["ours_ok"] for r in sub)
        v = {k: sum(r[f"off_{k}"] for r in sub) for k in VERIFIERS}
        agg["ours"][c] = ours
        for k in VERIFIERS:
            agg[k][c] = v[k]
        print(f"{c:<26}{ours:>6}{v['v1_ours']:>9}{v['v2_nb_noguard']:>11}"
              f"{v['v3_rs_guard']:>10}{v['v4_ult_norm']:>8}{len(sub):>5}")
    print("-" * 96)
    tot_ours = sum(agg["ours"].values())
    tots = {k: sum(agg[k].values()) for k in VERIFIERS}
    print(f"{'TOTAL':<26}{tot_ours:>6}{tots['v1_ours']:>9}{tots['v2_nb_noguard']:>11}"
          f"{tots['v3_rs_guard']:>10}{tots['v4_ult_norm']:>8}{sum(nT.values()):>5}")
    print(f"\n770 题面占比:ours={tot_ours/770:.4f}  v2_noguard={tots['v2_nb_noguard']/770:.4f}  "
          f"v4_ult={tots['v4_ult_norm']/770:.4f}")

    # ===== 表 2:离线虚高 = ours_ok ∧ 官方某变体 fail (逐类逐变体计数) =====
    print("\n" + "=" * 96)
    print("表2 — 离线虚高(我们判对 ∧ 官方判错)逐类计数")
    print("   (>0 = 该类离线 oracle 虚高;以最严官方变体为准)")
    print("-" * 96)
    print(f"{'cat':<26}{'vs v2_noguard':>14}{'vs v3_guard':>13}{'vs v4_ult':>11}")
    print("-" * 96)
    inflate_rows = defaultdict(list)
    for c in cats:
        sub = [r for r in rows if r["cat"] == c]
        cnt = {}
        for k in ["v2_nb_noguard", "v3_rs_guard", "v4_ult_norm"]:
            bad = [r for r in sub if r["ours_ok"] and not r[f"off_{k}"]]
            cnt[k] = len(bad)
            for r in bad:
                inflate_rows[(c, k)].append(r)
        print(f"{c:<26}{cnt['v2_nb_noguard']:>14}{cnt['v3_rs_guard']:>13}{cnt['v4_ult_norm']:>11}")
    print("-" * 96)

    # ===== 表 3:反向 — 官方更宽松(我们判错 ∧ 官方判对)=====
    print("\n" + "=" * 96)
    print("表3 — 官方更宽(我们判错 ∧ 官方判对)逐类计数  (>0 = 我们低估,LB 反而更高)")
    print("-" * 96)
    print(f"{'cat':<26}{'vs v2_noguard':>14}{'vs v3_guard':>13}{'vs v4_ult':>11}")
    print("-" * 96)
    lenient_rows = defaultdict(list)
    for c in cats:
        sub = [r for r in rows if r["cat"] == c]
        cnt = {}
        for k in ["v2_nb_noguard", "v3_rs_guard", "v4_ult_norm"]:
            good = [r for r in sub if (not r["ours_ok"]) and r[f"off_{k}"]]
            cnt[k] = len(good)
            for r in good:
                lenient_rows[(c, k)].append(r)
        print(f"{c:<26}{cnt['v2_nb_noguard']:>14}{cnt['v3_rs_guard']:>13}{cnt['v4_ult_norm']:>11}")
    print("-" * 96)

    # ===== 逐题明细:所有"离线虚高"行(任一官方变体判错)=====
    print("\n" + "=" * 96)
    print("明细 — 离线虚高逐题(我们判对,但 v2_noguard 或 v4_ult 判错)")
    print("   注:健康类/eq 在此出现 = 离线 oracle 虚高的直接铁证")
    print("-" * 96)
    any_inflate = []
    for r in rows:
        if r["ours_ok"] and (not r["off_v2_nb_noguard"] or not r["off_v4_ult_norm"]):
            any_inflate.append(r)
    if not any_inflate:
        print("(无 — 770 行里没有一题是'我们判对但官方判错';四健康类/eq 无离线虚高)")
    else:
        print(f"{'id':<10}{'cat':<24}{'v2':>4}{'v4':>4}  truth -> pred_off / pred_ours")
        for r in any_inflate:
            print(f"{r['id']:<10}{r['cat']:<24}"
                  f"{str(r['off_v2_nb_noguard'])[0]:>4}{str(r['off_v4_ult_norm'])[0]:>4}  "
                  f"{r['truth'][:18]!r} -> {r['pred_off'][:18]!r} / {r['pred_ours'][:18]!r}")
    print(f"\n离线虚高总行数(任一官方变体判错): {len(any_inflate)} / 770")

    # ===== crypt 反向救活:我们 0/110,官方 fallback 链可能救活几题 =====
    crypt = [r for r in rows if r["cat"] == "cryptarithm_deduce"]
    crypt_saved = {k: sum((not r["ours_ok"]) and r[f"off_{k}"] for r in crypt) for k in VERIFIERS}
    print(f"\ncrypt(我们 0/110)官方变体救活:{crypt_saved}")


if __name__ == "__main__":
    main()
