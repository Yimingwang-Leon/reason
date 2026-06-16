#!/usr/bin/env python3
"""bit 规则先验反学习 + 平局裁决实测。

发现:出题器不止用单层规则(atom / op(atom,atom) / MAJ / MUX),还会嵌套
2-3 个操作(例:sr3 ORN (rl1 XOR sl3))。本脚本枚举扩展规则族:

  T1  unary  : depth<=2 复合原子(24 基原子 + 两两复合,函数表 dedupe)
  T2  pair   : 8 个布尔二元 op × depth<=2 原子对
  T3  nested : op2( op1(a,b), c ) 与 op2( c, op1(a,b) ),a/b/c 为基原子
  T4  maj/mux: MAJ(a,b,c) / MUX(s;b,c) 及其取反,a/b/c/s 为基原子

在 train(带真值)上:
  - 覆盖率 = 真值落在一致候选集合的 query 预测里;
  - canonical 规则 = 最简的"既一致又命中真值"候选 → 出题器规则分布(先验);
  - 平局裁决 = 按 (签名计数, 粗签名计数, -复杂度) 排序候选。
在 holdout 110 道 bit 上实测裁决 oracle,与模型基线 91/110 对账。
"""

from __future__ import annotations

import collections
import json
import re
import sys

import numpy as np
import pandas as pd

ROOT = str(__import__("pathlib").Path(__file__).resolve().parent.parent)
N = 8
MASK = 0xFF

# ---------------------------------------------------------------- atoms ----

def _rl(x, k):
    return ((x << k) | (x >> (N - k))) & MASK

def _sl(x, k):
    return (x << k) & MASK

def _sr(x, k):
    return x >> k

def _nt(x):
    return ~x & MASK

def _rv(x):
    return int(format(x, "08b")[::-1], 2)

ATOMS: list[tuple[str, object]] = [("id", lambda x: x)]
ATOMS += [(f"rl{k}", lambda x, k=k: _rl(x, k)) for k in range(1, 8)]
ATOMS += [(f"sl{k}", lambda x, k=k: _sl(x, k)) for k in range(1, 8)]
ATOMS += [(f"sr{k}", lambda x, k=k: _sr(x, k)) for k in range(1, 8)]
ATOMS += [("nt", _nt), ("rv", _rv)]
NA = len(ATOMS)
L1 = [lab for lab, _ in ATOMS]

DOM = np.arange(256, dtype=np.int64)
A1 = np.array([[fn(int(x)) for x in DOM] for _, fn in ATOMS], dtype=np.int64)

def atom_cost(lab: str) -> int:
    # 复合原子 'a.b' 成本相加;id=0,rv=2,其余 1
    total = 0
    for part in lab.split("."):
        if part == "id":
            continue
        total += 2 if part == "rv" else 1
    return total

# depth-2 复合原子(b 先作用,a 后作用),函数表 dedupe,保留最简标签
_t2: dict[bytes, tuple[str, np.ndarray]] = {}
for _i in range(NA):
    for _j in range(NA):
        t = A1[_i][A1[_j]]
        if L1[_i] == "id":
            lab = L1[_j]
        elif L1[_j] == "id":
            lab = L1[_i]
        else:
            lab = f"{L1[_i]}.{L1[_j]}"
        key = t.tobytes()
        if key not in _t2 or atom_cost(lab) < atom_cost(_t2[key][0]):
            _t2[key] = (lab, t)
_pairs2 = sorted(_t2.values(), key=lambda v: (atom_cost(v[0]), v[0]))
L2 = [v[0] for v in _pairs2]
A2 = np.array([v[1] for v in _pairs2])
NA2 = len(L2)

PAIR_OPS = {
    "XOR": lambda a, b: a ^ b,
    "AND": lambda a, b: a & b,
    "OR": lambda a, b: a | b,
    "ANDN": lambda a, b: a & (~b & MASK),
    "ORN": lambda a, b: a | (~b & MASK),
    "XNOR": lambda a, b: ~(a ^ b) & MASK,
    "NAND": lambda a, b: ~(a & b) & MASK,
    "NOR": lambda a, b: ~(a | b) & MASK,
}
SYM_OPS = {"XOR", "AND", "OR", "XNOR", "NAND", "NOR"}
OP_COST = {
    "XOR": 1, "AND": 1, "OR": 1, "ANDN": 2, "ORN": 2,
    "XNOR": 2, "NAND": 2, "NOR": 2,
}

# 内层 pair 表(基原子 × 8 op),dedupe → 嵌套 T3 用
_ip: dict[bytes, tuple[str, int, np.ndarray]] = {}
for _op, _fn in PAIR_OPS.items():
    for _i in range(NA):
        rng = range(_i, NA) if _op in SYM_OPS else range(NA)
        for _j in rng:
            if _i == _j:
                continue
            t = _fn(A1[_i], A1[_j])
            cost = OP_COST[_op] + atom_cost(L1[_i]) + atom_cost(L1[_j])
            lab = f"{_op}({L1[_i]},{L1[_j]})"
            key = t.tobytes()
            if key not in _ip or cost < _ip[key][1]:
                _ip[key] = (lab, cost, t)
_ipl = sorted(_ip.values(), key=lambda v: (v[1], v[0]))
IPL = [v[0] for v in _ipl]
IPC = [v[1] for v in _ipl]
IPT = np.array([v[2] for v in _ipl])
NIP = len(IPL)

# ------------------------------------------------------------- parsing ----

EX_RE = re.compile(r"([01]{8})\s*->\s*([01]{8})")
Q_RE = re.compile(r"determine the output for:\s*([01]{8})")

def parse_problem(prompt: str):
    exs = EX_RE.findall(prompt)
    qm = Q_RE.search(prompt)
    if not exs or not qm:
        return None
    ins = [int(a, 2) for a, _ in exs]
    outs = [int(b, 2) for _, b in exs]
    return ins, outs, int(qm.group(1), 2)

# -------------------------------------------------------- enumeration ----

def enumerate_consistent(ins, outs):
    ex = np.array(ins, dtype=np.int64)
    out = np.array(outs, dtype=np.int64)
    cands = []
    seen: dict[bytes, int] = {}

    def add(tier, expr, ftab, cost):
        key = ftab.astype(np.uint8).tobytes()
        prev = seen.get(key)
        if prev is not None:
            if cost < cands[prev]["cost"]:
                cands[prev].update(tier=tier, expr=expr, cost=cost)
            return
        seen[key] = len(cands)
        cands.append({"tier": tier, "expr": expr, "ftab": ftab, "cost": cost})

    # ---- T1 unary over depth-2 atoms ----
    av2 = A2[:, ex]
    for i in np.where((av2 == out).all(axis=1))[0]:
        add("unary", L2[i], A2[i], atom_cost(L2[i]))

    # ---- T2 pairs over depth-2 atoms ----
    Aa = av2[:, None, :]
    Bb = av2[None, :, :]
    for op, fn in PAIR_OPS.items():
        M = fn(Aa, Bb)
        ok = (M == out).all(axis=2)
        ii, jj = np.where(ok)
        for i, j in zip(ii.tolist(), jj.tolist()):
            if op in SYM_OPS and j < i:
                continue
            cost = OP_COST[op] + atom_cost(L2[i]) + atom_cost(L2[j])
            add("pair", f"{op}({L2[i]},{L2[j]})", fn(A2[i], A2[j]), cost)

    # ---- T3 nested: op2(inner, c) / op2(c, inner) ----
    av1 = A1[:, ex]
    ipv = IPT[:, ex]
    for op, fn in PAIR_OPS.items():
        M = fn(ipv[:, None, :], av1[None, :, :])
        ok = (M == out).all(axis=2)
        for p, c in zip(*np.where(ok)):
            cost = OP_COST[op] + IPC[p] + atom_cost(L1[c])
            add("nested", f"{op}({IPL[p]},{L1[c]})", fn(IPT[p], A1[c]), cost)
        if op in SYM_OPS:
            continue
        M = fn(av1[:, None, :], ipv[None, :, :])
        ok = (M == out).all(axis=2)
        for c, p in zip(*np.where(ok)):
            cost = OP_COST[op] + IPC[p] + atom_cost(L1[c])
            add("nested", f"{op}({L1[c]},{IPL[p]})", fn(A1[c], IPT[p]), cost)

    # ---- T4 MAJ / MUX over base atoms(含取反)----
    for i in range(NA):
        ai, ti = av1[i], A1[i]
        for j in range(i + 1, NA):
            ab = ai & av1[j]
            aob = ai | av1[j]
            for k in range(j + 1, NA):
                m = ab | (aob & av1[k])
                lab = f"{L1[i]},{L1[j]},{L1[k]}"
                cst = 1 + atom_cost(L1[i]) + atom_cost(L1[j]) + atom_cost(L1[k])
                if (m == out).all():
                    ft = (ti & A1[j]) | ((ti | A1[j]) & A1[k])
                    add("maj", f"MAJ({lab})", ft, cst)
                if ((~m & MASK) == out).all():
                    ft = ~((ti & A1[j]) | ((ti | A1[j]) & A1[k])) & MASK
                    add("maj", f"NMAJ({lab})", ft, cst + 1)
    for s in range(NA):
        sv = av1[s]
        nsv = ~sv & MASK
        for b in range(NA):
            sb = sv & av1[b]
            for c in range(NA):
                if c == b:
                    continue
                m = sb | (nsv & av1[c])
                lab = f"{L1[s]};{L1[b]},{L1[c]}"
                cst = 1 + atom_cost(L1[s]) + atom_cost(L1[b]) + atom_cost(L1[c])
                if (m == out).all():
                    ft = (A1[s] & A1[b]) | ((~A1[s] & MASK) & A1[c])
                    add("mux", f"MUX({lab})", ft, cst)
                if ((~m & MASK) == out).all():
                    ft = ~((A1[s] & A1[b]) | ((~A1[s] & MASK) & A1[c])) & MASK
                    add("mux", f"NMUX({lab})", ft, cst + 1)

    return cands

# ---- T5 fallback: 任意 3 输入布尔函数 F(a(x),b(x),c(x)),a/b/c 为基原子 ----

def _bits_mat(vals):
    """ints -> (n,8) bit matrix, bit0=MSB。"""
    v = np.asarray(vals, dtype=np.int64)
    return ((v[:, None] >> (7 - np.arange(8))) & 1).astype(np.int8)

def enumerate_gf3(ins, outs, q):
    """返回 gf3 候选:与全部例子无冲突、query 全位可判定的 (triple, tt)。"""
    ex = np.array(ins, dtype=np.int64)
    AB = np.array([_bits_mat(A1[a][ex]) for a in range(NA)])  # (NA,nE,8)
    OB = _bits_mat(outs)  # (nE,8)
    QB = np.array([_bits_mat([int(A1[a][q])])[0] for a in range(NA)])  # (NA,8)
    cands = []
    for i in range(NA):
        for j in range(i + 1, NA):
            base = AB[i] * 4 + AB[j] * 2  # (nE,8)
            for k in range(j + 1, NA):
                idx = base + AB[k]
                tt = np.full(8, -1, dtype=np.int8)
                ok = True
                for v in range(8):
                    cell = idx == v
                    if not cell.any():
                        continue
                    has1 = bool((OB[cell] == 1).any())
                    has0 = bool((OB[cell] == 0).any())
                    if has1 and has0:
                        ok = False
                        break
                    tt[v] = 1 if has1 else 0
                if not ok:
                    continue
                qidx = QB[i] * 4 + QB[j] * 2 + QB[k]
                pred_bits = tt[qidx]
                if (pred_bits < 0).any():
                    continue  # query 不可判定
                pred = 0
                for b in range(8):
                    pred = (pred << 1) | int(pred_bits[b])
                tts = "".join("x" if t < 0 else str(t) for t in tt)
                cands.append(
                    {
                        "tier": "gf3",
                        "expr": f"F[{tts}]({L1[i]},{L1[j]},{L1[k]})",
                        "ftab": None,
                        "pred": pred,
                        "cost": 6
                        + atom_cost(L1[i])
                        + atom_cost(L1[j])
                        + atom_cost(L1[k]),
                    }
                )
    return cands

def cand_pred(c, q):
    if c["ftab"] is not None:
        return int(c["ftab"][q])
    return c["pred"]

# ------------------------------------------------------- prior / sigs ----

def signature(c) -> str:
    return f"{c['tier']}:{c['expr']}"

_DIG = re.compile(r"\d+")

def family_sig(c) -> str:
    return f"{c['tier']}:{_DIG.sub('', c['expr'])}"

def canonical_rule(rec):
    good = [c for c in rec["cands"] if cand_pred(c, rec["q"]) == rec["truth"]]
    if not good:
        return None
    return min(good, key=lambda c: (c["cost"], signature(c)))

def learn_prior(recs):
    sig_cnt = collections.Counter()
    fam_cnt = collections.Counter()
    for r in recs:
        if not r.get("parsed") or not r["covered"]:
            continue
        c = canonical_rule(r)
        if c is None:
            continue
        sig_cnt[signature(c)] += 1
        fam_cnt[family_sig(c)] += 1
    return sig_cnt, fam_cnt

def score_candidate(c, sig_cnt, fam_cnt):
    s = sig_cnt.get(signature(c), 0) + 0.1 * fam_cnt.get(family_sig(c), 0)
    return (-s, c["cost"], signature(c))

def decide(rec, sig_cnt, fam_cnt):
    if not rec.get("parsed") or not rec["cands"]:
        return None
    best = min(rec["cands"], key=lambda c: score_candidate(c, sig_cnt, fam_cnt))
    return cand_pred(best, rec["q"])

def decide_mincost(rec):
    if not rec.get("parsed") or not rec["cands"]:
        return None
    best = min(rec["cands"], key=lambda c: (c["cost"], signature(c)))
    return cand_pred(best, rec["q"])

# ------------------------------------------------------------- driver ----

def load_problems():
    df = pd.read_csv(f"{ROOT}/data/train_cat.csv")
    bit = df[df["cat"] == "bit_manipulation"]
    hold_ids = set()
    hold_rows = []
    with open(f"{ROOT}/data/run012_holdout_decode.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if d["cat"] == "bit_manipulation":
                hold_ids.add(d["id"])
                hold_rows.append(d)
    hdf = pd.read_csv(f"{ROOT}/data/holdout.csv")
    hbit = hdf[hdf["id"].isin(hold_ids)]
    model_ok = {d["id"]: d["ok"] for d in hold_rows}
    return bit, hbit, model_ok

def solve_set(df, label):
    recs = []
    for _, row in df.iterrows():
        parsed = parse_problem(row["prompt"])
        if parsed is None:
            recs.append({"id": row["id"], "parsed": False})
            continue
        ins, outs, q = parsed
        truth = int(str(row["answer"]).strip(), 2)
        cands = enumerate_consistent(ins, outs)
        used_gf3 = False
        if not cands:
            cands = enumerate_gf3(ins, outs, q)
            used_gf3 = True
        preds = sorted({cand_pred(c, q) for c in cands})
        recs.append(
            {
                "id": row["id"],
                "parsed": True,
                "n_ex": len(ins),
                "q": q,
                "truth": truth,
                "cands": cands,
                "preds": preds,
                "covered": truth in preds,
                "ambiguous": len(preds) > 1,
                "gf3": used_gf3,
            }
        )
    par = [r for r in recs if r["parsed"]]
    cov = sum(r["covered"] for r in par)
    amb = sum(r["ambiguous"] for r in par)
    amb_cov = sum(r["ambiguous"] and r["covered"] for r in par)
    ngf = sum(r.get("gf3", False) for r in par)
    gf_cov = sum(r.get("gf3", False) and r["covered"] for r in par)
    print(
        f"[{label}] n={len(recs)} parsed={len(par)} covered={cov} "
        f"({cov/len(par):.3f}) ambiguous={amb} (covered among amb={amb_cov}) "
        f"gf3-fallback={ngf} (covered={gf_cov})"
    )
    return recs

def main():
    bit, hbit, model_ok = load_problems()
    # 防泄漏:train_cat.csv 包含 holdout 题,先验学习集剔除 holdout id
    bit = bit[~bit["id"].isin(set(hbit["id"]))]
    if "--quick" in sys.argv:
        bit = bit.head(300)

    train_recs = solve_set(bit, "train(去 holdout)")
    hold_recs = solve_set(hbit, "holdout")

    sig_cnt, fam_cnt = learn_prior(train_recs)
    total = sum(sig_cnt.values())
    print(f"\n== 出题器规则分布(canonical,top40 / {len(sig_cnt)} 签名,{total} 题)==")
    for sig, n in sig_cnt.most_common(40):
        print(f"  {sig:50s} {n:5d}  {n/total:.3f}")
    print("\n== 粗签名分布(top25)==")
    for sig, n in fam_cnt.most_common(25):
        print(f"  {sig:50s} {n:5d}")

    def evaluate(recs, sc, fc, tag):
        ok_p = ok_m = tot = 0
        for r in recs:
            if not r.get("parsed"):
                continue
            tot += 1
            ok_p += decide(r, sc, fc) == r["truth"]
            ok_m += decide_mincost(r) == r["truth"]
        print(f"[{tag}] n={tot} prior={ok_p} ({ok_p/tot:.3f}) mincost={ok_m} ({ok_m/tot:.3f})")

    evaluate(train_recs, sig_cnt, fam_cnt, "train in-sample")

    parsed = [r for r in train_recs if r.get("parsed")]
    fo = fn_ = 0
    for f in range(5):
        tr = [r for i, r in enumerate(parsed) if i % 5 != f]
        te = [r for i, r in enumerate(parsed) if i % 5 == f]
        sc, fc = learn_prior(tr)
        for r in te:
            fn_ += 1
            fo += decide(r, sc, fc) == r["truth"]
    print(f"[train 5-fold] {fo}/{fn_} = {fo/fn_:.3f}")

    print("\n== holdout 110 ==")
    n_prior = n_min = n_ceil = 0
    fixed, broken, details = [], [], []
    for r in hold_recs:
        if not r.get("parsed"):
            continue
        p = decide(r, sig_cnt, fam_cnt)
        ok_prior = p == r["truth"]
        n_prior += ok_prior
        n_min += decide_mincost(r) == r["truth"]
        n_ceil += r["covered"]
        was_ok = model_ok.get(r["id"], False)
        if ok_prior and not was_ok:
            fixed.append(r["id"])
        if (not ok_prior) and was_ok:
            broken.append(r["id"])
        if not ok_prior or not was_ok:
            best = (
                min(r["cands"], key=lambda c: score_candidate(c, sig_cnt, fam_cnt))
                if r["cands"]
                else None
            )
            good = canonical_rule(r)
            details.append(
                f"  {r['id']} model_ok={was_ok} prior_ok={ok_prior} "
                f"covered={r['covered']} npred={len(r['preds'])} "
                f"pick={signature(best) if best else 'NONE'} "
                f"truthrule={signature(good) if good else 'NONE'}"
            )
    print(f"oracle ceiling: {n_ceil}/110")
    print(f"prior-tiebreak oracle: {n_prior}/110")
    print(f"mincost-tiebreak oracle: {n_min}/110")
    print(f"模型基线 91/110;prior 相对净增 = {n_prior - 91}")
    print(f"修复(模型错→prior对): {len(fixed)} {fixed}")
    print(f"打破(模型对→prior错): {len(broken)} {broken}")
    print("\n== holdout 难题细节(模型错或 prior 错)==")
    for d in details:
        print(d)

    unc = [r for r in train_recs if r.get("parsed") and not r["covered"]]
    print(f"\n== train 未覆盖 {len(unc)} 题 ==")
    for r in unc[:10]:
        print(f"  {r['id']} n_cands={len(r['cands'])} preds={len(r['preds'])}")

if __name__ == "__main__":
    main()
