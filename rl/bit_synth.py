#!/usr/bin/env python3
"""bit 新族合成数据增广(第四版,$0 本地)。

R3c 验尸:行为载体已通(扩展候选执行率 95/110),缺口在"例子→规则"的选择
准确率,根因 = 新族训练样本只有 195 个。本脚本按 rl/bit_prior.py 学到的出题器
规则先验合成新族题:

  1. 规则采样:对 data/rft/bit_new_traces.jsonl 的 196 道真·新族 train 题重算
     truth-consistent 先验首选(与 v3 渲染器逐题一致),得到新族规则池
     (pair/nested/MAJ/MUX/gf3 + 各自计数 = train 实测占比);unary(1 道)与
     无 truth 候选的 ccf02dee 剔除。legacy 老族不采样(老族不缺样本)。
  2. 出题:每题随机 n_ex ∈ {7,8,9,10}(train 实测均匀)、example 输入 =
     无放回均匀字节(train 实测:逐位 1 率 ~0.50、无重复、query 不在 example
     里),全部输出用引擎函数表正向计算(真值天然正确);gf3 规则的未定义
     cell 逐题随机补全。题面 = train_cat.csv 三道真题 diff 出的逐字节固定模板,
     只有位串可变。
  3. 自检(无歧义裁决):truth ∈ 一致候选预测集,且先验首选候选的预测 ==
     truth(候选唯一情形被包含)→ 推理期(无真值)同一裁决会选中同一 winner;
     legacy 排除:legacy 解题器 boxed == truth 的样题丢弃重采(那是老族)。
  4. 渲染:v3 同构混入发射器(src/reasoners/bit_manipulation.py)生成全长
     trace,boxed==truth 逐题硬性 assert;长度纪律同 v3。
  5. 质检:模板 800/800 等同性、5 道与真题并排、独立行校验(独立重写的
     原子/op/MAJ/MUX/gf3 语义,不复用引擎表)。
  6. 重建 data/rft/bit_corpus_mix.jsonl:195 真新族 + 800 合成 + 700 legacy
     (沿用上轮 seed=0 抽样,直接复用现有 mix 行)+ 178 非 bit 镇纸。

用法:python3 rl/bit_synth.py [--n 800] [--no-mix] [--seed 12345]
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.corpus import GEN_LIMIT, TOKEN_LIMIT, tokenize_prompt  # noqa: E402
from src.reasoners import bit_manipulation as bm  # noqa: E402
from src.reasoners.store_types import Example, Problem  # noqa: E402

TRAIN_CAT = ROOT / "data" / "train_cat.csv"
NEW_TRACES = ROOT / "data" / "rft" / "bit_new_traces.jsonl"
SYNTH_TRACES = ROOT / "data" / "rft" / "bit_synth_traces.jsonl"
MIX_PATH = ROOT / "data" / "rft" / "bit_corpus_mix.jsonl"

SYNTH_TIERS = ("pair", "nested", "maj", "mux", "gf3")  # 新族;unary/legacy 排除

_EX_RE = re.compile(r"([01]{8}) -> ([01]{8})")
_Q_RE = re.compile(r"Now, determine the output for:\s*([01]{8})")
_B8 = re.compile(r"[01]{8}")


def b8(x: int) -> str:
    return format(x & 0xFF, "08b")


# ---------------------------------------------------------------- 模板提取 ----

EX_MARK = "Here are some examples of input -> output:\n"
Q_MARK = "\n\nNow, determine the output for: "


def extract_template(bit_df) -> str:
    """从 3 道真题 diff 出固定模板(逐字节克隆;只有位串是变量)。"""
    heads = set()
    for _, row in bit_df.head(3).iterrows():
        p = row["prompt"]
        head, rest = p.split(EX_MARK, 1)
        exblock, q = rest.split(Q_MARK, 1)
        assert _B8.fullmatch(q), f"query 段残留非位串: {q!r}"
        for ln in exblock.rstrip("\n").splitlines():
            assert re.fullmatch(r"[01]{8} -> [01]{8}", ln), repr(ln)
        heads.add(head)
    assert len(heads) == 1, "3 道真题模板头不一致"
    return heads.pop() + EX_MARK


def render_prompt(tmpl_head: str, ins: list[int], outs: list[int], q: int) -> str:
    exlines = "\n".join(f"{b8(i)} -> {b8(o)}" for i, o in zip(ins, outs))
    return f"{tmpl_head}{exlines}{Q_MARK}{b8(q)}"


# ------------------------------------------------------------ 规则池(先验) ----

def parse_real(prompt: str):
    exs = _EX_RE.findall(prompt)
    q = _Q_RE.search(prompt)
    ins = [int(a, 2) for a, _ in exs]
    outs = [int(b, 2) for _, b in exs]
    return ins, outs, int(q.group(1), 2)


def build_pool(eng, prior, bit_by_id: dict) -> tuple[dict, dict]:
    """对 196 道真·新族题重算 truth-consistent 先验首选(与 v3 渲染器同款
    裁决),返回 {tier: [(expr, kind, weight), ...]} 与 tier 计数。"""
    new_ids = []
    with open(NEW_TRACES) as f:
        for line in f:
            new_ids.append(json.loads(line)["problem_id"])
    assert len(new_ids) == 196, len(new_ids)

    picks = []
    skipped = []
    for pid in new_ids:
        row = bit_by_id[pid]
        ins, outs, q = parse_real(row["prompt"])
        truth = int(str(row["answer"]).strip(), 2)
        cands = eng.enumerate_consistent(ins, outs)
        if not cands:
            cands = eng.enumerate_gf3(ins, outs, q)
        ranked = sorted(cands, key=lambda c: eng.score(c, prior))
        pick = next((c for c in ranked if eng.cand_pred(c, q) == truth), None)
        if pick is None:
            skipped.append(pid)  # ccf02dee:族不覆盖,无 truth 候选
            continue
        picks.append((pid, pick))

    tier_cnt = collections.Counter(p["kind"][0] for _, p in picks)
    cnt = collections.Counter()
    kind_of = {}
    for _, p in picks:
        t = p["kind"][0]
        if t not in SYNTH_TIERS:
            continue  # unary:老族邻接,样本充足,不合成
        key = (t, p["expr"])
        cnt[key] += 1
        kind_of[key] = p["kind"]
    pool: dict = {t: [] for t in SYNTH_TIERS}
    for (t, expr), w in sorted(cnt.items()):
        pool[t].append((expr, kind_of[(t, expr)], w))
    print(f"[pool] picks={len(picks)} skipped(no-truth-cand)={skipped} "
          f"tier_cnt={dict(tier_cnt)} unique_rules="
          f"{ {t: len(v) for t, v in pool.items()} }")
    return pool, {t: tier_cnt[t] for t in SYNTH_TIERS}


def quotas(counts: dict, total: int) -> dict:
    s = sum(counts.values())
    raw = {t: total * c / s for t, c in counts.items()}
    base = {t: int(raw[t]) for t in raw}
    rem = total - sum(base.values())
    for t in sorted(raw, key=lambda t: raw[t] - base[t], reverse=True)[:rem]:
        base[t] += 1
    return base


# ------------------------------------------------------- 规则 -> 函数表 ----

def kind_to_ftab(eng, kind, rng):
    """引擎正向函数表(256 字节);gf3 的未定义 cell 用 rng 补全。
    返回 (ftab, label) — label 仅用于审计。"""
    MASK = 0xFF
    t = kind[0]
    if t == "unary":
        return eng.A2[kind[1]].copy(), eng.L2[kind[1]]
    if t == "pair":
        _, op, i, j = kind
        return (eng.PAIR_OPS[op](eng.A2[i], eng.A2[j]) & MASK,
                f"{op}({eng.L2[i]},{eng.L2[j]})")
    if t == "nested":
        _, op2, p, c, c_first = kind
        fn = eng.PAIR_OPS[op2]
        ti, tc = eng.IPT[p], eng.A1[c]
        tab = (fn(tc, ti) if c_first else fn(ti, tc)) & MASK
        lab = (f"{op2}({eng.L1[c]},{eng.IPL[p]})" if c_first
               else f"{op2}({eng.IPL[p]},{eng.L1[c]})")
        return tab, lab
    if t == "maj":
        _, neg, i, j, k = kind
        ta, tb, tc = eng.A1[i], eng.A1[j], eng.A1[k]
        ft = (ta & tb) | ((ta | tb) & tc)
        if neg:
            ft = ~ft & MASK
        return ft, (f"{'N' if neg else ''}MAJ({eng.L1[i]},{eng.L1[j]},"
                    f"{eng.L1[k]})")
    if t == "mux":
        _, neg, s, b, c = kind
        ts, tb, tc = eng.A1[s], eng.A1[b], eng.A1[c]
        ft = (ts & tb) | ((~ts & MASK) & tc)
        if neg:
            ft = ~ft & MASK
        return ft, (f"{'N' if neg else ''}MUX({eng.L1[s]};{eng.L1[b]},"
                    f"{eng.L1[c]})")
    if t == "gf3":
        tt, i, j, k = kind[1], kind[2], kind[3], kind[4]
        tt_f = [v if v >= 0 else rng.randint(0, 1) for v in tt]
        np = eng.np
        out = np.zeros(256, dtype=np.int64)
        for x in range(256):
            a, b, c = int(eng.A1[i][x]), int(eng.A1[j][x]), int(eng.A1[k][x])
            v = 0
            for bit in range(8):
                idx = ((((a >> (7 - bit)) & 1) << 2)
                       | (((b >> (7 - bit)) & 1) << 1)
                       | ((c >> (7 - bit)) & 1))
                v = (v << 1) | tt_f[idx]
            out[x] = v
        tts = "".join(str(v) for v in tt_f)
        return out, f"F[{tts}]({eng.L1[i]},{eng.L1[j]},{eng.L1[k]})"
    raise ValueError(f"unknown kind {kind!r}")


# --------------------------------------------------------------- 出题主环 ----

# n_ex 按真·新族 196 道实测分布采样(7:64 / 8:53 / 9:35 / 10:44;全量 1602
# 道是均匀 7-10,但合成集补的是新族 cohort,跟它同分布,顺带把 trace 长度
# 钉在 v3 长度带:头部体积随 n_ex 单调)。
_N_EX_VALS = (7, 8, 9, 10)
_N_EX_WEIGHTS = (64, 53, 35, 44)


def synth_one(eng, prior, tmpl_head: str, kind, rng,
              seen: set, real_keys: set):
    """采样一道题;任一自检不过返回 (None, 原因)。"""
    ftab, rule_label = kind_to_ftab(eng, kind, rng)
    n_ex = rng.choices(_N_EX_VALS, weights=_N_EX_WEIGHTS, k=1)[0]
    ins = rng.sample(range(256), n_ex)
    pool_q = [v for v in range(256) if v not in set(ins)]
    q = rng.choice(pool_q)
    key = (tuple(ins), q)
    if key in seen or key in real_keys:
        return None, "dup"
    outs = [int(ftab[x]) for x in ins]
    truth_i = int(ftab[q])
    truth = b8(truth_i)

    # 无歧义自检:truth 可恢复 且 先验首选 == truth(推理期同款裁决)
    cands = eng.enumerate_consistent(ins, outs)
    if not cands:
        cands = eng.enumerate_gf3(ins, outs, q)
    if not cands:
        return None, "no_cand"
    preds = {eng.cand_pred(c, q) for c in cands}
    if truth_i not in preds:
        return None, "not_covered"
    top = min(cands, key=lambda c: eng.score(c, prior))
    if eng.cand_pred(top, q) != truth_i:
        return None, "ambiguous"

    prompt = render_prompt(tmpl_head, ins, outs, q)
    prob = Problem(
        id="synth_tmp", category="bit_manipulation",
        examples=[Example(b8(i), b8(o)) for i, o in zip(ins, outs)],
        question=b8(q), answer=truth, prompt=prompt,
    )
    legacy = bm._reasoning_legacy(prob)
    if legacy is not None and bm._ev_boxed(legacy) == truth:
        return None, "legacy_solves"  # 老族,排除
    trace = bm._reasoning_prior_family(prob)
    if trace is None:
        return None, "render_none"
    if bm._ev_boxed(trace) != truth:
        return None, "box_mismatch"
    if not bm._ev_completion_fits(trace, truth):
        return None, "too_long"
    seen.add(key)
    return {
        "prompt": prompt, "trace": trace, "answer": truth,
        "ins": ins, "outs": outs, "q": q,
        "tier": top["kind"][0], "rule": rule_label,
        "winner": top["expr"],
    }, "ok"


# ------------------------------------------------- 独立行校验(同 v3 方法) ----
# 独立重写的原子/op/MAJ/MUX/gf3 语义(纯字符串位运算,不复用引擎表)。

def _q_atom(label: str, s: str) -> str:
    for part in reversed(label.split(".")):
        if part == "id":
            continue
        if part == "nt":
            s = "".join("1" if c == "0" else "0" for c in s)
        elif part == "rv":
            s = s[::-1]
        elif part.startswith("rl"):
            k = int(part[2:]); s = s[k:] + s[:k]
        elif part.startswith("sl"):
            k = int(part[2:]); s = s[k:] + "0" * k
        elif part.startswith("sr"):
            k = int(part[2:]); s = "0" * k + s[: len(s) - k]
        else:
            raise ValueError(f"bad atom {label!r}")
    return s


def _q_op(op: str, a: str, b: str) -> str:
    out = []
    for ca, cb in zip(a, b):
        x, y = ca == "1", cb == "1"
        v = {"XOR": x ^ y, "AND": x and y, "OR": x or y,
             "ANDN": x and not y, "ORN": x or not y, "XNOR": not (x ^ y),
             "NAND": not (x and y), "NOR": not (x or y)}[op]
        out.append("1" if v else "0")
    return "".join(out)


def _split_top(s: str, sep: str) -> list[str]:
    parts, depth, cur = [], 0, []
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(cur)); cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return parts

_OPS = ("XOR", "AND", "OR", "ANDN", "ORN", "XNOR", "NAND", "NOR")
_OP_RE = re.compile(r"^(XOR|AND|OR|ANDN|ORN|XNOR|NAND|NOR)\((.*)\)$")
_MAJ_RE = re.compile(r"^(N?)MAJ\((.*)\)$")
_MUX_RE = re.compile(r"^(N?)MUX\((.*)\)$")
_GFW_RE = re.compile(r"^F\[([01x]{8})\]\((.*)\)$")
_GFR_RE = re.compile(r"^F\((.*)\)$")


def _seg_expected_rows(label: str, ins_s: list[str], outs_s: list[str]):
    """独立语义重算:返回 [(full_row, slim_row, ok)],至首个 mismatch 止。"""
    rows = []

    def push(k, ops, pv):
        ok = pv == outs_s[k]
        mark = "ok" if ok else "x"
        slim = f"{k} {pv} vs {outs_s[k]} {mark}"
        full = (f"{k} {' '.join(ops)} -> {pv} vs {outs_s[k]} {mark}"
                if ops else slim)
        rows.append((full, slim, ok))
        return ok

    m = _GFW_RE.match(label)
    if m:
        tt = [None if c == "x" else int(c) for c in m.group(1)]
        a, b, c = _split_top(m.group(2), ",")
        for k, s in enumerate(ins_s):
            av, bv, cv = _q_atom(a, s), _q_atom(b, s), _q_atom(c, s)
            cells = [tt[(int(x) << 2) | (int(y) << 1) | int(z)]
                     for x, y, z in zip(av, bv, cv)]
            assert all(v is not None for v in cells), \
                f"gf3 winner 例子命中未定义 cell: {label}"
            pv = "".join(str(v) for v in cells)
            if not push(k, (av, bv, cv), pv):
                break
        return rows
    m = _MAJ_RE.match(label)
    if m:
        neg = m.group(1) == "N"
        a, b, c = _split_top(m.group(2), ",")
        for k, s in enumerate(ins_s):
            av, bv, cv = _q_atom(a, s), _q_atom(b, s), _q_atom(c, s)
            pv = "".join("1" if (x + y + z >= 2) != neg else "0"
                         for x, y, z in zip(map(int, av), map(int, bv),
                                            map(int, cv)))
            if not push(k, (av, bv, cv), pv):
                break
        return rows
    m = _MUX_RE.match(label)
    if m:
        neg = m.group(1) == "N"
        s_lab, rest = _split_top(m.group(2), ";")
        b_lab, c_lab = _split_top(rest, ",")
        for k, s in enumerate(ins_s):
            sv, bv, cv = (_q_atom(s_lab, s), _q_atom(b_lab, s),
                          _q_atom(c_lab, s))
            pv = "".join((y if x == "1" else z) for x, y, z in zip(sv, bv, cv))
            if neg:
                pv = "".join("1" if c == "0" else "0" for c in pv)
            if not push(k, (sv, bv, cv), pv):
                break
        return rows
    m = _OP_RE.match(label)
    if m and "(" in m.group(2):  # nested: OP2(OP1(a,b),c) / OP2(c,OP1(a,b))
        op2 = m.group(1)
        x, y = _split_top(m.group(2), ",")
        c_first = "(" not in x
        inner_lab = y if c_first else x
        c_lab = x if c_first else y
        mi = _OP_RE.match(inner_lab)
        op1 = mi.group(1)
        a_lab, b_lab = _split_top(mi.group(2), ",")
        for k, s in enumerate(ins_s):
            av, bv = _q_atom(a_lab, s), _q_atom(b_lab, s)
            inner = _q_op(op1, av, bv)
            cv = _q_atom(c_lab, s)
            first, second = (cv, inner) if c_first else (inner, cv)
            pv = _q_op(op2, first, second)
            ok = pv == outs_s[k]
            mark = "ok" if ok else "x"
            full = (f"{k} {op1} {av} {bv} -> {inner} "
                    f"{op2} {first} {second} -> {pv} vs {outs_s[k]} {mark}")
            slim = f"{k} {pv} vs {outs_s[k]} {mark}"
            rows.append((full, slim, ok))
            if not ok:
                break
        return rows
    if m:  # plain pair over (composite) atoms
        op = m.group(1)
        a_lab, b_lab = _split_top(m.group(2), ",")
        for k, s in enumerate(ins_s):
            av, bv = _q_atom(a_lab, s), _q_atom(b_lab, s)
            pv = _q_op(op, av, bv)
            if not push(k, (av, bv), pv):
                break
        return rows
    # unary (composite) atom
    for k, s in enumerate(ins_s):
        pv = _q_atom(label, s)
        if not push(k, (), pv):
            break
    return rows


def verify_trace(trace: str, ins_s: list[str], outs_s: list[str],
                 q_s: str) -> tuple[int, int]:
    """独立行校验整个扫描区。返回 (候选段数, 校验行数);任何不一致即抛错。"""
    lines = trace.split("\n")
    last_best = max(i for i, ln in enumerate(lines) if ln.startswith("Best: "))
    assert lines[last_best + 1] == "", "Best 后应为空行"
    i_apply = lines.index(f"Applying to {q_s}")
    scan = lines[last_best + 2:i_apply]
    while scan and scan[-1] == "":
        scan.pop()
    assert scan, "扫描区为空"
    # 切分候选段
    segs, cur = [], None
    for ln in scan:
        if cur is None:
            cur = {"label": ln, "rows": [], "cells": [], "match": False,
                   "cellrej": None}
            continue
        if ln == "match":
            cur["match"] = True
            segs.append(cur); cur = None
        elif ln.startswith("Cells ("):
            pass  # 表头
        elif re.fullmatch(r"[01]{3} [01-]", ln):
            cur["cells"].append(ln)
        elif re.fullmatch(r"cell [01]{3} -> 0 1 x", ln):
            cur["cellrej"] = ln
            segs.append(cur); cur = None
        elif re.match(r"^\d+ ", ln) and " vs " in ln:
            cur["rows"].append(ln)
        else:
            segs.append(cur)
            cur = {"label": ln, "rows": [], "cells": [], "match": False,
                   "cellrej": None}
    assert cur is None, "末段未以 match 收尾"
    assert segs and segs[-1]["match"], "无 winner 段"
    assert all(not s["match"] for s in segs[:-1]), "match 只许出现在末段"

    n_rows = 0
    for si, seg in enumerate(segs):
        label = seg["label"]
        is_winner = seg["match"]
        if seg["cellrej"] is not None:  # gf3 否决段:验证 cell 冲突
            m = _GFR_RE.match(label)
            assert m, f"cellrej 段 label 非 F(...): {label!r}"
            a, b, c = _split_top(m.group(1), ",")
            v = int(re.match(r"cell ([01]{3})", seg["cellrej"]).group(1), 2)
            seen_out = set()
            for s, o in zip(ins_s, outs_s):
                av, bv, cv = _q_atom(a, s), _q_atom(b, s), _q_atom(c, s)
                for x, y, z, ob in zip(av, bv, cv, o):
                    if (int(x) << 2) | (int(y) << 1) | int(z) == v:
                        seen_out.add(ob)
            assert seen_out == {"0", "1"}, f"cell {v:03b} 无冲突: {label}"
            n_rows += 1
            continue
        if seg["cells"]:  # gf3 winner 的 Cells 表:逐 cell 对账
            m = _GFW_RE.match(label)
            assert m, label
            tt = m.group(1)
            a, b, c = _split_top(m.group(2), ",")
            observed = {}
            for s, o in zip(ins_s, outs_s):
                av, bv, cv = _q_atom(a, s), _q_atom(b, s), _q_atom(c, s)
                for x, y, z, ob in zip(av, bv, cv, o):
                    idx = (int(x) << 2) | (int(y) << 1) | int(z)
                    assert observed.setdefault(idx, ob) == ob, "cell 冲突"
            assert len(seg["cells"]) == 8
            for ln in seg["cells"]:
                idx = int(ln[:3], 2)
                shown = ln[4]
                want = observed.get(idx, "-")
                assert shown == want != "-" or (shown == "-" and idx not in
                                                observed), \
                    f"Cells 行不符: {ln!r} want {want}"
                # 表内容必须与 label 的 tt 一致(winner 的推导根据)
                if idx in observed:
                    assert tt[idx] == observed[idx], "tt 与观测不一致"
                n_rows += 1
        expected = _seg_expected_rows(label, ins_s, outs_s)
        got = seg["rows"]
        if is_winner:
            assert all(ok for _, _, ok in expected) and \
                len(expected) == len(ins_s), f"winner 段非全 ok: {label}"
        else:
            assert expected and not expected[-1][2], \
                f"否决段末行应为 x: {label}"
        assert len(got) == len(expected), \
            f"{label}: 行数 {len(got)} != 期望 {len(expected)}(止于首 x)"
        slim_used = got and expected and got[0] == expected[0][1] != \
            expected[0][0]
        for g, (full, slim, _ok) in zip(got, expected):
            want = slim if (is_winner and slim_used) else full
            assert g == want, (f"行不符 [{label}]\n  got : {g!r}"
                               f"\n  want: {want!r}")
            n_rows += 1
    return len(segs), n_rows


# ------------------------------------------------------------------ main ----

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--no-mix", action="store_true",
                    help="只合成+质检,不重建 bit_corpus_mix.jsonl")
    args = ap.parse_args()

    import pandas as pd
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    t0 = time.time()
    df = pd.read_csv(TRAIN_CAT)
    bit_df = df[df["cat"] == "bit_manipulation"]
    bit_by_id = {str(r["id"]): r for _, r in bit_df.iterrows()}
    tmpl_head = extract_template(bit_df)
    real_keys = {(tuple(parse_real(r["prompt"])[0]), parse_real(r["prompt"])[2])
                 for _, r in bit_df.iterrows()}

    eng = bm._ext_engine()
    print(f"[{time.time()-t0:.0f}s] learning prior (train_split, in-process)…")
    prior = eng.prior()
    print(f"[{time.time()-t0:.0f}s] prior ready: {len(prior[0])} signatures")

    pool, tier_cnt = build_pool(eng, prior, bit_by_id)
    quota = quotas(tier_cnt, args.n)
    print(f"[quota] {quota} (sum={sum(quota.values())})")

    rng = random.Random(args.seed)
    seen: set = set()
    made: list[dict] = []
    rej = collections.Counter()
    for tier in SYNTH_TIERS:
        entries = pool[tier]
        weights = [w for _, _, w in entries]
        got = 0
        attempts = 0
        while got < quota[tier]:
            attempts += 1
            assert attempts < quota[tier] * 60 + 1000, \
                f"{tier}: 重采淹没,检查自检条件"
            _, kind, _ = rng.choices(entries, weights=weights, k=1)[0]
            rec, why = synth_one(eng, prior, tmpl_head, kind, rng, seen,
                                 real_keys)
            if rec is None:
                rej[f"{tier}:{why}"] += 1
                continue
            rec["problem_id"] = f"synth_{len(made):06d}"
            made.append(rec)
            got += 1
        print(f"[{time.time()-t0:.0f}s] tier {tier}: {got}/{quota[tier]} "
              f"(attempts {attempts})")
    print(f"[reject] {dict(rej)}")
    assert len(made) == args.n

    # ---- 硬性自检:boxed == truth 全量(生成期已查,再独立复查一遍)----
    n_box = sum(bm._ev_boxed(r["trace"]) == r["answer"] for r in made)
    assert n_box == args.n, f"boxed==truth {n_box}/{args.n}"

    # ---- 质检 1:模板等同性(全量)+ 5 道并排 ----
    real_norm = re.sub(r"[01]{8}", "B", bit_df.iloc[0]["prompt"])
    real_norm = re.sub(r"(B -> B\n)+", "EX\n", real_norm)
    n_tmpl = 0
    for r in made:
        norm = re.sub(r"[01]{8}", "B", r["prompt"])
        norm = re.sub(r"(B -> B\n)+", "EX\n", norm)
        n_tmpl += norm == real_norm
    assert n_tmpl == args.n, f"模板等同 {n_tmpl}/{args.n}"
    print(f"[tmpl] 模板逐字节等同(位串归一后): {n_tmpl}/{args.n}")
    qc_rng = random.Random(0)
    for r in qc_rng.sample(made, 5):
        real_row = bit_df.iloc[qc_rng.randrange(len(bit_df))]
        print(f"\n===== 并排对比 {r['problem_id']} (rule {r['rule']}) vs 真题 "
              f"{real_row['id']} =====")
        for a, b in zip(r["prompt"].split("\n"),
                        str(real_row["prompt"]).split("\n")):
            print(f"  S| {a}\n  R| {b}")

    # ---- 质检 2:独立行校验(抽 3 道 + 顺手全量)----
    picks3 = random.Random(1).sample(made, 3)
    for r in picks3:
        ins_s = [b8(i) for i in r["ins"]]
        outs_s = [b8(o) for o in r["outs"]]
        nseg, nrow = verify_trace(r["trace"], ins_s, outs_s, b8(r["q"]))
        print(f"[verify-3] {r['problem_id']} ({r['tier']}): "
              f"{nseg} 段 / {nrow} 行,全部一致")
    tot_seg = tot_row = 0
    for r in made:
        nseg, nrow = verify_trace(
            r["trace"], [b8(i) for i in r["ins"]],
            [b8(o) for o in r["outs"]], b8(r["q"]))
        tot_seg += nseg
        tot_row += nrow
    print(f"[verify-all] {args.n} 道全量独立行校验通过: {tot_seg} 段 / "
          f"{tot_row} 行")

    # ---- tokenize + 落盘 ----
    print(f"[{time.time()-t0:.0f}s] tokenizing…")
    tok = Tokenizer.from_file(str(ROOT / "src" / "tokenizer.json"))
    chat_tok = AutoTokenizer.from_pretrained(
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", trust_remote_code=True)
    gen_toks = []
    with open(SYNTH_TRACES, "w") as f:
        for r in made:
            completion = (f"{r['trace']}\n</think>\n"
                          f"\\boxed{{{r['answer']}}}<|im_end|>")
            comp_ids = tok.encode(completion, add_special_tokens=False).ids
            assert len(comp_ids) <= GEN_LIMIT
            prompt_ids = tokenize_prompt(r["prompt"], chat_tok)
            tokens = prompt_ids + list(comp_ids)
            assert len(tokens) <= TOKEN_LIMIT
            mask = [0] * len(prompt_ids) + [1] * len(comp_ids)
            gen_toks.append(len(comp_ids))
            f.write(json.dumps({
                "problem_id": r["problem_id"],
                "category": "bit_manipulation", "sample_idx": 0,
                "tokens": tokens, "mask": mask, "text": completion,
                "n_prompt_tok": len(prompt_ids), "n_gen_tok": len(comp_ids),
                "tier": r["tier"], "rule": r["rule"], "winner": r["winner"],
                "answer": r["answer"], "prompt": r["prompt"],
            }) + "\n")
    gen_toks.sort()
    p50 = gen_toks[len(gen_toks) // 2]
    p95 = gen_toks[int(len(gen_toks) * 0.95)]
    tier_made = collections.Counter(r["tier"] for r in made)
    print(f"[len] completion tok p50={p50} p95={p95} "
          f"min={gen_toks[0]} max={gen_toks[-1]}")
    print(f"[tiers] {dict(tier_made)}")
    print(f"[out] {SYNTH_TRACES} ({args.n} 行)")

    if args.no_mix:
        return

    # ---- 重建 bit_corpus_mix.jsonl:沿用上轮 195 真新族 + 700 legacy +
    #      178 非 bit(直接复用现有行,保证“沿用上轮抽样”逐字节成立)----
    with open(NEW_TRACES) as f:
        new_ids = {json.loads(l)["problem_id"] for l in f}
    old_rows = [json.loads(l) for l in open(MIX_PATH)]
    nonbit = [r for r in old_rows if r["category"] != "bit_manipulation"]
    nf = [r for r in old_rows if r["category"] == "bit_manipulation"
          and r["problem_id"] in new_ids]
    legacy = [r for r in old_rows if r["category"] == "bit_manipulation"
              and r["problem_id"] not in new_ids]
    assert (len(nonbit), len(nf), len(legacy)) == (178, 195, 700), \
        (len(nonbit), len(nf), len(legacy))
    synth_rows = []
    with open(SYNTH_TRACES) as f:
        for line in f:
            d = json.loads(line)
            synth_rows.append({
                "problem_id": d["problem_id"], "category": d["category"],
                "tokens": d["tokens"], "mask": d["mask"]})
    mix = nf + synth_rows + legacy + nonbit
    random.Random(0).shuffle(mix)

    trainable = 0
    comp_lens = []
    for r in mix:
        assert len(r["tokens"]) == len(r["mask"]) <= TOKEN_LIMIT
        m = r["mask"]
        k = m.index(1)
        assert all(v == 0 for v in m[:k]) and all(v == 1 for v in m[k:])
        comp = len(m) - k
        assert comp <= GEN_LIMIT
        comp_lens.append(comp)
        trainable += comp
    with open(MIX_PATH, "w") as f:
        for r in mix:
            f.write(json.dumps(r) + "\n")
    comp_lens.sort()
    print(f"[mix] {len(mix)} 行 = 真新族 {len(nf)} + 合成 {len(synth_rows)} + "
          f"legacy {len(legacy)} + 非 bit {len(nonbit)};"
          f"trainable tokens {trainable:,}")
    print(f"[mix] completion p50={comp_lens[len(comp_lens)//2]} "
          f"p95={comp_lens[int(len(comp_lens)*0.95)]} max={comp_lens[-1]}")

    # decode 抽检:合成行 prompt 段以 <think>\n 收尾、completion 段收尾正确
    d = next(json.loads(l) for l in open(MIX_PATH)
             if json.loads(l)["problem_id"].startswith("synth_"))
    k = d["mask"].index(1)
    ptxt = tok.decode(d["tokens"][:k], skip_special_tokens=False)
    ctxt = tok.decode(d["tokens"][k:], skip_special_tokens=False)
    assert ptxt.endswith("<think>\n"), repr(ptxt[-40:])
    assert re.search(r"</think>\n\\boxed\{[01]{8}\}<\|im_end\|>$", ctxt), \
        repr(ctxt[-60:])
    print(f"[decode] synth 行抽检 ok ({d['problem_id']}): prompt 尾 "
          f"'<think>\\n',completion 尾 '</think>\\n\\boxed{{…}}<|im_end|>'")
    print(f"[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
