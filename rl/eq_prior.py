"""eq 运算符先验反学习(监督版 E2 复盘).

问题:equation_numeric_deduce 的解题器在多个 (rule, mode) 候选都与示例一致时,
用手写 _MODE_ORDER + _ORDER 做 tiebreak。E2 用另一个手工排序得 train +2 / holdout +0。
这里改为监督学习:用 train 真值统计出题器实际选了哪条规则(= 与示例 AND 真值都
一致的规则),学出 P(rule) 先验,按后验 argmax 重排 survivors。

隔离:先验只在 train_cat eq − holdout.csv eq (586 题) 上学;
评测在 run012 decode 的 110 题 holdout(模型基线 84/110)+ 全 146 题 holdout。

用法: python3 rl/eq_prior.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.problems import load_problems  # noqa: E402
from src.reasoners.equation_numeric_deduce import (  # noqa: E402
    _MODE_ORDER, _detect_fmt, _parse, _pick_survivor, _reattach, _survivors,
    _value,
)

DECODE = ROOT / "data/run012_holdout_decode.jsonl"
HOLDOUT_CSV = ROOT / "data/holdout.csv"
TRAIN_CSV = ROOT / "data/train_cat.csv"


# ---------------------------------------------------------------- per-problem
class Item:
    """One eq problem with its full survivor analysis."""

    def __init__(self, prob):
        self.id = prob.id
        self.truth = str(prob.answer)
        q = _parse(str(prob.question))
        self.qa, self.qop, self.qb = q
        by_op = defaultdict(list)
        for ex in prob.examples:
            p = _parse(str(ex.input_value))
            by_op[p[1]].append((p[0], p[2], str(ex.output_value)))
        self.group = by_op.get(self.qop, [])
        self.n_ex = len(self.group)
        self.correct_set: set = set()
        if not self.group:
            self.kind = "qop_unseen"
            self.survivors, self.answers = [], {}
            return
        self.fmt, self.tmap = _detect_fmt(self.qop, self.group)
        self.survivors = _survivors(self.group, self.tmap)
        if not self.survivors:
            self.kind = "no_survivor"
            self.answers = {}
            return
        # query answer for every survivor (after sign reattach)
        self.answers = {}
        for s in self.survivors:
            v = _value(s[0], self.qa, self.qb, s[1], s[2])
            self.answers[s] = None if v is None else _reattach(v, self.fmt, self.qop)
        self.correct_set = {s for s, a in self.answers.items() if a == self.truth}
        n_ans = len({a for a in self.answers.values() if a is not None})
        self.kind = "decisive" if n_ans > 1 else "single_answer"
        # baseline pick = production solver tiebreak
        top, _ = _pick_survivor(self.survivors, self.group, self.tmap, self.qa, self.qb)
        self.base_pick = top

    def base_correct(self) -> bool:
        return bool(self.survivors) and self.answers.get(self.base_pick) == self.truth

    def ceiling(self) -> bool:
        return bool(self.survivors) and bool(self.correct_set)


def rule_key(s, bucket: str = "rule"):  # s = (name, ro, rr)
    name, ro, rr = s
    fam = name.split("%")[0] if "%" in name else name
    if bucket == "rule":
        return (name, ro, rr)
    if bucket == "fam_mode":
        return (fam, ro, rr)
    if bucket == "name":
        return name
    raise ValueError(bucket)


# ----------------------------------------------------------------- evaluation
def pick_with_prior(it: Item, prior: dict, bucket: str, base_order: dict):
    """argmax prior among survivors; tie / unseen-rule -> production order."""
    best, best_score = None, None
    for s in it.survivors:
        if it.answers.get(s) is None:
            continue
        sc = prior.get(rule_key(s, bucket), 0.0)
        key = (-sc, base_order[s])
        if best is None or key < best_score:
            best, best_score = s, key
    return best if best is not None else it.base_pick


def evaluate(items, prior, bucket):
    n_ok = 0
    for it in items:
        if not it.survivors:
            continue
        order = {s: i for i, s in enumerate(it.survivors)}
        pick = pick_with_prior(it, prior, bucket, order)
        if it.answers.get(pick) == it.truth:
            n_ok += 1
    return n_ok


def base_score(items):
    return sum(it.base_correct() for it in items)


def ceiling_score(items):
    return sum(it.ceiling() for it in items)


# ----------------------------------------------------------------------- main
def main() -> None:
    probs = load_problems(TRAIN_CSV)
    eq = [p for p in probs if p.category == "equation_numeric_deduce"]
    ho_ids_all = set(pd.read_csv(HOLDOUT_CSV).query("cat=='equation_numeric_deduce'")["id"])
    dec = [json.loads(l) for l in DECODE.open()]
    dec_eq = {d["id"]: d for d in dec if d["cat"] == "equation_numeric_deduce"}
    ho110_ids = set(dec_eq)

    items = {p.id: Item(p) for p in eq}
    train = [items[p.id] for p in eq if p.id not in ho_ids_all]          # 586
    ho110 = [items[i] for i in ho110_ids]                                # 110
    ho146 = [items[p.id] for p in eq if p.id in ho_ids_all]              # 146

    for tag, S in [("train586", train), ("holdout110", ho110), ("holdout146", ho146)]:
        kinds = Counter(it.kind for it in S)
        print(f"[{tag}] n={len(S)} kinds={dict(kinds)} "
              f"base_oracle={base_score(S)} ceiling={ceiling_score(S)}")

    # ---- 出题器规则分布(与示例 AND 真值一致的规则;多解时分数计 1/k)----
    gen_counts: Counter = Counter()
    gen_hard: Counter = Counter()
    for it in train:
        if not it.survivors or not it.correct_set:
            continue
        k = len(it.correct_set)
        for s in it.correct_set:
            gen_counts[rule_key(s)] += 1.0 / k
            gen_hard[rule_key(s)] += 1
    print("\n=== 出题器规则分布 top25 (train586, 分数计) ===")
    for r, c in gen_counts.most_common(25):
        print(f"  {r}: {c:.1f}")

    # ---- 诊断:op 字符 / 数值范围与真规则的关联(只看 train)----
    op_rule: Counter = Counter()
    for it in train:
        if it.correct_set:
            fams = {rule_key(s, 'fam_mode')[0] for s in it.correct_set}
            for f in fams:
                op_rule[(it.qop, f)] += 1
    op_tot = Counter()
    for (op, _f), c in op_rule.items():
        op_tot[op] += c
    print("\n=== op字符→规则family 倾斜 (top op chars) ===")
    for op, tot in op_tot.most_common(8):
        tops = sorted(((c, f) for (o, f), c in op_rule.items() if o == op),
                      reverse=True)[:3]
        print(f"  '{op}' n={tot}: " + ", ".join(f"{f}={c}" for c, f in tops))

    # ---- 先验变体 ----
    # P1: 全局规则频率 (rule,mode) 分数计
    p1 = dict(gen_counts)
    # P2: family+mode 桶(更粗,抗过拟合)
    p2: Counter = Counter()
    for it in train:
        if it.correct_set:
            k = len(it.correct_set)
            for s in it.correct_set:
                p2[rule_key(s, "fam_mode")] += 1.0 / k
    # P3: 精度先验 = wins / survives(Laplace 平滑),只在 decisive 题上统计
    surv_cnt: Counter = Counter()
    win_cnt: Counter = Counter()
    for it in train:
        if it.kind != "decisive":
            continue
        for s in it.survivors:
            surv_cnt[rule_key(s)] += 1
            if s in it.correct_set:
                win_cnt[rule_key(s)] += 1
    p3 = {r: (win_cnt[r] + 1.0) / (surv_cnt[r] + 2.0) for r in surv_cnt}
    # P4: P1 但按 n_ex==1 / n_ex>=2 分桶
    p4_buckets = {True: Counter(), False: Counter()}
    for it in train:
        if it.correct_set:
            k = len(it.correct_set)
            for s in it.correct_set:
                p4_buckets[it.n_ex == 1][rule_key(s)] += 1.0 / k

    def eval_p4(items_):
        n_ok = 0
        for it in items_:
            if not it.survivors:
                continue
            order = {s: i for i, s in enumerate(it.survivors)}
            pick = pick_with_prior(it, p4_buckets[it.n_ex == 1], "rule", order)
            n_ok += it.answers.get(pick) == it.truth
        return n_ok

    # P5: 精确 op 字符桶(bucket_n>=15 才启用,否则回退全局 P1)—— 唯一 holdout 正向变体
    p5_char: dict = defaultdict(Counter)
    for it in train:
        if it.correct_set:
            k = len(it.correct_set)
            for s in it.correct_set:
                p5_char[it.qop][rule_key(s)] += 1.0 / k

    def eval_p5(items_, char_buckets, glob, min_n=15):
        n_ok = 0
        for it in items_:
            if not it.survivors:
                continue
            order = {s: i for i, s in enumerate(it.survivors)}
            pri = (char_buckets[it.qop]
                   if sum(char_buckets[it.qop].values()) >= min_n else glob)
            pick = pick_with_prior(it, pri, "rule", order)
            n_ok += it.answers.get(pick) == it.truth
        return n_ok

    print("\n=== tiebreak 对比 (oracle 道数) ===")
    rows = []
    for tag, S in [("train586", train), ("holdout110", ho110), ("holdout146", ho146)]:
        b, c = base_score(S), ceiling_score(S)
        r1 = evaluate(S, p1, "rule")
        r2 = evaluate(S, p2, "fam_mode")
        r3 = evaluate(S, p3, "rule")
        r4 = eval_p4(S)
        r5 = eval_p5(S, p5_char, p1)
        rows.append((tag, len(S), b, r1, r2, r3, r4, r5, c))
        print(f"  {tag:11s} n={len(S):3d} base={b:3d} P1_rulefreq={r1:3d} "
              f"P2_fammode={r2:3d} P3_precision={r3:3d} P4_nex_bucket={r4:3d} "
              f"P5_opchar={r5:3d} ceil={c:3d}")

    # ---- holdout110 上与模型 84/110 的交集:新先验能否救模型丢的题 ----
    model_wrong = {i for i, d in dec_eq.items() if not d["ok"]}
    print(f"\n模型 holdout110: 84 对, {len(model_wrong)} 错")
    for name_, prior_, bucket_ in [("base", None, None), ("P1", p1, "rule")]:
        saved = []
        for i in sorted(model_wrong):
            it = items[i]
            if not it.survivors:
                continue
            order = {s: i2 for i2, s in enumerate(it.survivors)}
            pick = (it.base_pick if prior_ is None
                    else pick_with_prior(it, prior_, bucket_, order))
            if it.answers.get(pick) == it.truth:
                saved.append(i)
        print(f"  {name_}: 解题器答对模型错题 {len(saved)}/{len(model_wrong)}")

    # ---- decisive 题上的逐题翻转明细 (holdout110, P1 vs base) ----
    print("\n=== holdout110 P1 vs base 翻转明细 ===")
    for it in ho110:
        if not it.survivors:
            continue
        order = {s: i for i, s in enumerate(it.survivors)}
        pick = pick_with_prior(it, p1, "rule", order)
        bc, nc = it.base_correct(), it.answers.get(pick) == it.truth
        if bc != nc:
            print(f"  {it.id} {'FIX' if nc else 'BREAK'} q={it.qa}{it.qop}{it.qb} "
                  f"n_ex={it.n_ex} base={it.base_pick}->{it.answers.get(it.base_pick)} "
                  f"new={pick}->{it.answers.get(pick)} truth={it.truth}")

    # ================== 补充 1:5-fold CV(只在 train586 内,估真泛化) ==================
    import random
    rng = random.Random(0)
    idx = list(range(len(train)))
    rng.shuffle(idx)
    folds = [idx[k::5] for k in range(5)]

    def learn_p1(sub):
        c: Counter = Counter()
        for it in sub:
            if it.correct_set:
                k = len(it.correct_set)
                for s in it.correct_set:
                    c[rule_key(s)] += 1.0 / k
        return dict(c)

    def learn_p3(sub):
        sc_, wc_ = Counter(), Counter()
        for it in sub:
            if it.kind != "decisive":
                continue
            for s in it.survivors:
                sc_[rule_key(s)] += 1
                if s in it.correct_set:
                    wc_[rule_key(s)] += 1
        return {r: (wc_[r] + 1.0) / (sc_[r] + 2.0) for r in sc_}

    print("\n=== 5-fold CV on train586 (估泛化, 不看 holdout) ===")
    cv = {"base": 0, "P1": 0, "P3": 0}
    for k in range(5):
        te = [train[i] for i in folds[k]]
        tr = [train[i] for j in range(5) if j != k for i in folds[j]]
        cv["base"] += base_score(te)
        cv["P1"] += evaluate(te, learn_p1(tr), "rule")
        cv["P3"] += evaluate(te, learn_p3(tr), "rule")
    print(f"  CV-sum/586: base={cv['base']} P1={cv['P1']} P3={cv['P3']}")

    # ================== 补充 2:任务点名分桶(op类型 / 数值范围 / 符号约定) ==================
    def op_class(ch: str) -> str:
        return "arith" if ch in "+-*/%^" else "symbol"

    def mag_bucket(it) -> str:
        try:
            return "small" if max(abs(int(it.qa)), abs(int(it.qb))) < 100 else "big"
        except ValueError:
            return "big"

    def ctx_key(it, mode: str):
        if mode == "opclass":
            return op_class(it.qop)
        if mode == "mag":
            return mag_bucket(it)
        if mode == "fmt":
            return getattr(it, "fmt", "num")
        raise ValueError(mode)

    def learn_ctx(sub, mode):
        buckets: dict = defaultdict(Counter)
        for it in sub:
            if it.correct_set:
                k = len(it.correct_set)
                for s in it.correct_set:
                    buckets[ctx_key(it, mode)][rule_key(s)] += 1.0 / k
        return buckets

    def eval_ctx(items_, buckets, mode):
        n_ok = 0
        for it in items_:
            if not it.survivors:
                continue
            order = {s: i for i, s in enumerate(it.survivors)}
            pick = pick_with_prior(it, buckets[ctx_key(it, mode)], "rule", order)
            n_ok += it.answers.get(pick) == it.truth
        return n_ok

    print("\n=== 分桶先验 (op类型/数值范围/符号约定) ===")
    for mode in ["opclass", "mag", "fmt"]:
        b_full = learn_ctx(train, mode)
        tr_s = eval_ctx(train, b_full, mode)
        ho_s = eval_ctx(ho110, b_full, mode)
        h146 = eval_ctx(ho146, b_full, mode)
        cv_s = 0
        for k in range(5):
            te = [train[i] for i in folds[k]]
            tr_ = [train[i] for j in range(5) if j != k for i in folds[j]]
            cv_s += eval_ctx(te, learn_ctx(tr_, mode), mode)
        print(f"  ctx={mode:8s} train={tr_s} (base 434)  CV={cv_s} (base {cv['base']})  "
              f"ho110={ho_s} (base 88)  ho146={h146} (base 113)")

    # ====== 补充 2b:P5 (op-char 桶) 的 CV / 翻转 / bootstrap 稳定性 ======
    def learn_char(sub):
        glob_, bk_ = Counter(), defaultdict(Counter)
        for it in sub:
            if it.correct_set:
                k = len(it.correct_set)
                for s in it.correct_set:
                    glob_[rule_key(s)] += 1.0 / k
                    bk_[it.qop][rule_key(s)] += 1.0 / k
        return dict(glob_), bk_

    cv5 = 0
    for k in range(5):
        te = [train[i] for i in folds[k]]
        tr_ = [train[i] for j in range(5) if j != k for i in folds[j]]
        g_, b_ = learn_char(tr_)
        cv5 += eval_p5(te, b_, g_)
    print(f"\nP5_opchar CV-sum/586: {cv5} (base {cv['base']})")

    def pick_p5(it):
        order = {s: i for i, s in enumerate(it.survivors)}
        pri = (p5_char[it.qop] if sum(p5_char[it.qop].values()) >= 15 else p1)
        return pick_with_prior(it, pri, "rule", order)

    flip_items = []
    for it in ho146:  # ho110 ⊂ ho146
        if it.survivors and (it.answers.get(pick_p5(it)) == it.truth) != it.base_correct():
            flip_items.append(it)
            d110 = "ho110+146" if it.id in ho110_ids else "ho146only"
            nc = it.answers.get(pick_p5(it)) == it.truth
            print(f"  P5 {'FIX' if nc else 'BREAK'} [{d110}] {it.id} q={it.qa}{it.qop}{it.qb} "
                  f"n_ex={it.n_ex} base={it.base_pick} new={pick_p5(it)} truth={it.truth}")
    import random as _rd
    stab = Counter()
    for t in range(200):
        rng_ = _rd.Random(1000 + t)
        samp = [train[rng_.randrange(len(train))] for _ in range(len(train))]
        g2, b2 = learn_char(samp)
        for it in flip_items:
            order = {s: i for i, s in enumerate(it.survivors)}
            pri = b2[it.qop] if sum(b2[it.qop].values()) >= 15 else g2
            pk = pick_with_prior(it, pri, "rule", order)
            if (it.answers.get(pk) == it.truth) == (it.answers.get(pick_p5(it)) == it.truth):
                stab[it.id] += 1
    for i, c in stab.items():
        print(f"  P5 翻转题 bootstrap 同向: {i} {c}/200")

    # ================== 补充 3:holdout110 headroom 解剖 (base 错但 ceiling 可达) ==================
    print("\n=== holdout110 headroom: base 错 / 正确规则在 survivors 里 ===")
    for it in ho110:
        if not it.survivors or it.base_correct() or not it.correct_set:
            continue
        n_in_train = {s: gen_hard.get(rule_key(s), 0) for s in it.correct_set}
        base_n = gen_hard.get(rule_key(it.base_pick), 0)
        print(f"  {it.id} q={it.qa}{it.qop}{it.qb} n_ex={it.n_ex} kind={it.kind} fmt={it.fmt}")
        print(f"    base_pick={it.base_pick} (train胜场={base_n}) -> {it.answers.get(it.base_pick)}")
        for s in sorted(it.correct_set):
            print(f"    correct: {s} (train胜场={n_in_train[s]}) -> {it.answers[s]}  truth={it.truth}")
        print(f"    survivors({len(it.survivors)}): {it.survivors[:8]}")


if __name__ == "__main__":
    main()
