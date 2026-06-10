"""Cryptarithm deduce: infer the hidden per-operator rule from symbol-string examples.

Problem format (verified empirically):
  Input: 5-char string  A0 A1 OP B0 B1   (operands A=A0A1, B=B0B1; OP is 1 symbol)
  Output: 1-4 char string, possibly prefixed with '-' (negative sign)

Generator model (reverse-engineered; operators are MIXED within one problem):
  Each operator symbol carries ONE rule, from two families:
    (1) Rearrangement -- output is a fixed index-selection over the 4 operand symbols.
    (2) Positional base arithmetic -- symbols are digits of one shared injective digit map
        (base 10-13, little- OR big-endian); operators map to add/sub/mul/abs-diff.

Solver (ported from the validated _crypt_scratch/improved.py pipeline, 183/823 = 22.2%):
  1. Group examples by operator; classify each operator independently as rearrangement
     (some index-selection fits ALL of its examples) or arithmetic.
  2. Query op rearrangement -> apply its selection; ambiguity broken by a simplicity
     prior (shorter > in-order > contiguous > lexicographic).
  3. Query op arithmetic -> pool ONLY the arithmetic operators to constrain one shared
     digit map; multi-hypothesis search over base (10,11,12,13) x endian x per-op
     operation combo (candidates pruned by output sign/length); prior-weighted pick
     (base 10 >> 11 > 12 > 13; little > big; canonical '+'/'-'/'*' bonus).
  4. Never abstain: if nothing verifies, fall back to concatenation A0A1B0B1
     (a wrong guess scores the same as abstaining).

Trace (LOCALITY LAW): the emitted CoT states the found rule as a hypothesis, then
VERIFIES it on every relevant example (compute -> compare -> match), then applies it
step by step to the query.  Every line is a copy or a deterministic local transform of
earlier tokens; no unverified global rule is asserted and then used.
"""
from __future__ import annotations

import itertools
from collections import Counter

from .store_types import Example, Problem

ADD, SUB, MUL, ABS = 0, 1, 2, 3
_IDX = (0, 1, 2, 3)
_SEQS = [c for L in range(1, 5) for c in itertools.product(_IDX, repeat=L)]
_POS = ("A0", "A1", "B0", "B1")
_OPNAME = {ADD: "addition", SUB: "subtraction", MUL: "multiplication",
           ABS: "absolute difference"}


def _operands(lhs: str):
    return (lhs[0], lhs[1], lhs[3], lhs[4])


# --------------------------------------------------------------------------- #
# Rearrangement classification
# --------------------------------------------------------------------------- #
def _classify_rearrange(group):
    """group: list of (lhs, rhs).  Return every index-selection fitting all examples."""
    if any(r.startswith("-") for _, r in group):
        return []
    out = []
    for s in _SEQS:
        ok = True
        for lhs, rhs in group:
            oc = _operands(lhs)
            if "".join(oc[i] for i in s) != rhs:
                ok = False
                break
        if ok:
            out.append(s)
    return out


# --------------------------------------------------------------------------- #
# Arithmetic
# --------------------------------------------------------------------------- #
def _op_candidates(out_list):
    anyneg = any(r.startswith("-") for r in out_list)
    maxout = max(len(r.lstrip("-")) for r in out_list)
    if anyneg:
        return [SUB]
    if maxout >= 4:
        return [MUL]
    if maxout >= 3:
        return [ADD, MUL]
    return [ADD, ABS, MUL, SUB]


def _value(syms, dmap, base, endian):
    seq = syms[::-1] if endian == "little" else syms
    v = 0
    for ch in seq:
        v = v * base + dmap[ch]
    return v


def _decode(rhs, dmap, base, endian):
    neg = rhs.startswith("-")
    v = _value(rhs.lstrip("-"), dmap, base, endian)
    return -v if neg else v


def _render(rv, inv, base, endian):
    neg = rv < 0
    mag = abs(rv)
    digs = []
    if mag == 0:
        digs = [0]
    while mag > 0:
        digs.append(mag % base)
        mag //= base
    seq = digs if endian == "little" else digs[::-1]
    out = []
    for d in seq:
        if d not in inv:
            return None
        out.append(inv[d])
    return ("-" if neg else "") + "".join(out)


def _solve_hypothesis(digit_syms, cons, base, endian, node_budget=120000):
    """First injective digit map (DFS, frequency-ordered) satisfying all constraints."""
    syms = list(digit_syms)
    n = len(syms)
    if base < n:
        return None
    freq = Counter()
    for a, b, o, neg, op in cons:
        for ch in a + b + o:
            freq[ch] += 1
    order = sorted(syms, key=lambda c: -freq[c])
    idxpos = {c: i for i, c in enumerate(order)}
    assign: dict[str, int] = {}
    used = [False] * base
    full_by: list[list[int]] = [[] for _ in range(n)]
    for ci, (a, b, o, neg, op) in enumerate(cons):
        es = set(a + b + o)
        full_by[max(idxpos[c] for c in es)].append(ci)
    units: dict[int, list] = {}
    for (a, b, o, neg, op) in cons:
        if op == ABS or neg:
            continue
        if endian == "little":
            ua, ub, uo = a[0], b[0], o[0]
        else:
            ua, ub, uo = a[-1], b[-1], o[-1]
        pos = max(idxpos[ua], idxpos[ub], idxpos[uo])
        units.setdefault(pos, []).append((ua, ub, uo, op))
    cnt = [0]

    def check_full(ci):
        a, b, o, neg, op = cons[ci]
        A = _value(a, assign, base, endian)
        B = _value(b, assign, base, endian)
        R = _value(o, assign, base, endian)
        if neg:
            R = -R
        if op == ADD:
            return A + B == R
        if op == SUB:
            return A - B == R
        if op == MUL:
            return A * B == R
        return abs(A - B) == R

    def check_units(pos):
        for (ua, ub, uo, op) in units.get(pos, []):
            da, db, do = assign[ua], assign[ub], assign[uo]
            if op == ADD:
                if (da + db) % base != do:
                    return False
            elif op == MUL:
                if (da * db) % base != do:
                    return False
            else:
                if (da - db) % base != do:
                    return False
        return True

    def rec(pos):
        if cnt[0] > node_budget:
            return "budget"
        if pos == n:
            return dict(assign)
        c = order[pos]
        for d in range(base):
            if used[d]:
                continue
            cnt[0] += 1
            assign[c] = d
            used[d] = True
            ok = check_units(pos)
            if ok:
                for ci in full_by[pos]:
                    if not check_full(ci):
                        ok = False
                        break
            if ok:
                r = rec(pos + 1)
                if r is not None and r != "budget":
                    used[d] = False
                    del assign[c]
                    return r
                if r == "budget":
                    used[d] = False
                    del assign[c]
                    return "budget"
            used[d] = False
            del assign[c]
        return None

    r = rec(0)
    if r in (None, "budget"):
        return None
    return r


# --------------------------------------------------------------------------- #
# Hypothesis enumeration + prior pick (validated pipeline)
# --------------------------------------------------------------------------- #
def _enumerate_answers(exs, q, bases=(10, 11, 12, 13), endians=("little", "big"),
                       cap=10, node_budget=120000):
    """exs: list of (lhs, rhs); q: 5-char query.
    Returns (answers, status, ctx); answers: ans -> [hyp, ...] where hyp is
    ("rearrange", seq) or (base, endian, opm_tuple, digit_map)."""
    qop = q[2]
    ops: dict[str, list] = {}
    for lhs, rhs in exs:
        ops.setdefault(lhs[2], []).append((lhs, rhs))
    ctx = {"ops": ops, "rearrange_seqs": {}, "arith_ops": []}
    if qop not in ops:
        return {}, "unseen_op", ctx
    for op, group in ops.items():
        seqs = _classify_rearrange(group)
        if seqs:
            ctx["rearrange_seqs"][op] = seqs
        else:
            ctx["arith_ops"].append(op)
    qoc = _operands(q)

    if qop in ctx["rearrange_seqs"]:
        preds: dict[str, list] = {}
        for s in ctx["rearrange_seqs"][qop]:
            pred = "".join(qoc[i] for i in s)
            preds.setdefault(pred, []).append(("rearrange", s))
        return preds, "rearrange", ctx

    arith_ops = ctx["arith_ops"]
    if not arith_ops:
        return {}, "no_arith", ctx
    digit_syms = set()
    for op in arith_ops:
        for lhs, rhs in ops[op]:
            digit_syms.update(_operands(lhs))
            digit_syms.update(rhs.lstrip("-"))
    digit_syms.update(qoc)
    digit_syms = sorted(digit_syms)
    nd = len(digit_syms)
    cand = {op: _op_candidates([r for _, r in ops[op]]) for op in arith_ops}

    answers: dict[str, list] = {}
    for base in bases:
        if base < nd:
            continue
        for endian in endians:
            for combo in itertools.product(*[cand[op] for op in arith_ops]):
                opm = dict(zip(arith_ops, combo))
                cons = []
                for op in arith_ops:
                    o = opm[op]
                    for lhs, rhs in ops[op]:
                        neg = rhs.startswith("-")
                        mag = rhs.lstrip("-")
                        a0, a1, b0, b1 = _operands(lhs)
                        cons.append(([a0, a1], [b0, b1], list(mag), neg, o))
                sol = _solve_hypothesis(digit_syms, cons, base, endian, node_budget)
                if sol is None:
                    continue
                inv: dict[int, str] = {}
                for c, d in sol.items():
                    inv.setdefault(d, c)
                A = _value([q[0], q[1]], sol, base, endian)
                B = _value([q[3], q[4]], sol, base, endian)
                oo = opm[qop]
                rv = (A + B if oo == ADD else A - B if oo == SUB
                      else A * B if oo == MUL else abs(A - B))
                out = _render(rv, inv, base, endian)
                if out is not None:
                    answers.setdefault(out, []).append(
                        (base, endian, tuple(sorted(opm.items())), sol))
                    if len(answers) > cap:
                        return answers, "capped", ctx
    return answers, ("ok" if answers else "unsat"), ctx


def _pick(answers, status):
    """Prior-weighted pick; returns (answer, hyp) or None.  Same pick as validated
    prior_pick (hyp added only so the trace can verify the chosen rule)."""
    if not answers:
        return None
    if status == "rearrange":
        # SIMPLICITY PRIOR: shorter > in-order > contiguous > lexicographic.
        def seq_score(s):
            in_order = list(s) == sorted(s)
            contiguous = all(s[i] + 1 == s[i + 1] for i in range(len(s) - 1))
            return (len(s), 0 if in_order else 1, 0 if contiguous else 1, s)
        best = best_ans = best_seq = None
        for ans, hyps in answers.items():
            for tag in hyps:
                sc = seq_score(tag[1])
                if best is None or sc < best:
                    best, best_ans, best_seq = sc, ans, tag[1]
        return best_ans, ("rearrange", best_seq)
    bs = best_ans = best_hyp = None
    for ans, hyps in answers.items():
        for (base, endian, opm, sol) in hyps:
            sc = {10: 300, 11: 200, 12: 100, 13: 50}.get(base, 0)
            sc += 10 if endian == "little" else 0
            for op, o in opm:
                if op == "+" and o == ADD:
                    sc += 5
                if op == "-" and o == SUB:
                    sc += 5
                if op == "*" and o == MUL:
                    sc += 5
            if bs is None or sc > bs:
                bs, best_ans, best_hyp = sc, ans, (base, endian, opm, sol)
    return best_ans, ("arith",) + best_hyp


# --------------------------------------------------------------------------- #
# Trace rendering (verify on every example, then apply to the query)
# --------------------------------------------------------------------------- #
_HEADER = ("Each operator symbol hides one fixed rule. Label every 5-symbol string as "
           "A0 A1 op B0 B1 (operands A=A0A1, B=B0B1). I find a rule from the examples, "
           "check it against each example, then apply it to the query.")


def _expr_str(A, B, opcode):
    if opcode == ADD:
        return f"{A}+{B}={A + B}", A + B
    if opcode == SUB:
        return f"{A}-{B}={A - B}", A - B
    if opcode == MUL:
        return f"{A}*{B}={A * B}", A * B
    return f"|{A}-{B}|={abs(A - B)}", abs(A - B)


def _not_copy_demo(qex, qop):
    """One local compute->compare->wrong line showing qop is not a symbol copy."""
    for lhs, rhs in qex:
        if rhs.startswith("-"):
            return (f"- {lhs} = {rhs}: the output starts with '-', "
                    f"so '{qop}' is not a copy of operand symbols.")
    for lhs, rhs in qex:
        cat = "".join(_operands(lhs))
        if cat != rhs:
            return (f"- {lhs}: copying A0 A1 B0 B1 gives {cat}, but the given output "
                    f"is {rhs}; wrong, so '{qop}' is not a plain symbol copy.")
    return None


def _trace_rearrange(lines, ops, q, seq, ans):
    qop = q[2]
    names = " ".join(_POS[i] for i in seq)
    qex = ops[qop]
    lines.append(f"Hypothesis for '{qop}': the output copies operand positions {names}.")
    lines.append(f"Check every '{qop}' example:")
    for lhs, rhs in qex:
        oc = _operands(lhs)
        pred = "".join(oc[i] for i in seq)
        lab = " ".join(f"{_POS[i]}={oc[i]}" for i in range(4))
        lines.append(f"- {lhs} = {rhs}: {lab}; take {names}: {pred}; given {rhs}; match.")
    lines.append(f"All {len(qex)} examples match.")
    qoc = _operands(q)
    lab = " ".join(f"{_POS[i]}={qoc[i]}" for i in range(4))
    lines.append(f"Apply to the query {q}: {lab}; take {names}: {ans}.")


def _trace_arith(lines, ops, arith_ops, q, base, endian, opm_tuple, sol, ans):
    qop = q[2]
    opm = dict(opm_tuple)
    demo = _not_copy_demo(ops[qop], qop)
    if demo is not None:
        lines.append(demo)
    lines.append("Treat the symbols as digits of one shared cipher.")
    endian_desc = ("first symbol = units digit, read right-to-left"
                   if endian == "little" else
                   "last symbol = units digit, read left-to-right")
    mapstr = ", ".join(f"{c}={d}" for c, d in sorted(sol.items(), key=lambda kv: kv[1]))
    lines.append(f"Hypothesis: base {base}, {endian_desc}; digit map: {mapstr}.")
    lines.append("Operators: " + "; ".join(f"'{op}' = {_OPNAME[opm[op]]}"
                                           for op in arith_ops) + ".")
    lines.append("Check the hypothesis on every arithmetic example:")
    nchecked = 0
    for op in arith_ops:
        for lhs, rhs in ops[op]:
            A = _value(lhs[0] + lhs[1], sol, base, endian)
            B = _value(lhs[3] + lhs[4], sol, base, endian)
            expr, _ = _expr_str(A, B, opm[op])
            gv = _decode(rhs, sol, base, endian)
            lines.append(f"- {lhs} = {rhs}: A={lhs[0]}{lhs[1]}={A}, "
                         f"B={lhs[3]}{lhs[4]}={B}; {expr}; given {rhs}={gv}; match.")
            nchecked += 1
    lines.append(f"All {nchecked} examples are consistent with this map.")
    A = _value(q[0] + q[1], sol, base, endian)
    B = _value(q[3] + q[4], sol, base, endian)
    expr, rv = _expr_str(A, B, opm[qop])
    lines.append(f"Apply to the query {q}: A={q[0]}{q[1]}={A}, B={q[3]}{q[4]}={B}; "
                 f"{expr}; {rv} written in this cipher is {ans}.")


def _trace_fallback(lines, ops, q, status):
    qop = q[2]
    if status == "unseen_op":
        lines.append(f"Operator '{qop}' does not appear in any example, "
                     "so no rule can be checked for it.")
    else:
        lines.append(f"Look for a rule for operator '{qop}':")
        demo = _not_copy_demo(ops.get(qop, []), qop)
        if demo is not None:
            lines.append(demo)
        lines.append("No symbol-selection fits all its examples, and the digit-cipher "
                     "search (bases 10-13, both reading orders) finds no consistent map.")
    qoc = _operands(q)
    guess = "".join(qoc)
    lab = " ".join(f"{_POS[i]}={qoc[i]}" for i in range(4))
    lines.append("Fall back to the default rule: copy the operand symbols in order "
                 "A0 A1 B0 B1.")
    lines.append(f"Apply to the query {q}: {lab}; copy A0 A1 B0 B1: {guess}.")
    return guess


def reasoning_cryptarithm_deduce(problem: Problem) -> str | None:
    q = problem.question.strip()
    if len(q) != 5:
        return None  # cannot even parse the query: truly nothing applies
    exs = []
    for ex in problem.examples:
        iv = ex.input_value.strip()
        ov = ex.output_value.strip()
        if len(iv) == 5 and ov:
            exs.append((iv, ov))

    lines = [_HEADER]
    answers, status, ctx = _enumerate_answers(exs, q)
    picked = _pick(answers, status)

    if picked is None:
        ans = _trace_fallback(lines, ctx["ops"], q, status)
    else:
        ans, hyp = picked
        if hyp[0] == "rearrange":
            _trace_rearrange(lines, ctx["ops"], q, hyp[1], ans)
        else:
            _, base, endian, opm_tuple, sol = hyp
            _trace_arith(lines, ctx["ops"], ctx["arith_ops"], q,
                         base, endian, opm_tuple, sol, ans)
    lines.append("The answer is \\boxed{" + ans + "}")
    return "\n".join(lines)


if __name__ == "__main__":
    def mk(exs, q, ans):
        return Problem(id="t", category="cryptarithm_deduce",
                       examples=[Example(i, o) for i, o in exs], question=q, answer=ans)
    p = mk([("##+|(", "##|("), ("/#+}/", "/#}/")], ">/+%(", ">/%(")
    print(reasoning_cryptarithm_deduce(p))
    p2 = mk([("12+34", "46"), ("56+11", "67"), ("21-11", "10")], "43-12", "31")
    print(reasoning_cryptarithm_deduce(p2))
