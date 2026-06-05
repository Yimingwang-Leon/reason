"""Cryptarithm deduce: infer the hidden per-operator rule from symbol-string examples.

Problem format (verified empirically):
  Input: 5-char string  A0 A1 OP B0 B1   (operands A=A0A1, B=B0B1; OP is 1 symbol)
  Output: 1-4 char string, possibly prefixed with '-' (negative sign)

Generator model (reverse-engineered):
  Each PROBLEM defines, per operator symbol, ONE transformation.  Two families cover the
  scoreable problems:
    (1) Rearrangement / set ops  -- output is a selection/permutation of the 4 operand
        symbols (e.g. concatenation A+B), or a set difference / intersection.
    (2) Positional base arithmetic -- each symbol is a digit (per-problem digit map, a
        bijection at a small base, base ~10-12), little-endian OR big-endian.  Operators map
        to {add, multiply, signed-subtract, absolute-difference}.  '-' uniquely yields a
        signed (negative) result; other operators stay non-negative.
  The digit map is shared across all arithmetic operators of a problem, so we pool every
  arithmetic example to constrain it, while skipping operators that are pure rearrangements
  (otherwise one concat operator would break the arithmetic fit).

Solver: try rearrangement/set first; then little-endian arithmetic; then big-endian.
Return a concatenation A+B fallback when no consistent rule is found (a wrong guess scores
the same as abstaining, so guessing is pure upside).
"""
from __future__ import annotations

import itertools
import re
from collections import Counter

from .store_types import Example, Problem

_PARSE_RE = re.compile(r"^(.)(.)(.)(.)(.)$")

_IDX = (0, 1, 2, 3)
_SEQS = [c for L in range(1, 5) for c in itertools.product(_IDX, repeat=L)]


def _parse(s: str):
    s = s.strip()
    m = _PARSE_RE.match(s)
    if not m:
        return None
    a0, a1, op, b0, b1 = m.groups()
    return a0, a1, op, b0, b1


# --------------------------------------------------------------------------- #
# Rearrangement / set rules
# --------------------------------------------------------------------------- #
def _is_rearrange(ex):
    """ex: list of (a0,a1,b0,b1,out,neg).  Return matching index-seqs (no negatives)."""
    if any(neg for *_, neg in ex):
        return []
    return [
        s for s in _SEQS
        if all("".join((a0, a1, b0, b1)[i] for i in s) == o
               for a0, a1, b0, b1, o, neg in ex)
    ]


def _set_rule(ex):
    for nm in ("A-B", "B-A", "A&B"):
        def ap(a0, a1, b0, b1, nm=nm):
            A, B = (a0, a1), (b0, b1)
            if nm == "A-B":
                return "".join(c for c in A if c not in B)
            if nm == "B-A":
                return "".join(c for c in B if c not in A)
            return "".join(c for c in A if c in B)
        if all(ap(a0, a1, b0, b1) == o for a0, a1, b0, b1, o, neg in ex):
            return nm
    return None


# --------------------------------------------------------------------------- #
# Arithmetic
# op codes: 0=add, 1=mul, 2=signed-sub (a-b), 3=abs-diff |a-b|
# --------------------------------------------------------------------------- #
def _op_candidates(ex):
    if any(neg for *_, neg in ex):
        return [2]
    maxlen = max(len(o) for *_, o, neg in ex)
    if maxlen >= 4:
        return [1]
    if maxlen >= 3:
        return [0, 1]
    return [0, 3, 1]


def _dfs(order, elist, base, endian, node_budget):
    n = len(order)
    idxpos = {c: i for i, c in enumerate(order)}
    assign: dict[str, int] = {}
    used = [False] * base

    exsyms = [set(a0 + a1 + b0 + b1 + o) for a0, a1, b0, b1, o, neg, op in elist]
    full_by = [[] for _ in range(n)]
    for i, es in enumerate(exsyms):
        full_by[max(idxpos[c] for c in es)].append(i)

    # units-column constraint (skip abs-diff)
    units_ready: dict[int, list] = {}
    for a0, a1, b0, b1, o, neg, op in elist:
        if op in (0, 1, 2) and not neg:
            if endian == "little":
                ua, ub, uo = a0, b0, o[0]
            else:
                ua, ub, uo = a1, b1, o[-1]
            r = max(idxpos[ua], idxpos[ub], idxpos[uo])
            units_ready.setdefault(r, []).append((ua, ub, uo, op))

    cnt = [0]

    def val(s):
        v = 0
        seq = s[::-1] if endian == "little" else s
        for ch in seq:
            v = v * base + assign[ch]
        return v

    def check_full(i):
        a0, a1, b0, b1, o, neg, op = elist[i]
        A = val(a0 + a1); B = val(b0 + b1); R = val(o); R = -R if neg else R
        if op == 0:
            return A + B == R
        if op == 1:
            return A * B == R
        if op == 2:
            return A - B == R
        return abs(A - B) == R

    def check_units(pos):
        for (ua, ub, uo, op) in units_ready.get(pos, []):
            da, db, do = assign[ua], assign[ub], assign[uo]
            if op == 0:
                if (da + db) % base != do:
                    return False
            elif op == 1:
                if (da * db) % base != do:
                    return False
            else:
                if (da - db) % base != do:
                    return False
        return True

    def rec(pos):
        if cnt[0] > node_budget:
            return False
        if pos == n:
            return dict(assign)
        c = order[pos]
        for d in range(base):
            if used[d]:
                continue
            cnt[0] += 1
            assign[c] = d; used[d] = True
            ok = check_units(pos)
            if ok:
                for ei in full_by[pos]:
                    if not check_full(ei):
                        ok = False
                        break
            if ok:
                r = rec(pos + 1)
                if isinstance(r, dict):
                    return r
                if r is False:
                    used[d] = False; del assign[c]
                    return False
            used[d] = False; del assign[c]
        return None

    r = rec(0)
    return r if isinstance(r, dict) else None


def _render(rv, sol, base, endian):
    inv: dict[int, str] = {}
    for c, d in sol.items():
        inv.setdefault(d, c)
    neg = rv < 0
    mag = abs(rv)
    digs = []
    if mag == 0:
        digs = [0]
    while mag > 0:
        digs.append(mag % base)
        mag //= base
    seq = digs if endian == "little" else digs[::-1]
    try:
        return ("-" if neg else "") + "".join(inv[d] for d in seq)
    except KeyError:
        return None


def _value(s, sol, base, endian):
    v = 0
    seq = s[::-1] if endian == "little" else s
    for ch in seq:
        v = v * base + sol[ch]
    return v


def _arith(byop, q, endian, base_list=(10, 11, 12), node_budget=400000):
    arith_ops = [op for op in byop if not _is_rearrange(byop[op])]
    if q[2] not in arith_ops:
        return None
    pooled = []
    syms = set()
    for op in arith_ops:
        for a0, a1, b0, b1, o, neg in byop[op]:
            syms.update([a0, a1, b0, b1]); syms.update(o)
            pooled.append((a0, a1, b0, b1, o, neg, op))
    syms.update([q[0], q[1], q[3], q[4]])
    exonly = sorted({c for tup in pooled for c in (tup[0] + tup[1] + tup[2] + tup[3] + tup[4])})
    nd = len(exonly)
    cand = {op: _op_candidates(byop[op]) for op in arith_ops}
    for base in base_list:
        if base < nd:
            continue
        freq = Counter()
        for a0, a1, b0, b1, o, neg, op in pooled:
            for c in a0 + a1 + b0 + b1 + o:
                freq[c] += 1
        order = sorted(exonly, key=lambda c: -freq[c])
        for combo in itertools.product(*[cand[op] for op in arith_ops]):
            opm = dict(zip(arith_ops, combo))
            elist = [(a0, a1, b0, b1, o, neg, opm[op]) for a0, a1, b0, b1, o, neg, op in pooled]
            sol = _dfs(order, elist, base, endian, node_budget)
            if sol:
                if any(c not in sol for c in [q[0], q[1], q[3], q[4]]):
                    return None
                A = _value(q[0] + q[1], sol, base, endian)
                B = _value(q[3] + q[4], sol, base, endian)
                o = opm[q[2]]
                rv = (A + B if o == 0 else A * B if o == 1 else A - B if o == 2 else abs(A - B))
                out = _render(rv, sol, base, endian)
                if out is not None:
                    return out, base, endian, opm, sol
    return None


def _solve(exs, q):
    """exs: list of (a0,a1,op,b0,b1, out_string).  Returns (answer, explanation) or None."""
    byop: dict[str, list] = {}
    for a0, a1, op, b0, b1, out in exs:
        bb = out[1:] if out.startswith("-") else out
        byop.setdefault(op, []).append((a0, a1, b0, b1, bb, out.startswith("-")))
    if q[2] not in byop:
        return None

    qex = byop[q[2]]
    # rearrangement
    seqs = _is_rearrange(qex)
    if seqs:
        preds = {"".join((q[0], q[1], q[3], q[4])[i] for i in s) for s in seqs}
        if len(preds) == 1:
            return next(iter(preds)), ("rearrange", seqs[0])
    # set rules
    if not any(neg for *_, neg in qex):
        sr = _set_rule(qex)
        if sr is not None:
            A, B = (q[0], q[1]), (q[3], q[4])
            if sr == "A-B":
                ans = "".join(c for c in A if c not in B)
            elif sr == "B-A":
                ans = "".join(c for c in B if c not in A)
            else:
                ans = "".join(c for c in A if c in B)
            return ans, ("set", sr)
    # arithmetic, little then big
    for endian in ("little", "big"):
        r = _arith(byop, q, endian)
        if r is not None:
            return r[0], ("arith", r[1], r[2], r[3])
    return None


_OPNAME = {0: "addition", 1: "multiplication", 2: "subtraction", 3: "absolute difference"}


def reasoning_cryptarithm_deduce(problem: Problem) -> str | None:
    lines: list[str] = []
    lines.append("Each operator hides a fixed rule; I deduce it from the examples.")

    exs = []
    for ex in problem.examples:
        p = _parse(ex.input_value)
        if p is None:
            continue  # skip unparseable example, keep going
        exs.append((*p, ex.output_value))
    q = _parse(problem.question)
    if q is None:
        return None  # cannot form an answer without a parseable query
    if not exs:
        # No usable examples: fall back to concatenation A+B.
        guess = q[0] + q[1] + q[3] + q[4]
        lines.append(f"Query: {q[0]+q[1]} {q[2]} {q[3]+q[4]}")
        lines.append("No usable examples; defaulting to concatenation A+B.")
        lines.append(f"The answer is \\boxed{{{guess}}}")
        return "\n".join(lines)

    # group display
    byop_disp: dict[str, list] = {}
    for a0, a1, op, b0, b1, out in exs:
        byop_disp.setdefault(op, []).append((a0 + a1, b0 + b1, out))
    for op, lst in byop_disp.items():
        lines.append(f"Operator '{op}':")
        for a, b, out in lst[:6]:
            lines.append(f"  {a} {op} {b} = {out!r}")

    result = _solve(exs, q)
    qa, qb = q[0] + q[1], q[3] + q[4]
    lines.append(f"Query: {qa} {q[2]} {qb}")

    if result is None:
        # Fallback: concatenation A+B is the single most common rule; guess it.
        guess = q[0] + q[1] + q[3] + q[4]
        lines.append("No exact rule confirmed; defaulting to concatenation A+B.")
        lines.append(f"The answer is \\boxed{{{guess}}}")
        return "\n".join(lines)

    ans, info = result
    kind = info[0]
    if kind == "rearrange":
        seq = info[1]
        picks = "".join({0: "A0", 1: "A1", 2: "B0", 3: "B1"}[i] for i in seq)
        lines.append(f"The rule rearranges operand symbols as {picks}.")
    elif kind == "set":
        lines.append(f"The rule is the set operation {info[1]}.")
    else:
        base, endian, opm = info[1], info[2], info[3]
        lines.append(
            f"The symbols encode digits in base {base} ({endian}-endian); "
            f"operator '{q[2]}' is {_OPNAME[opm[q[2]]]}."
        )
    lines.append(f"Applying the rule to {qa} {q[2]} {qb} gives {ans!r}.")
    lines.append(f"The answer is \\boxed{{{ans}}}")
    return "\n".join(lines)


if __name__ == "__main__":
    def mk(exs, q, ans):
        return Problem(id="t", category="cryptarithm_deduce",
                       examples=[Example(i, o) for i, o in exs], question=q, answer=ans)
    p = mk([("##+|(", "##|("), ("/#+}/", "/#}/")], ">/+%(", ">/%(")
    print(reasoning_cryptarithm_deduce(p))
