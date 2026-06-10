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

Trace (FORWARD FAIL-FAST ENUMERATION, run-011 R-fix): the old hypothesis-first
format ("Hypothesis: digit map A=3,B=7...") trained to NLL~0.002 but reproduced at
3% greedy -- the map was pulled from thin air.  The emitter now mirrors
equation_numeric_deduce's enumerative transcript (measured 100% greedy
reproduction): candidates are scanned in the solver's priority order
(rearrangement selections first, then base x endian x op-meaning ciphers); each
candidate's consequences are derived ON THE PAGE and the candidate is discarded at
its first contradiction; the winner's digit map is BUILT line by line (forced
column pins + narrated lexicographic search identical to the solver's DFS), every
example is re-checked, and the query is decoded/encoded symbol by symbol.
LOCALITY LAW: no line contains a fact that is not derivable from earlier printed
tokens by copy or one local operation.  Over-budget traces compress only the
DISCARDED candidates (abbreviation, then single-witness chains); the winner's
derivation is never compressed.  Unsolved problems keep the legacy fallback trace
byte-for-byte (wrong answers never enter the corpus).
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
                       cap=10, node_budget=120000, rec=None):
    """exs: list of (lhs, rhs); q: 5-char query.
    Returns (answers, status, ctx); answers: ans -> [hyp, ...] where hyp is
    ("rearrange", seq) or (base, endian, opm_tuple, digit_map).
    rec (emitter instrumentation, no behavior change): when a list is passed, every
    tried arithmetic candidate is appended as (base, endian, opm_tuple, sol|None,
    out|None) in enumeration order, so the trace can replay the scan."""
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
                    if rec is not None:
                        rec.append((base, endian, tuple(sorted(opm.items())),
                                    None, None))
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
                if rec is not None:
                    rec.append((base, endian, tuple(sorted(opm.items())), sol, out))
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
# Trace rendering.
#
# Fallback (unsolved) traces keep the legacy wording byte-for-byte (they never
# enter the corpus: the correctness filter drops wrong answers).  Solved traces
# use the FORWARD FAIL-FAST ENUMERATION emitter below (run-011 R-fix): the old
# hypothesis-first format ("Hypothesis: digit map A=3,B=7...") trained to
# NLL~0.002 but reproduced at 3% greedy, because the map was pulled from thin
# air; the new format mirrors equation_numeric_deduce's enumerative transcript
# (measured 100% greedy reproduction) and derives everything forward.
# --------------------------------------------------------------------------- #
_HEADER = ("Each operator symbol hides one fixed rule. Label every 5-symbol string as "
           "A0 A1 op B0 B1 (operands A=A0A1, B=B0B1). I find a rule from the examples, "
           "check it against each example, then apply it to the query.")


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


# --------------------------------------------------------------------------- #
# Forward fail-fast emitter: shared helpers
# --------------------------------------------------------------------------- #
_ENDIAN_NAME = {"little": "units-first", "big": "units-last"}
_OPCH = "+-*"  # display char for ADD/SUB/MUL (index = opcode)

_FF_HEADER = (
    "Each operator symbol hides one fixed rule. Read every 5-symbol string as "
    "A0 A1 op B0 B1 (left operand A=A0A1, right operand B=B0B1). Rules come from "
    "two families: (1) the output copies a fixed selection of the operand "
    "positions A0,A1,B0,B1 (such outputs are never marked negative); (2) every "
    "symbol is a digit of one shared cipher (distinct symbols = distinct digits) "
    "in base 10, 11, 12 or 13, words written units-first (first symbol = units "
    "digit) or units-last, and each operator means +, -, * or |difference|. "
    "I scan candidate rules in a fixed priority order (selections first, "
    "preferring fewer, in-order, contiguous positions; then ciphers, preferring "
    "base 10 over 11 over 12 over 13, units-first over units-last, and natural "
    "readings of +, -, *), derive each candidate's consequences from the "
    "examples, and discard a candidate at its first contradiction. The first "
    "candidate consistent with every example is applied to the query.")


def _seq_score(s):
    """Mirror of _pick's rearrangement simplicity prior (assertion-checked)."""
    in_order = list(s) == sorted(s)
    contiguous = all(s[i] + 1 == s[i + 1] for i in range(len(s) - 1))
    return (len(s), 0 if in_order else 1, 0 if contiguous else 1, s)


def _cand_score(base, endian, opm_tuple):
    """Mirror of _pick's arithmetic prior (assertion-checked)."""
    sc = {10: 300, 11: 200, 12: 100, 13: 50}.get(base, 0)
    sc += 10 if endian == "little" else 0
    for op, o in opm_tuple:
        if op == "+" and o == ADD:
            sc += 5
        if op == "-" and o == SUB:
            sc += 5
        if op == "*" and o == MUL:
            sc += 5
    return sc


def _labeled_ops(exs):
    """op -> [(lhs, rhs, eid)] in the same insertion order as _enumerate_answers."""
    ops: dict[str, list] = {}
    for i, (lhs, rhs) in enumerate(exs, start=1):
        ops.setdefault(lhs[2], []).append((lhs, rhs, f"e{i}"))
    return ops


def _cons_for(ops_lbl, arith_ops, opm):
    """Labeled constraints, same order _solve_hypothesis receives them."""
    cons = []
    for opch in arith_ops:
        o = opm[opch]
        for lhs, rhs, eid in ops_lbl[opch]:
            neg = rhs.startswith("-")
            mag = rhs.lstrip("-")
            a0, a1, b0, b1 = _operands(lhs)
            cons.append(([a0, a1], [b0, b1], list(mag), neg, o, eid, lhs, rhs))
    return cons


def _units_for(cons, endian):
    """Units-column rows. mode 'std' = solver's rows; 'neg' (B-A=mag) and 'abs'
    rows are extra SOUND consequences used only to shorten the printed search."""
    rows = []
    for a, b, o, neg, opc, eid, lhs, rhs in cons:
        if endian == "little":
            ua, ub, uo = a[0], b[0], o[0]
        else:
            ua, ub, uo = a[-1], b[-1], o[-1]
        if opc == ABS:
            rows.append((ua, ub, uo, opc, eid, "abs"))
        elif neg:
            rows.append((ua, ub, uo, opc, eid, "neg"))
        else:
            rows.append((ua, ub, uo, opc, eid, "std"))
    return rows


def _op_apply(opc, A, B):
    if opc == ADD:
        return A + B
    if opc == SUB:
        return A - B
    if opc == MUL:
        return A * B
    return abs(A - B)


def _op_text(opc, A, B):
    R = _op_apply(opc, A, B)
    if opc == ABS:
        return f"|{A}-{B}|={R}", R
    return f"{A}{_OPCH[opc]}{B}={R}", R


def _val_expr(word, dmap, base, endian):
    """(digits_txt, place_expr, value) for a symbol word under dmap."""
    digs = [dmap[c] for c in word]
    n = len(word)
    if endian == "little":
        places = [base ** i for i in range(n)]
    else:
        places = [base ** (n - 1 - i) for i in range(n)]
    val = sum(d * p for d, p in zip(digs, places))
    seen = set()
    dparts = []
    for c, d in zip(word, digs):
        if c not in seen:
            seen.add(c)
            dparts.append(f"{c}={d}")
    if n == 1:
        return ",".join(dparts), str(digs[0]), val
    eparts = [str(d) if p == 1 else f"{d}*{p}" for d, p in zip(digs, places)]
    return ",".join(dparts), "+".join(eparts), val


def _meanings_text(opm_tuple):
    return ", ".join(f"'{op}'={_OPNAME[o]}" for op, o in opm_tuple)


# --------------------------------------------------------------------------- #
# Forward fail-fast emitter: family (1) -- selection scan
# --------------------------------------------------------------------------- #
def _sel_scan(lines, opch, group_lbl, winner_seq=None):
    """Scan selection candidates for operator `opch` in the solver's priority
    order, discarding each at its first contradiction. With winner_seq, stop at
    it and verify every example (rearrangement winner); otherwise every
    candidate must die (cipher path). Returns True iff the winner was reached."""
    for lhs, rhs, eid in group_lbl:
        if rhs.startswith("-"):
            if winner_seq is not None:
                raise RuntimeError("rearrange winner with negative output")
            lines.append(f"  {eid}'s output {rhs} is marked negative; family (1) "
                         "outputs never are -> no selection rule.")
            return False
    lens = [len(rhs) for _, rhs, _ in group_lbl]
    if len(set(lens)) > 1:
        if winner_seq is not None:
            raise RuntimeError("rearrange winner with varying output lengths")
        i = next(k for k in range(len(lens)) if lens[k] != lens[0])
        lines.append(f"  {group_lbl[0][2]}'s output has {lens[0]} symbols but "
                     f"{group_lbl[i][2]}'s has {lens[i]}; one fixed selection "
                     "always copies the same number of positions -> no selection rule.")
        return False
    lhs1, rhs1, eid1 = group_lbl[0]
    L = len(rhs1)
    oc = _operands(lhs1)
    lab = " ".join(f"{_POS[i]}={oc[i]}" for i in range(4))
    lines.append(f"  Every '{opch}' output has {L} symbol(s), so a selection "
                 f"must pick exactly {L} position(s).")
    lines.append(f"  From {eid1} ({lhs1} = {rhs1}): {lab}.")
    if L > 4:
        lines.append(f"  A selection picks at most 4 positions but {eid1} needs "
                     f"{L} -> no selection rule.")
        return False
    slots = []
    for j, ch in enumerate(rhs1, start=1):
        sl = [i for i in range(4) if oc[i] == ch]
        if not sl:
            if winner_seq is not None:
                raise RuntimeError("rearrange winner with unmatched symbol")
            lines.append(f"  Output symbol {j} of {eid1} is {ch}, which is none "
                         f"of A0,A1,B0,B1 -> no selection can produce it -> no "
                         "selection rule.")
            return False
        slots.append(sl)
        lines.append(f"  Output symbol {j} is {ch} -> copied from "
                     + " or ".join(_POS[i] for i in sl) + ".")
    cands = sorted(itertools.product(*slots), key=_seq_score)
    for s in cands:
        names = " ".join(_POS[i] for i in s)
        if winner_seq is not None and s == tuple(winner_seq):
            lines.append(f"  Try {names}:")
            for lhs, rhs, eid in group_lbl:
                ocx = _operands(lhs)
                pred = "".join(ocx[i] for i in s)
                labx = " ".join(f"{_POS[i]}={ocx[i]}"
                                for i in sorted(set(s)))
                if pred != rhs:
                    raise RuntimeError("selection winner fails an example")
                lines.append(f"    {eid} ({lhs} = {rhs}): {labx} -> {pred}; "
                             f"given {rhs} -> match.")
            lines.append(f"  {names} reproduces every '{opch}' example -> rule "
                         "found, stop the scan.")
            return True
        fail = None
        for lhs, rhs, eid in group_lbl[1:]:
            ocx = _operands(lhs)
            pred = "".join(ocx[i] for i in s)
            if pred != rhs:
                fail = (eid, lhs, pred, rhs)
                break
        if fail is None:
            raise RuntimeError(
                f"selection scan/solver divergence for '{opch}': {s} survives")
        eid, lhs, pred, rhs = fail
        labx = " ".join(f"{_POS[i]}={_operands(lhs)[i]}" for i in sorted(set(s)))
        lines.append(f"  Try {names}: {eid}: {labx} -> {pred}, given {rhs} -> no.")
    if winner_seq is not None:
        raise RuntimeError("rearrange winner not among ex1-consistent selections")
    lines.append(f"  Every selection candidate fails -> '{opch}' is not a "
                 "selection rule.")
    return False


def _emit_sel_fit(lines, opch, group_lbl, seqs):
    """Exhibit the simplest selection fitting ALL of opch's examples (justifies
    excluding this operator from the cipher pool)."""
    seq = min(seqs, key=_seq_score)
    names = " ".join(_POS[i] for i in seq)
    lines.append(f"  Operator '{opch}': the selection {names} fits all its "
                 "examples:")
    for lhs, rhs, eid in group_lbl:
        ocx = _operands(lhs)
        pred = "".join(ocx[i] for i in seq)
        if pred != rhs:
            raise RuntimeError("rearrange fit fails")
        labx = " ".join(f"{_POS[i]}={ocx[i]}" for i in sorted(set(seq)))
        lines.append(f"    {eid} ({lhs} = {rhs}): {labx} -> {pred} -> match.")
    lines.append(f"  So '{opch}' is a selection rule; its examples are not part "
                 "of the cipher.")


# --------------------------------------------------------------------------- #
# Forward fail-fast emitter: family (2) -- pin engine (one derivation step)
# --------------------------------------------------------------------------- #
def _unit_residue(row, g):
    """Value that must be ≡ g(uo) (mod base) for a unit row, given digit lookup g."""
    ua, ub, uo, opc, eid, mode = row
    if mode == "neg":
        return g(ub) - g(ua)
    if opc == ADD:
        return g(ua) + g(ub)
    if opc == MUL:
        return g(ua) * g(ub)
    return g(ua) - g(ub)


def _unit_lhs_repr(row, known, s=None):
    """Printable LHS of a unit row with knowns substituted and `s` left as itself."""
    ua, ub, uo, opc, eid, mode = row
    def r(x):
        if x == s:
            return x
        return str(known[x]) if x in known else x
    if mode == "neg":
        return f"{r(ub)}-{r(ua)}"
    return f"{r(ua)}{_OPCH[opc]}{r(ub)}"


def _full_check_repr(c, known, base, endian):
    """(ok, one-line repr) for a fully-known constraint (repr built lazily)."""
    a, b, o, neg, opc, eid, lhs, rhs = c
    A = _value(a, known, base, endian)
    B = _value(b, known, base, endian)
    R = _value(o, known, base, endian)
    if neg:
        R = -R
    v = _op_apply(opc, A, B)
    if v == R:
        return True, ""
    txt, _v = _op_text(opc, A, B)
    return False, f"{eid}: A={A}, B={B}: {txt} but the given output reads {R}"


def _next_pin(cons, units, base, endian, known):
    """One forward derivation step from the current partial map.
    Returns ("pin", sym, digit, text), ("dead", text) or None. Deterministic."""
    used = set(known.values())

    # A. completed unit rows must hold
    for row in units:
        ua, ub, uo, opc, eid, mode = row
        if mode == "abs":
            if all(x in known for x in (ua, ub, uo)):
                da, db, do = known[ua], known[ub], known[uo]
                if (da - db) % base != do and (db - da) % base != do:
                    return ("dead", f"{eid} units: |{da}-{db}| ends in "
                            f"{(da - db) % base} or {(db - da) % base} "
                            f"(mod {base}), but {uo}={do}")
            continue
        if all(x in known for x in (ua, ub, uo)):
            v = _unit_residue(row, lambda x: known[x]) % base
            if v != known[uo]:
                return ("dead", f"{eid} units: ({_unit_lhs_repr(row, known)}) "
                        f"mod {base} = {v}, but {uo}={known[uo]}")
    # B. completed full constraints must hold
    for c in cons:
        a, b, o, neg, opc, eid, lhs, rhs = c
        if all(x in known for x in a + b + o):
            ok, txt = _full_check_repr(c, known, base, endian)
            if not ok:
                return ("dead", txt)

    # B2. interval (size) check: bound each word by its known digits (unknown
    #     digits range over 0..base-1); disjoint ranges kill the branch.
    def wrange(word):
        n = len(word)
        lo = hi = 0
        for i, ch in enumerate(word):
            p = base ** i if endian == "little" else base ** (n - 1 - i)
            if ch in known:
                lo += known[ch] * p
                hi += known[ch] * p
            else:
                hi += (base - 1) * p
        return lo, hi

    for c in cons:
        a, b, o, neg, opc, eid, lhs, rhs = c
        if all(x in known for x in a + b + o):
            continue
        A1, A2 = wrange(a)
        B1, B2 = wrange(b)
        R1, R2 = wrange(o)
        if neg:
            R1, R2 = -R2, -R1
        if opc == ADD:
            L1, L2 = A1 + B1, A2 + B2
        elif opc == SUB:
            L1, L2 = A1 - B2, A2 - B1
        elif opc == MUL:
            L1, L2 = A1 * B1, A2 * B2
        else:
            L1 = max(0, A1 - B2, B1 - A2)
            L2 = max(A2 - B1, B2 - A1, 0)
        opn = "A+B" if opc == ADD else ("A-B" if opc == SUB else
                                        ("A*B" if opc == MUL else "|A-B|"))
        ar = str(A1) if A1 == A2 else f"{A1}..{A2}"
        br = str(B1) if B1 == B2 else f"{B1}..{B2}"
        if L2 < R1:
            return ("dead", f"{eid}: A={ar}, B={br} -> {opn} <= {L2} < "
                    f"output >= {R1}")
        if L1 > R2:
            return ("dead", f"{eid}: A={ar}, B={br} -> {opn} >= {L1} > "
                    f"output <= {R2}")

    def pin_or_dead(s, d, why):
        if s in known:
            return None  # already pinned elsewhere; skip
        if d in used:
            holder = next(c for c, dd in known.items() if dd == d)
            return ("dead", f"{why}, but {holder}={d} already -> two symbols "
                    "would share a digit")
        return ("pin", s, d, why + f" -> {s}={d}")

    # C. structural unit pins (no prior digits needed)
    for row in units:
        ua, ub, uo, opc, eid, mode = row
        if mode == "abs" or opc == MUL:
            continue
        eff_a, eff_b = (ub, ua) if mode == "neg" else (ua, ub)
        # (x OP y) mod base = x  -> y = 0 for ADD/SUB on the y side
        if opc == ADD:
            if eff_a == uo and eff_b not in known:
                r = pin_or_dead(eff_b, 0, f"{eid} units: ({eff_a}+{eff_b}) mod "
                                f"{base} = {uo} = {eff_a}")
                if r:
                    return r
            if eff_b == uo and eff_a not in known:
                r = pin_or_dead(eff_a, 0, f"{eid} units: ({eff_a}+{eff_b}) mod "
                                f"{base} = {uo} = {eff_b}")
                if r:
                    return r
        else:  # SUB (std or neg)
            if eff_a == uo and eff_b not in known:
                r = pin_or_dead(eff_b, 0, f"{eid} units: ({eff_a}-{eff_b}) mod "
                                f"{base} = {uo} = {eff_a}")
                if r:
                    return r
            if eff_a == eff_b and uo not in known:
                r = pin_or_dead(uo, 0, f"{eid} units: ({eff_a}-{eff_b}) mod "
                                f"{base} = {uo} with equal sides")
                if r:
                    return r
    # C2. both operands fully known -> the whole result is computable; read
    #     its digits off against the output symbols (pin the first unknown
    #     one, or die on a sign/length/digit mismatch).
    for c in cons:
        a, b, o, neg, opc, eid, lhs, rhs = c
        if any(x not in known for x in a + b):
            continue
        if all(x in known for x in o):
            continue  # fully known: handled by check B
        A = _value(a, known, base, endian)
        B = _value(b, known, base, endian)
        txt, R = _op_text(opc, A, B)
        if neg and R > 0:
            return ("dead", f"{eid}: A={A}, B={B} -> {txt} is positive, "
                    f"but the output {rhs} is marked negative")
        if not neg and R < 0:
            return ("dead", f"{eid}: A={A}, B={B} -> {txt} is negative, but "
                    f"the output {rhs} has no negative mark")
        mag = abs(R)
        digs = [0] if mag == 0 else []
        while mag > 0:
            digs.append(mag % base)
            mag //= base
        if len(digs) > len(o):
            return ("dead", f"{eid}: A={A}, B={B} -> {txt} needs {len(digs)} "
                    f"digits in base {base} but the output has only "
                    f"{len(o)} symbols")
        digs += [0] * (len(o) - len(digs))
        ow = list(o) if endian == "little" else list(o[::-1])
        for i, (sym, dv) in enumerate(zip(ow, digs)):
            if sym in known:
                if known[sym] != dv:
                    return ("dead", f"{eid}: A={A}, B={B} -> {txt}; digit "
                            f"#{i + 1} (units first) is {dv} but {sym}="
                            f"{known[sym]}")
                continue
            why = (f"{eid}: A={A}, B={B} -> {txt}; its digit #{i + 1} "
                   f"(units first) in base {base} is {dv}")
            r = pin_or_dead(sym, dv, why)
            if r:
                return r

    # D. range facts about the high output digits (operands are < base^2):
    #    ADD: value < 2*base^2 -> a 3rd digit can only be the carry, 0 or 1;
    #    SUB/|.|/neg: value < base^2 -> digits beyond the 2nd must be 0.
    for c in cons:
        a, b, o, neg, opc, eid, lhs, rhs = c
        if len(o) < 3:
            continue
        vw = list(o) if endian == "little" else list(o[::-1])  # units first
        if opc == ADD:
            lead = vw[2]
            if lead in known and known[lead] > 1:
                return ("dead", f"{eid}: the 3rd digit of A+B is the carry of "
                        f"a 2-word sum, at most 1, but {lead}={known[lead]}")
            continue
        if opc == MUL:
            continue
        # SUB / ABS (and the negative-marked B-A reading): value < base^2
        kind = "|A-B|" if opc == ABS else ("B-A" if neg else "A-B")
        for pos in range(2, len(vw)):
            hi = vw[pos]
            r = pin_or_dead(hi, 0, f"{eid}: {kind} of 2-symbol words is < "
                            f"{base}^2, so the output digit at place "
                            f"{base}^{pos} must be 0")
            if r:
                return r
    # E. unit row with exactly one unknown symbol
    for row in units:
        ua, ub, uo, opc, eid, mode = row
        unk = {x for x in (ua, ub, uo) if x not in known}
        if len(unk) != 1:
            continue
        s = next(iter(unk))
        if mode == "abs":
            sat = []
            for d in range(base):
                g = lambda x: d if x == s else known[x]
                if ((g(ua) - g(ub)) % base == g(uo)
                        or (g(ub) - g(ua)) % base == g(uo)):
                    sat.append(d)
            free = [d for d in sat if d not in used]
            def rabs(x):
                if x == s:
                    return x
                return str(known[x])
            eqn = (f"{eid} units: |{rabs(ua)}-{rabs(ub)}| mod {base} "
                   f"(either order) = {rabs(uo)}")
            if not free:
                if sat:
                    taken = ", ".join(f"{d} (taken)" for d in sat)
                    return ("dead", f"{eqn}: the only fitting digits are "
                            f"{taken}")
                return ("dead", f"{eqn}: no digit fits {s}")
            if len(free) == 1:
                d = free[0]
                note = ""
                if len(sat) > 1:
                    note = "; " + ", ".join(f"{x} is taken"
                                            for x in sat if x != d)
                return ("pin", s, d, f"{eqn}: only {s}={d} fits{note}")
            continue
        sat = []
        for d in range(base):
            g = lambda x: d if x == s else known[x]
            if _unit_residue(row, g) % base == g(uo):
                sat.append(d)
        free = [d for d in sat if d not in used]
        lhs_repr = _unit_lhs_repr(row, known, s)
        uo_repr = str(known[uo]) if uo in known else uo
        eqn = f"{eid} units: ({lhs_repr}) mod {base} = {uo_repr}"
        if not free:
            if sat:
                taken = ", ".join(f"{d} (taken)" for d in sat)
                return ("dead", f"{eqn}: the only fitting digits are {taken}")
            return ("dead", f"{eqn}: no digit fits {s}")
        if len(free) == 1:
            d = free[0]
            note = ""
            if len(sat) > 1:
                note = "; " + ", ".join(f"{x} is taken" for x in sat if x != d)
            return ("pin", s, d, f"{eqn}: only {s}={d} fits{note}")
        # multiple possibilities: not a pin
    # E2. unit row with exactly TWO unknown symbols whose relation degenerates
    #     (e.g. a known 0 operand under * fixes the output digit no matter the
    #     other; a known 1 operand under * forces two distinct symbols to one
    #     digit). Detected generically by enumerating the pair relation.
    for row in units:
        ua, ub, uo, opc, eid, mode = row
        unk = sorted({x for x in (ua, ub, uo) if x not in known})
        if len(unk) != 2:
            continue
        s1, s2 = unk
        pairs = []
        for d1 in range(base):
            for d2 in range(base):
                if d1 == d2:
                    continue
                g = lambda x: d1 if x == s1 else (d2 if x == s2 else known[x])
                if mode == "abs":
                    ok = ((g(ua) - g(ub)) % base == g(uo)
                          or (g(ub) - g(ua)) % base == g(uo))
                else:
                    ok = _unit_residue(row, g) % base == g(uo)
                if ok:
                    pairs.append((d1, d2))
        lhs_repr = (f"|{_unit_lhs_repr((ua, ub, uo, SUB, eid, 'std'), known)}|"
                    if mode == "abs" else _unit_lhs_repr(row, known))
        uo_repr = str(known[uo]) if uo in known else uo
        eqn = f"{eid} units: ({lhs_repr}) mod {base} = {uo_repr}"
        if not pairs:
            return ("dead", f"{eqn}: no pair of distinct unused digits fits "
                    f"{s1},{s2}")
        for si, other in ((0, s2), (1, s1)):
            vals = sorted({p[si] for p in pairs})
            if len(vals) == 1:
                s = (s1, s2)[si]
                r = pin_or_dead(s, vals[0],
                                f"{eqn}: whatever {other} is, only "
                                f"{s}={vals[0]} fits")
                if r:
                    return r
    #    is either fixed (units digits known) or ranges over {0,1} -- both
    #    cases are sound and the bounded carry often pins or kills regardless
    #    of the unknown units.
    for c in cons:
        a, b, o, neg, opc, eid, lhs, rhs = c
        if opc not in (ADD, SUB):
            continue
        if endian == "little":
            a0, a1, b0, b1 = a[0], a[1], b[0], b[1]
            ow = list(o)
        else:
            a0, a1, b0, b1 = a[1], a[0], b[1], b[0]
            ow = list(o[::-1])  # ow[0] = units, ow[1] = tens, ...
        ea, eb = (b0, a0) if neg else (a0, b0)   # neg: B - A = mag
        ta, tb = (b1, a1) if neg else (a1, b1)
        if ea in known and eb in known:
            da, db = known[ea], known[eb]
            if opc == ADD:
                cset = [1 if da + db >= base else 0]
                cdesc = f"from units {da}+{db}={da + db}"
            else:
                cset = [1 if da < db else 0]
                cdesc = f"from units {da}-{db}"
        else:
            cset = [0, 1]
            cdesc = "0 or 1, units not fixed yet"
        if opc == ADD:
            if len(ow) > 3:
                continue
            syms3 = [ta, tb] + list(ow[1:])  # tens column + carry-out digit
            unk = {x for x in syms3 if x not in known}
            if len(unk) != 1:
                continue
            s = next(iter(unk))
            sat = []
            for d in range(base):
                g = lambda x: d if x == s else known[x]
                for cy in cset:
                    tot = g(ta) + g(tb) + cy
                    o1 = g(ow[1]) if len(ow) >= 2 else 0
                    lead = g(ow[2]) if len(ow) == 3 else 0
                    if tot % base == o1 and tot // base == lead:
                        sat.append(d)
                        break
            free = [d for d in sat if d not in used]
            def rsym(x):
                if x == s:
                    return x
                return str(known[x])
            o1r = rsym(ow[1]) if len(ow) >= 2 else "0 (no tens symbol)"
            leadr = rsym(ow[2]) if len(ow) == 3 else "0"
            cr = "/".join(str(cy) for cy in cset)
            eqn = (f"{eid} tens: {rsym(ta)}+{rsym(tb)}+carry({cr}, {cdesc}) = "
                   f"{o1r} + {base}*{leadr}")
            if not free:
                return ("dead", f"{eqn}: no unused digit fits")
            if len(free) == 1:
                d = free[0]
                note = ""
                if len(sat) > 1:
                    note = "; " + ", ".join(f"{x} is taken" for x in sat if x != d)
                return ("pin", s, d, f"{eqn}: only {s}={d} fits{note}")
            continue
        # SUB / neg-SUB: the result value is < base^2, so only two columns
        # matter; any higher output symbols were already pinned to 0 by D.
        if any(x not in known for x in ow[2:]):
            continue
        if any(known[x] != 0 for x in ow[2:]):
            continue  # D will report the contradiction
        syms3 = [ta, tb] + ([ow[1]] if len(ow) >= 2 else [])
        unk = {x for x in syms3 if x not in known}
        if len(unk) != 1:
            continue
        s = next(iter(unk))
        sat = []
        for d in range(base):
            g = lambda x: d if x == s else known[x]
            for br in cset:
                t = g(ta) - g(tb) - br
                o1 = g(ow[1]) if len(ow) >= 2 else 0
                if t == o1:
                    sat.append(d)
                    break
        free = [d for d in sat if d not in used]
        def rsym2(x):
            if x == s:
                return x
            return str(known[x])
        o1r = rsym2(ow[1]) if len(ow) >= 2 else "0 (no tens symbol)"
        br_r = "/".join(str(cy) for cy in cset)
        eqn = (f"{eid} tens: {rsym2(ta)}-{rsym2(tb)}-borrow({br_r}, {cdesc}) "
               f"= {o1r}")
        if not free:
            return ("dead", f"{eqn}: no unused digit fits")
        if len(free) == 1:
            d = free[0]
            note = ""
            if len(sat) > 1:
                note = "; " + ", ".join(f"{x} is taken" for x in sat if x != d)
            return ("pin", s, d, f"{eqn}: only {s}={d} fits{note}")
        continue
    # G. full constraint with exactly one unknown symbol
    for c in cons:
        a, b, o, neg, opc, eid, lhs, rhs = c
        unk = {x for x in a + b + o if x not in known}
        if len(unk) != 1:
            continue
        s = next(iter(unk))
        sat = []
        for d in range(base):
            trial = dict(known)
            trial[s] = d
            _, _, A = _val_expr(a, trial, base, endian)
            _, _, B = _val_expr(b, trial, base, endian)
            _, _, R = _val_expr(o, trial, base, endian)
            if neg:
                R = -R
            if _op_apply(opc, A, B) == R:
                sat.append(d)
        free = [d for d in sat if d not in used]
        opname = _OPNAME[opc]
        if not free:
            if sat:
                taken = ", ".join(str(d) for d in sat)
                return ("dead", f"{eid} ({lhs} = {rhs}): only {s} is unknown, "
                        f"and the digits that satisfy the {opname} ({taken}) "
                        "are already taken")
            return ("dead", f"{eid} ({lhs} = {rhs}): only {s} is unknown, and "
                    f"no digit satisfies the {opname}")
        if len(free) == 1:
            d = free[0]
            trial = dict(known)
            trial[s] = d
            _, _, A = _val_expr(a, trial, base, endian)
            _, _, B = _val_expr(b, trial, base, endian)
            txt, _v = _op_text(opc, A, B)
            note = ""
            if len(sat) > 1:
                note = "; " + ", ".join(f"{x} is taken" for x in sat if x != d)
            return ("pin", s, d, f"{eid} ({lhs} = {rhs}): only {s} is unknown; "
                    f"only {s}={d} works: A={A}, B={B}, {txt} = the given "
                    f"output{note}")
    # G2. full constraint with exactly TWO unknown symbols: enumerate the
    #     digit pairs; a forced single value (or no pair at all) pins or kills.
    for c in cons:
        a, b, o, neg, opc, eid, lhs, rhs = c
        unk = sorted({x for x in a + b + o if x not in known})
        if len(unk) != 2:
            continue
        s1, s2 = unk
        vals1 = set()
        vals2 = set()
        npairs = 0
        for d1 in range(base):
            if d1 in used:
                continue
            for d2 in range(base):
                if d2 in used or d2 == d1:
                    continue
                trial = dict(known)
                trial[s1] = d1
                trial[s2] = d2
                A = _value(a, trial, base, endian)
                B = _value(b, trial, base, endian)
                R = _value(o, trial, base, endian)
                if neg:
                    R = -R
                if _op_apply(opc, A, B) == R:
                    npairs += 1
                    vals1.add(d1)
                    vals2.add(d2)
        if npairs == 0:
            return ("dead", f"{eid} ({lhs} = {rhs}): only {s1},{s2} are "
                    "unknown and no unused digit pair satisfies it")
        for s, vals, other in ((s1, vals1, s2), (s2, vals2, s1)):
            if len(vals) == 1 and s not in known:
                v = next(iter(vals))
                return ("pin", s, v, f"{eid} ({lhs} = {rhs}): only {s1},{s2} "
                        f"are unknown; every unused digit pair that satisfies "
                        f"it has {s}={v}")
    return None


# --------------------------------------------------------------------------- #
# Forward fail-fast emitter: narrated search = solver lex order + forced pins
# --------------------------------------------------------------------------- #
def _propagate(cons, units, base, endian, known):
    """Run the pin engine to fixpoint on a COPY of known.
    Returns (new_known, steps, dead_text|None); steps = printable pin lines."""
    known = dict(known)
    steps = []
    while True:
        r = _next_pin(cons, units, base, endian, known)
        if r is None:
            return known, steps, None
        if r[0] == "dead":
            return known, steps, r[1]
        _, s, d, txt = r
        known[s] = d
        steps.append(txt)


def _dpll_narrate(lines, cons, units, base, endian, known, con_order, indent,
                  attempt_cap=2000, choose_split=None, digit_order=None):
    """Narrated search over the CONSTRAINED symbols: propagate forced digits to
    fixpoint, then split on an unassigned symbol, digits ascending, recursing
    per branch.  Default split = first unassigned symbol in `con_order` (the
    solver's frequency order): then the search finds exactly the same map as
    _solve_hypothesis restricted to constrained symbols (identical
    lexicographic order over split symbols, ascending digits; FORCED pins are
    sound, so no solution is skipped and the first leaf reached is still the
    lexicographically smallest solution).  `choose_split` / `digit_order` may
    deviate ONLY when the caller has proven the solution unique (then any
    complete search returns it), re-validates the result, or knows no solution
    exists (exhaustion is order-independent).  Prints every step; returns the
    completed map or None."""
    n_attempt = [0]
    dorder_digits = list(digit_order) if digit_order is not None \
        else list(range(base))

    def flush_quick(pend, indent, s):
        """Merge consecutive quickly-dying branches; identical chains are
        packed into one line listing all their digits."""
        i = 0
        while i < len(pend):
            j = i
            while j + 1 < len(pend) and pend[j + 1][1] == pend[i][1]:
                j += 1
            ds = "/".join(str(d) for d, _c in pend[i:j + 1])
            lines.append(f"{indent}try {s}={ds}: {pend[i][1]} -> no.")
            i = j + 1
        pend.clear()

    def rec2(known, indent):
        known, steps, dead = _propagate(cons, units, base, endian, known)
        for t in steps:
            lines.append(indent + t + ".")
        if dead is not None:
            lines.append(indent + dead + " -> contradiction.")
            return None
        rest = [s for s in con_order if s not in known]
        if not rest:
            return known
        s = choose_split(known, rest) if choose_split else rest[0]
        used = set(known.values())
        pend: list = []
        for d in dorder_digits:
            if d in used:
                continue
            n_attempt[0] += 1
            if n_attempt[0] > attempt_cap:
                raise RuntimeError("narrated search exceeded the line budget")
            sub = dict(known)
            sub[s] = d
            # peek: branches that die within three derivation steps are
            # merged into single printed lines (and packed when identical).
            _k2, steps2, dead2 = _propagate(cons, units, base, endian, sub)
            if dead2 is not None and len(steps2) <= 3:
                pend.append((d, "; ".join(steps2 + [dead2])))
                continue
            flush_quick(pend, indent, s)
            lines.append(f"{indent}try {s}={d}:")
            r = rec2(sub, indent + "  ")
            if r is not None:
                return r
        flush_quick(pend, indent, s)
        lines.append(f"{indent}no digit works for {s} -> backtrack.")
        return None

    return rec2(dict(known), indent)


def _count_solutions(cons, units, base, endian, con_order, limit=2,
                     node_cap=2000000):
    """Count injective digit maps over the constrained symbols (up to limit),
    with checks indexed by the position where they complete (same scheme as
    _solve_hypothesis). node_cap exceeded -> returns limit (conservative:
    treated as 'not provably unique' / 'not provably unsatisfiable')."""
    n = len(con_order)
    idxpos = {s: i for i, s in enumerate(con_order)}
    units_by: list[list] = [[] for _ in range(n)]
    for row in units:
        ua, ub, uo, opc, eid, mode = row
        units_by[max(idxpos[x] for x in (ua, ub, uo))].append(row)
    full_by: list[list] = [[] for _ in range(n)]
    for c in cons:
        a, b, o, neg, opc, eid, lhs, rhs = c
        full_by[max(idxpos[x] for x in set(a + b + o))].append(c)
    found = [0]
    nodes = [0]
    known: dict[str, int] = {}
    used = [False] * base

    def check(pos):
        for ua, ub, uo, opc, eid, mode in units_by[pos]:
            da, db, do = known[ua], known[ub], known[uo]
            if mode == "abs":
                if (da - db) % base != do and (db - da) % base != do:
                    return False
            elif mode == "neg":
                if (db - da) % base != do:
                    return False
            elif opc == ADD:
                if (da + db) % base != do:
                    return False
            elif opc == MUL:
                if (da * db) % base != do:
                    return False
            else:
                if (da - db) % base != do:
                    return False
        for a, b, o, neg, opc, eid, lhs, rhs in full_by[pos]:
            A = _value(a, known, base, endian)
            B = _value(b, known, base, endian)
            R = _value(o, known, base, endian)
            if neg:
                R = -R
            v = (A + B if opc == ADD else A - B if opc == SUB
                 else A * B if opc == MUL else abs(A - B))
            if v != R:
                return False
        return True

    def rec(i):
        if i == n:
            found[0] += 1
            return
        s = con_order[i]
        for d in range(base):
            if used[d]:
                continue
            nodes[0] += 1
            if nodes[0] > node_cap:
                found[0] = limit
                return
            known[s] = d
            used[d] = True
            if check(i):
                rec(i + 1)
            used[d] = False
            del known[s]
            if found[0] >= limit:
                return

    rec(0)
    return found[0]


def _sweep_refute(cons, units, base, endian, known, s, max_branch_steps=4):
    """Try to refute by sweeping every unused digit of `s`: each branch must
    die by propagation alone within max_branch_steps pins. Returns the list of
    printable branch lines, or None."""
    used = set(known.values())
    out = []
    for d in range(base):
        if d in used:
            continue
        sub = dict(known)
        sub[s] = d
        _k, steps, dead = _propagate(cons, units, base, endian, sub)
        if dead is None or len(steps) > max_branch_steps:
            return None
        chain = "; ".join(steps + [dead])
        out.append(f"{s}={d}: {chain} -> no.")
    return out


# --------------------------------------------------------------------------- #
# Forward fail-fast emitter: quick one-line refutation certificates
# --------------------------------------------------------------------------- #
def _value_cert(cons):
    """Same operands, same meaning, different same-length same-sign outputs ->
    impossible (equal-length words that differ in a symbol differ in value:
    the highest differing digit cannot be cancelled by lower places)."""
    seen = {}
    for a, b, o, neg, opc, eid, lhs, rhs in cons:
        k = (tuple(a), tuple(b), opc, neg, len(o))
        v = tuple(o)
        if k in seen and seen[k][0] != v:
            e0 = seen[k][1]
            return (f"{e0} and {eid} apply the same operation to the same "
                    f"operands but show different outputs of equal length, "
                    f"which cannot decode to the same value -> impossible")
        seen.setdefault(k, (v, eid))
    return None


def _unit_pair_cert(units, base):
    """Two unit rows forcing one residue onto two different symbols."""
    seen = {}
    for ua, ub, uo, opc, eid, mode in units:
        if mode == "abs":
            continue
        ea, eb = (ub, ua) if mode == "neg" else (ua, ub)
        if opc in (ADD, MUL):
            key = (opc, tuple(sorted((ea, eb))))
        else:
            key = (opc, ea, eb)
        if key in seen:
            uo0, e0, lhs0 = seen[key]
            if uo0 != uo:
                return (f"{e0} and {eid} units: ({lhs0}) mod {base} would have "
                        f"to equal both {uo0} and {uo}, two different symbols "
                        f"with distinct digits -> impossible")
        else:
            lhs_r = f"{ea}{_OPCH[opc] if mode != 'neg' else '-'}{eb}"
            seen[key] = (uo, eid, lhs_r)
    return None


# --------------------------------------------------------------------------- #
# Forward fail-fast emitter: cipher candidate derivation / refutation
# --------------------------------------------------------------------------- #
def _solver_order(cons, digit_syms):
    """EXACT variable order of _solve_hypothesis: frequency-desc over the
    constraint symbols, ties by the sorted(digit_syms) base order."""
    freq = Counter()
    for a, b, o, neg, opc, eid, lhs, rhs in cons:
        for ch in a + b + o:
            freq[ch] += 1
    return sorted(list(digit_syms), key=lambda c: -freq[c])


def _guided_chooser(cons, units, base, endian, sol, kill_depth=2, prio=None):
    """Split chooser ordering the derivation with the solver's known map: pick
    the symbol whose wrong digits below sol[s] all die fastest by propagation
    (the search descends at sol[s] and never looks back). Used only where the
    first-found solution is independent of the split order (unique map) or
    where the caller re-validates the result (relaxed answer check).
    prio: optional symbol set ranked first (e.g. query-relevant symbols)."""
    def choose(known, rest):
        best = None
        used = set(known.values())
        for s in rest:
            target = sol.get(s)
            cost = 0
            feasible = target is not None
            for d in range(base):
                if not feasible or d >= target:
                    break
                if d in used:
                    continue
                sub = dict(known)
                sub[s] = d
                _k, steps, dead = _propagate(cons, units, base, endian, sub)
                if dead is None or len(steps) > kill_depth:
                    feasible = False
                    break
                cost += 1
            if not feasible:
                continue
            key = ((0 if prio and s in prio else 1) if prio else 0,
                   cost, rest.index(s))
            if best is None or key < best[0]:
                best = (key, s)
        if best is not None:
            return best[1]
        return _min_branch_chooser(cons, units, base, endian)(known, rest)
    return choose


def _min_branch_chooser(cons, units, base, endian):
    """Split chooser for UNSAT narration (order-free): minimize surviving
    branches, then prefer symbols that activate the most unit rows."""
    def choose(known, rest):
        best = None
        used = set(known.values())
        for s in rest:
            alive = quick = 0
            for d in range(base):
                if d in used:
                    continue
                sub = dict(known)
                sub[s] = d
                _k, steps, dead = _propagate(cons, units, base, endian, sub)
                if dead is None:
                    alive += 1
                elif len(steps) <= 1:
                    quick += 1
            urows = sum(1 for ua, ub, uo, opc, eid, mode in units
                        if s in (ua, ub, uo))
            key = (alive, -urows, -quick, rest.index(s))
            if best is None or key < best[0]:
                best = (key, s)
        return best[1]
    return choose


def _answer_from(m, q, opc, base, endian):
    A = _value([q[0], q[1]], m, base, endian)
    B = _value([q[3], q[4]], m, base, endian)
    rv = _op_apply(opc, A, B)
    inv: dict[int, str] = {}
    for c, d in m.items():
        inv.setdefault(d, c)
    return _render(rv, inv, base, endian)


def _opt_root_chooser(cons, units, base, endian, con_order):
    """At the first split, pick the symbol whose whole simulated search tree
    (min-branch policy below) prints the fewest characters; afterwards behave
    like the min-branch chooser. Used only in re-validated (relaxed) runs."""
    mb = _min_branch_chooser(cons, units, base, endian)
    state = {"root_pick": None, "root_known": None}

    def choose(known, rest):
        if state["root_pick"] is not None and \
                state["root_known"] == sorted(known.items()):
            return state["root_pick"]
        if state["root_known"] is not None:
            return mb(known, rest)
        state["root_known"] = sorted(known.items())
        best = None
        for s in rest:
            forced = {"first": True}

            def chooser_s(k2, r2):
                if forced["first"]:
                    forced["first"] = False
                    return s
                return mb(k2, r2)
            sim: list[str] = []
            try:
                _dpll_narrate(sim, cons, units, base, endian, dict(known),
                              list(con_order), "", attempt_cap=900,
                              choose_split=chooser_s)
            except RuntimeError:
                continue
            cost = sum(len(t) for t in sim)
            if best is None or cost < best[0]:
                best = (cost, s)
        state["root_pick"] = best[1] if best else rest[0]
        return state["root_pick"]
    return choose


def _derive_map(lines, cons, units, base, endian, dorder, expect,
                accept=None, prio_syms=None, indent="  "):
    """Forward derivation of the digit map: forced pins + narrated split
    search over the constrained symbols, then the smallest-unused-digit
    convention for query-only symbols. When the constrained map is provably
    unique, the split order is optimized for brevity (any order finds the same
    unique map = the solver's). Otherwise: if the brevity-ordered search finds
    a (different) consistent map that still yields the solver's exact answer,
    it is kept -- the trace is a valid first-survivor scan for this same
    candidate and boxes the same string; else the solver's frequency order is
    replayed so the first-found map is the solver's own. Result is asserted
    against `expect` (or against the answer in the relaxed case)."""
    con_syms = set()
    for a, b, o, neg, opc, eid, lhs, rhs in cons:
        con_syms.update(a + b + o)
    con_order = [s for s in dorder if s in con_syms]

    def run(target_lines, chooser, cons_v=None, units_v=None,
            digit_order=None):
        cons_v = cons if cons_v is None else cons_v
        units_v = units if units_v is None else units_v
        syms_v = {x for a, b, o, *_r in cons_v for x in a + b + o}
        order_v = [s for s in dorder if s in syms_v]
        m = _dpll_narrate(target_lines, cons_v, units_v, base, endian, {},
                          order_v, indent, choose_split=chooser,
                          digit_order=digit_order)
        if m is None:
            raise RuntimeError("trace search refuted the solver's candidate")
        free = [s for s in dorder if s not in m]
        if free:
            parts = []
            for s in free:
                d = min(set(range(base)) - set(m.values()))
                m[s] = d
                parts.append(f"{s}={d}")
            if all(s not in con_syms for s in free):
                why = ("appear only in the query, never in an example -> "
                       "unconstrained")
            else:
                why = "are still unpinned"
            target_lines.append(
                indent + f"Symbol(s) {', '.join(free)} {why}; give them the "
                "smallest unused digits in symbol order: "
                + ", ".join(parts) + ".")
        return m

    nsol = _count_solutions(cons, units, base, endian, con_order)
    if nsol == 1:
        m = run(lines, _guided_chooser(cons, units, base, endian, expect))
        if m != expect:
            raise RuntimeError(f"derived map differs from solver map: "
                               f"{m} vs {expect}")
        return m
    if accept is not None:
        # Not provably unique: any consistent map satisfying `accept` (same
        # boxed answer for the winner; same render failure for a render-fail
        # discard) makes a valid first-survivor scan for this same candidate.
        # Variants: search against the full constraint set or against subsets
        # with one example deferred to the final re-check (every emitted map
        # is verified offline against ALL constraints AND `accept` before the
        # narration is kept; the trace itself re-checks every example too).
        variants = [(cons, units, None)]
        if len(cons) > 1:
            for i in range(len(cons)):
                sub = cons[:i] + cons[i + 1:]
                note = ("Search the digits against "
                        + ", ".join(c[5] for c in sub)
                        + f" first ({cons[i][5]} is re-checked afterwards):")
                variants.append((sub, _units_for(sub, endian), note))
        if len(cons) >= 3:
            for i in range(len(cons)):
                sub = [cons[i]]
                note = (f"Search the digits against {cons[i][5]} first (the "
                        "other examples are re-checked afterwards):")
                variants.append((sub, _units_for(sub, endian), note))
        best = None
        for cons_v, units_v, note in variants:
            syms_v = {x for a, b, o, *_r in cons_v for x in a + b + o}
            order_v = [s for s in dorder if s in syms_v]
            choosers = [_guided_chooser(cons_v, units_v, base, endian,
                                        expect),
                        _guided_chooser(cons_v, units_v, base, endian,
                                        expect, kill_depth=3),
                        _min_branch_chooser(cons_v, units_v, base, endian),
                        _opt_root_chooser(cons_v, units_v, base, endian,
                                          order_v),
                        None]
            if prio_syms:
                choosers.insert(0, _guided_chooser(
                    cons_v, units_v, base, endian, expect, kill_depth=3,
                    prio=set(prio_syms)))
            for chooser in choosers:
                for digit_order in (None, range(base - 1, -1, -1)):
                    try:
                        scratch: list[str] = []
                        if note is not None:
                            scratch.append(indent + note)
                        m = run(scratch, chooser, cons_v, units_v,
                                digit_order)
                    except RuntimeError:
                        continue
                    if any(not _full_check_repr(c, m, base, endian)[0]
                           for c in cons):
                        continue
                    if not accept(m):
                        continue
                    if best is None or len("".join(scratch)) < \
                            len("".join(best[0])):
                        best = (scratch, m)
        if best is not None:
            lines.extend(best[0])
            return best[1]
    m = run(lines, None)
    if m != expect:
        raise RuntimeError(f"derived map differs from solver map: "
                           f"{m} vs {expect}")
    return m


def _refute_narrate(lines, cons, units, base, endian, con_order, indent="  "):
    """Narrate why an (offline-verified) UNSAT system has no digit map:
    one-line certificate -> propagation contradiction -> one-symbol sweep ->
    full narrated search with branch-minimizing splits."""
    cert = _value_cert(cons) or _unit_pair_cert(units, base)
    if cert is not None:
        lines.append(indent + cert + ".")
        return
    known, steps, dead = _propagate(cons, units, base, endian, {})
    if dead is not None:
        for t in steps:
            lines.append(indent + t + ".")
        lines.append(indent + dead + " -> contradiction.")
        return
    best = None
    for s in [x for x in con_order if x not in known]:
        br = _sweep_refute(cons, units, base, endian, known, s)
        if br is not None:
            sz = sum(len(t) for t in br)
            if best is None or sz < best[0]:
                best = (sz, s, br)
    if best is not None:
        for t in steps:
            lines.append(indent + t + ".")
        _sz, s, br = best
        lines.append(indent + f"whichever digit {s} takes, a contradiction "
                     "follows:")
        for t in br:
            lines.append(indent + "  " + t)
        lines.append(indent + f"no digit works for {s} -> no consistent "
                     "cipher.")
        return
    m = _dpll_narrate(lines, cons, units, base, endian, {}, con_order, indent,
                      choose_split=_min_branch_chooser(cons, units, base,
                                                       endian))
    if m is not None:
        raise RuntimeError("refute narration found a map")


_ABBR = [
    (r"(\be\d+) units: ", r"\1u "),
    (r" mod \d+ \(either order\) = ", r" =- "),
    (r" mod \d+ = ", r" == "),
    (r"the only fitting digits are ", "only "),
    (r" \(taken\)", " taken"),
    (r"no digit fits (\S+)", r"none for \1"),
    (r" fits \([^)]*\)", ""),
    (r" fits\b", ""),
    (r"whatever (.+?) is, only", r"any \1: only"),
    (r"carry\(0/1, 0 or 1, units not fixed yet\)", "carry 0/1"),
    (r"carry\((\d), from units ([^)]*)\)", r"carry \1 (\2)"),
    (r"borrow\(0/1, 0 or 1, units not fixed yet\)", "borrow 0/1"),
    (r"borrow\((\d), from units ([^)]*)\)", r"borrow \1 (\2)"),
    (r"A is (-?\d+)\.\.(-?\d+) and B is (-?\d+)\.\.(-?\d+), so ",
     r"A=\1..\2, B=\3..\4: "),
    (r", but the output reads at least (-?\d+)", r" < output >= \1"),
    (r", but the output reads at most (-?\d+)", r" > output <= \1"),
    (r" = the given output", r" checks"),
    (r"only (\S+) is unknown; only", "only"),
    (r"only (\S+) is unknown, and no digit satisfies the \w+( \w+)?",
     r"no digit for \1"),
    (r"no unused digit", "none"),
    (r" -> contradiction\.", " -> no."),
    (r"and the two sides are the same symbol, so", "->"),
    (r" are already taken", " taken"),
    (r"(\S+) and (\S+) are different symbols \(different digits\), and no "
     r"such pair fits", r"no distinct digits for \1,\2 fit"),
    (r"'=multiplication", "'=mul"),
    (r"'=addition", "'=add"),
    (r"'=subtraction", "'=sub"),
    (r"'=absolute difference", "'=|diff|"),
    (r"\(refuted for base (\d+), units-(\w+) above\)",
     r"(refuted above, base \1 units-\2)"),
    (r"\(its alternatives fail the same way\)", "(others fail too)"),
]


def _abbreviate(line):
    import re as _re
    for pat, rep in _ABBR:
        line = _re.sub(pat, rep, line)
    return line


def _witness_narrate(lines, cons, units, base, endian, con_order, indent="  "):
    """Compressed discard story (level 2): forced pins; at a stall, follow the
    single quickest-dying branch and say so. The candidate's failure was
    verified exhaustively offline; the page shows one concrete dead end per
    split instead of the whole tree."""
    def witness(known, depth):
        known2, steps, dead = _propagate(cons, units, base, endian, known)
        out = [t + "." for t in steps]
        if dead is not None:
            out.append(dead + " -> no.")
            return out
        if depth >= 6:
            return None
        rest = [s for s in con_order if s not in known2]
        if not rest:
            raise RuntimeError("witness narration found a map")
        s = rest[0]
        used = set(known2.values())
        ranked = []
        for d in range(base):
            if d in used:
                continue
            _k3, st3, dd3 = _propagate(cons, units, base, endian,
                                       {**known2, s: d})
            ranked.append((0 if dd3 is not None else 1, len(st3), d))
        ranked.sort()
        for _alive, _ns, d in ranked:
            sub = witness({**known2, s: d}, depth + 1)
            if sub is not None:
                head = f"try {s}={d} (its alternatives fail the same way):"
                return out + [head] + ["  " + t for t in sub]
        return None

    r = witness({}, 0)
    if r is None:
        raise RuntimeError("witness narration failed (no short dead end)")
    for t in r:
        lines.append(indent + t)


def _emit_discard(lines, ops_lbl, arith_ops, base, endian, opm_tuple, rec_sol,
                  rec_out, q, dorder, memo, sat_cache, style,
                  short_intro=False, var_ops=None):
    """Discard one pre-winner cipher candidate. A minimal UNSAT core (smallest
    subset of operator readings that already has no digit map) is derived ONCE
    and memoized; later candidates containing the same core are discarded with
    a one-line reference to that earlier derivation (a copy of printed facts).
    style: {'abbr': bool, 'witness': bool} -- discard-only compression levels
    used when a trace would exceed the token budget (winner never compressed)."""
    if short_intro:
        shown = [kv for kv in opm_tuple
                 if var_ops is None or kv[0] in var_ops]
        intro = f"Try {_meanings_text(tuple(shown))}:"
    else:
        intro = (f"Try base {base}, {_ENDIAN_NAME[endian]}, "
                 f"{_meanings_text(opm_tuple)}:")
    opm = dict(opm_tuple)
    if rec_sol is not None:
        # The candidate admits a map but cannot WRITE the query's result
        # (solver render failure): derive a map with the same failure, then
        # show it. Discard-grade content: abbreviated at level >= 1.
        if rec_out is not None:
            raise RuntimeError("discard path reached a renderable candidate")
        body: list[str] = []
        cons = _cons_for(ops_lbl, arith_ops, opm)
        units = _units_for(cons, endian)
        opc = opm[q[2]]
        if style.get("witness"):
            # compressed: forced pins, then the completing assignment of the
            # solver's map with a full per-example verification (every line
            # still checkable; only the dead search branches are omitted).
            known, steps, dead = _propagate(cons, units, base, endian, {})
            if dead is not None:
                raise RuntimeError("render-fail candidate refuted by pins")
            if any(rec_sol.get(s) != d for s, d in known.items()):
                raise RuntimeError("forced pin disagrees with solver map")
            for t in steps:
                body.append("  " + t + ".")
            m = dict(rec_sol)
            rest = [s for s in dorder if s not in known]
            asg = ", ".join(f"{s}={m[s]}" for s in rest)
            body.append("  Completing the search as before (dead branches "
                        f"omitted) settles the rest: {asg}; verify:")
            _emit_verify(body, cons, m, base, endian)
        else:
            m = _derive_map(body, cons, units, base, endian, dorder, rec_sol,
                            accept=lambda mm: _answer_from(mm, q, opc, base,
                                                           endian) is None)
        _emit_map_recap(body, m, dorder)
        qa, qb = [q[0], q[1]], [q[3], q[4]]
        da, ea, A = _val_expr(qa, m, base, endian)
        db, eb, B = _val_expr(qb, m, base, endian)
        txt, rv = _op_text(opc, A, B)
        body.append(f"  Apply to the query {q}: A={q[0]}{q[1]}: {da} -> "
                    f"{ea}={A}; B={q[3]}{q[4]}: {db} -> {eb}={B}; {txt}.")
        inv = {}
        for c, d in m.items():
            inv.setdefault(d, c)
        mag = abs(rv)
        digs = [0] if mag == 0 else []
        while mag > 0:
            digs.append(mag % base)
            mag //= base
        missing = next(d for d in digs if d not in inv)
        body.append(f"  Writing {abs(rv)} in base {base} needs digit "
                    f"{missing}, but no symbol has digit {missing} -> the "
                    "result cannot be written in this cipher -> discard.")
        lines.append(_abbreviate(intro) if style.get("abbr") else intro)
        if style.get("abbr"):
            body = [_abbreviate(t) for t in body]
        lines.extend(body)
        return
    # Enumerate PROVABLY UNSAT sub-readings (proofs shared across compression
    # levels via sat_cache) and pick the cheapest story: a core already
    # printed above -> one-line reference; otherwise the shortest fresh
    # narration. Node-capped solution counting treats 'unproven' as
    # satisfiable, never the reverse.
    best = None
    order_idx = 0
    for r in range(1, len(arith_ops) + 1):
        for ops_sub in itertools.combinations(arith_ops, r):
            order_idx += 1
            key = (base, endian, tuple((op, opm[op]) for op in ops_sub))
            proof = sat_cache.get(key)
            if proof is None:
                cons_sub = _cons_for(ops_lbl, list(ops_sub), opm)
                syms_sub = sorted({x for a, b, o, *_r2 in cons_sub
                                   for x in a + b + o})
                units_sub = _units_for(cons_sub, endian)
                order_sub = _solver_order(cons_sub, syms_sub)
                cap = 2000000 if r == len(arith_ops) else 120000
                if _count_solutions(cons_sub, units_sub, base, endian,
                                    order_sub, limit=1, node_cap=cap) != 0:
                    proof = ("sat",)
                else:
                    # also keep a minimal UNSAT example subset (fewer symbols
                    # can shorten the story; fewer constraints can weaken the
                    # pruning, so both systems get narrated later).
                    kept = list(range(len(cons_sub)))
                    i = 0
                    while i < len(kept) and len(kept) > 1:
                        trial = [cons_sub[j] for j in
                                 kept[:i] + kept[i + 1:]]
                        syms_t = sorted({x for a, b, o, *_r3 in trial
                                         for x in a + b + o})
                        units_t = _units_for(trial, endian)
                        order_t = _solver_order(trial, syms_t)
                        if _count_solutions(trial, units_t, base, endian,
                                            order_t, limit=1,
                                            node_cap=120000) == 0:
                            kept = kept[:i] + kept[i + 1:]
                        else:
                            i += 1
                    proof = ("unsat", kept)
                sat_cache[key] = proof
            if proof[0] == "sat":
                continue
            st = memo.get(key)
            if st is None:
                cons_sub = _cons_for(ops_lbl, list(ops_sub), opm)
                kept_cons = [cons_sub[j] for j in proof[1]]
                label = ", ".join(f"'{op}'={_OPNAME[opm[op]]}"
                                  for op in ops_sub)
                narration = None
                systems = ([kept_cons, cons_sub]
                           if len(kept_cons) < len(cons_sub) else [cons_sub])
                for sys_cons in systems:
                    syms_k = sorted({x for a, b, o, *_r3 in sys_cons
                                     for x in a + b + o})
                    units_k = _units_for(sys_cons, endian)
                    order_k = _solver_order(sys_cons, syms_k)
                    cand_n: list[str] = []
                    try:
                        if style.get("witness"):
                            _witness_narrate(cand_n, sys_cons, units_k, base,
                                             endian, order_k)
                        else:
                            _refute_narrate(cand_n, sys_cons, units_k, base,
                                            endian, order_k)
                    except RuntimeError:
                        continue
                    if narration is None or len("".join(cand_n)) < \
                            len("".join(narration)):
                        narration = cand_n
                if narration is not None and style.get("abbr"):
                    narration = [_abbreviate(t) for t in narration]
                memo[key] = [narration, label, False]
                st = memo[key]
            narration, label, emitted = st
            if emitted:
                cost = (0, order_idx)
            elif narration is None:
                continue
            else:
                cost = (sum(len(t) + 1 for t in narration), order_idx)
            if best is None or cost < best[0]:
                best = (cost, key, narration, label, emitted,
                        len(key[2]) < len(arith_ops))
    abbr = (lambda t: _abbreviate(t)) if style.get("abbr") else (lambda t: t)
    if best is None:
        # The solver's own bounded search gave this candidate up (node budget)
        # without finding a map; mirror that policy on the page: show the
        # forced start of the derivation, then set the candidate aside.
        cons = _cons_for(ops_lbl, arith_ops, opm)
        units = _units_for(cons, endian)
        known, steps, dead = _propagate(cons, units, base, endian, {})
        if dead is not None:
            # forced pins alone refute it (the offline UNSAT prover had given
            # up under its node cap): print that refutation directly.
            lines.append(abbr(intro))
            for t in steps:
                lines.append("  " + abbr(t) + ".")
            lines.append("  " + abbr(dead) + " -> contradiction -> discard.")
            return
        lines.append(abbr(intro))
        for t in steps[:4]:
            lines.append("  " + abbr(t) + ".")
        lines.append("  Nothing else is forced and the remaining search does "
                     "not finish within the step budget -> set this candidate "
                     "aside and continue the scan.")
        return
    _cost, key, narration, label, emitted, proper = best
    if emitted:
        return ("ref", key, label, intro)
    if proper and not style.get("abbr"):
        lines.append(f"{intro} already the partial reading {label} fails:")
    else:
        lines.append(abbr(intro))
    lines.extend(narration)
    if style.get("abbr") and lines:
        lines[-1] += " -> discard."
    else:
        lines.append("  -> discard this candidate.")
    memo[key][2] = True


def _emit_map_recap(lines, m, dorder):
    recap = ", ".join(f"{s}={m[s]}" for s in dorder)
    lines.append(f"  Digit map complete: {recap}.")


def _emit_verify(lines, cons, m, base, endian):
    lines.append("  Re-check every cipher example end-to-end:")
    for a, b, o, neg, opc, eid, lhs, rhs in cons:
        da, ea, A = _val_expr(a, m, base, endian)
        db, eb, B = _val_expr(b, m, base, endian)
        do, eo, R = _val_expr(o, m, base, endian)
        sgn = ""
        if neg:
            R = -R
            sgn = "-"
        txt, v = _op_text(opc, A, B)
        if v != R:
            raise RuntimeError("winner fails verification")
        lines.append(f"    {eid} ({lhs} = {rhs}): A: {da} -> {ea}={A}; "
                     f"B: {db} -> {eb}={B}; {txt}; output {rhs}: {do} -> "
                     f"{sgn}({eo})={R} -> match.")
    lines.append(f"  All {len(cons)} cipher examples are reproduced.")


def _emit_apply_cipher(lines, q, m, base, endian, opc, ans):
    qa, qb = [q[0], q[1]], [q[3], q[4]]
    da, ea, A = _val_expr(qa, m, base, endian)
    db, eb, B = _val_expr(qb, m, base, endian)
    txt, rv = _op_text(opc, A, B)
    lines.append(f"Apply the cipher to the query {q} ('{q[2]}' = {_OPNAME[opc]}):")
    lines.append(f"  A={q[0]}{q[1]}: {da} -> {ea}={A}; B={q[3]}{q[4]}: "
                 f"{db} -> {eb}={B}; {txt}.")
    inv = {}
    for c, d in m.items():
        inv.setdefault(d, c)
    neg = rv < 0
    mag = abs(rv)
    digs = [0] if mag == 0 else []
    steps = []
    while mag > 0:
        steps.append(f"{mag} = {mag // base}*{base} + {mag % base} -> "
                     f"digit {mag % base} = {inv[mag % base]}")
        digs.append(mag % base)
        mag //= base
    if not steps:
        steps.append(f"0 -> digit 0 = {inv[0]}")
    lines.append(f"  Write {abs(rv)} in base {base}, units digit first: "
                 + "; ".join(steps) + ".")
    seq = digs if endian == "little" else digs[::-1]
    word = "".join(inv[d] for d in seq)
    order_txt = ("units digit first" if endian == "little"
                 else "leading digit first")
    sgn_txt = ""
    if neg:
        sgn_txt = f"; the result is negative, prefix '-'"
    lines.append(f"  Symbols written {order_txt}: {word}{sgn_txt}.")
    final = ("-" if neg else "") + word
    if final != ans:
        raise RuntimeError(f"apply step produced {final}, solver answer {ans}")


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


# --------------------------------------------------------------------------- #
# Forward fail-fast emitter: top-level traces
# --------------------------------------------------------------------------- #
def _emit_echo(lines, exs, q):
    lines.append("Examples:")
    for i, (lhs, rhs) in enumerate(exs, start=1):
        lines.append(f"  e{i}: {lhs} = {rhs}")
    lines.append(f"Query: {q} -- its operator is '{q[2]}'.")


def _emit_rearrange_trace(lines, exs, q, seq, ans):
    qop = q[2]
    ops_lbl = _labeled_ops(exs)
    _emit_echo(lines, exs, q)
    lines.append(f"Family (1) scan for '{qop}' (only its own examples matter "
                 "for a selection rule):")
    if not _sel_scan(lines, qop, ops_lbl[qop], winner_seq=seq):
        raise RuntimeError("rearrange winner not reached in scan")
    qoc = _operands(q)
    names = " ".join(_POS[i] for i in seq)
    lab = " ".join(f"{_POS[i]}={qoc[i]}" for i in range(4))
    pred = "".join(qoc[i] for i in seq)
    if pred != ans:
        raise RuntimeError("rearrange apply mismatch")
    lines.append(f"Apply to the query {q}: {lab}; take {names}: {ans}.")


def _meaning_lines(lines, opch, group_lbl):
    """Derive the possible meanings of one cipher operator on the page;
    asserted equal to _op_candidates' pruning."""
    outs = [rhs for _, rhs, _ in group_lbl]
    anyneg = any(r.startswith("-") for r in outs)
    maxout = max(len(r.lstrip("-")) for r in outs)
    if anyneg:
        eid = next(e for _, r, e in group_lbl if r.startswith("-"))
        lines.append(f"  '{opch}': output of {eid} is marked negative; +, * "
                     f"and |difference| of values are never negative -> "
                     f"'{opch}' = subtraction.")
        cand = [SUB]
    elif maxout >= 4:
        eid = next(e for _, r, e in group_lbl if len(r.lstrip("-")) == maxout)
        lines.append(f"  '{opch}': output of {eid} has {maxout} symbols; "
                     f"2-symbol words are < base^2, so A+B < 2*base^2 "
                     f"(at most 3 digits) and A-B, |A-B| are < base^2 "
                     f"(at most 2); only A*B reaches 4 digits -> "
                     f"'{opch}' = multiplication.")
        cand = [MUL]
    elif maxout >= 3:
        eid = next(e for _, r, e in group_lbl if len(r.lstrip("-")) == maxout)
        lines.append(f"  '{opch}': output of {eid} has 3 symbols; A-B and "
                     f"|A-B| of 2-symbol words stay < base^2 (at most 2 "
                     f"digits) -> '{opch}' = addition or multiplication "
                     "(tried in that order).")
        cand = [ADD, MUL]
    else:
        lines.append(f"  '{opch}': outputs stay within 2 symbols and none is "
                     f"negative -> '{opch}' could be addition, |difference|, "
                     "multiplication or subtraction (tried in that order).")
        cand = [ADD, ABS, MUL, SUB]
    if cand != _op_candidates(outs):
        raise RuntimeError("meaning derivation diverges from _op_candidates")
    return cand


def _emit_arith_trace(lines, exs, q, ctx, rec, picked, sat_cache, style):
    ans, hyp = picked
    _, base_w, endian_w, opm_w, sol_w = hyp
    qop = q[2]
    ops_lbl = _labeled_ops(exs)
    arith_ops = ctx["arith_ops"]
    _emit_echo(lines, exs, q)

    # ---- step 1: classify every operator (selection vs cipher) -------------
    lines.append("Step 1 -- which operators are selection rules?")
    lines.append(f"Family (1) scan for '{qop}':")
    if _sel_scan(lines, qop, ops_lbl[qop], winner_seq=None):
        raise RuntimeError("query op classified rearrange in arith path")
    for opch in ops_lbl:
        if opch == qop:
            continue
        if opch in ctx["rearrange_seqs"]:
            _emit_sel_fit(lines, opch, ops_lbl[opch],
                          ctx["rearrange_seqs"][opch])
        else:
            lines.append(f"Family (1) scan for '{opch}':")
            if _sel_scan(lines, opch, ops_lbl[opch], winner_seq=None):
                raise RuntimeError("non-query op scan diverges from solver")
    cipher_ops_txt = ", ".join(f"'{o}'" for o in arith_ops)
    lines.append(f"So family (2) must explain operator(s) {cipher_ops_txt}: "
                 "one shared digit cipher covering all their examples plus "
                 "the query symbols.")

    # ---- step 2: possible meanings per cipher operator ---------------------
    lines.append("Step 2 -- what can each cipher operator mean?")
    cand = {}
    for opch in arith_ops:
        cand[opch] = _meaning_lines(lines, opch, ops_lbl[opch])

    digit_syms = set()
    for opch in arith_ops:
        for lhs, rhs, _eid in ops_lbl[opch]:
            digit_syms.update(_operands(lhs))
            digit_syms.update(rhs.lstrip("-"))
    digit_syms.update(_operands(q))
    digit_syms = sorted(digit_syms)
    nd = len(digit_syms)
    if nd > 10:
        skipped = [b for b in (10, 11, 12, 13) if b < nd]
        lines.append(f"  The cipher must cover {nd} distinct symbols with "
                     f"distinct digits, so base(s) {skipped} are impossible.")

    # ---- step 3: scan cipher candidates in priority order ------------------
    lines.append("Step 3 -- scan the cipher candidates (a negative-marked "
                 "output -xy stands for the value -(xy)):")
    if style.get("abbr"):
        lines.append("(notation in discards: 'eNu' = units column of eN; "
                     "'x == y' = x mod base is y; '|a-b| =- z' = z matches "
                     "a-b or b-a mod base.)")
    # Mirror the solver's pick to order the scan (assertion-checked).
    mirror: dict[str, list] = {}
    for idx, (b, e, o, sol, out) in enumerate(rec):
        if out is not None:
            mirror.setdefault(out, []).append((idx, b, e, o, sol))
    best = None
    for out, hyps in mirror.items():
        for idx, b, e, o, sol in hyps:
            sc = _cand_score(b, e, o)
            if best is None or sc > best[0]:
                best = (sc, out, b, e, o, sol)
    if best is None:
        raise RuntimeError("no consistent candidate recorded for arith winner")
    wscore = _cand_score(base_w, endian_w, opm_w)
    if (best[1], best[2], best[3], best[4]) != (ans, base_w, endian_w, opm_w):
        raise RuntimeError("scan-order mirror diverges from _pick")
    pre = [(idx, b, e, o, sol, out)
           for idx, (b, e, o, sol, out) in enumerate(rec)
           if _cand_score(b, e, o) > wscore]
    if any(out is not None for *_x, out in pre):
        raise RuntimeError("consistent candidate outranks the picked winner")
    pre.sort(key=lambda r: (-_cand_score(r[1], r[2], r[3]), r[0]))
    memo: dict = {}
    abbr = (lambda t: _abbreviate(t)) if style.get("abbr") else (lambda t: t)
    refbuf: list = []

    def flush_refs():
        if not refbuf:
            return
        if len(refbuf) == 1:
            _k, label, intro, b2, e2 = refbuf[0]
            where = ("" if style.get("abbr")
                     else f" for base {b2}, {_ENDIAN_NAME[e2]}")
            lines.append(abbr(f"{intro} contains {label} (refuted{where} "
                              "above) -> discard."))
        else:
            _k, label, _intro, b2, e2 = refbuf[0]
            lines.append(abbr(f"Try the next {len(refbuf)} candidates: each "
                              f"contains {label}, refuted above -> discard "
                              "them all."))
        refbuf.clear()

    cur_block = [None]
    for _idx, b, e, o, sol, out in pre:
        cons_c = _cons_for(ops_lbl, arith_ops, dict(o))
        dorder_c = _solver_order(cons_c, digit_syms)
        short = bool(style.get("abbr"))
        sub: list = []
        r = _emit_discard(sub, ops_lbl, arith_ops, b, e, o, sol, out, q,
                          dorder_c, memo, sat_cache, style, short_intro=short,
                          var_ops=[op for op in arith_ops
                                   if len(cand[op]) > 1])
        if short and cur_block[0] != (b, e):
            flush_refs()
            fixed = [(op, dict(o)[op]) for op in arith_ops
                     if len(cand[op]) == 1]
            fx = (" (" + _abbreviate(_meanings_text(tuple(fixed)))
                  + " is forced)") if fixed else ""
            lines.append(f"Cipher candidates for base {b}, "
                         f"{_ENDIAN_NAME[e]}{fx}:")
            cur_block[0] = (b, e)
        if r is not None and r[0] == "ref":
            _tag, key, label, intro = r
            if refbuf and refbuf[0][0] != (key, b, e):
                flush_refs()
            refbuf.append(((key, b, e), label, intro, b, e))
        else:
            flush_refs()
            lines.extend(sub)
    flush_refs()

    # ---- the winning candidate: full forward derivation --------------------
    cons_w = _cons_for(ops_lbl, arith_ops, dict(opm_w))
    units_w = _units_for(cons_w, endian_w)
    dorder_w = _solver_order(cons_w, digit_syms)
    lines.append(f"Try base {base_w}, {_ENDIAN_NAME[endian_w]}, "
                 f"{_meanings_text(opm_w)}:")
    qopc_w = dict(opm_w)[qop]
    m = _derive_map(lines, cons_w, units_w, base_w, endian_w, dorder_w, sol_w,
                    accept=lambda mm: _answer_from(mm, q, qopc_w, base_w,
                                                   endian_w) == ans,
                    prio_syms=set(_operands(q)))
    _emit_map_recap(lines, m, dorder_w)
    _emit_verify(lines, cons_w, m, base_w, endian_w)
    lines.append("This candidate survives every example -> stop the scan and "
                 "use it.")
    _emit_apply_cipher(lines, q, m, base_w, endian_w, dict(opm_w)[qop], ans)


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

    rec: list = []
    answers, status, ctx = _enumerate_answers(exs, q, rec=rec)
    picked = _pick(answers, status)

    if picked is None:
        # Unsolved: keep the legacy trace byte-for-byte (these answers are
        # wrong on train, so the correctness filter keeps them out of the
        # corpus; behavior of which problems get traces/boxes is unchanged).
        lines = [_HEADER]
        ans = _trace_fallback(lines, ctx["ops"], q, status)
        lines.append("The answer is \\boxed{" + ans + "}")
        return "\n".join(lines)

    ans, hyp = picked
    if hyp[0] == "rearrange":
        lines = [_FF_HEADER]
        _emit_rearrange_trace(lines, exs, q, hyp[1], ans)
        lines.append("The answer is \\boxed{" + ans + "}")
        return "\n".join(lines)

    # Arithmetic winner: emit at the lightest compression level that fits the
    # token budget (discard sections only; the winner derivation, the example
    # verification and the query application are never compressed).
    sat_cache: dict = {}
    best = None
    last_err = None
    for level in (0, 1, 2):
        style = {"abbr": level >= 1, "witness": level >= 2}
        lines = [_FF_HEADER]
        try:
            _emit_arith_trace(lines, exs, q, ctx, rec, picked, sat_cache,
                              style)
        except RuntimeError as e:
            last_err = e
            continue
        lines.append("The answer is \\boxed{" + ans + "}")
        trace = "\n".join(lines)
        ntok = _completion_tokens(trace, ans)
        if best is None or ntok < best[0]:
            best = (ntok, trace)
        if ntok <= _TOKEN_BUDGET:
            break
    if best is None:
        raise last_err
    return best[1]


_TOKEN_BUDGET = 6500
_TOKENIZER: list = ["unset"]


def _get_tokenizer():
    if _TOKENIZER[0] == "unset":
        try:
            from pathlib import Path
            from tokenizers import Tokenizer
            path = Path(__file__).resolve().parent.parent / "tokenizer.json"
            _TOKENIZER[0] = Tokenizer.from_file(str(path))
        except Exception:
            _TOKENIZER[0] = None
    return _TOKENIZER[0]


def _completion_tokens(trace, ans):
    completion = f"{trace}\n</think>\n\\boxed{{{ans}}}<|im_end|>"
    tok = _get_tokenizer()
    if tok is None:
        return len(completion) // 2  # conservative fallback
    return len(tok.encode(completion, add_special_tokens=False).ids)


if __name__ == "__main__":
    def mk(exs, q, ans):
        return Problem(id="t", category="cryptarithm_deduce",
                       examples=[Example(i, o) for i, o in exs], question=q, answer=ans)
    p = mk([("##+|(", "##|("), ("/#+}/", "/#}/")], ">/+%(", ">/%(")
    print(reasoning_cryptarithm_deduce(p))
    p2 = mk([("12+34", "46"), ("56+11", "67"), ("21-11", "10")], "43-12", "31")
    print(reasoning_cryptarithm_deduce(p2))
