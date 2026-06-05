"""Equation numeric deduce: infer the hidden per-operator rule, apply to the query.

Ports the competition champion's mechanics on top of our stricter all-examples
validation:
  - base-op vocabulary (concat / arithmetic / digit-wise) CROSSED with 4 reversal
    modes: reverse each operand before computing, and/or reverse the result string
    (reversal alone explains ~73% of the champion's wins)
  - per-operator sign convention: an operator whose outputs carry the op-char (or a
    bare '-') as prefix/suffix is marking negatives; solve on the signed domain and
    re-attach the marker only when the computed answer is negative
  - +-1 offset ops (add/sub/mul +-1)
  - our extra modular / bitwise rules at LOW priority (recover ~10 the champion misses)
  - deterministic priority order. The OUTER key is the reversal *mode*: we prefer
    the columnar reading (reverse operands, compute, reverse result == right-to-left
    place-value arithmetic) over plain left-to-right arithmetic, because columnar is
    the most common hidden convention here and wins ambiguous ties more often. The
    two single-reversal modes never win on their own, so they are demoted to a
    last-resort fallback (kept only for hidden-test coverage). Greedy submission
    still gets byte-stable traces because the ordering is fully deterministic.
  - query-op-unseen fallback (absolute difference) instead of emitting '?'

`solve_equation()` exposes the survivor count and the query-op example count for
optional label-confidence analysis. We currently KEEP single-example traces: the
offline holdout showed the priority-ordered first-consistent-rule heuristic still
generalizes (~80% realistic on 1-example query-ops), and we only ever train on
grader-correct traces, so the heuristic taught is a sound one rather than noise.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from .store_types import Example, Problem

_EXPR_RE = re.compile(r"^(\d+)([^\d\s])(\d+)$")

# Modular bases worth trying (our extra over the champion's set).
_MODS = [9, 10, 11, 19, 89, 90, 91, 99, 100, 101]


def _base_candidates(sa: str, sb: str) -> dict[str, str]:
    """name -> value for operands given as strings (may carry leading zeros)."""
    a, b = int(sa), int(sb)
    r: dict[str, str] = {}
    # champion "common"
    r["concat"] = sa + sb
    r["revconcat"] = sb + sa
    r["add"] = str(a + b)
    r["absdiff"] = str(abs(a - b))
    r["negabsdiff"] = str(-abs(a - b))
    r["sub"] = str(a - b)
    r["revsub"] = str(b - a)
    r["mul"] = str(a * b)
    # champion "rare"
    r["mul+1"] = str(a * b + 1)
    r["mul-1"] = str(a * b - 1)
    r["add+1"] = str(a + b + 1)
    r["add-1"] = str(a + b - 1)
    r["sub+1"] = str(a - b + 1)
    r["sub-1"] = str(a - b - 1)
    if a and b:
        r["maxmodmin"] = str(max(a, b) % min(a, b))
    if b:
        r["intdiv"] = str(a // b)
        r["mod"] = str(a % b)
    if a:
        r["revdiv"] = str(b // a)
        r["revmod"] = str(b % a)
    if len(sa) == 2 and len(sb) == 2:
        d1, d2, d3, d4 = int(sa[0]), int(sa[1]), int(sb[0]), int(sb[1])
        r["digabsdiff"] = str(abs(d1 - d3)) + str(abs(d2 - d4))
        r["digaddmod10"] = str((d1 + d3) % 10) + str((d2 + d4) % 10)
        r["digsubmod10"] = str((d1 - d3) % 10) + str((d2 - d4) % 10)
        r["crossmul"] = str(d1 * d3 + d2 * d4)
        r["crossmulrev"] = str(d1 * d4 + d2 * d3)
        r["digmul"] = str(d1 * d3) + str(d2 * d4)
        r["digmulrev"] = str(d1 * d4) + str(d2 * d3)
        r["digsumdiff"] = str((d1 + d2) - (d3 + d4))
        r["digsumsum"] = str((d1 + d2) + (d3 + d4))
        r["digproddiff"] = str(d1 * d2 - d3 * d4)
        r["digprodsum"] = str(d1 * d2 + d3 * d4)
        det = d1 * d4 - d2 * d3
        r["det"] = str(det)
        r["absdet"] = str(abs(det))
    # our extras (LOW priority): modular + bitwise
    for m in _MODS:
        r[f"add%{m}"] = str((a + b) % m)
        r[f"sub%{m}"] = str((a - b) % m)
        r[f"revsub%{m}"] = str((b - a) % m)
    r["xor"] = str(a ^ b)
    r["and"] = str(a & b)
    r["or"] = str(a | b)
    r["max"] = str(max(a, b))
    r["min"] = str(min(a, b))
    return r


# Deterministic priority among base ops: simpler/more-common first. The reversal
# *mode* (columnar vs plain, see _MODE_ORDER below) is the OUTER sort key, so the
# preferred mode always wins first and this op order only breaks ties within a mode.
_ORDER = [
    # natural signed arithmetic first; abs/neg-abs are special cases that
    # coincidentally tie on limited data, so they rank below sub/revsub.
    "concat", "revconcat", "add", "sub", "revsub", "mul", "absdiff", "negabsdiff",
    "mul+1", "mul-1", "add+1", "add-1", "sub+1", "sub-1", "maxmodmin",
    "intdiv", "mod", "revdiv", "revmod",
    "digabsdiff", "digaddmod10", "digsubmod10", "crossmul", "crossmulrev",
    "digmul", "digmulrev", "digsumdiff", "digsumsum", "digproddiff", "digprodsum",
    "det", "absdet",
    # our extras last
    "xor", "and", "or", "max", "min",
    *[f"add%{m}" for m in _MODS],
    *[f"sub%{m}" for m in _MODS],
    *[f"revsub%{m}" for m in _MODS],
]
_ORDER_IDX = {name: i for i, name in enumerate(_ORDER)}

# Mode preference, learned from the holdout (and from the structure of the task):
#
#   (True, True)  -- reverse each operand, compute, reverse the result. This is
#                    exactly *columnar / place-value* arithmetic done right-to-left
#                    (e.g. 97-65 -> 79-56=23 -> "32" == per-column 9-6,7-5). It is by
#                    far the most common hidden convention in this family.
#   (False, False) -- plain left-to-right arithmetic on the literal numbers.
#
# These two account for EVERY mode that ever produces a correct answer on the
# holdout (60 vs 48 of the wins). The single-reversal modes never win a problem on
# their own and merely *shadow* the columnar interpretation when they happen to fit
# a sparse example set, so we demote them to a last-resort fallback that only fires
# when neither columnar nor plain arithmetic is consistent.
#
# When BOTH columnar and plain arithmetic fit the demonstrated examples (an
# inherently ambiguous tie, common with a single example), we break toward the
# columnar reading: on the holdout it is the correct hidden rule ~55% more often
# than the plain reading, and it is the more "deliberate" of the two conventions
# (plain arithmetic rarely needs to be *taught* with examples).
_MODE_ORDER = [(True, True), (False, False), (False, True), (True, False)]
_MODE_IDX = {m: i for i, m in enumerate(_MODE_ORDER)}


def _rev(s: str) -> str:
    return ("-" + s[1:][::-1]) if s.startswith("-") else s[::-1]


def _value(name: str, sa: str, sb: str, rev_ops: bool, rev_res: bool) -> str | None:
    ta, tb = (sa[::-1], sb[::-1]) if rev_ops else (sa, sb)
    c = _base_candidates(ta, tb)
    if name not in c:
        return None
    return _rev(c[name]) if rev_res else c[name]


def _parse(s: str) -> tuple[str, str, str] | None:
    m = _EXPR_RE.match(s.strip())
    return (m.group(1), m.group(2), m.group(3)) if m else None


def _detect_fmt(op_char: str, group: list[tuple[str, str, str]]) -> tuple[str, dict]:
    """Detect the per-operator sign convention. Returns (fmt, transformed_map)."""
    neg_suf = op_char != "-" and any(o.endswith("-") and len(o) > 1 for _, _, o in group)
    neg_pre = op_char != "-" and any(o.startswith("-") and len(o) > 1 for _, _, o in group)
    sym_suf = any(o.endswith(op_char) and len(o) > 1 for _, _, o in group)
    sym_pre = any(o.startswith(op_char) and len(o) > 1 for _, _, o in group)
    fmt = "num"
    tmap: dict[tuple[str, str], str] = {}
    for a, b, o in group:
        t = o
        if neg_suf and o.endswith("-") and len(o) > 1:
            t, fmt = "-" + o[:-1], "neg_suffix"
        elif neg_pre and o.startswith("-") and len(o) > 1:
            fmt = "neg_prefix"
        elif sym_suf and o.endswith(op_char) and len(o) > 1:
            t, fmt = "-" + o[: -len(op_char)], "sym_suffix"
        elif sym_pre and o.startswith(op_char) and len(o) > 1:
            t, fmt = "-" + o[len(op_char):], "sym_prefix"
        tmap[(a, b)] = t
    return fmt, tmap


def _reattach(final: str, fmt: str, op_char: str) -> str:
    if fmt in ("neg_suffix", "sym_suffix") and final.startswith("-"):
        return final[1:] + (op_char if fmt == "sym_suffix" else "-")
    if fmt in ("neg_prefix", "sym_prefix") and final.startswith("-"):
        return (op_char if fmt == "sym_prefix" else "-") + final[1:]
    return final


def _survivors(group: list[tuple[str, str, str]], tmap: dict) -> list[tuple[str, bool, bool]]:
    """All (name, rev_ops, rev_res) consistent with EVERY transformed example,
    sorted by deterministic priority (mode first, then op order)."""
    found: list[tuple[str, bool, bool]] = []
    for name in _ORDER:
        for ro, rr in _MODE_ORDER:
            ok = True
            for a, b, _o in group:
                if _value(name, a, b, ro, rr) != tmap[(a, b)]:
                    ok = False
                    break
            if ok:
                found.append((name, ro, rr))
    found.sort(key=lambda x: (_MODE_IDX[(x[1], x[2])], _ORDER_IDX[x[0]]))
    return found


@dataclass
class SolveResult:
    answer: str
    rule: str
    fmt: str
    op_char: str
    n_examples_qop: int
    n_survivors: int
    fallback: bool


def solve_equation(problem: Problem) -> SolveResult | None:
    """Deduce the query operator's rule and apply it. None if unparseable."""
    by_op: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for ex in problem.examples:
        p = _parse(str(ex.input_value))
        if p is None:
            return None
        a, op, b = p
        by_op[op].append((a, b, str(ex.output_value)))

    q = _parse(str(problem.question))
    if q is None:
        return None
    qa, qop, qb = q
    group = by_op.get(qop, [])

    if not group:
        # query operator never demonstrated: champion's abs-diff prior (rarely right,
        # so these traces get dropped by the correctness filter — harmless)
        return SolveResult(str(abs(int(qa) - int(qb))), "absdiff(fallback)", "num",
                           qop, 0, 0, True)

    fmt, tmap = _detect_fmt(qop, group)
    survivors = _survivors(group, tmap)
    if not survivors:
        return SolveResult(str(abs(int(qa) - int(qb))), "absdiff(fallback)", fmt,
                           qop, len(group), 0, True)

    name, ro, rr = survivors[0]
    val = _value(name, qa, qb, ro, rr)
    if val is None:
        return SolveResult(str(abs(int(qa) - int(qb))), "absdiff(fallback)", fmt,
                           qop, len(group), len(survivors), True)
    answer = _reattach(val, fmt, qop)
    mode = {"": ""}.get("", "")
    mode = (" | rev_ops" if ro else "") + (" | rev_res" if rr else "")
    return SolveResult(answer, name + mode, fmt, qop, len(group), len(survivors), False)


_NICE = {
    "concat": "concatenate a,b", "revconcat": "concatenate b,a", "add": "a+b",
    "absdiff": "|a-b|", "negabsdiff": "-|a-b|", "sub": "a-b", "revsub": "b-a",
    "mul": "a*b", "mul+1": "a*b+1", "mul-1": "a*b-1", "add+1": "a+b+1",
    "add-1": "a+b-1", "sub+1": "a-b+1", "sub-1": "a-b-1", "maxmodmin": "max%min",
    "intdiv": "a//b", "mod": "a%b", "revdiv": "b//a", "revmod": "b%a",
    "xor": "a XOR b", "and": "a AND b", "or": "a OR b", "max": "max(a,b)", "min": "min(a,b)",
}


def reasoning_equation_numeric_deduce(problem: Problem) -> str | None:
    """Generate the chain-of-thought for an equation_numeric_deduce problem."""
    res = solve_equation(problem)
    if res is None:
        return None

    q = _parse(str(problem.question))
    if q is None:
        return None
    qa, qop, qb = q

    L: list[str] = []
    L.append("We need to infer the hidden transformation rule for each operator from the examples.")
    L.append("I will put my final answer inside \\boxed{}.")
    L.append("")
    L.append("Examples:")
    for ex in problem.examples:
        L.append(f"  {ex.input_value} = {ex.output_value}")
    L.append("")

    # group + sign format for the query operator
    by_op: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for ex in problem.examples:
        p = _parse(str(ex.input_value))
        if p:
            by_op[p[1]].append((p[0], p[2], str(ex.output_value)))
    group = by_op.get(qop, [])

    L.append(f"The query uses operator '{qop}'.")
    if not group:
        L.append("This operator never appears in the examples, so its rule cannot be "
                 "inferred; falling back to the absolute difference.")
        L.append(f"|{qa} - {qb}| = {res.answer}")
        L.append("")
        L.append(f"The answer is \\boxed{{{res.answer}}}")
        return "\n".join(L)

    fmt, tmap = _detect_fmt(qop, group)
    L.append(f"Examples for '{qop}':")
    for a, b, o in group:
        L.append(f"  {a} {qop} {b} = {o}")
    if fmt != "num":
        if fmt in ("sym_suffix", "neg_suffix"):
            mk = qop if fmt == "sym_suffix" else "-"
            L.append(f"Some outputs carry a trailing '{mk}' marking a negative value; "
                     f"we solve on the signed value and re-attach '{mk}' when negative.")
        else:
            mk = qop if fmt == "sym_prefix" else "-"
            L.append(f"Some outputs carry a leading '{mk}' marking a negative value; "
                     f"we solve on the signed value and re-attach '{mk}' when negative.")
        L.append("Signed outputs: " + ", ".join(tmap[(a, b)] for a, b, _ in group))
    L.append("")

    # show the deduction
    if res.fallback:
        L.append("No single rule is consistent with all examples; falling back to "
                 f"the absolute difference: |{qa} - {qb}| = {res.answer}")
        L.append("")
        L.append(f"The answer is \\boxed{{{res.answer}}}")
        return "\n".join(L)

    name = res.rule.split(" | ")[0]
    ro = "rev_ops" in res.rule
    rr = "rev_res" in res.rule
    nice = _NICE.get(name, name)
    L.append("Testing candidate rules against every example for this operator:")
    # show the winning rule verified on each example
    for a, b, o in group:
        v = _value(name, a, b, ro, rr)
        L.append(f"  {a},{b}: {nice}"
                 + (" with reversed operands" if ro else "")
                 + (" then reverse the result" if rr else "")
                 + f" -> {v}  (target {tmap[(a, b)]})")
    L.append(f"Consistent rule for '{qop}': {nice}"
             + (" [reverse operands]" if ro else "")
             + (" [reverse result]" if rr else "") + ".")
    L.append("")

    # apply to query
    ta, tb = (qa[::-1], qb[::-1]) if ro else (qa, qb)
    raw = _value(name, qa, qb, ro, rr)
    L.append(f"Apply to {qa} {qop} {qb}:")
    if ro:
        L.append(f"  reverse operands -> {ta}, {tb}")
    L.append(f"  {nice} = {raw}")
    if res.answer != raw:
        L.append(f"  re-attach sign marker -> {res.answer}")
    L.append("")
    L.append(f"The answer is \\boxed{{{res.answer}}}")
    return "\n".join(L)


if __name__ == "__main__":
    def mk(exs, q, ans):
        return Problem(id="t", category="equation_numeric_deduce",
                       examples=[Example(i, o) for i, o in exs], question=q, answer=ans)

    # reversal: 64-65 computed as 46-56 = -10 -> abs? test a real reversed case
    print(reasoning_equation_numeric_deduce(mk(
        [("12@34", "1234"), ("56@78", "5678")], "69@52", "6952")))
