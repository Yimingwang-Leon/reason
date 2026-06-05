"""Reasoning generator for 8-bit bit-manipulation tasks.

Two-stage solver:

1. STRUCTURAL search (primary). Each problem is a single global transform of the
   8-bit word. We search, in simplicity order, a library of position-relative
   primitives that *extrapolate* to unseen inputs by construction (they are
   closed-form boolean/shift formulas, not memorised truth tables):
       - unary: identity, circular rotate L/R by k, logical shift L/R by k,
         bitwise NOT, full reverse, plus one-step compositions of those;
       - binary: OP(u1(x), u2(x)) for OP in XOR/AND/OR/XNOR/NAND/NOR and the
         NOT-second variants, with u1, u2 unary primitives;
       - ternary: XOR3 / majority of three rotations, and templates such as
         multiplex (a&b)|(~a&c), (a&b)|c, (a|b)&c, (a&b)^c, ...
   The first formula consistent with *every* example wins (Occam / simplicity
   prior); it is evaluated on the query directly, so it generalises even when the
   query's bit pattern never appears among the examples.

2. LEGACY per-bit fallback. When no global structural formula fits, fall back to
   the column-matching trace (kept verbatim below as ``_reasoning_legacy``).
"""

from __future__ import annotations

import itertools as _it
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple

from .store_types import Problem

N_BITS = 8

_MASK = 0xFF


def _bits_to_int(bits: str) -> int:
    """MSB-first 8-bit string -> int (string index 0 is the high bit)."""
    v = 0
    for ch in bits:
        v = (v << 1) | (1 if ch == "1" else 0)
    return v


def _int_to_bits(v: int) -> str:
    return "".join(str((v >> (N_BITS - 1 - j)) & 1) for j in range(N_BITS))


def _u_rotl(v: int, k: int) -> int:
    k %= N_BITS
    if k == 0:
        return v & _MASK
    return ((v << k) | (v >> (N_BITS - k))) & _MASK


def _u_rotr(v: int, k: int) -> int:
    return _u_rotl(v, (N_BITS - (k % N_BITS)) % N_BITS)


def _u_shl(v: int, k: int) -> int:
    return (v << k) & _MASK


def _u_shr(v: int, k: int) -> int:
    return (v >> k) & _MASK


def _u_not(v: int) -> int:
    return (~v) & _MASK


def _u_rev(v: int) -> int:
    r = 0
    for j in range(N_BITS):
        r = (r << 1) | ((v >> j) & 1)
    return r


def _unary_atoms() -> List[Tuple[str, object, int]]:
    """Unary primitives ordered by cost (lower == simpler == preferred)."""
    atoms: List[Tuple[str, object, int]] = [("identity", lambda v: v & _MASK, 0)]
    for k in range(1, N_BITS):
        atoms.append((f"rotate-left-{k}", (lambda v, k=k: _u_rotl(v, k)), 1))
        atoms.append((f"rotate-right-{k}", (lambda v, k=k: _u_rotr(v, k)), 1))
    for k in range(1, N_BITS):
        atoms.append((f"shift-left-{k}", (lambda v, k=k: _u_shl(v, k)), 2))
        atoms.append((f"shift-right-{k}", (lambda v, k=k: _u_shr(v, k)), 2))
    atoms.append(("NOT", _u_not, 2))
    atoms.append(("reverse", _u_rev, 3))
    return atoms


_UATOM = _unary_atoms()


def _unary_ext() -> List[Tuple[str, object, int]]:
    """Unary operands for binary ops: atoms plus 'atom then NOT'."""
    ext = list(_UATOM)
    for name, fn, cost in _UATOM:
        if name in ("identity", "NOT"):
            continue
        ext.append((f"NOT({name})", (lambda v, fn=fn: _u_not(fn(v))), cost + 2))
    return ext


_UEXT = _unary_ext()

_BINOPS: List[Tuple[str, object, int]] = [
    ("XOR", lambda a, b: a ^ b, 1),
    ("AND", lambda a, b: a & b, 1),
    ("OR", lambda a, b: a | b, 1),
    ("XNOR", lambda a, b: _u_not(a ^ b), 2),
    ("AND-NOT", lambda a, b: a & _u_not(b), 2),
    ("OR-NOT", lambda a, b: a | _u_not(b), 2),
    ("NAND", lambda a, b: _u_not(a & b), 2),
    ("NOR", lambda a, b: _u_not(a | b), 2),
]

_TERNARY: List[Tuple[str, object]] = [
    ("MUX(sel={0}, then={1}, else={2})", lambda a, b, c: (a & b) | (_u_not(a) & c)),
    ("({0} AND {1}) OR {2}", lambda a, b, c: (a & b) | c),
    ("({0} OR {1}) AND {2}", lambda a, b, c: (a | b) & c),
    ("({0} AND {1}) XOR {2}", lambda a, b, c: (a & b) ^ c),
    ("({0} OR {1}) XOR {2}", lambda a, b, c: (a | b) ^ c),
    ("({0} AND-NOT {1}) OR {2}", lambda a, b, c: (a & _u_not(b)) | c),
]

# Atoms used as ternary operands (drop the rarely-useful full reverse to bound cost).
_TATOM = [(n, f) for (n, f, c) in _UATOM if n != "reverse"]


class _Rule:
    """A discovered global transform: label, evaluator, cost."""

    __slots__ = ("kind", "label", "fn", "cost")

    def __init__(self, kind: str, label: str, fn: object, cost: int):
        self.kind = kind
        self.label = label
        self.fn = fn
        self.cost = cost

    def apply(self, v: int) -> int:
        return self.fn(v) & _MASK  # type: ignore[operator]


def _search_unary(ivs: List[int], ovs: List[int]) -> List[_Rule]:
    rules: List[_Rule] = []
    n = len(ivs)
    for name, fn, cost in _UATOM:
        if all(fn(ivs[e]) == ovs[e] for e in range(n)):
            rules.append(_Rule("unary", name, fn, cost))
    return rules


def _search_unary_compose(ivs: List[int], ovs: List[int]) -> List[_Rule]:
    rules: List[_Rule] = []
    n = len(ivs)
    for n1, f1, c1 in _UATOM:
        t = [f1(ivs[e]) for e in range(n)]
        for n2, f2, c2 in _UATOM:
            if n2 == "identity":
                continue
            if all(f2(t[e]) == ovs[e] for e in range(n)):
                rules.append(
                    _Rule(
                        "compose",
                        f"{n1} then {n2}",
                        (lambda v, f1=f1, f2=f2: f2(f1(v))),
                        c1 + c2 + 1,
                    )
                )
    return rules


def _search_binary(ivs: List[int], ovs: List[int]) -> List[_Rule]:
    rules: List[_Rule] = []
    n = len(ivs)
    for bn, bf, bc in _BINOPS:
        for n1, f1, c1 in _UEXT:
            t1 = [f1(ivs[e]) for e in range(n)]
            for n2, f2, c2 in _UEXT:
                if all(bf(t1[e], f2(ivs[e])) == ovs[e] for e in range(n)):
                    rules.append(
                        _Rule(
                            "binary",
                            f"{bn}({n1}, {n2})",
                            (lambda v, f1=f1, f2=f2, bf=bf: bf(f1(v), f2(v))),
                            bc + c1 + c2 + 3,
                        )
                    )
    return rules


def _search_ternary(ivs: List[int], ovs: List[int]) -> List[_Rule]:
    n = len(ivs)
    atoms = [(name, fn) for (name, fn, _c) in _UATOM]
    for combo in _it.combinations(range(len(atoms)), 3):
        f1 = atoms[combo[0]][1]
        f2 = atoms[combo[1]][1]
        f3 = atoms[combo[2]][1]
        if all(f1(ivs[e]) ^ f2(ivs[e]) ^ f3(ivs[e]) == ovs[e] for e in range(n)):
            label = (
                f"XOR3({atoms[combo[0]][0]}, {atoms[combo[1]][0]}, "
                f"{atoms[combo[2]][0]})"
            )
            return [
                _Rule(
                    "ternary",
                    label,
                    (lambda v, f1=f1, f2=f2, f3=f3: f1(v) ^ f2(v) ^ f3(v)),
                    10,
                )
            ]
    for combo in _it.combinations(range(len(atoms)), 3):
        f1 = atoms[combo[0]][1]
        f2 = atoms[combo[1]][1]
        f3 = atoms[combo[2]][1]

        def _maj(v: int, f1=f1, f2=f2, f3=f3) -> int:
            return (f1(v) & f2(v)) | (f1(v) & f3(v)) | (f2(v) & f3(v))

        if all(_maj(ivs[e]) == ovs[e] for e in range(n)):
            label = (
                f"MAJ({atoms[combo[0]][0]}, {atoms[combo[1]][0]}, "
                f"{atoms[combo[2]][0]})"
            )
            return [_Rule("ternary", label, _maj, 11)]
    return []


def _search_ternary_templates(ivs: List[int], ovs: List[int]) -> List[_Rule]:
    n = len(ivs)
    for fmt, tf in _TERNARY:
        for na, fa in _TATOM:
            ta = [fa(ivs[e]) for e in range(n)]
            for nb, fb in _TATOM:
                tb = [fb(ivs[e]) for e in range(n)]
                for nc, fc in _TATOM:
                    if all(
                        tf(ta[e], tb[e], fc(ivs[e])) == ovs[e] for e in range(n)
                    ):
                        return [
                            _Rule(
                                "template",
                                fmt.format(na, nb, nc),
                                (
                                    lambda v, fa=fa, fb=fb, fc=fc, tf=tf: tf(
                                        fa(v), fb(v), fc(v)
                                    )
                                ),
                                20,
                            )
                        ]
    return []


def _find_structural_rule(
    inputs: List[str], outputs: List[str]
) -> Optional[_Rule]:
    """Return the simplest global transform consistent with all examples, else None."""
    ivs = [_bits_to_int(b) for b in inputs]
    ovs = [_bits_to_int(b) for b in outputs]

    rules = _search_unary(ivs, ovs)
    rules += _search_unary_compose(ivs, ovs)
    rules += _search_binary(ivs, ovs)
    if rules:
        rules.sort(key=lambda r: (r.cost, r.label))
        return rules[0]

    tern = _search_ternary(ivs, ovs)
    if tern:
        return tern[0]

    tmpl = _search_ternary_templates(ivs, ovs)
    if tmpl:
        return tmpl[0]
    return None


def _structural_trace(
    inputs: List[str], outputs: List[str], question_bits: str, rule: _Rule
) -> str:
    """Deterministic CoT for a discovered global structural rule."""
    lines: List[str] = []
    lines.append(
        "We need to deduce the single bit-manipulation rule that maps each "
        "8-bit input to its output."
    )
    lines.append("I will put my final answer inside \\boxed{}.")
    lines.append("")
    lines.append("Examples (input -> output):")
    for inp, out in zip(inputs, outputs):
        lines.append(f"  {inp} -> {out}")
    lines.append("")
    lines.append(
        "Searching position-relative operations (rotations, shifts, NOT, "
        "reverse and their boolean combinations), simplest first."
    )
    lines.append(
        f"The simplest rule consistent with every example is: {rule.label}."
    )
    lines.append("")
    lines.append("Verifying the rule on each example:")
    for inp, out in zip(inputs, outputs):
        got = _int_to_bits(rule.apply(_bits_to_int(inp)))
        mark = "ok" if got == out else "MISMATCH"
        lines.append(f"  {inp} -> {got} (expected {out}) {mark}")
    lines.append("")
    answer = _int_to_bits(rule.apply(_bits_to_int(question_bits)))
    lines.append(f"Applying {rule.label} to the query {question_bits}:")
    lines.append(f"  {question_bits} -> {answer}")
    lines.append("")
    lines.append(f"The answer is \\boxed{{{answer}}}")
    return "\n".join(lines)


SYM_FAMILIES = ("XOR", "OR", "AND")
ASYM_FAMILIES = ("AND-NOT", "XOR-NOT", "OR-NOT")
PAIR_FAMILIES = SYM_FAMILIES + ASYM_FAMILIES
UNARY_FAMILIES = ("I", "NOT")
CONSTANT_FAMILIES = ("0", "1")
DEFAULT_FAMILY: RuleFamily = "DEFAULT"
SECTION_ORDER = (
    "Identity",
    "NOT",
    "Constant",
    "AND",
    "OR",
    "XOR",
    "AND-NOT",
    "OR-NOT",
    "XOR-NOT",
)

# Map section names to their constituent family codes.
_SECTION_TO_FAMILIES = {
    "Identity": ("I",),
    "NOT": ("NOT",),
    "Constant": ("0", "1"),
}

# Reverse map: family code → section name.
_FAMILY_TO_SECTION: dict[str, str] = {}
for _section in SECTION_ORDER:
    for _fam in _SECTION_TO_FAMILIES.get(_section, (_section,)):
        _FAMILY_TO_SECTION[_fam] = _section


RuleFamily = Literal[
    "I",
    "NOT",
    "0",
    "1",
    "XOR",
    "OR",
    "AND",
    "AND-NOT",
    "XOR-NOT",
    "OR-NOT",
    "DEFAULT",
]


@dataclass(frozen=True)
class RuleCandidate:
    family: RuleFamily
    primary: Optional[int]
    secondary: Optional[int]
    expr: str
    primary_stride: Optional[int] = None  # always +1 (stored as 1)
    secondary_stride: Optional[int] = None  # always +1 (stored as 1)
    primary_offset: Optional[int] = (
        None  # primary at bit 0: primary = (offset + bit * stride) % 8
    )
    secondary_offset: Optional[int] = (
        None  # secondary at bit 0: secondary = (offset + bit * stride) % 8
    )

    @property
    def is_default(self) -> bool:
        return self.family == DEFAULT_FAMILY


@dataclass(frozen=True)
class Record:
    label: str
    col: str
    hash_: str
    matches: Tuple[int, ...]


def _normalize_bits(value: str) -> str:
    bits = "".join(ch for ch in str(value) if ch in {"0", "1"})
    if len(bits) != N_BITS:
        return ""
    return bits


def _column_bits(values: Sequence[str], bit: int) -> str:
    return "".join(v[bit] for v in values)


def _bit_not(bit: str) -> str:
    return "1" if bit == "0" else "0"


def _invert(bits: str) -> str:
    return "".join(_bit_not(b) for b in bits)


def _column_hash(bits: str, total_examples: int) -> str:
    ones = bits.count("1")
    if ones == 0 or ones == total_examples:
        return "a"
    return format(ones, "x")


def _evaluate_binary(a: str, b: str, family: str) -> str:
    if family in ("AND", "AND-NOT"):
        return "1" if a == "1" and b == "1" else "0"
    if family in ("OR", "OR-NOT"):
        return "1" if a == "1" or b == "1" else "0"
    if family in ("XOR", "XOR-NOT"):
        return "1" if a != b else "0"
    raise ValueError(f"Unsupported family {family}")


def _apply_family(
    a_bits: str, b_bits: str, family: str, invert_second: bool = False
) -> str:
    b_eff = _invert(b_bits) if invert_second else b_bits
    out = []
    for x, y in zip(a_bits, b_eff):
        out.append(_evaluate_binary(x, y, family))
    return "".join(out)


def _find_match(
    candidates: List[RuleCandidate], fam: str, ep: Optional[int], es: Optional[int]
) -> Optional[RuleCandidate]:
    """Find candidate matching (fam, ep, es) by direct lookup."""
    for c in candidates:
        if c.family != fam:
            continue
        if c.primary == ep and (fam not in PAIR_FAMILIES or c.secondary == es):
            return c
    return None


def _exists_anywhere(
    all_matches: List[List[RuleCandidate]],
    fam: str,
    ep: Optional[int],
    es: Optional[int],
) -> bool:
    """Check if operand pair (ep, es) exists in any bit position for this family."""
    for bit_cands in all_matches:
        if _find_match(bit_cands, fam, ep, es) is not None:
            return True
    return False


def _fail_suffix(
    all_matches: List[List[RuleCandidate]],
    fam: str,
    ep: Optional[int],
    es: Optional[int],
) -> str:
    """Return 'y' if operand exists somewhere (wrong position), 'x' if nowhere."""
    if _exists_anywhere(all_matches, fam, ep, es):
        return "y"
    return "x"


def _find_all_left_runs(
    all_matches: List[List[RuleCandidate]],
) -> List[Tuple[List[RuleCandidate], Optional[str]]]:
    """All stride-consistent runs from bit 0, all stride combos per starter.

    Returns list of (chain, failed_next_expr) tuples.
    """
    if not all_matches or not all_matches[0]:
        return []
    runs: List[Tuple[List[RuleCandidate], Optional[str]]] = []
    for start_cand in all_matches[0]:
        fam = start_cand.family
        strides = [(1, 1)]
        for p_step, s_step in strides:
            chain = [start_cand]
            # Track expected position independently (don't use found candidate's operands)
            cur_p = start_cand.primary
            cur_s = start_cand.secondary
            failed_next: Optional[str] = None
            for b in range(1, len(all_matches)):
                ep = (cur_p + p_step) % N_BITS if cur_p is not None else None
                es = (cur_s + s_step) % N_BITS if cur_s is not None else None
                found = _find_match(all_matches[b], fam, ep, es)
                if found is None:
                    suffix = _fail_suffix(all_matches, fam, ep, es)
                    if ep is not None and es is not None:
                        failed_next = f"{ep}{es}{suffix}"
                    elif ep is not None:
                        failed_next = f"{ep}{suffix}"
                    break
                chain.append(found)
                cur_p, cur_s = ep, es
            runs.append((chain, failed_next))
    return runs


def _find_all_right_runs(
    all_matches: List[List[RuleCandidate]],
) -> List[Tuple[List[RuleCandidate], Optional[str]]]:
    """All stride-consistent runs ending at last bit, all stride combos per ender.

    Returns list of (chain, failed_next_expr) tuples.
    """
    n = len(all_matches)
    if not all_matches or not all_matches[-1]:
        return []
    runs: List[Tuple[List[RuleCandidate], Optional[str]]] = []
    for end_cand in all_matches[-1]:
        fam = end_cand.family
        strides = [(1, 1)]
        for p_step, s_step in strides:
            chain = [end_cand]
            # Track expected position independently
            cur_p = end_cand.primary
            cur_s = end_cand.secondary
            failed_next: Optional[str] = None
            for k in range(1, n):
                b = n - 1 - k
                pp = (cur_p - p_step) % N_BITS if cur_p is not None else None
                ps = (cur_s - s_step) % N_BITS if cur_s is not None else None
                found = _find_match(all_matches[b], fam, pp, ps)
                if found is None:
                    suffix = _fail_suffix(all_matches, fam, pp, ps)
                    if pp is not None and ps is not None:
                        failed_next = f"{pp}{ps}{suffix}"
                    elif pp is not None:
                        failed_next = f"{pp}{suffix}"
                    break
                chain.insert(0, found)
                cur_p, cur_s = pp, ps
            runs.append((chain, failed_next))
    return runs


def _lr_from_matches(
    all_matches: List[List[RuleCandidate]],
) -> Tuple[List[str], str, List[str], str]:
    """Compute Left/Right from full per-bit match lists.

    Returns (left_all_lines, left_best, right_all_lines, right_best).
    """
    all_left_runs = _find_all_left_runs(all_matches)
    all_right_runs = _find_all_right_runs(all_matches)
    left_run = max(all_left_runs, key=lambda t: len(t[0])) if all_left_runs else ([], None)
    right_run = max(all_right_runs, key=lambda t: len(t[0])) if all_right_runs else ([], None)

    left_lines = (
        [_format_list(chain, failed=failed) for chain, failed in all_left_runs]
        if all_left_runs
        else ["none"]
    )
    left_best = _format_list(left_run[0], with_count=True)
    right_lines = (
        [
            _format_list(list(reversed(chain)), failed=failed)
            for chain, failed in all_right_runs
        ]
        if all_right_runs
        else ["none"]
    )
    right_best = _format_list(list(reversed(right_run[0])), with_count=True)

    return left_lines, left_best, right_lines, right_best


def _format_list(
    cands: List[RuleCandidate],
    with_count: bool = False,
    failed: Optional[str] = None,
) -> str:
    if not cands:
        return "none"
    if with_count:
        parts = []
        for i, c in enumerate(cands):
            if i == 0:
                parts.append(c.expr)
            else:
                parts.append(_compact_rule(c))
        return " ".join(parts) + f": {len(cands)}"
    parts = [_compact_rule(c) for c in cands]
    if failed:
        parts.append(failed)
    return " ".join(parts)


def _compact_rule(c: RuleCandidate) -> str:
    """Compact display: just the operand indices without family prefix."""
    if c.primary is not None and c.secondary is not None:
        return f"{c.primary}{c.secondary}"
    if c.primary is not None:
        return str(c.primary)
    return c.family


def _evaluate_rule(bits: str, rule: RuleCandidate) -> str:
    if rule.family == "DEFAULT":
        return "1"
    if rule.family == "0":
        return "0"
    if rule.family == "1":
        return "1"
    if rule.family == "I":
        assert rule.primary is not None
        return bits[rule.primary]
    if rule.family == "NOT":
        assert rule.primary is not None
        return _bit_not(bits[rule.primary])
    if rule.family in PAIR_FAMILIES:
        assert rule.primary is not None and rule.secondary is not None
        a = bits[rule.primary]
        b = bits[rule.secondary]
        if "-NOT" in rule.family:
            b = _bit_not(b)
        return _evaluate_binary(a, b, rule.family)
    raise ValueError(f"Unknown family {rule.family}")


# Family preference for resolving ambiguous fills: simpler rules win ties.
_RESOLVE_FAM_RANK: Dict[str, int] = {
    fam: idx
    for idx, fam in enumerate(
        ("I", "NOT", "0", "1", "XOR", "OR", "AND", "XOR-NOT", "OR-NOT", "AND-NOT")
    )
}


def _all_consistent_per_bit(
    all_matches: Dict[str, List[List[RuleCandidate]]],
) -> List[List[RuleCandidate]]:
    """Flatten all per-section consistent candidates into one list per output bit.

    A candidate is consistent if its column equals the output column across every
    example (that is exactly what `all_matches` records). Returns, for each output
    bit, the full deduplicated candidate list in deterministic family/operand order.
    """
    per_bit: List[List[RuleCandidate]] = []
    for bit in range(N_BITS):
        seen: set[Tuple[str, Optional[int], Optional[int]]] = set()
        cands: List[RuleCandidate] = []
        for section in SECTION_ORDER:
            for c in all_matches[section][bit]:
                key = (c.family, c.primary, c.secondary)
                if key in seen:
                    continue
                seen.add(key)
                cands.append(c)
        cands.sort(
            key=lambda c: (
                _RESOLVE_FAM_RANK.get(c.family, 99),
                c.primary if c.primary is not None else -1,
                c.secondary if c.secondary is not None else -1,
            )
        )
        per_bit.append(cands)
    return per_bit


def _spine_rule_at(
    fam: str, p_off: int, q_off: Optional[int], bit: int
) -> Tuple[str, Optional[int], Optional[int]]:
    """Rule key produced by a stride-1 spine pattern at a given output bit."""
    p = (p_off + bit) % N_BITS
    if q_off is None:
        return (fam, p, None)
    return (fam, p, (q_off + bit) % N_BITS)


def _longest_spines(
    per_bit: List[List[RuleCandidate]],
) -> List[Tuple[int, str, int, Optional[int]]]:
    """Find stride-1 spine patterns ordered by longest contiguous coverage.

    A spine is (family, primary_offset, secondary_offset) such that for a run of
    consecutive output bits the operands advance by +1 each step and the resulting
    rule is consistent. Returns (max_run, family, p_off, q_off) tuples sorted by
    (max contiguous run desc, family rank, offsets) — most reliable spine first.
    """
    sets = [
        {(c.family, c.primary, c.secondary) for c in cands} for cands in per_bit
    ]
    scored: List[Tuple[int, int, int, int, Tuple[str, int, Optional[int]]]] = []
    # Unary spines.
    for fam in ("I", "NOT"):
        for off in range(N_BITS):
            best_run = 0
            bit = 0
            while bit < N_BITS:
                if _spine_rule_at(fam, off, None, bit) in sets[bit]:
                    start = bit
                    while (
                        bit < N_BITS
                        and _spine_rule_at(fam, off, None, bit) in sets[bit]
                    ):
                        bit += 1
                    best_run = max(best_run, bit - start)
                else:
                    bit += 1
            if best_run > 0:
                scored.append(
                    (best_run, _RESOLVE_FAM_RANK[fam], off, 0, (fam, off, None))
                )
    # Binary spines.
    for fam in PAIR_FAMILIES:
        for p_off in range(N_BITS):
            for d in range(1, N_BITS):
                q_off = (p_off + d) % N_BITS
                best_run = 0
                bit = 0
                while bit < N_BITS:
                    if _spine_rule_at(fam, p_off, q_off, bit) in sets[bit]:
                        start = bit
                        while (
                            bit < N_BITS
                            and _spine_rule_at(fam, p_off, q_off, bit) in sets[bit]
                        ):
                            bit += 1
                        best_run = max(best_run, bit - start)
                    else:
                        bit += 1
                if best_run > 0:
                    scored.append(
                        (
                            best_run,
                            _RESOLVE_FAM_RANK.get(fam, 99),
                            p_off,
                            q_off,
                            (fam, p_off, q_off),
                        )
                    )
    # Longest run first; then simpler family; then smallest offsets (deterministic).
    scored.sort(key=lambda s: (-s[0], s[1], s[2], s[3]))
    return [(s[0], s[4][0], s[4][1], s[4][2]) for s in scored]


def _spine_covers_with_neighbor(
    fam: str,
    p_off: int,
    q_off: Optional[int],
    bit: int,
    sets: List[set],
) -> bool:
    """True if `bit` sits in a contiguous stride run of length >= 2 for this spine.

    Requiring an adjacent matching bit rules out length-1 "spines" that are just an
    isolated coincidental candidate carrying no real positional evidence.
    """
    here = _spine_rule_at(fam, p_off, q_off, bit) in sets[bit]
    if not here:
        return False
    left = bit > 0 and _spine_rule_at(fam, p_off, q_off, bit - 1) in sets[bit - 1]
    right = (
        bit < N_BITS - 1
        and _spine_rule_at(fam, p_off, q_off, bit + 1) in sets[bit + 1]
    )
    return left or right


def _resolve_default(
    bit: int,
    per_bit: List[List[RuleCandidate]],
    spines: List[Tuple[int, str, int, Optional[int]]],
    question_bits: str,
) -> Optional[RuleCandidate]:
    """Choose a consistent rule for a bit the main pass left as DEFAULT.

    Priority: (1) extend a dominant stride spine (length >= 2) if it continues
    through this bit; (2) if a strict majority of all consistent candidates agree on
    the output value, use the simplest such candidate. Returns None when neither
    signal is strong enough (keep the conservative DEFAULT rather than risk a
    coincidental match).
    """
    cands = per_bit[bit]
    if not cands:
        return None
    cand_by_key = {(c.family, c.primary, c.secondary): c for c in cands}
    sets = [
        {(c.family, c.primary, c.secondary) for c in pb} for pb in per_bit
    ]
    # (1) Spine continuation — only spines that form a length>=2 run through `bit`.
    for run_len, fam, p_off, q_off in spines:
        if run_len < 2:
            break  # spines are sorted longest-first; nothing useful remains
        if not _spine_covers_with_neighbor(fam, p_off, q_off, bit, sets):
            continue
        key = _spine_rule_at(fam, p_off, q_off, bit)
        if key in cand_by_key:
            return cand_by_key[key]
    # (2) Majority value among all consistent candidates.
    votes: Dict[str, int] = {}
    for c in cands:
        v = _evaluate_rule(question_bits, c)
        votes[v] = votes.get(v, 0) + 1
    total = sum(votes.values())
    top_val, top_n = max(votes.items(), key=lambda kv: (kv[1], kv[0]))
    if top_n * 2 > total:
        for c in cands:  # cands already sorted simplest-first
            if _evaluate_rule(question_bits, c) == top_val:
                return c
    return None


def _emit_apply(
    lines: List[str], question_bits: str, vector: List[RuleCandidate]
) -> None:
    lines.append(f"Applying to {question_bits}")
    lines.append("Input")
    for i, bit in enumerate(question_bits):
        lines.append(f"{i} {bit}")
    lines.append("Output")

    answer_bits: List[str] = []
    for i, rule in enumerate(vector):
        if rule.family == "DEFAULT":
            lines.append(f"{i} default 1 = 1")
            answer_bits.append("1")
            continue
        if rule.family in CONSTANT_FAMILIES:
            lines.append(f"{i} {rule.expr} = {rule.family}")
            answer_bits.append(rule.family)
            continue
        if rule.family == "I":
            assert rule.primary is not None
            val = question_bits[rule.primary]
            lines.append(f"{i} {rule.expr} = {val}")
            answer_bits.append(val)
            continue
        if rule.family == "NOT":
            assert rule.primary is not None
            val = question_bits[rule.primary]
            nval = _bit_not(val)
            lines.append(f"{i} {rule.expr} = NOT({val}) = {nval}")
            answer_bits.append(nval)
            continue

        assert rule.primary is not None and rule.secondary is not None
        a = question_bits[rule.primary]
        b = question_bits[rule.secondary]
        if rule.family in SYM_FAMILIES:
            result = _evaluate_rule(question_bits, rule)
            lines.append(f"{i} {rule.expr} = {rule.family}({a},{b}) = {result}")
            answer_bits.append(result)
            continue

        base = rule.family.split("-")[0]
        result = _evaluate_rule(question_bits, rule)
        lines.append(f"{i} {rule.expr} = {base}({a},NOT({b})) = {result}")
        answer_bits.append(result)

    lines.append("")
    lines.append(f"The answer is \\boxed{{{''.join(answer_bits)}}}")


def _reasoning_legacy(problem: Problem) -> Optional[str]:
    examples = problem.examples
    if not examples:
        return None

    outputs = [_normalize_bits(ex.output_value) for ex in examples]
    inputs = [_normalize_bits(ex.input_value) for ex in examples]
    question_bits = _normalize_bits(problem.question)

    if any(not bits for bits in outputs + inputs) or not question_bits:
        return None

    if len(outputs[0]) != N_BITS or len(inputs[0]) != N_BITS:
        return None

    if len(outputs) != len(inputs):
        return None

    n_examples = len(outputs)

    # 1) Example columns.
    output_columns = [_column_bits(outputs, i) for i in range(N_BITS)]
    input_columns = [_column_bits(inputs, i) for i in range(N_BITS)]
    input_inverted = [_invert(col) for col in input_columns]

    all_records: Dict[str, List[Record]] = {name: [] for name in SECTION_ORDER}
    all_matches: Dict[str, List[List[RuleCandidate]]] = {
        name: [[] for _ in range(N_BITS)] for name in SECTION_ORDER
    }

    # Build unary records and matches.
    for out_idx, out_col in enumerate(output_columns):
        for i_col, in_col in enumerate(input_columns):
            if in_col == out_col:
                all_matches["Identity"][out_idx].append(
                    RuleCandidate("I", i_col, None, f"I{i_col}")
                )
            if input_inverted[i_col] == out_col:
                all_matches["NOT"][out_idx].append(
                    RuleCandidate("NOT", i_col, None, f"NOT{i_col}")
                )
        if out_col.count("1") == 0:
            all_matches["Constant"][out_idx].append(
                RuleCandidate("0", None, None, "C0")
            )
        if out_col.count("1") == n_examples:
            all_matches["Constant"][out_idx].append(
                RuleCandidate("1", None, None, "C1")
            )

    # Build unary raw records.
    for label, col in zip([str(i) for i in range(N_BITS)], input_columns):
        matches = tuple(i for i, oc in enumerate(output_columns) if col == oc)
        all_records["Identity"].append(
            Record(
                label=label,
                col=col,
                hash_=_column_hash(col, n_examples),
                matches=matches,
            )
        )
    for label, col in zip([str(i) for i in range(N_BITS)], input_inverted):
        matches = tuple(i for i, oc in enumerate(output_columns) if col == oc)
        all_records["NOT"].append(
            Record(
                label=label,
                col=col,
                hash_=_column_hash(col, n_examples),
                matches=matches,
            )
        )
    for val in ("0", "1"):
        col = val * n_examples
        matches = tuple(i for i, oc in enumerate(output_columns) if col == oc)
        all_records["Constant"].append(
            Record(
                label=val, col=col, hash_=_column_hash(col, n_examples), matches=matches
            )
        )

    # Build pair records (ordered by circular difference for symmetric ops).
    fam: RuleFamily
    for fam in ("XOR", "OR", "AND"):
        for circ_diff in range(1, N_BITS // 2 + 1):
            # For circ_diff == N_BITS/2, only half the circle to avoid duplicates
            n_pairs = N_BITS // 2 if circ_diff == N_BITS // 2 else N_BITS
            for a in range(n_pairs):
                b = (a + circ_diff) % N_BITS
                # Canonical pair for the operation: smaller index first
                lo, hi = min(a, b), max(a, b)
                col = _apply_family(input_columns[lo], input_columns[hi], fam)
                matches = tuple(
                    i for i, out_col in enumerate(output_columns) if col == out_col
                )
                all_records[fam].append(
                    Record(
                        label=f"{a}{b} {b}{a}",
                        col=col,
                        hash_=_column_hash(col, n_examples),
                        matches=matches,
                    )
                )
                for out_idx in matches:
                    all_matches[fam][out_idx].append(
                        RuleCandidate(fam, a, b, f"{fam}{a}{b}")
                    )
                    all_matches[fam][out_idx].append(
                        RuleCandidate(fam, b, a, f"{fam}{b}{a}")
                    )

    for fam in ("AND-NOT", "XOR-NOT", "OR-NOT"):
        for diff in range(1, N_BITS):
            for a in range(N_BITS):
                b = (a + diff) % N_BITS
                col = _apply_family(
                    input_columns[a], input_columns[b], fam, invert_second=True
                )
                matches = tuple(
                    i for i, out_col in enumerate(output_columns) if col == out_col
                )
                all_records[fam].append(
                    Record(
                        label=f"{a}{b}",
                        col=col,
                        hash_=_column_hash(col, n_examples),
                        matches=matches,
                    )
                )
                for out_idx in matches:
                    all_matches[fam][out_idx].append(
                        RuleCandidate(fam, a, b, f"{fam}{a}{b}")
                    )

    # Deterministic order for unary/constant records (pair records already ordered by diff).
    for name in ("Identity", "NOT", "Constant"):
        all_records[name].sort(key=lambda r: r.label)

    lines: List[str] = []

    # 1) header
    lines.append(
        "We need to deduce the transformation by matching the example outputs."
    )
    lines.append("I will put my final answer inside \\boxed{}.")
    lines.append("")

    # 2) output examples
    for i, out in enumerate(outputs):
        lines.append(f"Output {i}: {out}")
        for bit in range(N_BITS):
            lines.append(f"{bit} {out[bit]}")
        lines.append("")

    # 3) output bit columns
    lines.append("Output bit columns (with bitsum as hash)")
    for bit in range(N_BITS):
        lines.append(
            f"{bit} {output_columns[bit]} {_column_hash(output_columns[bit], n_examples)}"
        )

    # 4) input examples
    lines.append("")
    for i, inp in enumerate(inputs):
        lines.append(f"Input {i}: {inp}")
        for bit in range(N_BITS):
            lines.append(f"{bit} {inp[bit]}")
        lines.append("")

    # 5) Operation sections (raw data + matching + LRM)
    lines.append("When matching output")
    lines.append("x: not in operator")
    lines.append("y: wrong position")
    lines.append("")
    section_lefts: list[tuple[str, str]] = []  # (name, left_best)
    section_rights: list[tuple[str, str]] = []  # (name, right_best)

    def _add_section(name: str) -> None:
        records = all_records[name]
        per_bit = all_matches[name]
        # Raw data
        lines.append(name)
        prev_diff = None
        for rec in records:
            # Insert blank line between diff groups for pair operations
            if (
                len(rec.label) >= 2
                and rec.label[0].isdigit()
                and rec.label[1].isdigit()
            ):
                diff = (int(rec.label[1]) - int(rec.label[0])) % N_BITS
                if prev_diff is not None and diff != prev_diff:
                    lines.append("")
                prev_diff = diff
            line = f"{rec.label} {rec.col} {rec.hash_}"
            if rec.matches:
                line += " match " + " ".join(str(i) for i in rec.matches)
            lines.append(line)
        lines.append("")
        # Matching: per output bit, which candidates match
        lines.append("Matching output")
        for i in range(N_BITS):
            cands = per_bit[i]
            if cands:

                def _compact(c: RuleCandidate) -> str:
                    if c.primary is not None and c.secondary is not None:
                        return f"{c.primary}{c.secondary}"
                    if c.primary is not None:
                        return str(c.primary)
                    return c.expr

                lines.append(f"{i} " + " ".join(_compact(c) for c in cands))
            else:
                lines.append(f"{i} absent")
        lines.append("")
        left_lines, left_best, right_lines, right_best = _lr_from_matches(per_bit)
        section_lefts.append((name, left_best))
        section_rights.append((name, right_best))
        lines.append("Left")
        for ll in left_lines:
            lines.append(ll)
        lines.append(f"Best: {left_best}")
        lines.append("")
        lines.append("Right")
        for rl in right_lines:
            lines.append(rl)
        lines.append(f"Best: {right_best}")
        lines.append("")

    for name in all_records:
        _add_section(name)

    # 7) Selecting rule block.
    lines.append("Selecting")
    lines.append("")

    # Pick winners from per-section analysis
    def _parse_count(val: str) -> int:
        if val == "none":
            return 0
        try:
            return int(val.rsplit(": ", 1)[-1])
        except ValueError:
            return 0

    def _pick_winner(
        entries: list[tuple[str, str]],
    ) -> tuple[Optional[str], str, int]:
        best_name: Optional[str] = None
        best_text = "none"
        best_count = 0
        for name, val in entries:
            count = _parse_count(val)
            if count > best_count:
                best_count = count
                best_name = name
                best_text = val
        return best_name, best_text, best_count

    left_winner_name, left_winner_text, left_winner_count = _pick_winner(section_lefts)
    right_winner_name, right_winner_text, right_winner_count = _pick_winner(
        section_rights
    )

    # Get the actual left/right runs from per-section matches
    def _get_section_run(
        winner_name: Optional[str], direction: str
    ) -> List[RuleCandidate]:
        if winner_name is None:
            return []
        per_bit = all_matches[winner_name]
        if direction == "left":
            runs = _find_all_left_runs(per_bit)
        else:
            runs = _find_all_right_runs(per_bit)
        if not runs:
            return []
        best_chain, _ = max(runs, key=lambda t: len(t[0]))
        return best_chain

    left_run = _get_section_run(left_winner_name, "left")
    right_run = _get_section_run(right_winner_name, "right")

    lines.append("Lefts")
    for name, lb in section_lefts:
        lines.append(f"{name} {lb}")
    lines.append("")
    lines.append("Rights")
    for name, rb in section_rights:
        lines.append(f"{name} {rb}")
    lines.append("")
    lines.append(f"Left longest: {left_winner_count}")
    lines.append(f"Right longest: {right_winner_count}")
    lines.append("")

    def _matching_line(
        label: str,
        winner_name: Optional[str],
        entries: list[tuple[str, str]],
    ) -> str:
        parts = []
        for name, _val in entries:
            parts.append(f"{name} {'yes' if name == winner_name else 'no'}")
        return f"{label} winner: {', '.join(parts)}"

    if right_winner_count > left_winner_count:
        lines.append(_matching_line("Right", right_winner_name, section_rights))
        lines.append(_matching_line("Left", left_winner_name, section_lefts))
        lines.append("")
        lines.append(f"Best right: {right_winner_text}")
        lines.append(f"Best left: {left_winner_text}")
    else:
        lines.append(_matching_line("Left", left_winner_name, section_lefts))
        lines.append(_matching_line("Right", right_winner_name, section_rights))
        lines.append("")
        lines.append(f"Best left: {left_winner_text}")
        lines.append(f"Best right: {right_winner_text}")
    lines.append("")

    # Truncate if left + right > N_BITS: shorten the shorter one
    left_len_final = left_winner_count
    right_len_final = right_winner_count
    if left_len_final + right_len_final > N_BITS:
        if right_len_final > left_len_final:
            left_len_final = N_BITS - right_len_final
            left_run = left_run[:left_len_final]
        else:
            right_len_final = N_BITS - left_len_final
            right_run = right_run[-right_len_final:] if right_len_final else []
    left_was_truncated = left_len_final < left_winner_count
    right_was_truncated = right_len_final < right_winner_count
    trunc_left = f"Truncated left: {_format_list(left_run, with_count=True)}"
    if left_was_truncated:
        trunc_left += " truncated"
    trunc_right = f"Truncated right: {_format_list(list(reversed(right_run)), with_count=True)}"
    if right_was_truncated:
        trunc_right += " truncated"
    if right_winner_count > left_winner_count:
        lines.append(trunc_right)
        lines.append(trunc_left)
    else:
        lines.append(trunc_left)
        lines.append(trunc_right)
    lines.append("")

    right_start_final = N_BITS - right_len_final
    lines.append("Tentative from right")
    for i in range(N_BITS - 1, -1, -1):
        if i >= right_start_final and right_run:
            lines.append(f"{i} {right_run[i - right_start_final].expr}")
        else:
            lines.append(f"{i} pending")
    lines.append("")
    lines.append("Tentative")
    for i in range(N_BITS):
        if i < left_len_final:
            lines.append(f"{i} {left_run[i].expr}")
        elif i >= right_start_final and right_run:
            lines.append(f"{i} {right_run[i - right_start_final].expr}")
        else:
            lines.append(f"{i} pending")
    lines.append("")

    # Preferred: extrapolate left/right strides into pending slots
    def _extrap_from(
        run: List[RuleCandidate],
        bit: int,
        run_start_bit: int,
        side: str = "left",
    ) -> Optional[str]:
        if not run:
            return None
        r = run[0]
        # Derive offset from first candidate's position at run_start_bit
        # offset = primary - run_start_bit * stride (mod N_BITS), stride=1
        p = r.primary
        s = r.secondary
        if p is not None:
            p_off = (p - run_start_bit) % N_BITS
            ep = (p_off + bit) % N_BITS
        else:
            ep = None
        if s is not None:
            s_off = (s - run_start_bit) % N_BITS
            es = (s_off + bit) % N_BITS
        else:
            es = None
        if ep is not None and es is not None:
            return f"?{ep}{es}"
        if ep is not None:
            # Unary: show which slot is known
            if side == "left":
                return f"?{ep}?"
            else:
                return f"??{ep}"
        return None

    left_fam = left_run[0].family if left_run else None
    right_fam = right_run[0].family if right_run else None
    left_is_const = left_fam in CONSTANT_FAMILIES if left_fam else False
    right_is_const = right_fam in CONSTANT_FAMILIES if right_fam else False
    left_is_binary = left_fam in PAIR_FAMILIES if left_fam else False
    right_is_binary = right_fam in PAIR_FAMILIES if right_fam else False
    left_is_unary = left_fam in UNARY_FAMILIES if left_fam else False
    right_is_unary = right_fam in UNARY_FAMILIES if right_fam else False

    # Preferred: extrapolate from the longer side first, then fill from the other
    if right_winner_count > left_winner_count:
        # Right is longer: extrapolate from right first
        preferred: list[str] = []
        for i in range(N_BITS):
            if i >= right_start_final and right_run:
                preferred.append(right_run[i - right_start_final].expr)
            elif i < left_len_final:
                preferred.append(left_run[i].expr)
            elif right_is_binary or right_is_unary:
                preferred.append(
                    _extrap_from(right_run, i, right_start_final, "right") or "pending"
                )
            else:
                preferred.append("pending")

        lines.append("Preferred from right")
        for i in range(N_BITS - 1, -1, -1):
            lines.append(f"{i} {preferred[i]}")
        lines.append("")

        # Fill remaining pending from left; merge unary digits
        for i in range(N_BITS):
            if preferred[i] == "pending":
                if left_is_binary or left_is_unary:
                    preferred[i] = _extrap_from(left_run, i, 0, "left") or "?"
                else:
                    preferred[i] = "?"
            elif "?" in preferred[i][1:] and left_is_unary:
                el = _extrap_from(left_run, i, 0, "left")
                if el:
                    # Merge: fill unknown slots
                    merged = list(preferred[i])
                    el_chars = list(el)
                    for j in range(1, min(len(merged), len(el_chars))):
                        if merged[j] == "?" and el_chars[j] != "?":
                            merged[j] = el_chars[j]
                    preferred[i] = "".join(merged)

        lines.append("Preferred from left")
        for i in range(N_BITS):
            lines.append(f"{i} {preferred[i]}")
        lines.append("")
    else:
        # Left is longer or equal: extrapolate from left first
        preferred = []
        for i in range(N_BITS):
            if i < left_len_final:
                preferred.append(left_run[i].expr)
            elif i >= right_start_final and right_run:
                preferred.append(right_run[i - right_start_final].expr)
            elif left_is_binary or left_is_unary:
                preferred.append(
                    _extrap_from(left_run, i, 0, "left") or "pending"
                )
            else:
                preferred.append("pending")

        lines.append("Preferred from left")
        for i in range(N_BITS):
            lines.append(f"{i} {preferred[i]}")
        lines.append("")

        # Fill remaining pending from right; merge unary digits
        for i in range(N_BITS):
            if preferred[i] == "pending":
                if right_is_binary or right_is_unary:
                    preferred[i] = _extrap_from(right_run, i, right_start_final, "right") or "?"
                else:
                    preferred[i] = "?"
            elif "?" in preferred[i][1:] and right_is_unary:
                er = _extrap_from(right_run, i, right_start_final, "right")
                if er:
                    # Merge: fill unknown slots
                    merged = list(preferred[i])
                    er_chars = list(er)
                    for j in range(1, min(len(merged), len(er_chars))):
                        if merged[j] == "?" and er_chars[j] != "?":
                            merged[j] = er_chars[j]
                    preferred[i] = "".join(merged)

        lines.append("Preferred from right")
        for i in range(N_BITS - 1, -1, -1):
            lines.append(f"{i} {preferred[i]}")
        lines.append("")

    lines.append("Preferred")
    for i, pref in enumerate(preferred):
        if pref.startswith("?") and len(pref) == 3 and pref[1] != "?" and pref[2] != "?":
            lines.append(f"{i} {pref} ?{pref[2]}{pref[1]}")
        else:
            lines.append(f"{i} {pref}")
    lines.append("")

    # Build the final vector: left + middle selection + right
    default_cand = RuleCandidate(DEFAULT_FAMILY, None, None, "default 1")
    best: List[RuleCandidate] = [default_cand] * N_BITS

    # Place left and right runs
    for i, rc in enumerate(left_run):
        best[i] = rc
    for i, rc in enumerate(right_run):
        best[right_start_final + i] = rc

    # Fill middle (pending) slots via Matching + Perfect match logic
    lines.append("Matching")
    pending_indices: list[int] = []
    per_bit_cat: dict[str, dict[int, list[RuleCandidate]]] = {
        name: {} for name in SECTION_ORDER
    }

    for i in range(N_BITS):
        pref = preferred[i]
        if not pref.startswith("?") or pref == "?":
            lines.append(f"{i} {best[i].expr}")
            continue

        pending_indices.append(i)
        digits_str = pref[1:]
        pref_digits = [int(d) for d in digits_str if d != "?"]

        checks: list[str] = []
        for section_name in SECTION_ORDER:
            cands = all_matches[section_name][i]
            if section_name in ("Identity", "NOT"):
                found = [c for c in cands if c.primary in pref_digits]
                if found:
                    checks.append(section_name + " " + " ".join(c.expr for c in found))
                    per_bit_cat[section_name][i] = found
                else:
                    checks.append(f"{section_name} absent")
            elif section_name == "Constant":
                if cands:
                    checks.append("Constant " + " ".join(c.expr for c in cands))
                    per_bit_cat["Constant"][i] = list(cands)
                else:
                    checks.append("Constant absent")
            else:
                found_c: Optional[RuleCandidate] = None
                # Try both orderings; prefer the first (as shown in Preferred)
                orderings = []
                want_p = int(pref[1]) if len(pref) > 1 and pref[1] != "?" else None
                want_s = int(pref[2]) if len(pref) > 2 and pref[2] != "?" else None
                orderings.append((want_p, want_s))
                if want_p is not None and want_s is not None and want_p != want_s:
                    orderings.append((want_s, want_p))
                for wp, ws in orderings:
                    for c in cands:
                        if (wp is None or c.primary == wp) and (ws is None or c.secondary == ws):
                            found_c = c
                            break
                    if found_c is not None:
                        break
                if found_c is not None:
                    checks.append(found_c.expr)
                    per_bit_cat[section_name][i] = [found_c]
                else:
                    checks.append(f"{section_name} absent")
        if pref.startswith("?") and len(pref) == 3 and pref[1] != "?" and pref[2] != "?":
            pref_display = f"{pref} ?{pref[2]}{pref[1]}"
        else:
            pref_display = pref
        lines.append(f"{i} {pref_display} - {', '.join(checks)}")
    lines.append("")

    # Perfect match: first category that covers ALL pending bits wins
    lines.append("Perfect match")
    chosen_cat: Optional[str] = None
    for cat in SECTION_ORDER:
        is_perfect = (
            chosen_cat is None
            and bool(pending_indices)
            and all(i in per_bit_cat[cat] for i in pending_indices)
        )
        lines.append(f"{cat} {'yes' if is_perfect else 'no'}")
        if is_perfect:
            chosen_cat = cat
    lines.append("")

    # Matched: use perfect-match category to fill pending slots
    pending_set = set(pending_indices)
    lines.append("Matched")
    for i in range(N_BITS):
        if i in pending_set:
            if chosen_cat and i in per_bit_cat[chosen_cat]:
                best[i] = per_bit_cat[chosen_cat][i][0]
                lines.append(f"{i} {best[i].expr}")
            else:
                # No perfect match — list all candidates for this slot
                all_cands: list[RuleCandidate] = []
                for name in SECTION_ORDER:
                    if i in per_bit_cat[name]:
                        all_cands.extend(per_bit_cat[name][i])
                if all_cands:
                    lines.append(f"{i} " + " ".join(c.expr for c in all_cands))
                    best[i] = all_cands[0]
                else:
                    lines.append(f"{i} none")
                    best[i] = default_cand
        else:
            lines.append(f"{i} {best[i].expr}")
    lines.append("")

    # Resolve any bit still left as DEFAULT using the full consistent-candidate set:
    # extend the dominant stride spine, else take the majority value. This recovers
    # boundary/middle bits the left/right anchoring could not reach.
    per_bit_all = _all_consistent_per_bit(all_matches)
    spines = _longest_spines(per_bit_all)
    lines.append("")
    lines.append("Resolving defaults")
    any_default = False
    for i in range(N_BITS):
        if not best[i].is_default:
            continue
        any_default = True
        repl = _resolve_default(i, per_bit_all, spines, question_bits)
        if repl is not None:
            best[i] = repl
            lines.append(f"{i} {repl.expr}")
        else:
            lines.append(f"{i} keep default 1")
    if not any_default:
        lines.append("none")
    lines.append("")

    # Check if we have any non-default rules
    if all(r.is_default for r in best):
        return None

    lines.append("Selected")
    for i, rule in enumerate(best):
        lines.append(f"{i} {rule.expr}")

    # 8) Apply to question.
    lines.append("")
    _emit_apply(lines, question_bits, best)

    return "\n".join(lines)


def reasoning_bit_manipulation(problem: Problem) -> Optional[str]:
    """Solve an 8-bit transform problem.

    Primary path: find the simplest global structural formula (rotate/shift/NOT/
    reverse plus boolean combinations) consistent with all examples, then evaluate
    it on the query. Such formulas extrapolate to unseen bit patterns by
    construction. Fall back to the legacy per-bit column-matching trace when no
    global formula fits.
    """
    examples = problem.examples
    if not examples:
        return None

    inputs = [_normalize_bits(ex.input_value) for ex in examples]
    outputs = [_normalize_bits(ex.output_value) for ex in examples]
    question_bits = _normalize_bits(problem.question)

    if question_bits and all(inputs) and all(outputs) and len(inputs) == len(outputs):
        rule = _find_structural_rule(inputs, outputs)
        if rule is not None:
            return _structural_trace(inputs, outputs, question_bits, rule)

    return _reasoning_legacy(problem)
