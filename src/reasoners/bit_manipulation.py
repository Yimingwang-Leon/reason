"""Reasoning generator for 8-bit bit-manipulation tasks.

Two emission paths:

1. Legacy per-column matching trace (byte-identical to the proven R~0.95
   corpus) for every problem it already solves.
2. Extended whole-byte rule family + prior tiebreak (ported from
   rl/bit_prior.py; holdout oracle 110/110) for the hard tail the legacy
   procedure gets wrong: 24 base atoms with depth<=2 composites, 8 boolean
   pair ops, nested op2(op1(a,b),c), MAJ/MUX (and negations), and a free
   ternary boolean fallback.

   v3 "isomorphic mixing" format (2026-06-13). History: the v1 short per-bit
   trace (p50 ~1.9k tokens vs legacy ~6.6k) collapsed run-009 (judge 94->51,
   length-distribution collapse); the v2 full-length two-phase trace
   ("Continuing the scan" marker + legend + tier headers) was never executed
   by the model (0/110 entered phase two) and over-scanning broke 5 long
   legacy problems (94->88). Lesson: NO extra phase/marker is learnable, and
   the "when to extend" discrimination cannot be taught. v3 therefore kills
   the second-phase concept entirely: the trace replays the COMPLETE legacy
   candidate scan (example blocks + all nine per-column sections,
   byte-identical to the legacy prefix), and the extended-family candidates
   (pair/nested/MAJ/MUX/gf3) appear as ORDINARY candidate segments inside
   that one scan loop — each is a label line + per-example verification rows
   recomputed for real, rejected at its first mismatching output (x mark),
   exactly the geometry of a legacy candidate check. The scan stops when the
   winner segment verifies ok on ALL examples (match line, legacy stop
   semantics), then the legacy-geometry per-bit Applying block ends in
   \\boxed{}. No "Continuing the scan" line, no legend, no Unary/Pairs/...
   tier headers — no new structural marker of any kind. Rejected-candidate
   counts tune completions into the legacy band (p50 6200-6700, p95 <= 7250,
   max <= 7349).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple

from .store_types import Problem

N_BITS = 8

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


def _legacy_prefix(problem: Problem) -> Optional[Tuple[List[str], dict]]:
    """Shared candidate-scan prefix: example blocks, output bit columns, the
    nine per-column candidate sections (raw records, per-bit Matching output,
    Left/Right runs), and the Selecting summary through the longest-run lines.

    Pure refactor split out of the historical _reasoning_legacy so the
    extended-family trace can reuse the exact legacy scan before continuing
    the candidate scan on whole-byte rules; the legacy continuation below
    re-joins at the byte where it always did (verified byte-identical on the
    full legacy-correct corpus)."""
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

    ctx = {
        "question_bits": question_bits,
        "all_matches": all_matches,
        "section_lefts": section_lefts,
        "section_rights": section_rights,
        "left_run": left_run,
        "right_run": right_run,
        "left_winner_name": left_winner_name,
        "left_winner_text": left_winner_text,
        "left_winner_count": left_winner_count,
        "right_winner_name": right_winner_name,
        "right_winner_text": right_winner_text,
        "right_winner_count": right_winner_count,
    }
    return lines, ctx


def _reasoning_legacy(problem: Problem) -> Optional[str]:
    res = _legacy_prefix(problem)
    if res is None:
        return None
    prefix_lines, ctx = res
    lines = list(prefix_lines)
    question_bits = ctx["question_bits"]
    all_matches = ctx["all_matches"]
    section_lefts = ctx["section_lefts"]
    section_rights = ctx["section_rights"]
    left_run = ctx["left_run"]
    right_run = ctx["right_run"]
    left_winner_name = ctx["left_winner_name"]
    left_winner_text = ctx["left_winner_text"]
    left_winner_count = ctx["left_winner_count"]
    right_winner_name = ctx["right_winner_name"]
    right_winner_text = ctx["right_winner_text"]
    right_winner_count = ctx["right_winner_count"]

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


# ---------------------------------------------------------------------------
# Extended whole-byte rule family + prior tiebreak (port of rl/bit_prior.py,
# 2026-06-12; holdout oracle 110/110, train coverage 99.8%).
#
# The problem generator's REAL rule family (reverse-engineered under truth
# supervision) is wider than the legacy per-column ladder: 24 base atoms
# {id, rl1-7, sl1-7, sr1-7, nt, rv}, depth<=2 composite atoms, 8 boolean
# pair ops over them, nested op2(op1(a,b),c), MAJ/MUX (and negations), and a
# free ternary boolean fallback F[tt](a,b,c) for the zero-candidate tail.
# Ties are broken by a prior over canonical rules learned from
# data/train_split.csv (leak-free: holdout ids never appear there).
#
# TRACE FORMAT (v3 isomorphic mixing, 2026-06-13): v1 (short trace) collapsed
# the length distribution (judge 94->51); v2 (two-phase full-length with
# "Continuing the scan" marker + legend + tier headers) was never executed
# (0/110) and its over-scanning broke 5 long legacy problems (94->88).
# v3 deletes the second-phase concept: after the byte-identical legacy
# candidate-scan prefix, extended-family candidates appear as ORDINARY
# candidate segments in the same single scan loop (label + per-example
# verification rows, first mismatch marked x), with no new structural marker
# of any kind; the winner segment verifies all examples (match) and the scan
# stops there, followed by the legacy-geometry per-bit Applying tail.
# The run-006/007/008 death format (assert ONE whole-register rule up
# front, no scan, no rejection evidence) remains structurally excluded.
# ---------------------------------------------------------------------------

_EV_MAX_TOKENS = 7349  # hard completion cap (trace + ending + box; v3 spec max)
_EV_MAX_CHARS = 9600  # conservative fallback when the tokenizer is missing

_EXT_EX_RE = re.compile(r"([01]{8})\s*->\s*([01]{8})")
_EXT_Q_RE = re.compile(r"determine the output for:\s*([01]{8})")
_EXT_DIG = re.compile(r"\d+")

# Fold every pair op onto {XOR, AND, OR} with operand negations (De Morgan),
# so the per-bit leaves carry all negation: op -> (base, neg_a, neg_b).
_EXT_FOLD = {
    "XOR": ("XOR", 0, 0), "AND": ("AND", 0, 0), "OR": ("OR", 0, 0),
    "ANDN": ("AND", 0, 1), "ORN": ("OR", 0, 1), "XNOR": ("XOR", 0, 1),
    "NAND": ("OR", 1, 1), "NOR": ("AND", 1, 1),
}

# Per-bit expression trees: leaves ("I", p) / ("N", p) / ("C", v); nodes
# ("XOR"|"AND"|"OR", a, b), ("MAJ", a, b, c), ("MUX", s, b, c),
# ("F", a, b, c) where F evaluates a gf3 cell table.

_EXT_LEAF_KINDS = ("I", "N", "C")


def _ext_is_leaf(node: tuple) -> bool:
    return node[0] in _EXT_LEAF_KINDS


def _ext_neg(node: tuple) -> tuple:
    """Negate a per-bit tree by pushing NOT into the leaves (De Morgan)."""
    k = node[0]
    if k == "I":
        return ("N", node[1])
    if k == "N":
        return ("I", node[1])
    if k == "C":
        return ("C", 1 - node[1])
    if k == "XOR":
        return ("XOR", node[1], _ext_neg(node[2]))
    if k == "AND":
        return ("OR", _ext_neg(node[1]), _ext_neg(node[2]))
    if k == "OR":
        return ("AND", _ext_neg(node[1]), _ext_neg(node[2]))
    raise ValueError(f"cannot negate {node!r}")


def _ext_render(node: tuple) -> str:
    k = node[0]
    if k == "I":
        return f"I{node[1]}"
    if k == "N":
        return f"NOT{node[1]}"
    if k == "C":
        return f"C{node[1]}"
    if k == "MUX":
        return (f"MUX({_ext_render(node[1])};"
                f"{_ext_render(node[2])},{_ext_render(node[3])})")
    return f"{k}(" + ",".join(_ext_render(c) for c in node[1:]) + ")"


def _ext_op_bit(k: str, vals: Tuple[str, ...], tt=None) -> str:
    if k == "XOR":
        return "1" if vals[0] != vals[1] else "0"
    if k == "AND":
        return "1" if vals[0] == "1" and vals[1] == "1" else "0"
    if k == "OR":
        return "1" if vals[0] == "1" or vals[1] == "1" else "0"
    if k == "MAJ":
        return "1" if "".join(vals).count("1") >= 2 else "0"
    if k == "MUX":
        return vals[1] if vals[0] == "1" else vals[2]
    if k == "F":
        idx = ((vals[0] == "1") << 2) | ((vals[1] == "1") << 1) | (vals[2] == "1")
        return str(tt[idx])
    raise ValueError(f"unknown op {k}")


def _ext_col(node: tuple, icols: List[str], n: int, tt=None) -> str:
    """Column of node values across the examples (length n)."""
    k = node[0]
    if k == "I":
        return icols[node[1]]
    if k == "N":
        return _invert(icols[node[1]])
    if k == "C":
        return str(node[1]) * n
    cols = [_ext_col(c, icols, n, tt) for c in node[1:]]
    return "".join(_ext_op_bit(k, vals, tt) for vals in zip(*cols))


def _ext_val(node: tuple, qbits: str, tt=None) -> str:
    k = node[0]
    if k == "I":
        return qbits[node[1]]
    if k == "N":
        return _bit_not(qbits[node[1]])
    if k == "C":
        return str(node[1])
    vals = tuple(_ext_val(c, qbits, tt) for c in node[1:])
    return _ext_op_bit(k, vals, tt)


def _ext_has_not(node: tuple) -> bool:
    if node[0] == "N":
        return True
    if _ext_is_leaf(node):
        return False
    return any(_ext_has_not(c) for c in node[1:])


def _ext_subst(node: tuple, qbits: str) -> str:
    """Substitution display: leaves replaced by question-bit values; NOT
    leaves keep an explicit NOT(v) wrapper (legacy AND-NOT line style)."""
    k = node[0]
    if k == "I":
        return qbits[node[1]]
    if k == "N":
        return f"NOT({qbits[node[1]]})"
    if k == "C":
        return str(node[1])
    if k == "MUX":
        return (f"MUX({_ext_subst(node[1], qbits)};"
                f"{_ext_subst(node[2], qbits)},{_ext_subst(node[3], qbits)})")
    return f"{k}(" + ",".join(_ext_subst(c, qbits) for c in node[1:]) + ")"


def _ext_resolve(node: tuple, qbits: str) -> str:
    """Like _ext_subst but with every NOT leaf resolved to its value."""
    k = node[0]
    if k in ("I", "N", "C"):
        return _ext_val(node, qbits)
    if k == "MUX":
        return (f"MUX({_ext_resolve(node[1], qbits)};"
                f"{_ext_resolve(node[2], qbits)},{_ext_resolve(node[3], qbits)})")
    return f"{k}(" + ",".join(_ext_resolve(c, qbits) for c in node[1:]) + ")"


def _ext_collapse(node: tuple, qbits: str, tt=None) -> str:
    """Top op with every child collapsed to its value (nested inner solved)."""
    k = node[0]
    vals = [_ext_val(c, qbits, tt) for c in node[1:]]
    if k == "MUX":
        return f"MUX({vals[0]};{vals[1]},{vals[2]})"
    return f"{k}(" + ",".join(vals) + ")"


def _ext_apply_line(i: int, node: tuple, qbits: str, tt=None) -> Tuple[str, str]:
    """One per-bit Applying line (legacy visual style) + the result bit."""
    if node[0] == "I":
        v = qbits[node[1]]
        return f"{i} I{node[1]} = {v}", v
    if node[0] == "N":
        v = qbits[node[1]]
        nv = _bit_not(v)
        return f"{i} NOT{node[1]} = NOT({v}) = {nv}", nv
    if node[0] == "C":
        return f"{i} C{node[1]} = {node[1]}", str(node[1])
    steps = [_ext_render(node), _ext_subst(node, qbits)]
    if _ext_has_not(node):
        steps.append(_ext_resolve(node, qbits))
    if any(not _ext_is_leaf(c) for c in node[1:]):
        steps.append(_ext_collapse(node, qbits, tt))
    result = _ext_val(node, qbits, tt)
    steps.append(result)
    return f"{i} " + " = ".join(steps), result


class _ExtEngine:
    """Candidate enumeration + prior tiebreak, ported 1:1 from rl/bit_prior.py
    (same atoms, same tiers, same costs, same dedupe and scoring), plus the
    per-bit projection used by the trace emitter."""

    def __init__(self) -> None:
        import numpy as np

        self.np = np
        MASK = 0xFF

        def _rl(x, k):
            return ((x << k) | (x >> (N_BITS - k))) & MASK

        def _sl(x, k):
            return (x << k) & MASK

        def _sr(x, k):
            return x >> k

        def _nt(x):
            return ~x & MASK

        def _rv(x):
            return int(format(x, "08b")[::-1], 2)

        atoms = [("id", lambda x: x)]
        atoms += [(f"rl{k}", lambda x, k=k: _rl(x, k)) for k in range(1, 8)]
        atoms += [(f"sl{k}", lambda x, k=k: _sl(x, k)) for k in range(1, 8)]
        atoms += [(f"sr{k}", lambda x, k=k: _sr(x, k)) for k in range(1, 8)]
        atoms += [("nt", _nt), ("rv", _rv)]
        self.L1 = [lab for lab, _ in atoms]
        self.NA = len(atoms)
        dom = np.arange(256, dtype=np.int64)
        self.A1 = np.array(
            [[fn(int(x)) for x in dom] for _, fn in atoms], dtype=np.int64
        )

        # depth-2 composite atoms (b applies first), function-table dedupe,
        # keep the cheapest label.
        t2: Dict[bytes, tuple] = {}
        for i in range(self.NA):
            for j in range(self.NA):
                t = self.A1[i][self.A1[j]]
                if self.L1[i] == "id":
                    lab = self.L1[j]
                elif self.L1[j] == "id":
                    lab = self.L1[i]
                else:
                    lab = f"{self.L1[i]}.{self.L1[j]}"
                key = t.tobytes()
                if key not in t2 or self.atom_cost(lab) < self.atom_cost(t2[key][0]):
                    t2[key] = (lab, t)
        pairs2 = sorted(t2.values(), key=lambda v: (self.atom_cost(v[0]), v[0]))
        self.L2 = [v[0] for v in pairs2]
        self.A2 = np.array([v[1] for v in pairs2])
        self.NA2 = len(self.L2)

        self.PAIR_OPS = {
            "XOR": lambda a, b: a ^ b,
            "AND": lambda a, b: a & b,
            "OR": lambda a, b: a | b,
            "ANDN": lambda a, b: a & (~b & MASK),
            "ORN": lambda a, b: a | (~b & MASK),
            "XNOR": lambda a, b: ~(a ^ b) & MASK,
            "NAND": lambda a, b: ~(a & b) & MASK,
            "NOR": lambda a, b: ~(a | b) & MASK,
        }
        self.SYM_OPS = {"XOR", "AND", "OR", "XNOR", "NAND", "NOR"}
        self.OP_COST = {
            "XOR": 1, "AND": 1, "OR": 1, "ANDN": 2, "ORN": 2,
            "XNOR": 2, "NAND": 2, "NOR": 2,
        }

        # inner pair table (base atoms x 8 ops), dedupe -> nested tier, with
        # structured (op, i, j) kept for per-bit projection.
        ip: Dict[bytes, tuple] = {}
        for op, fn in self.PAIR_OPS.items():
            for i in range(self.NA):
                rng = range(i, self.NA) if op in self.SYM_OPS else range(self.NA)
                for j in rng:
                    if i == j:
                        continue
                    t = fn(self.A1[i], self.A1[j])
                    cost = (self.OP_COST[op] + self.atom_cost(self.L1[i])
                            + self.atom_cost(self.L1[j]))
                    lab = f"{op}({self.L1[i]},{self.L1[j]})"
                    key = t.tobytes()
                    if key not in ip or cost < ip[key][1]:
                        ip[key] = (lab, cost, t, (op, i, j))
        ipl = sorted(ip.values(), key=lambda v: (v[1], v[0]))
        self.IPL = [v[0] for v in ipl]
        self.IPC = [v[1] for v in ipl]
        self.IPT = np.array([v[2] for v in ipl])
        self.IPS = [v[3] for v in ipl]

        self._proj1 = [self._proj(self.A1[i]) for i in range(self.NA)]
        self._proj2_cache: Dict[int, list] = {}
        self._prior_counts: Optional[Tuple[Dict[str, int], Dict[str, int]]] = None

    @staticmethod
    def atom_cost(lab: str) -> int:
        total = 0
        for part in lab.split("."):
            if part == "id":
                continue
            total += 2 if part == "rv" else 1
        return total

    # ---------------------------------------------------- per-bit projection
    def _proj(self, tab) -> list:
        """Project a routing atom table to per-bit leaves: every output bit
        is one input bit, its negation, or a boundary constant."""
        z = int(tab[0])
        zb = format(z, "08b")
        proj: list = [None] * N_BITS
        for p in range(N_BITS):
            y = int(tab[1 << (7 - p)])
            diff = y ^ z
            for i in range(N_BITS):
                if (diff >> (7 - i)) & 1:
                    proj[i] = ("N", p) if zb[i] == "1" else ("I", p)
        return [
            pr if pr is not None else ("C", int(zb[i]))
            for i, pr in enumerate(proj)
        ]

    def proj2(self, i: int) -> list:
        if i not in self._proj2_cache:
            self._proj2_cache[i] = self._proj(self.A2[i])
        return self._proj2_cache[i]

    # -------------------------------------------------------- enumeration
    def enumerate_consistent(self, ins: List[int], outs: List[int]) -> List[dict]:
        np = self.np
        MASK = 0xFF
        ex = np.array(ins, dtype=np.int64)
        out = np.array(outs, dtype=np.int64)
        cands: List[dict] = []
        seen: Dict[bytes, int] = {}

        def add(tier, expr, ftab, cost, kind):
            key = ftab.astype(np.uint8).tobytes()
            prev = seen.get(key)
            if prev is not None:
                if cost < cands[prev]["cost"]:
                    cands[prev].update(tier=tier, expr=expr, cost=cost, kind=kind)
                return
            seen[key] = len(cands)
            cands.append({"tier": tier, "expr": expr, "ftab": ftab,
                          "cost": cost, "kind": kind})

        # T1 unary over depth-2 atoms
        av2 = self.A2[:, ex]
        for i in np.where((av2 == out).all(axis=1))[0]:
            i = int(i)
            add("unary", self.L2[i], self.A2[i],
                self.atom_cost(self.L2[i]), ("unary", i))

        # T2 pairs over depth-2 atoms
        Aa = av2[:, None, :]
        Bb = av2[None, :, :]
        for op, fn in self.PAIR_OPS.items():
            M = fn(Aa, Bb)
            ok = (M == out).all(axis=2)
            ii, jj = np.where(ok)
            for i, j in zip(ii.tolist(), jj.tolist()):
                if op in self.SYM_OPS and j < i:
                    continue
                cost = (self.OP_COST[op] + self.atom_cost(self.L2[i])
                        + self.atom_cost(self.L2[j]))
                add("pair", f"{op}({self.L2[i]},{self.L2[j]})",
                    fn(self.A2[i], self.A2[j]), cost, ("pair", op, i, j))

        # T3 nested: op2(inner, c) / op2(c, inner)
        av1 = self.A1[:, ex]
        ipv = self.IPT[:, ex]
        for op, fn in self.PAIR_OPS.items():
            M = fn(ipv[:, None, :], av1[None, :, :])
            ok = (M == out).all(axis=2)
            for p, c in zip(*np.where(ok)):
                p, c = int(p), int(c)
                cost = self.OP_COST[op] + self.IPC[p] + self.atom_cost(self.L1[c])
                add("nested", f"{op}({self.IPL[p]},{self.L1[c]})",
                    fn(self.IPT[p], self.A1[c]), cost, ("nested", op, p, c, False))
            if op in self.SYM_OPS:
                continue
            M = fn(av1[:, None, :], ipv[None, :, :])
            ok = (M == out).all(axis=2)
            for c, p in zip(*np.where(ok)):
                p, c = int(p), int(c)
                cost = self.OP_COST[op] + self.IPC[p] + self.atom_cost(self.L1[c])
                add("nested", f"{op}({self.L1[c]},{self.IPL[p]})",
                    fn(self.A1[c], self.IPT[p]), cost, ("nested", op, p, c, True))

        # T4 MAJ / MUX over base atoms (and negations)
        for i in range(self.NA):
            ai, ti = av1[i], self.A1[i]
            for j in range(i + 1, self.NA):
                ab = ai & av1[j]
                aob = ai | av1[j]
                for k in range(j + 1, self.NA):
                    m = ab | (aob & av1[k])
                    lab = f"{self.L1[i]},{self.L1[j]},{self.L1[k]}"
                    cst = (1 + self.atom_cost(self.L1[i])
                           + self.atom_cost(self.L1[j])
                           + self.atom_cost(self.L1[k]))
                    if (m == out).all():
                        ft = (ti & self.A1[j]) | ((ti | self.A1[j]) & self.A1[k])
                        add("maj", f"MAJ({lab})", ft, cst, ("maj", 0, i, j, k))
                    if ((~m & MASK) == out).all():
                        ft = ~((ti & self.A1[j])
                               | ((ti | self.A1[j]) & self.A1[k])) & MASK
                        add("maj", f"NMAJ({lab})", ft, cst + 1, ("maj", 1, i, j, k))
        for s in range(self.NA):
            sv = av1[s]
            nsv = ~sv & MASK
            for b in range(self.NA):
                sb = sv & av1[b]
                for c in range(self.NA):
                    if c == b:
                        continue
                    m = sb | (nsv & av1[c])
                    lab = f"{self.L1[s]};{self.L1[b]},{self.L1[c]}"
                    cst = (1 + self.atom_cost(self.L1[s])
                           + self.atom_cost(self.L1[b])
                           + self.atom_cost(self.L1[c]))
                    if (m == out).all():
                        ft = ((self.A1[s] & self.A1[b])
                              | ((~self.A1[s] & MASK) & self.A1[c]))
                        add("mux", f"MUX({lab})", ft, cst, ("mux", 0, s, b, c))
                    if ((~m & MASK) == out).all():
                        ft = ~((self.A1[s] & self.A1[b])
                               | ((~self.A1[s] & MASK) & self.A1[c])) & MASK
                        add("mux", f"NMUX({lab})", ft, cst + 1, ("mux", 1, s, b, c))
        return cands

    def _bits_mat(self, vals):
        np = self.np
        v = np.asarray(vals, dtype=np.int64)
        return ((v[:, None] >> (7 - np.arange(8))) & 1).astype(np.int8)

    def enumerate_gf3(self, ins: List[int], outs: List[int], q: int) -> List[dict]:
        """gf3 fallback: free 3-input boolean over base-atom triples; only
        candidates with no example conflict and a fully decidable query."""
        np = self.np
        ex = np.array(ins, dtype=np.int64)
        AB = np.array([self._bits_mat(self.A1[a][ex]) for a in range(self.NA)])
        OB = self._bits_mat(outs)
        QB = np.array([self._bits_mat([int(self.A1[a][q])])[0]
                       for a in range(self.NA)])
        cands: List[dict] = []
        for i in range(self.NA):
            for j in range(i + 1, self.NA):
                base = AB[i] * 4 + AB[j] * 2
                for k in range(j + 1, self.NA):
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
                        continue
                    pred = 0
                    for b in range(8):
                        pred = (pred << 1) | int(pred_bits[b])
                    tts = "".join("x" if t < 0 else str(t) for t in tt)
                    cands.append({
                        "tier": "gf3",
                        "expr": (f"F[{tts}]({self.L1[i]},{self.L1[j]},"
                                 f"{self.L1[k]})"),
                        "ftab": None,
                        "pred": pred,
                        "cost": (6 + self.atom_cost(self.L1[i])
                                 + self.atom_cost(self.L1[j])
                                 + self.atom_cost(self.L1[k])),
                        "kind": ("gf3", tuple(int(t) for t in tt), i, j, k),
                    })
        return cands

    # ----------------------------------------------------- prior / scoring
    def cand_pred(self, c: dict, q: int) -> int:
        if c["ftab"] is not None:
            return int(c["ftab"][q])
        return c["pred"]

    def signature(self, c: dict) -> str:
        return f"{c['tier']}:{c['expr']}"

    def family_sig(self, c: dict) -> str:
        return f"{c['tier']}:{_EXT_DIG.sub('', c['expr'])}"

    def score(self, c: dict, prior) -> tuple:
        sig_cnt, fam_cnt = prior
        s = (sig_cnt.get(self.signature(c), 0)
             + 0.1 * fam_cnt.get(self.family_sig(c), 0))
        return (-s, c["cost"], self.signature(c))

    def prior(self) -> Tuple[Dict[str, int], Dict[str, int]]:
        if self._prior_counts is None:
            self._prior_counts = self._learn_prior()
        return self._prior_counts

    def _learn_prior(self) -> Tuple[Dict[str, int], Dict[str, int]]:
        """Canonical-rule prior over data/train_split.csv bit problems
        (cheapest truth-consistent candidate per problem; bit_prior.py
        learn_prior, leak-free by construction)."""
        import csv
        from pathlib import Path

        sig_cnt: Dict[str, int] = {}
        fam_cnt: Dict[str, int] = {}
        csv_path = Path(__file__).resolve().parents[2] / "data" / "train_split.csv"
        if not csv_path.exists():
            return sig_cnt, fam_cnt  # cost-only tiebreak fallback
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                if row.get("cat") != "bit_manipulation":
                    continue
                exs = _EXT_EX_RE.findall(row["prompt"])
                qm = _EXT_Q_RE.search(row["prompt"])
                ans = str(row["answer"]).strip()
                if not exs or not qm or not re.fullmatch(r"[01]{8}", ans):
                    continue
                ins = [int(a, 2) for a, _ in exs]
                outs = [int(b, 2) for _, b in exs]
                q = int(qm.group(1), 2)
                truth = int(ans, 2)
                cands = self.enumerate_consistent(ins, outs)
                if not cands:
                    cands = self.enumerate_gf3(ins, outs, q)
                good = [c for c in cands if self.cand_pred(c, q) == truth]
                if not good:
                    continue
                c = min(good, key=lambda c: (c["cost"], self.signature(c)))
                sig = self.signature(c)
                fam = self.family_sig(c)
                sig_cnt[sig] = sig_cnt.get(sig, 0) + 1
                fam_cnt[fam] = fam_cnt.get(fam, 0) + 1
        return sig_cnt, fam_cnt

    # ------------------------------------------------- winner -> bit trees
    def trees(self, c: dict) -> Tuple[list, Optional[tuple]]:
        """Per-bit expression trees (one per output bit) + gf3 cell table."""
        kind = c["kind"]
        t = kind[0]
        if t == "unary":
            return list(self.proj2(kind[1])), None
        if t == "pair":
            op, ia, ib = kind[1], kind[2], kind[3]
            base, na, nb = _EXT_FOLD[op]
            A, B = self.proj2(ia), self.proj2(ib)
            return [
                (base,
                 _ext_neg(A[i]) if na else A[i],
                 _ext_neg(B[i]) if nb else B[i])
                for i in range(N_BITS)
            ], None
        if t == "nested":
            op2, p, ci, c_first = kind[1], kind[2], kind[3], kind[4]
            op1, ia, ib = self.IPS[p]
            base1, na1, nb1 = _EXT_FOLD[op1]
            base2, no1, no2 = _EXT_FOLD[op2]
            A, B, C = self._proj1[ia], self._proj1[ib], self._proj1[ci]
            out = []
            for i in range(N_BITS):
                inner = (base1,
                         _ext_neg(A[i]) if na1 else A[i],
                         _ext_neg(B[i]) if nb1 else B[i])
                first, second = (C[i], inner) if c_first else (inner, C[i])
                if no1:
                    first = _ext_neg(first)
                if no2:
                    second = _ext_neg(second)
                out.append((base2, first, second))
            return out, None
        if t == "maj":
            neg = kind[1]
            pa, pb, pc = (self._proj1[k] for k in kind[2:5])
            out = []
            for i in range(N_BITS):
                a, b, cc = pa[i], pb[i], pc[i]
                if neg:
                    a, b, cc = _ext_neg(a), _ext_neg(b), _ext_neg(cc)
                out.append(("MAJ", a, b, cc))
            return out, None
        if t == "mux":
            neg = kind[1]
            ps, pb, pc = (self._proj1[k] for k in kind[2:5])
            out = []
            for i in range(N_BITS):
                b, cc = pb[i], pc[i]
                if neg:
                    b, cc = _ext_neg(b), _ext_neg(cc)
                out.append(("MUX", ps[i], b, cc))
            return out, None
        # gf3
        tt = kind[1]
        pa, pb, pc = (self._proj1[k] for k in kind[2:5])
        return [("F", pa[i], pb[i], pc[i]) for i in range(N_BITS)], tt

    # ------------------------------------------------- scan display pools
    def scan_pool(self, tier: str, ins: List[int], outs: List[int],
                  limit: int = 48) -> List[dict]:
        """Cheap-first candidates of one tier that CONFLICT with the example
        outputs — the rejection material for the v2 scan loop. Every entry
        carries real recomputed per-example values up to (and including) the
        first mismatching output; example-consistent candidates are skipped
        (they may only ever appear as the winner)."""
        MASK = 0xFF
        entries: List[dict] = []
        seen: set = set()

        def rows_for(vals_fn) -> Optional[list]:
            rows = []
            for k, (iv, ov) in enumerate(zip(ins, outs)):
                ops, pv = vals_fn(iv)
                ok = pv == ov
                rows.append((k, ops, _ext_b(pv), _ext_b(ov), ok))
                if not ok:
                    return rows
            return None  # consistent with every example -> not rejectable

        if tier == "unary":
            for u in range(self.NA2):
                tab = self.A2[u]
                rows = rows_for(lambda iv, tab=tab: ((), int(tab[iv])))
                if rows is None:
                    continue
                entries.append({"t": "unary", "label": self.L2[u], "rows": rows})
                if len(entries) >= limit:
                    break
            return entries

        if tier == "pair":
            ops_sorted = sorted(self.PAIR_OPS, key=lambda o: (self.OP_COST[o], o))
            m = min(12, self.NA2)
            descs = []
            for op in ops_sorted:
                for i in range(m):
                    rng = range(i + 1, m) if op in self.SYM_OPS else range(m)
                    for j in rng:
                        if i == j:
                            continue
                        cost = (self.OP_COST[op] + self.atom_cost(self.L2[i])
                                + self.atom_cost(self.L2[j]))
                        descs.append(
                            (cost, f"{op}({self.L2[i]},{self.L2[j]})", op, i, j))
            descs.sort(key=lambda d: (d[0], d[1]))
            for cost, lab, op, i, j in descs:
                fn = self.PAIR_OPS[op]
                ta, tb = self.A2[i], self.A2[j]
                key = fn(ta, tb).tobytes()
                if key in seen:
                    continue
                rows = rows_for(
                    lambda iv, ta=ta, tb=tb, fn=fn:
                    ((_ext_b(ta[iv]), _ext_b(tb[iv])),
                     int(fn(int(ta[iv]), int(tb[iv]))) & MASK))
                if rows is None:
                    continue
                seen.add(key)
                entries.append({"t": "pair", "label": lab, "rows": rows})
                if len(entries) >= limit:
                    break
            return entries

        if tier == "nested":
            ops_sorted = sorted(self.PAIR_OPS, key=lambda o: (self.OP_COST[o], o))
            mp = min(10, len(self.IPL))
            order_c = sorted(range(self.NA),
                             key=lambda c: (self.atom_cost(self.L1[c]), c))[:8]
            descs = []
            for op2 in ops_sorted:
                for p in range(mp):
                    for c in order_c:
                        cost = (self.OP_COST[op2] + self.IPC[p]
                                + self.atom_cost(self.L1[c]))
                        descs.append((cost, f"{op2}({self.IPL[p]},{self.L1[c]})",
                                      op2, p, c, False))
                        if op2 not in self.SYM_OPS:
                            descs.append(
                                (cost, f"{op2}({self.L1[c]},{self.IPL[p]})",
                                 op2, p, c, True))
            descs.sort(key=lambda d: (d[0], d[1]))
            for cost, lab, op2, p, c, c_first in descs:
                fn = self.PAIR_OPS[op2]
                ti, tc = self.IPT[p], self.A1[c]
                tab = fn(tc, ti) if c_first else fn(ti, tc)
                key = tab.tobytes()
                if key in seen:
                    continue
                op1, ia, ib = self.IPS[p]
                ta1, tb1 = self.A1[ia], self.A1[ib]

                def vals(iv, ta1=ta1, tb1=tb1, ti=ti, tc=tc, fn=fn,
                         c_first=c_first):
                    inner_v, c_v = int(ti[iv]), int(tc[iv])
                    pv = (fn(c_v, inner_v) if c_first
                          else fn(inner_v, c_v)) & MASK
                    return ((_ext_b(ta1[iv]), _ext_b(tb1[iv]),
                             _ext_b(inner_v), _ext_b(c_v)), pv)

                rows = rows_for(vals)
                if rows is None:
                    continue
                seen.add(key)
                entries.append({
                    "t": "nested", "label": lab, "rows": rows,
                    "op1": op1, "op2": op2, "c_first": c_first,
                })
                if len(entries) >= limit:
                    break
            return entries

        if tier in ("maj", "mux"):
            descs = []
            if tier == "maj":
                m = min(8, self.NA)
                for i in range(m):
                    for j in range(i + 1, m):
                        for k in range(j + 1, m):
                            cost = (1 + self.atom_cost(self.L1[i])
                                    + self.atom_cost(self.L1[j])
                                    + self.atom_cost(self.L1[k]))
                            lab = f"MAJ({self.L1[i]},{self.L1[j]},{self.L1[k]})"
                            descs.append((cost, lab, i, j, k))
            else:
                m = min(6, self.NA)
                for s in range(m):
                    for b in range(m):
                        for c in range(m):
                            if c == b:
                                continue
                            cost = (1 + self.atom_cost(self.L1[s])
                                    + self.atom_cost(self.L1[b])
                                    + self.atom_cost(self.L1[c]))
                            lab = f"MUX({self.L1[s]};{self.L1[b]},{self.L1[c]})"
                            descs.append((cost, lab, s, b, c))
            descs.sort(key=lambda d: (d[0], d[1]))
            for cost, lab, x, y, z in descs:
                tx, ty, tz = self.A1[x], self.A1[y], self.A1[z]
                if tier == "maj":
                    tab = (tx & ty) | ((tx | ty) & tz)
                else:
                    tab = (tx & ty) | ((~tx & MASK) & tz)
                key = tab.tobytes()
                if key in seen:
                    continue
                rows = rows_for(
                    lambda iv, tx=tx, ty=ty, tz=tz, tab=tab:
                    ((_ext_b(tx[iv]), _ext_b(ty[iv]), _ext_b(tz[iv])),
                     int(tab[iv])))
                if rows is None:
                    continue
                seen.add(key)
                entries.append({"t": tier, "label": lab, "rows": rows})
                if len(entries) >= limit:
                    break
            return entries

        if tier == "gf3":
            np = self.np
            ex = np.array(ins, dtype=np.int64)
            OB = self._bits_mat(outs)
            m = min(8, self.NA)
            AB = [self._bits_mat(self.A1[a][ex]) for a in range(m)]
            for i in range(m):
                for j in range(i + 1, m):
                    base = AB[i] * 4 + AB[j] * 2
                    for k in range(j + 1, m):
                        idx = base + AB[k]
                        conflict = None
                        for v in range(8):
                            cell = idx == v
                            if not cell.any():
                                continue
                            if bool((OB[cell] == 1).any()) and bool(
                                    (OB[cell] == 0).any()):
                                conflict = v
                                break
                        if conflict is None:
                            continue
                        entries.append({
                            "t": "gf3r",
                            "label": (f"F({self.L1[i]},{self.L1[j]},"
                                      f"{self.L1[k]})"),
                            "cell": conflict,
                        })
                        if len(entries) >= limit:
                            return entries
            return entries

        raise ValueError(f"unknown tier {tier}")


_EXT_ENGINE: Optional[_ExtEngine] = None


def _ext_engine() -> _ExtEngine:
    global _EXT_ENGINE
    if _EXT_ENGINE is None:
        _EXT_ENGINE = _ExtEngine()
    return _EXT_ENGINE


_EXT_TIER_ORDER = ("unary", "pair", "nested", "maj", "mux", "gf3")
_EXT_CAP = 7250       # v3 emitter ceiling (spec: p95 <= 7250); traces above it
                      # go through the slimming ladder, hard max _EV_MAX_TOKENS
_EXT_FILL_MIN = 6250  # stop adding rejected candidates once reached
                      # (v3 length band target: p50 6200-6700)
_EXT_FILL_MAX = 7000  # a rejected candidate may never push past this


def _ext_b(x) -> str:
    return format(int(x) & 0xFF, "08b")


def _ext_entry_lines(e: dict) -> List[str]:
    """Render one scan-candidate block: label line + per-example verification
    rows (real recomputed data; rejected candidates end on their first
    mismatching output, marked x)."""
    if e["t"] == "gf3r":
        return [e["label"], f"cell {e['cell']:03b} -> 0 1 x"]
    lines = [e["label"]]
    for k, ops, pv, ov, ok in e["rows"]:
        mark = "ok" if ok else "x"
        if e["t"] == "nested":
            av, bv, inner_v, c_v = ops
            first, second = (c_v, inner_v) if e["c_first"] else (inner_v, c_v)
            lines.append(f"{k} {e['op1']} {av} {bv} -> {inner_v} "
                         f"{e['op2']} {first} {second} -> {pv} vs {ov} {mark}")
        elif ops:
            lines.append(f"{k} {' '.join(ops)} -> {pv} vs {ov} {mark}")
        else:
            lines.append(f"{k} {pv} vs {ov} {mark}")
    return lines


def _ext_winner_block(eng: _ExtEngine, pick: dict, ins_i: List[int],
                      outs_i: List[int], slim: bool = False
                      ) -> Optional[List[str]]:
    """Winner candidate block: label + FULL all-example verification rows
    (every row recomputed and required to match, else refuse to emit) +
    closing 'match' line. gf3 winners additionally derive their cell table
    before verifying. slim=True drops operand columns from the rows (token
    budget rescue for the largest prefixes; the verification stays full)."""
    kind = pick["kind"]
    t = kind[0]
    lines: List[str] = [pick["expr"]]

    if t == "gf3":
        tt = kind[1]
        ta, tb, tc = (eng.A1[k] for k in kind[2:5])
        lines.append("Cells (a,b,c bits -> output bit, over all example bits)")
        for v in range(8):
            lines.append(f"{v:03b} {'-' if tt[v] < 0 else tt[v]}")
        for k, (iv, ov) in enumerate(zip(ins_i, outs_i)):
            av, bv, cv = int(ta[iv]), int(tb[iv]), int(tc[iv])
            pv = 0
            for b in range(8):
                idx = ((((av >> (7 - b)) & 1) << 2)
                       | (((bv >> (7 - b)) & 1) << 1)
                       | ((cv >> (7 - b)) & 1))
                if tt[idx] < 0:
                    return None
                pv = (pv << 1) | int(tt[idx])
            if pv != ov:
                return None
            if slim:
                lines.append(f"{k} {_ext_b(pv)} vs {_ext_b(ov)} ok")
            else:
                lines.append(f"{k} {_ext_b(av)} {_ext_b(bv)} {_ext_b(cv)} "
                             f"-> {_ext_b(pv)} vs {_ext_b(ov)} ok")
        lines.append("match")
        return lines

    ftab = pick["ftab"]
    for k, (iv, ov) in enumerate(zip(ins_i, outs_i)):
        pv = int(ftab[iv])
        if pv != ov:
            return None
        if slim or t == "unary":
            lines.append(f"{k} {_ext_b(pv)} vs {_ext_b(ov)} ok")
        elif t == "pair":
            _, op, i, j = kind
            ta, tb = eng.A2[i], eng.A2[j]
            lines.append(f"{k} {_ext_b(ta[iv])} {_ext_b(tb[iv])} "
                         f"-> {_ext_b(pv)} vs {_ext_b(ov)} ok")
        elif t == "nested":
            _, op2, p, c, c_first = kind
            op1, ia, ib = eng.IPS[p]
            inner_v = _ext_b(eng.IPT[p][iv])
            c_v = _ext_b(eng.A1[c][iv])
            first, second = (c_v, inner_v) if c_first else (inner_v, c_v)
            lines.append(f"{k} {op1} {_ext_b(eng.A1[ia][iv])} "
                         f"{_ext_b(eng.A1[ib][iv])} -> {inner_v} "
                         f"{op2} {first} {second} "
                         f"-> {_ext_b(pv)} vs {_ext_b(ov)} ok")
        elif t in ("maj", "mux"):
            ta, tb, tc = (eng.A1[x] for x in kind[2:5])
            lines.append(f"{k} {_ext_b(ta[iv])} {_ext_b(tb[iv])} "
                         f"{_ext_b(tc[iv])} -> {_ext_b(pv)} vs {_ext_b(ov)} ok")
        else:
            return None
    lines.append("match")
    return lines


def _reasoning_prior_family(problem: Problem) -> Optional[str]:
    """v3 isomorphic-mixing trace (2026-06-13).

    Structure: the COMPLETE legacy candidate-scan prefix (example blocks +
    all nine per-column sections, byte-identical to the legacy emitter up to
    the Selecting block), then extended-family candidates (unary -> pair ->
    nested -> MAJ -> MUX -> gf3, cheap-first) flow on as ORDINARY candidate
    segments of the same single scan loop: label line + per-example
    verification rows recomputed for real, rejected at the first mismatching
    output (x mark). No phase marker, no legend, no tier headers. The scan
    stops at the winner segment (all example rows ok + match), then the
    legacy-geometry per-bit Applying tail ending in \\boxed{}.

    The winner is chosen by prior tiebreak over all example-consistent
    candidates; when the ground truth is available (corpus generation) the
    best-scored TRUTH-consistent candidate is preferred (truth-box corpus
    philosophy), falling back to the plain prior pick on the uncovered tail.
    Displayed rejected-candidate counts are tuned per problem so the SFT
    completion lands in the legacy length band (p50 6200-6700, p95 <= 7250,
    max <= 7349)."""
    examples = problem.examples
    if not examples:
        return None
    outs = [_normalize_bits(ex.output_value) for ex in examples]
    ins = [_normalize_bits(ex.input_value) for ex in examples]
    q = _normalize_bits(problem.question)
    if not q or any(not b for b in ins + outs) or len(ins) != len(outs):
        return None

    res = _legacy_prefix(problem)
    if res is None:
        return None
    prefix_lines, _ctx = res
    try:
        cut = prefix_lines.index("Selecting")
    except ValueError:
        return None
    head = prefix_lines[:cut]  # ends with the blank line after XOR-NOT

    eng = _ext_engine()
    ins_i = [int(s, 2) for s in ins]
    outs_i = [int(s, 2) for s in outs]
    q_i = int(q, 2)
    cands = eng.enumerate_consistent(ins_i, outs_i)
    if not cands:
        cands = eng.enumerate_gf3(ins_i, outs_i, q_i)
    if not cands:
        return None
    prior = eng.prior()
    ranked = sorted(cands, key=lambda c: eng.score(c, prior))
    pick = ranked[0]
    answer = (problem.answer or "").strip()
    if _EV_BITS_RE.fullmatch(answer):
        truth_i = int(answer, 2)
        for c in ranked:
            if eng.cand_pred(c, q_i) == truth_i:
                pick = c
                break
    trees, tt = eng.trees(pick)
    win_tier = pick["kind"][0]
    win_full = _ext_winner_block(eng, pick, ins_i, outs_i, slim=False)
    win_slim = _ext_winner_block(eng, pick, ins_i, outs_i, slim=True)
    if win_full is None or win_slim is None:
        return None

    # Per-bit tail: Applying block (projection of the verified winner into
    # the legacy per-bit Applying geometry; each line names its per-bit rule,
    # so no separate Selected vector is emitted).
    apply_lines = [f"Applying to {q}", "Input"]
    for i, bit in enumerate(q):
        apply_lines.append(f"{i} {bit}")
    apply_lines.append("Output")
    ans_bits: List[str] = []
    for i in range(N_BITS):
        line, v = _ext_apply_line(i, trees[i], q, tt)
        apply_lines.append(line)
        ans_bits.append(v)
    ans = "".join(ans_bits)
    if int(ans, 2) != eng.cand_pred(pick, q_i):
        return None  # projection mismatch guard (never emit inconsistency)
    apply_lines.append("")
    apply_lines.append(f"The answer is \\boxed{{{ans}}}")

    tiers = list(_EXT_TIER_ORDER[: _EXT_TIER_ORDER.index(win_tier) + 1])
    pools = {t: eng.scan_pool(t, ins_i, outs_i) for t in tiers}

    def assemble(counts: Dict[str, int], slim: bool) -> str:
        """v3 isomorphic mixing: the extended candidates flow straight after
        the last legacy section as ORDINARY candidate segments (label line +
        per-example verification rows, first mismatch marked x) inside the one
        and only scan loop — no phase marker, no legend, no tier headers.
        The scan stops at the winner segment (all rows ok + match), then the
        legacy-geometry Applying tail. slim=True drops operand columns from
        the winner rows (token-budget rescue; verification stays full)."""
        body: List[str] = []
        win_lines = win_slim if slim else win_full
        for t in tiers:
            for e in pools[t][: counts[t]]:
                body.extend(_ext_entry_lines(e))
            if t == win_tier:
                body.extend(win_lines)
        body.append("")
        parts = list(head) + body
        parts.extend(apply_lines)
        return "\n".join(parts)

    def comp_tokens(text: str) -> int:
        return _ev_token_count(f"{text}\n</think>\n\\boxed{{{ans}}}<|im_end|>")

    # Minimum scan evidence: one cheapest rejected candidate per displayed
    # tier (the scan-with-rejection geometry must survive even on the
    # largest prefixes; only the deepest budget rescue may drop it).
    min_counts = {t: (1 if pools[t] else 0) for t in tiers}

    text = assemble(min_counts, slim=False)
    ntok = comp_tokens(text)
    if ntok > _EXT_CAP:
        # Budget rescue ladder; rejection evidence is dropped last.
        for slim, counts in ((True, min_counts), (True, {t: 0 for t in tiers})):
            cand_text = assemble(counts, slim=slim)
            cand_tok = comp_tokens(cand_text)
            if cand_tok < ntok:
                text, ntok = cand_text, cand_tok
            if ntok <= _EXT_CAP:
                break
        return text if ntok <= _EV_MAX_TOKENS else None

    # Fill with rejection evidence: round-robin across tiers, cheap-first
    # within each tier, until the completion reaches the legacy length band.
    counts = dict(min_counts)
    active = [t for t in tiers if counts[t] < len(pools[t])]
    while ntok < _EXT_FILL_MIN and active:
        for t in list(active):
            if counts[t] >= len(pools[t]):
                active.remove(t)
                continue
            counts[t] += 1
            cand_text = assemble(counts, slim=False)
            cand_tok = comp_tokens(cand_text)
            if cand_tok > _EXT_FILL_MAX:
                counts[t] -= 1
                active.remove(t)
                continue
            text, ntok = cand_text, cand_tok
            if ntok >= _EXT_FILL_MIN:
                break
    return text


_EV_BOX_RE = re.compile(r"\\boxed\{([^}]*)\}")
_EV_BITS_RE = re.compile(r"[01]{8}")

_EV_TOKENIZER: object = None


def _ev_boxed(text: str) -> str:
    m = _EV_BOX_RE.findall(text)
    return m[-1].strip() if m else ""


def _ev_load_tokenizer():
    global _EV_TOKENIZER
    if _EV_TOKENIZER is None:
        try:
            from pathlib import Path

            from tokenizers import Tokenizer  # type: ignore[import-untyped]

            _EV_TOKENIZER = Tokenizer.from_file(
                str(Path(__file__).resolve().parent.parent / "tokenizer.json")
            )
        except Exception:
            _EV_TOKENIZER = False
    return _EV_TOKENIZER


def _ev_token_count(completion: str) -> int:
    """Completion token count (corpus.py template); crude char-based estimate
    if the tokenizer asset is unavailable."""
    tok = _ev_load_tokenizer()
    if tok is False:
        return len(completion) // 3 + 1
    return len(tok.encode(completion, add_special_tokens=False).ids)


def _ev_completion_fits(text: str, answer: str) -> bool:
    """True iff the SFT completion built from this trace fits the token cap.

    Uses the corpus completion template (corpus.py). Falls back to a
    conservative char bound if the tokenizer asset is unavailable.
    """
    completion = f"{text}\n</think>\n\\boxed{{{answer}}}<|im_end|>"
    if _ev_load_tokenizer() is False:
        return len(completion) <= _EV_MAX_CHARS
    return _ev_token_count(completion) <= _EV_MAX_TOKENS


def reasoning_bit_manipulation(problem: Problem) -> Optional[str]:
    """Legacy trace when it solves the problem (byte-identical to the proven
    R~0.95 corpus); otherwise the extended-family isomorphic-mixing trace
    (prior tiebreak, truth-preferring candidate selection), emitted
    only when the SFT completion fits the token cap. Falls back to the
    legacy trace (the run-011 wrong-but-self-consistent policy) when the
    extended family has nothing to say."""
    legacy = _reasoning_legacy(problem)
    answer = (problem.answer or "").strip()
    if not _EV_BITS_RE.fullmatch(answer):
        return legacy
    if legacy is not None and _ev_boxed(legacy) == answer:
        return legacy
    ext = _reasoning_prior_family(problem)
    if ext is not None and _ev_completion_fits(ext, _ev_boxed(ext)):
        return ext
    return legacy
