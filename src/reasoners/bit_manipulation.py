"""Reasoning generator for 8-bit bit-manipulation tasks.

Two emission paths:

1. Legacy per-column matching trace (byte-identical to the proven R~0.95
   corpus) for every problem it already solves.
2. Enumerate-verify whole-register scan for the hard tail the legacy
   procedure gets wrong: a fixed-priority candidate ladder (atoms; XOR/AND/
   OR/AND-NOT/OR-NOT pairs; MAJ triples; MUX) where every candidate is
   checked against printed example bits and dies at its first mismatch, the
   winner is verified on every example, and the query is computed bit by
   bit. Same enumerative local geometry equation_numeric_deduce validated
   at 100% greedy reproduction. Emitted only when the winner's query output
   equals ground truth; otherwise the legacy behavior is preserved exactly.
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
# Enumerate-verify whole-register extension (hard-tail only).
#
# Hidden-search content (rotations/shifts and their boolean combinations,
# cf. commit 71ce0a5) rendered in enumerative local geometry: nothing on the
# page asserts a whole-byte rule; candidates are TRIED in a fixed priority
# order, every elimination cites a printed mismatch, every filter is an
# exact necessary condition computed from printed strings, and the winner
# only wins after reproducing every example on the page.
#
# Key exact identities driving the page-affordable scan (D := atom(in) XOR
# out, the per-example mismatch mask):
#   unary    atom == out             <=> D = 0
#   XOR(a,b) == out                  <=> v(b) = D(a)
#   AND  needs out inside a and b    (first out=1,v=0 bit is the witness)
#   OR   needs a and b inside out    (first out=0,v=1 bit is the witness)
#   AND-NOT(a,b)=a&~b needs out inside a, b zero on out's ones
#   OR-NOT(a,b)=a|~b  needs a inside out, b one on out's zeros
#   MAJ(a,b,c) == out                <=> D-masks pairwise disjoint
#   MUX(s;b,c): D(b),D(c) disjoint; where b!=c the output forces s.
# ---------------------------------------------------------------------------

_EV_MAX_TOKENS = 7400  # hard completion cap (trace + ending + box)
_EV_MAX_CHARS = 9600  # conservative fallback when the tokenizer is missing


def _ev_sl(s: str, k: int) -> str:
    return s[k:] + "0" * k


def _ev_sr(s: str, k: int) -> str:
    return "0" * k + s[: N_BITS - k]


def _ev_rl(s: str, k: int) -> str:
    return s[k:] + s[:k]


def _ev_atom_list() -> List[Tuple[str, object]]:
    atoms: List[Tuple[str, object]] = [("id", lambda s: s)]
    for k in range(1, N_BITS):
        atoms.append((f"rl{k}", (lambda s, k=k: _ev_rl(s, k))))
    for k in range(1, N_BITS):
        atoms.append((f"sl{k}", (lambda s, k=k: _ev_sl(s, k))))
    for k in range(1, N_BITS):
        atoms.append((f"sr{k}", (lambda s, k=k: _ev_sr(s, k))))
    atoms.append(("nt", _invert))
    atoms.append(("rv", lambda s: s[::-1]))
    return atoms


_EV_ATOMS = _ev_atom_list()
_EV_N = len(_EV_ATOMS)


def _ev_xor(a: str, b: str) -> str:
    return "".join("1" if x != y else "0" for x, y in zip(a, b))


def _ev_and(a: str, b: str) -> str:
    return "".join("1" if x == "1" and y == "1" else "0" for x, y in zip(a, b))


def _ev_or(a: str, b: str) -> str:
    return "".join("1" if x == "1" or y == "1" else "0" for x, y in zip(a, b))


def _ev_maj3(a: str, b: str, c: str) -> str:
    return "".join(
        "1" if (x + y + z).count("1") >= 2 else "0" for x, y, z in zip(a, b, c)
    )


def _ev_mux3(s: str, b: str, c: str) -> str:
    return "".join(y if x == "1" else z for x, y, z in zip(s, b, c))


_EV_PAIR_OPS: Dict[str, object] = {
    "XOR": _ev_xor,
    "AND": _ev_and,
    "OR": _ev_or,
    "AND-NOT": lambda a, b: _ev_and(a, _invert(b)),
    "OR-NOT": lambda a, b: _ev_or(a, _invert(b)),
}

_EV_PAIR_SYM = {"XOR": "^", "AND": "&", "OR": "|", "AND-NOT": "&~", "OR-NOT": "|~"}


def _ev_first_diff(a: str, b: str) -> int:
    for j in range(N_BITS):
        if a[j] != b[j]:
            return j
    return -1


def _ev_disjoint(a: str, b: str) -> bool:
    return all(not (x == "1" and y == "1") for x, y in zip(a, b))


@dataclass(frozen=True)
class _EvCand:
    kind: str  # "unary" | "pair" | "maj" | "mux"
    op: str  # atom label (unary) or pair-op name or "MAJ"/"MUX"
    atoms: Tuple[Tuple[str, object], ...]

    def value(self, bits: str) -> str:
        vals = [fn(bits) for _, fn in self.atoms]  # type: ignore[operator]
        if self.kind == "unary":
            return vals[0]
        if self.kind == "pair":
            return _EV_PAIR_OPS[self.op](vals[0], vals[1])  # type: ignore[operator]
        if self.kind == "maj":
            return _ev_maj3(*vals)
        return _ev_mux3(*vals)

    def expr(self) -> str:
        labs = [lab for lab, _ in self.atoms]
        if self.kind == "unary":
            return labs[0]
        if self.kind == "pair":
            return f"{self.op}({labs[0]},{labs[1]})"
        if self.kind == "maj":
            return f"MAJ({labs[0]},{labs[1]},{labs[2]})"
        return f"MUX({labs[0]};{labs[1]},{labs[2]})"


def _ev_verify(
    lines: List[str], cand: _EvCand, ins: List[str], outs: List[str]
) -> bool:
    """Check candidate example by example, printing every check; fail-fast.

    e0/e1 lines print only the combined value (its operands sit verbatim in
    the printed atom table); e2+ lines reprint the operand strings, which
    are one shift/rotate step from the printed example input. The format
    depends only on the example index, never on the verdict.
    """
    for e in range(len(ins)):
        pred = cand.value(ins[e])
        prefix = f"{cand.expr()} e0:" if e == 0 else f"e{e}:"
        if e <= 1 or cand.kind == "unary":
            body = pred
        else:
            vals = [fn(ins[e]) for _, fn in cand.atoms]  # type: ignore[operator]
            body = " ".join(vals) + f" -> {pred}"
        if pred == outs[e]:
            lines.append(f"{prefix} {body} vs {outs[e]} ok")
        else:
            j = _ev_first_diff(pred, outs[e])
            lines.append(f"{prefix} {body} vs {outs[e]} x@{j}")
            return False
    return True


def _ev_apply(lines: List[str], cand: _EvCand, q: str) -> str:
    lines.append("")
    lines.append(f"{cand.expr()} reproduces every example; scan stops.")
    lines.append("")
    lines.append(f"Applying to {q}")
    vals = [fn(q) for _, fn in cand.atoms]  # type: ignore[operator]
    lines.append(
        "operands: "
        + " ".join(f"{lab}={v}" for (lab, _), v in zip(cand.atoms, vals))
    )
    bits: List[str] = []
    for j in range(N_BITS):
        if cand.kind == "unary":
            r = vals[0][j]
            lines.append(f"b{j}: {r}")
        elif cand.kind == "pair":
            a, b = vals[0][j], vals[1][j]
            r = _EV_PAIR_OPS[cand.op](a * N_BITS, b * N_BITS)[0]  # type: ignore[operator]
            lines.append(f"b{j}: {a}{_EV_PAIR_SYM[cand.op]}{b} -> {r}")
        elif cand.kind == "maj":
            a, b, c = vals[0][j], vals[1][j], vals[2][j]
            r = "1" if (a + b + c).count("1") >= 2 else "0"
            lines.append(f"b{j}: {a},{b},{c} -> {r}")
        else:
            s, b, c = vals[0][j], vals[1][j], vals[2][j]
            r = b if s == "1" else c
            src = "b" if s == "1" else "c"
            lines.append(f"b{j}: s={s} -> {src}={r}")
        bits.append(r)
    ans = "".join(bits)
    lines.append(f"Output: {ans}")
    lines.append("")
    lines.append(f"The answer is \\boxed{{{ans}}}")
    return ans


def _reasoning_enum_verify(problem: Problem) -> Optional[str]:
    examples = problem.examples
    if not examples:
        return None
    outs = [_normalize_bits(ex.output_value) for ex in examples]
    ins = [_normalize_bits(ex.input_value) for ex in examples]
    q = _normalize_bits(problem.question)
    if (
        not q
        or any(not b for b in ins + outs)
        or len(ins) != len(outs)
        or len(ins) < 3
    ):
        return None
    n = len(ins)

    # Atom values and mismatch masks on e0/e1 (the scan's working set).
    V = [[fn(ins[e]) for _, fn in _EV_ATOMS] for e in range(2)]  # type: ignore[operator]
    D = [[_ev_xor(V[e][i], outs[e]) for i in range(_EV_N)] for e in range(2)]
    M = [[D[e][i].count("1") for i in range(_EV_N)] for e in range(2)]

    L: List[str] = []
    L.append(
        "We need to deduce the transformation by scanning whole-register "
        "candidate rules in a fixed priority order."
    )
    L.append("I will put my final answer inside \\boxed{}.")
    L.append("")
    L.append(
        "Order: atoms; XOR, AND, OR, AND-NOT, OR-NOT pairs; MAJ triples; "
        "MUX. Atoms: id, rl1-7 rotate left, sl1-7 shift left (zeros in "
        "right), sr1-7 shift right (zeros in left), nt NOT, rv reverse. A "
        "candidate dies at its first mismatch with the printed bits."
    )
    L.append("")
    L.append("Examples")
    for e in range(n):
        L.append(f"e{e}: {ins[e]} -> {outs[e]}")
    L.append(f"q: {q}")
    L.append("")
    L.append(
        "Atom table, columns v0 d0 m0 v1 d1 (v = atom(in), d = v XOR out, "
        "m = ones(d))."
    )
    for i, (lab, _) in enumerate(_EV_ATOMS):
        L.append(
            f"{lab}: {V[0][i]} {D[0][i]} {M[0][i]} {V[1][i]} {D[1][i]}"
        )
    L.append("")

    winner: Optional[_EvCand] = None

    def _flush(header: str, items: List[str]) -> None:
        if items:
            L.append(header + " " + ", ".join(items))
            items.clear()

    # ---- Stage 1: unary atoms (need m0 = 0 and d1 all zero). ----
    hdr = "Unary scan (need m0=0 and d1=0):"
    items: List[str] = []
    for i, (lab, fn) in enumerate(_EV_ATOMS):
        s = M[0][i]
        if s > 0:
            items.append(f"{lab} {s}")
            continue
        if "1" in D[1][i]:
            items.append(f"{lab} 0 d1 no")
            continue
        items.append(f"{lab} 0 -> try")
        _flush(hdr, items)
        cand = _EvCand("unary", lab, ((lab, fn),))
        if _ev_verify(L, cand, ins, outs):
            winner = cand
            break
    if winner is None:
        items.append("none survive")
        _flush(hdr, items)
        L.append("")

    # ---- Stage 2: XOR pairs via residual lookup (v(b) must equal d(a)). ----
    if winner is None:
        L.append(
            "XOR(a,b): b = a XOR out, so d(a) must equal some atom's v on "
            "e0 and e1 (a=b marks a full hit, a~b a d0-only hit)."
        )
        hdr = "XOR scan:"
        items = []
        for i, (lab, fn) in enumerate(_EV_ATOMS):
            if winner is not None:
                break
            hits = [j for j in range(_EV_N) if j != i and V[0][j] == D[0][i]]
            if not hits:
                items.append(f"{lab} -")
                continue
            for j in hits:
                labj = _EV_ATOMS[j][0]
                if V[1][j] != D[1][i]:
                    items.append(f"{lab}~{labj}")
                    continue
                items.append(f"{lab}={labj} -> try")
                _flush(hdr, items)
                cand = _EvCand(
                    "pair", "XOR", ((lab, fn), (labj, _EV_ATOMS[j][1]))
                )
                if _ev_verify(L, cand, ins, outs):
                    winner = cand
                    break
        if winner is None:
            items.append("no pair")
            _flush(hdr, items)
            L.append("")

    # Helper: filtered survivor scan + pair trials for AND/OR/AND-NOT/OR-NOT.
    def _filter_scan(
        title: str, viol: object, reuse: Optional[List[int]] = None
    ) -> List[int]:
        """Print a per-atom witness scan; return surviving atom indices.

        viol(e, i) -> first violating bit index or -1, computed from the
        printed strings V[e][i] and outs[e].
        """
        if reuse is not None:
            L.append(
                title
                + " survivors (from above): "
                + (", ".join(_EV_ATOMS[i][0] for i in reuse) or "none")
            )
            return reuse
        surv: List[int] = []
        its: List[str] = []
        for i, (lab, _) in enumerate(_EV_ATOMS):
            j0 = viol(0, i)  # type: ignore[operator]
            if j0 >= 0:
                its.append(f"{lab} x{j0}")
                continue
            j1 = viol(1, i)  # type: ignore[operator]
            if j1 >= 0:
                its.append(f"{lab} y{j1}")
                continue
            surv.append(i)
            its.append(f"{lab} ok")
        L.append(title + " " + ", ".join(its))
        # Cap survivor blowup: extend the same witness test to later examples;
        # stop as soon as a round eliminates nobody (no progress, save ink).
        r = 2
        while len(surv) > 6 and r < n:
            its = []
            kept: List[int] = []
            for i in surv:
                v = _EV_ATOMS[i][1](ins[r])  # type: ignore[operator]
                jr = -1
                for jj in range(N_BITS):
                    if viol_cell(v[jj], outs[r][jj]):  # type: ignore[operator]
                        jr = jj
                        break
                if jr >= 0:
                    its.append(f"{_EV_ATOMS[i][0]} x{jr}")
                else:
                    kept.append(i)
            L.append(
                f"still {len(surv)} candidates; e{r} kills: "
                + (", ".join(its) if its else "none")
            )
            no_progress = len(kept) == len(surv)
            surv = kept
            r += 1
            if no_progress:
                break
        return surv

    def _try_pairs(op: str, lefts: List[int], rights: List[int], sym: bool) -> None:
        nonlocal winner
        todo: List[Tuple[int, int]] = []
        for i in lefts:
            for j in rights:
                if j == i or (sym and j <= i):
                    continue
                todo.append((i, j))
        if not todo:
            L.append("no pair to try.")
            return
        if len(todo) > 12:
            L.append(f"{len(todo)} pairs; trying the first 12 in order only.")
            todo = todo[:12]
        for i, j in todo:
            cand = _EvCand(
                "pair",
                op,
                (
                    (_EV_ATOMS[i][0], _EV_ATOMS[i][1]),
                    (_EV_ATOMS[j][0], _EV_ATOMS[j][1]),
                ),
            )
            if _ev_verify(L, cand, ins, outs):
                winner = cand
                return

    # ---- Stage 3: AND pairs. ----
    and_surv: List[int] = []
    if winner is None:
        def _viol_and(e: int, i: int) -> int:
            for j in range(N_BITS):
                if outs[e][j] == "1" and V[e][i][j] == "0":
                    return j
            return -1

        viol_cell = lambda vb, ob: ob == "1" and vb == "0"  # noqa: E731
        and_surv = _filter_scan(
            "AND(a,b): out must lie inside both operands; witness bit "
            "(x=e0, y=e1):",
            _viol_and,
        )
        _try_pairs("AND", and_surv, and_surv, sym=True)
        if winner is None:
            L.append("")

    # ---- Stage 4: OR pairs. ----
    or_surv: List[int] = []
    if winner is None:
        def _viol_or(e: int, i: int) -> int:
            for j in range(N_BITS):
                if outs[e][j] == "0" and V[e][i][j] == "1":
                    return j
            return -1

        viol_cell = lambda vb, ob: ob == "0" and vb == "1"  # noqa: E731
        or_surv = _filter_scan(
            "OR(a,b): both operands must lie inside out; witness bit "
            "(x=e0, y=e1):",
            _viol_or,
        )
        _try_pairs("OR", or_surv, or_surv, sym=True)
        if winner is None:
            L.append("")

    # ---- Stage 5: AND-NOT pairs (a&~b): a covers out, b zero on out's ones.
    if winner is None:
        def _viol_andn_b(e: int, i: int) -> int:
            for j in range(N_BITS):
                if outs[e][j] == "1" and V[e][i][j] == "1":
                    return j
            return -1

        viol_cell = lambda vb, ob: ob == "1" and vb == "1"  # noqa: E731
        L.append("AND-NOT(a,b) = a&~b: a must cover out (AND scan above); "
                 "b must be 0 on out's ones.")
        a_side = _filter_scan("a-side", None, reuse=and_surv)
        b_side = _filter_scan(
            "b-side witness bit (x=e0, y=e1):", _viol_andn_b
        )
        _try_pairs("AND-NOT", a_side, b_side, sym=False)
        if winner is None:
            L.append("")

    # ---- Stage 6: OR-NOT pairs (a|~b): a inside out, b one on out's zeros.
    if winner is None:
        def _viol_orn_b(e: int, i: int) -> int:
            for j in range(N_BITS):
                if outs[e][j] == "0" and V[e][i][j] == "0":
                    return j
            return -1

        viol_cell = lambda vb, ob: ob == "0" and vb == "0"  # noqa: E731
        L.append("OR-NOT(a,b) = a|~b: a must lie inside out (OR scan above); "
                 "b must be 1 on out's zeros.")
        a_side = _filter_scan("a-side", None, reuse=or_surv)
        b_side = _filter_scan(
            "b-side witness bit (x=e0, y=e1):", _viol_orn_b
        )
        _try_pairs("OR-NOT", a_side, b_side, sym=False)
        if winner is None:
            L.append("")

    # ---- Stage 7/8 shared: disjoint d-mask pairs on e0 and e1. ----
    pairs: List[Tuple[int, int]] = []
    if winner is None:
        L.append(
            "MAJ(a,b,c): the output is the majority vote, so at most one "
            "operand may differ from out at any cell -> d-masks must be "
            "pairwise disjoint (checked on e0 and e1)."
        )
        partner_items: List[str] = []
        for i in range(_EV_N):
            ps = [
                j
                for j in range(i + 1, _EV_N)
                if _ev_disjoint(D[0][i], D[0][j])
                and _ev_disjoint(D[1][i], D[1][j])
            ]
            for j in ps:
                pairs.append((i, j))
            partner_items.append(
                f"{_EV_ATOMS[i][0]} "
                + (" ".join(_EV_ATOMS[j][0] for j in ps) if ps else "-")
            )
        L.append("Disjoint partners (j>i): " + "; ".join(partner_items))
        # Cap blowup: prune the pair list on later examples; stop after two
        # consecutive no-progress rounds (a single plateau can still break).
        r = 2
        stall = 0
        while len(pairs) > 10 and r < n and stall < 2:
            its = []
            kept_pairs: List[Tuple[int, int]] = []
            for (i, j) in pairs:
                di = _ev_xor(_EV_ATOMS[i][1](ins[r]), outs[r])  # type: ignore[operator]
                dj = _ev_xor(_EV_ATOMS[j][1](ins[r]), outs[r])  # type: ignore[operator]
                li, lj = _EV_ATOMS[i][0], _EV_ATOMS[j][0]
                if _ev_disjoint(di, dj):
                    kept_pairs.append((i, j))
                else:
                    ov = next(
                        jj
                        for jj in range(N_BITS)
                        if di[jj] == "1" and dj[jj] == "1"
                    )
                    its.append(f"{li}-{lj} x{ov}")
            L.append(
                f"{len(pairs)} disjoint pairs; e{r} kills: "
                + (", ".join(its) if its else "none")
            )
            stall = stall + 1 if len(kept_pairs) == len(pairs) else 0
            pairs = kept_pairs
            r += 1

    # ---- Stage 7: MAJ triples (all three pairs disjoint). ----
    if winner is None:
        pair_set = {(i, j) for (i, j) in pairs}
        tried: List[str] = []
        for (i, j) in pairs:
            if winner is not None:
                break
            thirds = [
                k
                for k in range(j + 1, _EV_N)
                if (i, k) in pair_set and (j, k) in pair_set
            ]
            li, lj = _EV_ATOMS[i][0], _EV_ATOMS[j][0]
            if not thirds:
                tried.append(f"({li},{lj}) no third")
                continue
            for k in thirds:
                lk = _EV_ATOMS[k][0]
                tried.append(f"({li},{lj},{lk}) -> try")
                if tried:
                    L.append("Triples: " + ", ".join(tried))
                    tried = []
                cand = _EvCand(
                    "maj",
                    "MAJ",
                    (
                        (li, _EV_ATOMS[i][1]),
                        (lj, _EV_ATOMS[j][1]),
                        (lk, _EV_ATOMS[k][1]),
                    ),
                )
                if _ev_verify(L, cand, ins, outs):
                    winner = cand
                    break
        if winner is None:
            tried.append("no surviving triple")
            L.append("Triples: " + ", ".join(tried))
            L.append("")

    # ---- Stage 8: MUX over disjoint pairs; forced-sel pattern lookup.
    if winner is None:
        L.append(
            "MUX(s;b,c): out follows b where s=1 and c where s=0; (b,c) must "
            "be a disjoint pair, and every cell where b!=c forces a bit of s "
            "('.' = free). An atom matching the forced pattern is a sel for "
            "(b,c); one matching its complement is a sel for the swapped "
            "pair."
        )

        def _mux_pattern(bi: int, ci: int, e: int) -> Optional[str]:
            """Forced sel bits on example e; None if the pair is impossible
            there (b == c != out at some cell)."""
            vb_s = _EV_ATOMS[bi][1](ins[e]) if e >= 2 else V[e][bi]  # type: ignore[operator]
            vc_s = _EV_ATOMS[ci][1](ins[e]) if e >= 2 else V[e][ci]  # type: ignore[operator]
            pat = []
            for jj in range(N_BITS):
                vb, vc = vb_s[jj], vc_s[jj]
                if vb == vc:
                    if outs[e][jj] != vb:
                        return None
                    pat.append(".")
                else:
                    pat.append("1" if outs[e][jj] == vb else "0")
            return "".join(pat)

        def _pat_match(val: str, pat: str) -> bool:
            return all(pc == "." or pc == val[jj] for jj, pc in enumerate(pat))

        def _pat_comp(pat: str) -> str:
            return "".join(
                "." if c == "." else ("1" if c == "0" else "0") for c in pat
            )

        for (pi, pj) in pairs:
            if winner is not None:
                break
            lb, lc = _EV_ATOMS[pi][0], _EV_ATOMS[pj][0]
            pats = [_mux_pattern(pi, pj, e) for e in range(2)]
            # Disjointness on e0/e1 guarantees the pattern exists there.
            assert pats[0] is not None and pats[1] is not None
            # cands: (atom index, swapped?) in atom order, straight before
            # swapped at the same atom.
            cands: List[Tuple[int, bool]] = []
            for k in range(_EV_N):
                if _pat_match(V[0][k], pats[0]) and _pat_match(V[1][k], pats[1]):
                    cands.append((k, False))
                if _pat_match(V[0][k], _pat_comp(pats[0])) and _pat_match(
                    V[1][k], _pat_comp(pats[1])
                ):
                    cands.append((k, True))
            L.append(
                f"b={lb} c={lc}: forced s e0 {pats[0]} e1 {pats[1]}; matches: "
                + (
                    ", ".join(
                        f"{_EV_ATOMS[k][0]}{' swapped' if sw else ''}"
                        for k, sw in cands
                    )
                    if cands
                    else "none"
                )
            )
            # Extend the forced pattern example by example until at most one
            # sel remains: pattern eliminations cost one short line, while a
            # late trial death costs a full verify chain. A sel that matches
            # the forced pattern on every example is exactly consistent.
            r = 2
            killed = False
            while len(cands) > 1 and r < n:
                vb_r = _EV_ATOMS[pi][1](ins[r])  # type: ignore[operator]
                vc_r = _EV_ATOMS[pj][1](ins[r])  # type: ignore[operator]
                pr = _mux_pattern(pi, pj, r)
                if pr is None:
                    jj = next(
                        j
                        for j in range(N_BITS)
                        if vb_r[j] == vc_r[j] != outs[r][j]
                    )
                    L.append(
                        f"extend to e{r}: b {vb_r} c {vc_r}; b=c!=out at "
                        f"b{jj}; pair dies."
                    )
                    killed = True
                    break
                kept_c = [
                    (k, sw)
                    for k, sw in cands
                    if _pat_match(
                        _EV_ATOMS[k][1](ins[r]),  # type: ignore[operator]
                        _pat_comp(pr) if sw else pr,
                    )
                ]
                L.append(
                    f"{len(cands)} sels; extend to e{r}: b {vb_r} c {vc_r} "
                    f"forced {pr}; keep: "
                    + (
                        ", ".join(
                            f"{_EV_ATOMS[k][0]}{' swapped' if sw else ''}"
                            for k, sw in kept_c
                        )
                        if kept_c
                        else "none"
                    )
                )
                cands = kept_c
                r += 1
            if killed:
                continue
            for k, sw in cands:
                bi, ci = (pj, pi) if sw else (pi, pj)
                cand = _EvCand(
                    "mux",
                    "MUX",
                    (
                        (_EV_ATOMS[k][0], _EV_ATOMS[k][1]),
                        (_EV_ATOMS[bi][0], _EV_ATOMS[bi][1]),
                        (_EV_ATOMS[ci][0], _EV_ATOMS[ci][1]),
                    ),
                )
                if _ev_verify(L, cand, ins, outs):
                    winner = cand
                    break

    if winner is None:
        return None

    _ev_apply(L, winner, q)
    return "\n".join(L)


_EV_BOX_RE = re.compile(r"\\boxed\{([^}]*)\}")
_EV_BITS_RE = re.compile(r"[01]{8}")

_EV_TOKENIZER: object = None


def _ev_boxed(text: str) -> str:
    m = _EV_BOX_RE.findall(text)
    return m[-1].strip() if m else ""


def _ev_completion_fits(text: str, answer: str) -> bool:
    """True iff the SFT completion built from this trace fits the token cap.

    Uses the corpus completion template (corpus.py). Falls back to a
    conservative char bound if the tokenizer asset is unavailable.
    """
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
    completion = f"{text}\n</think>\n\\boxed{{{answer}}}<|im_end|>"
    if _EV_TOKENIZER is False:
        return len(completion) <= _EV_MAX_CHARS
    ids = _EV_TOKENIZER.encode(completion, add_special_tokens=False).ids  # type: ignore[union-attr]
    return len(ids) <= _EV_MAX_TOKENS


def reasoning_bit_manipulation(problem: Problem) -> Optional[str]:
    """Legacy trace when it solves the problem (byte-identical); otherwise
    the enumerate-verify whole-register trace, emitted only when its winner
    reproduces the ground-truth query answer and the completion fits the
    token cap."""
    legacy = _reasoning_legacy(problem)
    answer = (problem.answer or "").strip()
    if not _EV_BITS_RE.fullmatch(answer):
        return legacy
    if legacy is not None and _ev_boxed(legacy) == answer:
        return legacy
    ev = _reasoning_enum_verify(problem)
    if ev is not None and _ev_boxed(ev) == answer and _ev_completion_fits(ev, answer):
        return ev
    return legacy
