"""Cryptarithm deduce: find hidden operator rules from character-string examples.

Problem format:
  Input: 5-char string  A[0]A[1] op B[0]B[1]  (A,B are 2-char strings, op is 1 char)
  Output: variable-length string (1–4 chars)

Strategy:
  For each operator seen in examples, try a ranked list of candidate rules:
    1. Concat/positional rules (A+B, B+A, permutations of the 4 input chars)
    2. Set-difference / intersection rules
    3. Interleave rules
    4. Single-char extraction rules
  Accept a rule only if it is consistent with ALL examples for that operator.
  Prefer unique matches; if multiple rules match, pick the highest-priority one.
  Return None if no consistent rule can be found for the query operator.
"""
from __future__ import annotations

import itertools
import re
from collections import defaultdict

from .store_types import Example, Problem

_PARSE_RE = re.compile(r"^(.{2})([^a-zA-Z0-9])(.{2})$")


def _parse_input(s: str) -> tuple[str, str, str] | None:
    m = _PARSE_RE.match(s.strip())
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3)


# ---- Rule catalogue --------------------------------------------------------
# Each rule is (name, fn) where fn(a, b) -> str.
# Rules are tried in order; first consistent one wins.

def _build_rules() -> list[tuple[str, "function"]]:  # type: ignore[type-arg]
    rules: list[tuple[str, object]] = []

    # Concat
    rules.append(("A+B",        lambda a, b: a + b))
    rules.append(("B+A",        lambda a, b: b + a))
    rules.append(("rev(A)+B",   lambda a, b: a[::-1] + b))
    rules.append(("A+rev(B)",   lambda a, b: a + b[::-1]))
    rules.append(("rev(A+B)",   lambda a, b: (a + b)[::-1]))
    rules.append(("rev(B+A)",   lambda a, b: (b + a)[::-1]))

    # Interleave
    rules.append(("interleave(A,B)", lambda a, b: a[0] + b[0] + a[1] + b[1]))
    rules.append(("interleave(B,A)", lambda a, b: b[0] + a[0] + b[1] + a[1]))

    # Single-char
    rules.append(("A",       lambda a, b: a))
    rules.append(("B",       lambda a, b: b))
    rules.append(("rev(A)",  lambda a, b: a[::-1]))
    rules.append(("rev(B)",  lambda a, b: b[::-1]))

    # Set operations (preserving input order)
    rules.append(("A-B(set)",  lambda a, b: "".join(c for c in a if c not in b)))
    rules.append(("B-A(set)",  lambda a, b: "".join(c for c in b if c not in a)))
    rules.append(("A&B(set)",  lambda a, b: "".join(c for c in a if c in b)))
    rules.append(("A|B(set)",  lambda a, b: a + "".join(c for c in b if c not in a)))

    # All 4-choose-k positional reorderings of {A[0], A[1], B[0], B[1]}
    # (lengths 1 through 4)
    src_names = ["A[0]", "A[1]", "B[0]", "B[1]"]
    for length in range(1, 5):
        for mapping in itertools.product(range(4), repeat=length):
            # Avoid duplicating already-added rules by skipping identity-like patterns
            # that were already covered above (A+B, B+A, A, B)
            rule_name = "+".join(src_names[m] for m in mapping)
            # Already covered: A+B = (0,1,2,3), B+A = (2,3,0,1), A=(0,1), B=(2,3)
            skip = {(0,1,2,3), (2,3,0,1), (0,1), (2,3)}
            if mapping in skip:
                continue
            mp = mapping  # capture
            rules.append((rule_name, lambda a, b, mp=mp: "".join(
                [a[0], a[1], b[0], b[1]][i] for i in mp
            )))

    return rules  # type: ignore[return-value]


_RULES = _build_rules()


def _find_consistent_rule(
    exs: list[tuple[str, str, str]],  # (a, op, b) triples
    outs: list[str],
) -> tuple[str, object] | tuple[None, None]:
    """Return (rule_name, rule_fn) for the first rule consistent with all examples."""
    if not exs:
        return None, None
    for name, fn in _RULES:
        try:
            if all(fn(a, b) == out for (a, _, b), out in zip(exs, outs)):  # type: ignore[arg-type]
                return name, fn
        except Exception:
            continue
    return None, None


# ---- Display helpers --------------------------------------------------------

def _describe(rule_name: str, a: str, b: str) -> str:
    """Translate a rule name to a readable computation string."""
    subs = {
        "A+B":             f"concat({a},{b})",
        "B+A":             f"concat({b},{a})",
        "rev(A)+B":        f"concat(rev({a}),{b})",
        "A+rev(B)":        f"concat({a},rev({b}))",
        "rev(A+B)":        f"rev(concat({a},{b}))",
        "rev(B+A)":        f"rev(concat({b},{a}))",
        "interleave(A,B)": f"{a[0]}+{b[0]}+{a[1]}+{b[1]}",
        "interleave(B,A)": f"{b[0]}+{a[0]}+{b[1]}+{a[1]}",
        "A":               a,
        "B":               b,
        "rev(A)":          f"rev({a})",
        "rev(B)":          f"rev({b})",
        "A-B(set)":        f"chars_in_{a}_not_in_{b}",
        "B-A(set)":        f"chars_in_{b}_not_in_{a}",
        "A&B(set)":        f"chars_in_both_{a}_and_{b}",
        "A|B(set)":        f"union({a},{b})",
    }
    return subs.get(rule_name, rule_name)


# ---- Main reasoner ---------------------------------------------------------

_DISPLAY_RULES = [
    "A+B", "B+A", "rev(A)+B", "A+rev(B)", "rev(A+B)",
    "interleave(A,B)", "A", "B",
    "A-B(set)", "B-A(set)", "A&B(set)",
]


def reasoning_cryptarithm_deduce(problem: Problem) -> str | None:
    """Generate chain-of-thought for cryptarithm_deduce problems."""
    lines: list[str] = []
    lines.append(
        "We need to figure out the hidden operation rule for each operator from the examples."
    )
    lines.append("I will put my final answer inside \\boxed{}.")
    lines.append("")

    # Parse all examples
    op_exs: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    op_outs: dict[str, list[str]] = defaultdict(list)

    for ex in problem.examples:
        parsed = _parse_input(ex.input_value)
        if parsed is None:
            return None
        a, op, b = parsed
        op_exs[op].append((a, op, b))
        op_outs[op].append(ex.output_value)

    if not op_exs:
        return None

    # Parse query
    q_parsed = _parse_input(problem.question)
    if q_parsed is None:
        return None
    q_a, q_op, q_b = q_parsed

    lines.append("Grouping examples by operator:")
    lines.append("")

    op_rules: dict[str, tuple[str | None, object]] = {}

    for op in op_exs:
        exs = op_exs[op]
        outs = op_outs[op]
        lines.append(f"Operator '{op}':")
        for (a, _, b), out in zip(exs, outs):
            lines.append(f"  {a} {op} {b} = {repr(out)}")
        lines.append("")

        lines.append(f"Testing candidate rules for '{op}':")

        rule_name, rule_fn = _find_consistent_rule(exs, outs)

        # Display top representative rules plus the winner
        display = list(_DISPLAY_RULES)
        if rule_name and rule_name not in display:
            display.append(rule_name)

        for rn in display:
            # Find the rule function
            rfn = next((fn for n, fn in _RULES if n == rn), None)
            if rfn is None:
                continue
            parts = []
            all_match = True
            for (a, _, b), out in zip(exs, outs):
                try:
                    val = rfn(a, b)  # type: ignore[arg-type]
                    parts.append(f"{a}{op}{b}={repr(val)}")
                    if val != out:
                        all_match = False
                except Exception:
                    parts.append(f"{a}{op}{b}=err")
                    all_match = False
            match_str = "yes" if all_match else "no"
            lines.append(f"  Rule '{rn}': {', '.join(parts)}  [match: {match_str}]")
            if all_match and rn == rule_name:
                break

        if rule_name:
            lines.append(f"  → Consistent rule for '{op}': {rule_name}")
        else:
            lines.append(f"  → No consistent rule found for '{op}'")
        lines.append("")

        op_rules[op] = (rule_name, rule_fn)

    # Apply to query
    lines.append(f"Query: {q_a} {q_op} {q_b}")

    rule_name, rule_fn = op_rules.get(q_op, (None, None))
    if rule_fn is None:
        lines.append("No consistent rule found for query operator. Cannot determine answer.")
        lines.append("")
        lines.append("The answer is \\boxed{?}")
        return "\n".join(lines)

    try:
        result = rule_fn(q_a, q_b)  # type: ignore[arg-type]
    except Exception:
        return None

    desc = _describe(rule_name, q_a, q_b)
    lines.append(f"Applying rule '{rule_name}': {desc} = {repr(result)}")
    lines.append("")
    lines.append(f"The answer is \\boxed{{{result}}}")
    return "\n".join(lines)


if __name__ == "__main__":
    # Smoke tests

    def make_p(exs: list[tuple[str, str]], q: str, ans: str) -> Problem:
        return Problem(
            id="t",
            category="cryptarithm_deduce",
            examples=[Example(inp, out) for inp, out in exs],
            question=q,
            answer=ans,
        )

    print("=" * 60)
    print("Test 1: A+B rule")
    p1 = make_p(
        [("%|*\"|", "%|\"|"), ("\\(*[^", "\\([^")],
        "\\(*[#", "\\([#",
    )
    cot = reasoning_cryptarithm_deduce(p1)
    print(cot)
    print()

    print("=" * 60)
    print("Test 2: B+A rule")
    p2 = make_p(
        [("ab*cd", "cdab"), ("ef*gh", "ghef")],
        "ij*kl", "klij",
    )
    cot = reasoning_cryptarithm_deduce(p2)
    print(cot)
    print()

    print("=" * 60)
    print("Test 3: positional rule A[0]+B[1]")
    p3 = make_p(
        [("ab+cd", "ad"), ("ef+gh", "eh")],
        "ij+kl", "il",
    )
    cot = reasoning_cryptarithm_deduce(p3)
    print(cot)
