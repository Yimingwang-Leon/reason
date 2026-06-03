"""Equation numeric deduce: find hidden operator rules from examples and apply to query."""

from __future__ import annotations

import re
from collections import defaultdict

from .store_types import Example, Problem


def _get_candidates(a_str: str, b_str: str) -> dict[str, str]:
    """Compute all candidate rule outputs for the given 2-digit inputs."""
    a, b = int(a_str), int(b_str)
    a1, a2 = int(a_str[0]), int(a_str[1])
    b1, b2 = int(b_str[0]), int(b_str[1])
    r: dict[str, str] = {}

    # Standard arithmetic
    r["a+b"] = str(a + b)
    r["a-b"] = str(a - b)
    r["b-a"] = str(b - a)
    r["a*b"] = str(a * b)
    r["abs(a-b)"] = str(abs(a - b))
    if b:
        r["a//b"] = str(a // b)
    if a:
        r["b//a"] = str(b // a)
    if b:
        r["a%b"] = str(a % b)
    if a:
        r["b%a"] = str(b % a)
    r["max(a,b)"] = str(max(a, b))
    r["min(a,b)"] = str(min(a, b))
    r["a^b"] = str(a ^ b)   # XOR
    r["a&b"] = str(a & b)
    r["a|b"] = str(a | b)

    # Modular arithmetic
    for mod in [9, 10, 11, 19, 89, 90, 91, 99, 100, 101]:
        r[f"(a-b)%{mod}"] = str((a - b) % mod)
        r[f"(b-a)%{mod}"] = str((b - a) % mod)
        r[f"(a+b)%{mod}"] = str((a + b) % mod)

    # String concat variants
    r["str(a)+str(b)"] = a_str + b_str
    r["str(b)+str(a)"] = b_str + a_str
    r["rev(a)+str(b)"] = a_str[::-1] + b_str
    r["str(a)+rev(b)"] = a_str + b_str[::-1]
    r["rev(a+b)"] = (a_str + b_str)[::-1]
    r["rev(b+a)"] = (b_str + a_str)[::-1]

    # Digit-wise operations
    r["a1+a2+b1+b2"] = str(a1 + a2 + b1 + b2)
    r["a1*a2+b1*b2"] = str(a1 * a2 + b1 * b2)
    r["a1*b1+a2*b2"] = str(a1 * b1 + a2 * b2)
    r["a1*b2+a2*b1"] = str(a1 * b2 + a2 * b1)
    r["(a1+b1)*10+(a2+b2)"] = str((a1 + b1) * 10 + (a2 + b2))
    r["str(a1+b1)+str(a2+b2)"] = str(a1 + b1) + str(a2 + b2)
    r["str(a1*b1)+str(a2*b2)"] = str(a1 * b1) + str(a2 * b2)
    r["str(a1*b2)+str(a2*b1)"] = str(a1 * b2) + str(a2 * b1)
    r["str(a1-b1)+str(a2-b2)"] = str(a1 - b1) + str(a2 - b2)
    r["str(abs(a1-b1))+str(abs(a2-b2))"] = str(abs(a1 - b1)) + str(abs(a2 - b2))
    r["str(a1+b2)+str(a2+b1)"] = str(a1 + b2) + str(a2 + b1)
    r["str(a1*b2)+str(a2+b1)"] = str(a1 * b2) + str(a2 + b1)
    r["str(a1+b1)+str(a2*b2)"] = str(a1 + b1) + str(a2 * b2)
    r["a1*10+b1"] = str(a1 * 10 + b1)
    r["a2*10+b2"] = str(a2 * 10 + b2)
    r["a1*10+b2"] = str(a1 * 10 + b2)
    r["a2*10+b1"] = str(a2 * 10 + b1)
    r["b1*10+a1"] = str(b1 * 10 + a1)
    r["b2*10+a2"] = str(b2 * 10 + a2)
    r["b1*10+b2"] = str(b1 * 10 + b2)   # = b itself
    r["a1*10+a2"] = str(a1 * 10 + a2)   # = a itself
    r["(a1+a2)*b"] = str((a1 + a2) * b)
    r["a*(b1+b2)"] = str(a * (b1 + b2))
    r["(a1+a2)*(b1+b2)"] = str((a1 + a2) * (b1 + b2))
    r["a1*(b1+b2)"] = str(a1 * (b1 + b2))
    r["(a1+a2)*b1"] = str((a1 + a2) * b1)
    r["a2*(b1+b2)"] = str(a2 * (b1 + b2))
    r["(a1+a2)*b2"] = str((a1 + a2) * b2)
    r["a1*b1"] = str(a1 * b1)
    r["a2*b2"] = str(a2 * b2)
    r["a1*b2"] = str(a1 * b2)
    r["a2*b1"] = str(a2 * b1)
    r["a1+b1"] = str(a1 + b1)
    r["a2+b2"] = str(a2 + b2)
    r["a1+b2"] = str(a1 + b2)
    r["a2+b1"] = str(a2 + b1)
    r["abs(a1-b1)+abs(a2-b2)"] = str(abs(a1 - b1) + abs(a2 - b2))
    r["abs(a1-b2)+abs(a2-b1)"] = str(abs(a1 - b2) + abs(a2 - b1))
    if b2:
        r["a1//b2*10+a2"] = str(a1 // b2 * 10 + a2)
    if a2:
        r["b1//a2*10+b2"] = str(b1 // a2 * 10 + b2)
    if b1:
        r["a1//b1"] = str(a1 // b1)
    if a1:
        r["b1//a1"] = str(b1 // a1)
    if b2:
        r["a2//b2"] = str(a2 // b2)
    if a2:
        r["b2//a2"] = str(b2 // a2)

    # Mod 10 digit sums
    r["(a1+b1)%10"] = str((a1 + b1) % 10)
    r["(a2+b2)%10"] = str((a2 + b2) % 10)
    r["str((a1+b1)%10)+str((a2+b2)%10)"] = str((a1 + b1) % 10) + str((a2 + b2) % 10)

    # a*b string concat variants
    r["str(a*b)+str(a+b)"] = str(a * b) + str(a + b)
    r["str(a+b)+str(a*b)"] = str(a + b) + str(a * b)
    r["str(a1*b1)+str(a2+b1)"] = str(a1 * b1) + str(a2 + b1)
    if b2 >= a2:
        r["str(a1*b1)+str(b2-a2)"] = str(a1 * b1) + str(b2 - a2)
    if a2:
        r["str(a1*b1)+str(b2//a2)"] = str(a1 * b1) + str(b2 // a2)

    return {k: v for k, v in r.items() if v != ""}


# The most representative candidate rules to display in CoT (ordered by commonness)
_DISPLAY_RULES = [
    "a+b",
    "a-b",
    "abs(a-b)",
    "a*b",
    "a//b",
    "a%b",
    "str(a)+str(b)",
    "(a1+b1)*10+(a2+b2)",
    "str(abs(a1-b1))+str(abs(a2-b2))",
    "a1+a2+b1+b2",
    "a1*b1+a2*b2",
    "(a-b)%9",
    "(a+b)%10",
    "a^b",
]

_PARSE_RE = re.compile(r"^(\d+)([^\d\s])(\d+)$")


def _parse_input(input_value: str) -> tuple[str, str, str] | None:
    """Parse 'AB op CD' into (a_str, op, b_str). Returns None if format doesn't match."""
    m = _PARSE_RE.match(input_value.strip())
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3)


def _find_consistent_rule(
    examples: list[tuple[str, str, str]],  # list of (a_str, op, b_str)
    outputs: list[str],
) -> str | None:
    """Return the first candidate rule name that matches all examples, or None."""
    if not examples:
        return None

    # Start from the candidate rules of the first example
    a0, _op, b0 = examples[0]
    candidates = _get_candidates(a0, b0)
    # Filter to rules that match the first example's output
    surviving = {name for name, val in candidates.items() if val == outputs[0]}

    for (a_str, _op2, b_str), expected in zip(examples[1:], outputs[1:]):
        cands = _get_candidates(a_str, b_str)
        # Keep only rules that match this example too
        surviving = {name for name in surviving if cands.get(name) == expected}
        if not surviving:
            return None

    # Return first match in a deterministic order (prefer simpler rules)
    all_names = list(_get_candidates(examples[0][0], examples[0][2]).keys())
    for name in all_names:
        if name in surviving:
            return name
    return None


def reasoning_equation_numeric_deduce(problem: Problem) -> str | None:
    """Generate chain-of-thought reasoning for equation_numeric_deduce problems."""
    lines: list[str] = []
    lines.append(
        "We need to figure out the hidden operation rule for each operator from the examples."
    )
    lines.append("I will put my final answer inside \\boxed{}.")
    lines.append("")

    # Parse all examples and group by operator
    op_examples: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    op_outputs: dict[str, list[str]] = defaultdict(list)

    for ex in problem.examples:
        parsed = _parse_input(ex.input_value)
        if parsed is None:
            return None
        a_str, op, b_str = parsed
        op_examples[op].append((a_str, op, b_str))
        op_outputs[op].append(ex.output_value)

    if not op_examples:
        return None

    # Parse the query
    query_parsed = _parse_input(problem.question)
    if query_parsed is None:
        return None
    q_a, q_op, q_b = query_parsed

    lines.append("Grouping examples by operator:")
    lines.append("")

    op_rules: dict[str, str | None] = {}

    for op, exs in op_examples.items():
        outs = op_outputs[op]
        lines.append(f"Operator '{op}':")
        for (a_str, _, b_str), out in zip(exs, outs):
            lines.append(f"  {a_str} {op} {b_str} = {out}")
        lines.append("")

        lines.append(f"Testing candidate rules for '{op}':")

        # Determine which rules to show: always show top display rules,
        # plus the winning rule once found
        consistent_rule = _find_consistent_rule(exs, outs)

        shown_rules = list(_DISPLAY_RULES)
        if consistent_rule and consistent_rule not in shown_rules:
            shown_rules.append(consistent_rule)

        for rule_name in shown_rules:
            # Compute this rule's value on each example and check match
            example_parts: list[str] = []
            all_match = True
            for (a_str, _, b_str), expected in zip(exs, outs):
                cands = _get_candidates(a_str, b_str)
                val = cands.get(rule_name)
                if val is None:
                    # Rule not defined for this input (e.g. division by zero)
                    example_parts.append(f"{a_str}{op}{b_str}=N/A")
                    all_match = False
                else:
                    example_parts.append(f"{a_str}{op}{b_str}={val}")
                    if val != expected:
                        all_match = False

            match_str = "yes" if all_match else "no"
            lines.append(
                f"  Rule '{rule_name}': {', '.join(example_parts)}  [match: {match_str}]"
            )

            # Stop displaying once we found the winning rule
            if all_match and rule_name == consistent_rule:
                break

        if consistent_rule:
            lines.append(f"  → Consistent rule for '{op}': {consistent_rule}")
        else:
            lines.append(f"  → No consistent rule found for '{op}' (best guess: closest match)")
        lines.append("")

        op_rules[op] = consistent_rule

    # Apply rule to the query
    lines.append(f"Query: {q_a} {q_op} {q_b}")

    rule = op_rules.get(q_op)
    if rule is None:
        # No consistent rule: make a best-effort guess using most votes
        lines.append("No consistent rule found for query operator. Cannot determine answer.")
        lines.append("")
        lines.append("I will now return the answer in \\boxed{}")
        lines.append("The answer in \\boxed{–} is \\boxed{?}")
        return "\n".join(lines)

    # Compute result
    q_cands = _get_candidates(q_a, q_b)
    result = q_cands.get(rule)

    if result is None:
        lines.append(f"Rule '{rule}' is undefined for query inputs. Cannot compute answer.")
        lines.append("")
        lines.append("I will now return the answer in \\boxed{}")
        lines.append("The answer in \\boxed{–} is \\boxed{?}")
        return "\n".join(lines)

    lines.append(f"Applying rule '{rule}': {_describe_computation(rule, q_a, q_b)} = {result}")
    lines.append("")
    lines.append("I will now return the answer in \\boxed{}")
    lines.append(f"The answer in \\boxed{{–}} is \\boxed{{{result}}}")
    return "\n".join(lines)


def _describe_computation(rule: str, a_str: str, b_str: str) -> str:
    """Produce a readable computation string for the CoT."""
    a, b = int(a_str), int(b_str)
    a1, a2 = int(a_str[0]), int(a_str[1])
    b1, b2 = int(b_str[0]), int(b_str[1])

    replacements = {
        "a+b": f"{a}+{b}",
        "a-b": f"{a}-{b}",
        "b-a": f"{b}-{a}",
        "a*b": f"{a}*{b}",
        "abs(a-b)": f"|{a}-{b}|",
        "a//b": f"{a}//{b}",
        "b//a": f"{b}//{a}",
        "a%b": f"{a}%{b}",
        "b%a": f"{b}%{a}",
        "max(a,b)": f"max({a},{b})",
        "min(a,b)": f"min({a},{b})",
        "a^b": f"{a} XOR {b}",
        "a&b": f"{a} AND {b}",
        "a|b": f"{a} OR {b}",
        "str(a)+str(b)": f"concat({a_str},{b_str})",
        "str(b)+str(a)": f"concat({b_str},{a_str})",
        "rev(a)+str(b)": f"concat(rev({a_str}),{b_str})",
        "str(a)+rev(b)": f"concat({a_str},rev({b_str}))",
        "rev(a+b)": f"rev(concat({a_str},{b_str}))",
        "rev(b+a)": f"rev(concat({b_str},{a_str}))",
        "a1+a2+b1+b2": f"{a1}+{a2}+{b1}+{b2}",
        "a1*a2+b1*b2": f"{a1}*{a2}+{b1}*{b2}",
        "a1*b1+a2*b2": f"{a1}*{b1}+{a2}*{b2}",
        "a1*b2+a2*b1": f"{a1}*{b2}+{a2}*{b1}",
        "(a1+b1)*10+(a2+b2)": f"({a1}+{b1})*10+({a2}+{b2})",
        "str(a1+b1)+str(a2+b2)": f"concat({a1}+{b1},{a2}+{b2})",
        "str(a1*b1)+str(a2*b2)": f"concat({a1}*{b1},{a2}*{b2})",
        "str(a1*b2)+str(a2*b1)": f"concat({a1}*{b2},{a2}*{b1})",
        "str(a1-b1)+str(a2-b2)": f"concat({a1}-{b1},{a2}-{b2})",
        "str(abs(a1-b1))+str(abs(a2-b2))": f"concat(|{a1}-{b1}|,|{a2}-{b2}|)",
        "str(a1+b2)+str(a2+b1)": f"concat({a1}+{b2},{a2}+{b1})",
        "a1*10+b1": f"{a1}*10+{b1}",
        "a2*10+b2": f"{a2}*10+{b2}",
        "a1*10+b2": f"{a1}*10+{b2}",
        "a2*10+b1": f"{a2}*10+{b1}",
        "b1*10+a1": f"{b1}*10+{a1}",
        "b2*10+a2": f"{b2}*10+{a2}",
        "(a1+a2)*b": f"({a1}+{a2})*{b}",
        "a*(b1+b2)": f"{a}*({b1}+{b2})",
        "(a1+a2)*(b1+b2)": f"({a1}+{a2})*({b1}+{b2})",
        "a1*b1": f"{a1}*{b1}",
        "a2*b2": f"{a2}*{b2}",
        "a1*b2": f"{a1}*{b2}",
        "a2*b1": f"{a2}*{b1}",
        "a1+b1": f"{a1}+{b1}",
        "a2+b2": f"{a2}+{b2}",
        "a1+b2": f"{a1}+{b2}",
        "a2+b1": f"{a2}+{b1}",
        "abs(a1-b1)+abs(a2-b2)": f"|{a1}-{b1}|+|{a2}-{b2}|",
        "abs(a1-b2)+abs(a2-b1)": f"|{a1}-{b2}|+|{a2}-{b1}|",
        "(a1+b1)%10": f"({a1}+{b1})%10",
        "(a2+b2)%10": f"({a2}+{b2})%10",
        "str((a1+b1)%10)+str((a2+b2)%10)": f"concat(({a1}+{b1})%10,({a2}+{b2})%10)",
    }

    # Handle parameterised mod rules
    for mod in [9, 10, 11, 19, 89, 90, 91, 99, 100, 101]:
        replacements[f"(a-b)%{mod}"] = f"({a}-{b})%{mod}"
        replacements[f"(b-a)%{mod}"] = f"({b}-{a})%{mod}"
        replacements[f"(a+b)%{mod}"] = f"({a}+{b})%{mod}"

    return replacements.get(rule, rule)


if __name__ == "__main__":
    # Quick smoke tests on 3 sample problems

    def make_problem(
        examples: list[tuple[str, str]], question: str, answer: str
    ) -> Problem:
        return Problem(
            id="test",
            category="equation_numeric_deduce",
            examples=[Example(inp, out) for inp, out in examples],
            question=question,
            answer=answer,
        )

    print("=" * 60)
    print("Test 1: rule = a-b  (op='/')")
    p1 = make_problem(
        [("34/44", "10"), ("41/32", "9"), ("55/23", "32")],
        question="69/52",
        answer="17",
    )
    cot1 = reasoning_equation_numeric_deduce(p1)
    print(cot1)
    print()

    print("=" * 60)
    print("Test 2: rule = str(a)+str(b)  (op='@')")
    p2 = make_problem(
        [("12@34", "1234"), ("56@78", "5678"), ("11@99", "1199")],
        question="69@52",
        answer="6952",
    )
    cot2 = reasoning_equation_numeric_deduce(p2)
    print(cot2)
    print()

    print("=" * 60)
    print("Test 3: rule = a1*b1+a2*b2  (op='*')")
    # a1*b1+a2*b2: 2*4+3*5=23, 1*3+2*6=15, 3*2+4*1=10
    p3 = make_problem(
        [("23*45", "23"), ("12*36", "15"), ("34*21", "10")],
        question="69*52",
        answer="30",
    )
    cot3 = reasoning_equation_numeric_deduce(p3)
    print(cot3)
    print()

    print("=" * 60)
    print("Test 4: rule = (a-b)%11  (op='/')")
    # (34-44)%11=1, (41-32)%11=9, (77-44)%11=0, answer for 69/52=(69-52)%11=6
    p4 = make_problem(
        [("34/44", "1"), ("41/32", "9"), ("77/44", "0")],
        question="69/52",
        answer="6",
    )
    cot4 = reasoning_equation_numeric_deduce(p4)
    print(cot4)
