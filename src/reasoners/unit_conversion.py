"""Unit conversion: output = factor * input reasoning generator.

Max-R redesign (text-only; boxed answer is byte-identical to the frozen
median-factor baseline). The boxed value is still computed exactly as before:
the median of the per-example factor strings (even count -> lower-middle index),
multiplied into the query, truncated to 3 dp. Only the REASONING TEXT changed,
to make every committed value a copy of a token already printed in the same
trace (locality law):

  * an explicit Examples anchor block restates every (input -> output) pair
    verbatim from the prompt before any arithmetic consumes it (R2);
  * each division / multiplication row is one fixed-template micro-step (R3),
    via the shared atomic helpers, unchanged from the legacy reasoner;
  * the median is selected MECHANICALLY (R4): the factor values are printed,
    then printed again in sorted order, then the middle position is named
    explicitly with a fixed tie-break, and the chosen factor is a verbatim
    copy of an entry in the printed sorted list -- never a bare "median is X";
  * the query and the chosen factor are restated on the multiply line before
    they are consumed (R2);
  * the boxed value is a char-for-char copy of the `= {boxed}` line directly
    above it (R6), which is the same self-consistent value the legacy reasoner
    boxed.
"""

from __future__ import annotations

from .store_types import (
    Problem,
    cast_dp_pair,
    long_division_lines,
    long_multiplication_lines,
    truncate_3dp,
)


def reasoning_unit_conversion(problem: Problem) -> str | None:
    lines: list[str] = []
    lines.append("We are told output = factor * input. The factor is the same for every line.")
    lines.append(
        "I will compute the factor from each example, take the median factor, "
        "then multiply the query by it."
    )
    lines.append("I will put my final answer inside \\boxed{}.")
    lines.append("")

    # --- Anchor block (R2): restate every example verbatim before any use. ---
    lines.append("Examples (input -> output):")
    usable: list[tuple[str, str]] = []
    for ex in problem.examples:
        lines.append(f"{ex.input_value} -> {ex.output_value}")
        if float(ex.input_value) != 0:
            usable.append((ex.input_value, ex.output_value))
    lines.append("")

    if not usable:
        return None

    # --- Per-example factor via the shared atomic long-division helper (R3). ---
    factor_strs: list[str] = []
    for inp_value, out_value in usable:
        out_str = truncate_3dp(out_value)
        inp_str = truncate_3dp(inp_value)
        inp_cast, out_cast, inp_dp, out_dp = cast_dp_pair(inp_str, out_str)
        lines.append(f"Use example {inp_value} -> {out_value}.")
        lines.append(
            f"Cast to equal decimal places: {inp_cast} -> {out_cast}"
        )
        lines.append(f"factor = {out_cast} / {inp_cast}")
        div_lines, factor_str = long_division_lines(out_cast, inp_cast)
        lines.extend(div_lines)
        lines.append(f"= {factor_str}")
        lines.append(f"factor = {factor_str}")
        factor_strs.append(factor_str)
        lines.append("")

    factors = [float(s) for s in factor_strs]

    # --- Mechanical median selection (R4): print the values, print the sorted
    # values, name the middle position with a fixed tie-break, then COPY the
    # chosen factor from the printed sorted list. Identical pick to legacy. ---
    lines.append(f"factor values: {', '.join(factor_strs)}")
    paired = sorted(zip(factors, factor_strs))
    sorted_strs = [s for _, s in paired]
    lines.append(f"factor values sorted: {', '.join(sorted_strs)}")
    n = len(paired)
    if n % 2 == 0 and n >= 2:
        mid_index = n // 2 - 1
        lines.append(
            f"Even count ({n}): take the lower-middle entry, "
            f"position {mid_index + 1} of {n} in the sorted list."
        )
    else:
        mid_index = n // 2
        lines.append(
            f"Odd count ({n}): take the middle entry, "
            f"position {mid_index + 1} of {n} in the sorted list."
        )
    med_factor_str = sorted_strs[mid_index]
    lines.append(f"median factor = {med_factor_str}")

    # --- Multiply the query by the median factor (R2 restate, R3 atomic rows).
    # The boxed value matches the legacy reasoner: it uses the same trailing-
    # zero-stripped factor display and the same 3dp truncation. ---
    q_str = problem.question
    med_display = med_factor_str.rstrip("0").rstrip(".")
    lines.append("")
    lines.append(f"Now convert the query {q_str}.")
    lines.append(f"{q_str} * {med_display}:")
    mult_lines, mult_result = long_multiplication_lines(q_str, med_display)
    lines.extend(mult_lines)
    dot = mult_result.index(".")
    boxed_answer = mult_result[: dot + 4]
    lines.append(f"= {boxed_answer}")

    # --- Self-consistent box (R6): char-for-char copy of the line above. ---
    lines.append("")
    lines.append(f"The answer is \\boxed{{{boxed_answer}}}")
    return "\n".join(lines)
