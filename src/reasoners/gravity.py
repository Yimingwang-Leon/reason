"""Gravity: d = k * t^2 reasoning generator.

Max-R redesign (design_gravity.json, Approach B + juror fixes):
- Same arithmetic path as before (long_multiplication_lines, long_division_lines,
  same sort/median tie-break, same trailing-zero strip, same final 3dp truncate),
  so the boxed value is BYTE-IDENTICAL to the prior reasoner on every problem.
- Surface restructured so every committed/derived token is a copy of a token
  already printed verbatim earlier in the same completion:
    * indexed per-example anchors  (k1 = ..., k2 = ...)
    * the result anchor `= {t_sq_full}` is emitted unconditionally (no dead branch)
    * the median selection is a MECHANICAL selection-sort ladder of explicit
      pairwise comparisons (juror needs-fix: removes the inferred-permutation
      `Sorted ascending` regime-(3) slot), each value tagged with its source
      index so tied lines are distinguishable to a greedy decoder
    * an explicit visible trailing-zero-strip line for k_display
    * the box copies the `= {box}` line directly above
"""

from __future__ import annotations

from .store_types import (
    Problem,
    cast_dp_pair,
    long_division_lines,
    long_multiplication_lines,
    truncate_3dp,
)


def reasoning_gravity(problem: Problem) -> str | None:
    lines: list[str] = []
    lines.append(
        "We need to determine the falling distance using d = k*t^2. "
        "Let me find k from the examples."
    )
    lines.append("I will put my final answer inside \\boxed{}.")
    lines.append("")

    k_strs: list[str] = []
    for idx, ex in enumerate(problem.examples, start=1):
        t = float(ex.input_value)
        if t <= 0:
            continue
        t_squared = round(t * t, 4)
        t_sq_full = str(t_squared)
        t_sq_str = truncate_3dp(t_sq_full)
        d_str = truncate_3dp(ex.output_value)

        # ANCHOR: restate the pair verbatim immediately before use (R2)
        lines.append(f"Example {idx}: t = {ex.input_value}s, d = {ex.output_value}m:")
        lines.append(f"t^2 = {ex.input_value} * {ex.input_value}:")
        sq_lines, _sq_result = long_multiplication_lines(
            ex.input_value, ex.input_value
        )
        lines.extend(sq_lines)
        # ALWAYS emit the t^2 result anchor (no optional/dead branch) (R3/R5)
        lines.append(f"= {t_sq_full}")

        d_cast, tsq_cast, _, _ = cast_dp_pair(d_str, t_sq_str)
        lines.append(
            f"k = {ex.output_value} / {ex.input_value}^2 "
            f"= {d_str} / {t_sq_full} = {d_cast} / {tsq_cast}"
        )
        div_lines, k_str = long_division_lines(d_cast, tsq_cast)
        lines.extend(div_lines)
        # UNIQUE INDEXED ANCHOR for this example's k (copy target for the
        # selection-sort ladder below) (R2)
        lines.append(f"k{len(k_strs) + 1} = {k_str}")
        k_strs.append(k_str)
        lines.append("")

    if not k_strs:
        return None

    k_values = [float(s) for s in k_strs]

    # --- MEDIAN BLOCK (mechanical selection-sort ladder -> copy median) ---
    # Collected line restates each k{i} anchor printed just above; each value is
    # tagged with its source index so duplicate-valued lines are distinguishable.
    collected = ", ".join(f"k{i + 1}={s}" for i, s in enumerate(k_strs))
    lines.append(f"Collected k values: {collected}")

    # Mechanical selection sort by (float, str) -- IDENTICAL ordering to
    # sorted(zip(k_values, k_strs)) so the chosen median string is unchanged.
    # remaining holds (value, str, original_index) of not-yet-selected items.
    remaining: list[tuple[float, str, int]] = [
        (v, s, i) for i, (v, s) in enumerate(zip(k_values, k_strs))
    ]
    sorted_items: list[tuple[float, str, int]] = []
    rank = 0
    while remaining:
        rank += 1
        # pick the minimum by (value, str) via explicit pairwise comparisons
        best = remaining[0]
        for cand in remaining[1:]:
            # compare on the SAME key tuple Python's sorted uses: (float, str)
            if (cand[0], cand[1]) < (best[0], best[1]):
                # show the comparison that updates the running minimum (R4)
                lines.append(
                    f"compare k{cand[2] + 1}={cand[1]} < k{best[2] + 1}={best[1]} "
                    f"-> min so far k{cand[2] + 1}={cand[1]}"
                )
                best = cand
        # one rank per line; value is a copy from the Collected line (R3/R4)
        lines.append(f"rank {rank}: k{best[2] + 1}={best[1]}")
        sorted_items.append(best)
        remaining = [r for r in remaining if r[2] != best[2]]

    n = len(sorted_items)
    if n % 2 == 0 and n >= 2:
        mid_rank = n // 2  # lower-middle (matches paired[n//2 - 1], 0-indexed)
        lines.append(
            f"Count = {n} (even). Lower-middle rank = {n}/2 = {mid_rank}."
        )
    else:
        mid_rank = n // 2 + 1  # matches paired[n//2], 0-indexed
        lines.append(
            f"Count = {n} (odd). Middle rank = ({n}+1)/2 = {mid_rank}."
        )
    k_fit_str = sorted_items[mid_rank - 1][1]
    # Median is a COPY of the `rank {mid_rank}: ...` line just above (R6).
    lines.append(f"Median k = value at rank {mid_rank} = {k_fit_str}.")

    # EXPLICIT visible trailing-zero-strip step (was a silent transform) (R2).
    k_display = k_fit_str.rstrip("0").rstrip(".")
    lines.append(f"Strip trailing zeros: k = {k_display}.")
    lines.append("")

    # --- TARGET BLOCK ---
    lines.append(f"Target: t = {problem.question}.")
    lines.append(f"t^2 = {problem.question} * {problem.question}:")
    sq_lines, t_sq_str = long_multiplication_lines(problem.question, problem.question)
    lines.extend(sq_lines)
    lines.append(f"= {t_sq_str}")
    lines.append("")

    # both operands restated verbatim on the line right before the multiply (R2):
    # k_display copies the strip line; t_sq_str copies the `= {t_sq_str}` line.
    lines.append(f"d = k * t^2 = {k_display} * {t_sq_str}:")
    mult_lines, mult_result = long_multiplication_lines(k_display, t_sq_str)
    lines.extend(mult_lines)
    # Truncate to 3 decimal places (unchanged primitive).
    dot = mult_result.index(".")
    boxed_answer = mult_result[: dot + 4]
    lines.append(f"= {boxed_answer}")

    lines.append("")
    # box is a character-for-character copy of the `= {box}` line above (R6).
    lines.append(f"The answer is \\boxed{{{boxed_answer}}}")
    return "\n".join(lines)
