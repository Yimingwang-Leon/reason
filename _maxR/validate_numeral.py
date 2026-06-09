"""Validate the redesigned numeral reasoner against the frozen baseline.

Gate: extracted \\boxed{} answer must be BYTE-IDENTICAL to baseline for every id.
Also: under 7680-token cap, deterministic, grep-clean of global-rule phrasing.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from tokenizers import Tokenizer

from src.problems import load_problems
from src.reasoning import extract_answer
from src.reasoners.numeral import reasoning_numeral

REPO = Path(__file__).resolve().parent.parent
BASELINE = json.loads((REPO / "_maxR" / "baseline_numeral.json").read_text())
TOK = Tokenizer.from_file(str(REPO / "src" / "tokenizer.json"))
GEN_LIMIT = 7680


def box_of(problem) -> str | None:
    text = reasoning_numeral(problem)
    if text is None:
        return None
    pred = extract_answer(text)
    return pred if pred != "" else None


def main() -> None:
    problems = {p.id: p for p in load_problems() if p.category == "numeral"}

    # --- 1. byte-identical box gate ---
    n_checked = 0
    n_mismatch = 0
    mismatch_ids: list[str] = []
    for pid, base_val in BASELINE.items():
        assert pid in problems, f"baseline id {pid} not found among numeral problems"
        n_checked += 1
        got = box_of(problems[pid])
        # treat null==null match; reasoner None -> box null
        base_norm = base_val if base_val is not None else None
        if got != base_norm:
            n_mismatch += 1
            if len(mismatch_ids) < 5:
                mismatch_ids.append(pid)

    # --- 2. token cap ---
    max_tokens = 0
    max_id = None
    for pid, p in problems.items():
        cot = reasoning_numeral(p)
        box = box_of(p)
        box_str = box if box is not None else ""
        completion = f"{cot}\n</think>\n\\boxed{{{box_str}}}<|im_end|>"
        ntok = len(TOK.encode(completion).ids)
        if ntok > max_tokens:
            max_tokens = ntok
            max_id = pid
    under_cap = max_tokens <= GEN_LIMIT

    # --- 3. determinism on 3 sampled problems ---
    sample_ids = sorted(problems.keys())[:3]
    deterministic = True
    for pid in sample_ids:
        a = reasoning_numeral(problems[pid])
        b = reasoning_numeral(problems[pid])
        if a != b:
            deterministic = False

    # --- 4. grep_clean: no global-rule-assertion phrasing ---
    BAD = [
        "the rule is", "the pattern is", "the operation is", "the formula is",
        "the shift is", "pattern is", "secret", "the rule:", "rule is",
    ]
    grep_clean = True
    bad_hits: list[str] = []
    # also flag a bare "Best:" without printed comparisons (rubric R4)
    for pid in sorted(problems.keys()):
        cot = reasoning_numeral(problems[pid]).lower()
        for phrase in BAD:
            if phrase in cot:
                grep_clean = False
                bad_hits.append(f"{pid}: {phrase}")
        if re.search(r"^\s*best:", cot, re.MULTILINE):
            grep_clean = False
            bad_hits.append(f"{pid}: bare Best:")

    print(json.dumps({
        "n_checked": n_checked,
        "n_mismatch": n_mismatch,
        "sample_mismatch_ids": mismatch_ids,
        "max_completion_tokens": max_tokens,
        "max_token_id": max_id,
        "under_cap": under_cap,
        "deterministic": deterministic,
        "grep_clean": grep_clean,
        "bad_hits": bad_hits[:10],
    }, indent=2))


if __name__ == "__main__":
    main()
