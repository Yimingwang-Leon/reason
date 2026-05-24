"""Parse train.csv into Problem objects (in-memory, no per-id files)."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.reasoners.store_types import Example, Problem, ProblemCategory

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def classify(p: str) -> ProblemCategory:
    pl = p.lower()
    if "secret bit manipulation rule transforms 8-bit binary" in pl:
        return "bit_manipulation"
    if "secret encryption rules are used on text" in pl:
        return "cipher"
    if "secret unit conversion is applied to measurements" in pl:
        return "unit_conversion"
    if "gravitational constant has been secretly changed" in pl:
        return "gravity"
    if "numbers are secretly converted into a different numeral system" in pl:
        return "numeral"
    if "secret set of transformation rules is applied to equations" in pl:
        body = p.split("Below are a few examples:")[-1].split("Now,")[0]
        symbolic = not bool(re.search(r"\d", body))
        now = p.split("Now,")[-1].lower()
        guess = any(t in now for t in (
            "what input", "find the input", "what value", "find the value",
            "guess", "what would produce", "what equation", "expression that"))
        if symbolic:
            return "cryptarithm_guess" if guess else "cryptarithm_deduce"
        return "equation_numeric_guess" if guess else "equation_numeric_deduce"
    raise ValueError(f"Unrecognized prompt: {p[:120]!r}")


_PATTERNS = {
    "bit_manipulation": (
        re.compile(r"([01]{8}) -> ([01]{8})"),
        re.compile(r"Now, determine the output for:\s*([01]{8})"),
    ),
    "cipher": (
        re.compile(r"^([a-z ]+?)\s*->\s*([a-z ]+)$", re.MULTILINE),
        re.compile(r"Now, decrypt the following text:\s*([a-z ]+)"),
    ),
    "numeral": (
        re.compile(r"(\d+)\s*->\s*([IVXLCDMivxlcdm]+)"),
        re.compile(r"Now, write the number\s*(\d+)"),
    ),
    "gravity": (
        re.compile(r"For t = ([\d.]+)s, distance = ([\d.]+)"),
        re.compile(r"Now, determine the falling distance for t = ([\d.]+)s"),
    ),
    "unit_conversion": (
        re.compile(r"([\d.]+) m becomes ([\d.]+)"),
        re.compile(r"Now, convert the following measurement:\s*([\d.]+)"),
    ),
}


def _parse_equation_like(prompt: str) -> tuple[list[Example], str]:
    """Parse equation_numeric_* and cryptarithm_*: lhs = rhs lines + query."""
    body = prompt.split("Below are a few examples:", 1)[-1].split("\nNow,", 1)[0]
    examples: list[Example] = []
    for line in body.strip().splitlines():
        line = line.strip()
        if "=" in line:
            lhs, rhs = line.rsplit("=", 1)
            examples.append(Example(lhs.strip(), rhs.strip()))
    now = prompt.split("Now,", 1)[-1]
    m = re.search(r"determine the result for:\s*(\S+)", now)
    question = m.group(1).strip() if m else now.strip()
    return examples, question


def parse_prompt(category: ProblemCategory, prompt: str) -> tuple[list[Example], str]:
    if category in ("equation_numeric_deduce", "equation_numeric_guess",
                    "cryptarithm_deduce", "cryptarithm_guess"):
        return _parse_equation_like(prompt)
    ex_re, q_re = _PATTERNS[category]
    examples = [Example(a, b) for a, b in ex_re.findall(prompt)]
    qm = q_re.search(prompt)
    question = qm.group(1).strip() if qm else ""
    return examples, question


def load_problems(csv_path: Path | None = None) -> list[Problem]:
    """Load and parse all problems from a CSV (train.csv or train_split.csv)."""
    if csv_path is None:
        csv_path = DATA_DIR / "train.csv"
    df = pd.read_csv(csv_path)
    df["answer"] = df["answer"].astype(str)
    problems: list[Problem] = []
    for _, row in df.iterrows():
        cat = classify(row["prompt"])
        examples, question = parse_prompt(cat, row["prompt"])
        problems.append(Problem(
            id=str(row["id"]), category=cat,
            examples=examples, question=question,
            answer=str(row["answer"]), prompt=str(row["prompt"]),
        ))
    return problems


if __name__ == "__main__":
    problems = load_problems()
    print(f"Parsed {len(problems)} problems")
    from collections import Counter
    cats = Counter(p.category for p in problems)
    for cat, n in sorted(cats.items()):
        sample = next(p for p in problems if p.category == cat)
        ex_str = ", ".join(f"{e.input_value!r}->{e.output_value!r}" for e in sample.examples[:2])
        print(f"  {cat:25s} n={n:5d}  q={sample.question[:40]!r}  ex=[{ex_str}]")
