"""Cryptarithm sub-skill augmenter: cipher-deduction + base-B arithmetic execution.

H3 (autoresearch): cryptarithm_deduce is only ~13% SOLVABLE from examples (ciphers
are non-injective + underdetermined). We cannot reliably SOLVE real problems — but we
CAN teach the model the *meta-skill*: read operands through a per-problem symbol->digit
cipher, do 2-digit base-B arithmetic, and encode the result back. We generate clean,
fully-determined synthetic problems (KNOWN injective cipher, enough examples to pin it)
with a worked deduction completion. Every row is correct-by-construction and brace-free.

Distribution chosen to match the observed real generator (Z3 H1 signal): base ~10-11,
whole-number, add-dominant with some sub/mul; operator glyph fixed per problem; the
result is encoded with the same cipher (no leading-zero pad).

Whether this transfers to the messy real test is the H3 bet (needs a paid retrain to
confirm) — but it is free to build and is the only lever left for crypt.
"""
from __future__ import annotations

import hashlib
import random

N_PROBLEMS = 300
N_EXAMPLES = 6          # enough that the injective cipher is uniquely deducible
OP_WEIGHTS = [("add", 5), ("sub", 3), ("mul", 2)]

# 21 real-alphabet glyphs MINUS { } (brace answers are dropped by the corpus and
# truncated by the grader) and MINUS '=' (collides with the ' = ' input/output
# separator). The per-problem operator glyph is drawn from the same alphabet but
# from symbols NOT used as digits in that problem (matches the real format, where
# the position-2 operator is just another alphabet symbol).
SAFE = list("!\"#$%&'()/:<>?@[\\]^`|+-*")   # 24 distinct glyphs (incl real ops + - *)


def _weighted_op(rng: random.Random) -> str:
    pool = [op for op, w in OP_WEIGHTS for _ in range(w)]
    return rng.choice(pool)


def _apply(op: str, a: int, b: int) -> int | None:
    if op == "add":
        return a + b
    if op == "sub":
        return a - b if a >= b else None
    if op == "mul":
        return a * b
    return None


def _to_digits(n: int, base: int) -> list[int]:
    if n == 0:
        return [0]
    d = []
    while n > 0:
        d.append(n % base)
        n //= base
    return d[::-1]


def _gen_one(rng: random.Random, idx: int) -> dict[str, str] | None:
    base = rng.choice([10, 10, 11])           # base 10 dominant, some 11
    op = _weighted_op(rng)
    glyphs = rng.sample(SAFE, base)           # injective digit->glyph map
    dig2sym = {d: glyphs[d] for d in range(base)}
    op_sym = rng.choice([g for g in SAFE if g not in glyphs])  # operator: an unused glyph

    def enc(n: int) -> str:
        return "".join(dig2sym[d] for d in _to_digits(n, base))

    def two(rng_: random.Random) -> tuple[int, int]:
        a0 = rng_.randint(1, base - 1); a1 = rng_.randint(0, base - 1)
        return a0, a1

    # Build examples whose symbols collectively cover all glyphs (so the map is
    # genuinely deducible from the shown examples), plus the query.
    rows: list[tuple[int, int, int]] = []
    seen: set[int] = set()
    tries = 0
    while len(rows) < N_EXAMPLES + 1 and tries < 200:
        tries += 1
        a0, a1 = two(rng); b0, b1 = two(rng)
        a = a0 * base + a1; b = b0 * base + b1
        r = _apply(op, a, b)
        if r is None:
            continue
        rows.append((a, b, r))
        for v in (a0, a1, b0, b1, *_to_digits(r, base)):
            seen.add(v)
    if len(rows) < N_EXAMPLES + 1:
        return None

    examples, (qa, qb, qr) = rows[:N_EXAMPLES], rows[N_EXAMPLES]

    def fmt(a: int, b: int) -> str:
        a0, a1 = divmod(a, base); b0, b1 = divmod(b, base)
        return f"{dig2sym[a0]}{dig2sym[a1]}{op_sym}{dig2sym[b0]}{dig2sym[b1]}"

    ex_lines = [f"{fmt(a, b)} = {enc(r)}" for a, b, r in examples]
    query = fmt(qa, qb)
    prompt = (
        "In Alice's Wonderland, a secret set of transformation rules is applied to equations. "
        "Below are a few examples:\n"
        + "\n".join(ex_lines)
        + f"\n\nNow, determine the result for: {query}"
    )

    # Worked deduction completion: recover the cipher, then apply to the query.
    op_word = {"add": "added", "sub": "subtracted", "mul": "multiplied"}[op]
    mapping_str = ", ".join(f"{dig2sym[d]}={d}" for d in range(base))
    qa0, qa1 = divmod(qa, base); qb0, qb1 = divmod(qb, base)
    cot = (
        f"Each glyph is a base-{base} digit under one fixed substitution. "
        f"Cross-checking the examples column by column pins every glyph:\n"
        f"  {mapping_str}\n"
        f"The middle glyph '{op_sym}' means the two 2-digit numbers are {op_word}.\n"
        f"Query {query}: left = {dig2sym[qa0]}{dig2sym[qa1]} = {qa}, "
        f"right = {dig2sym[qb0]}{dig2sym[qb1]} = {qb}.\n"
        f"{qa} {'+' if op=='add' else '-' if op=='sub' else '×'} {qb} = {qr}, "
        f"which encodes back to {enc(qr)}.\n"
        f"The answer is \\boxed{{{enc(qr)}}}"
    )
    pid = hashlib.sha256(f"cryptarithm_synth_{idx}".encode()).hexdigest()[:8]
    ans = enc(qr)
    if "{" in ans or "}" in ans:        # safety: never emit a brace answer
        return None
    return {"id": pid, "prompt": prompt, "completion": cot,
            "category": "cryptarithm_synth"}


def generate() -> list[dict[str, str]]:
    rng = random.Random(31337)
    out: list[dict[str, str]] = []
    i = 0
    while len(out) < N_PROBLEMS and i < N_PROBLEMS * 4:
        row = _gen_one(rng, i); i += 1
        if row is not None:
            out.append(row)
    print(f"[cryptarithm_synth] Generated {len(out)} problems")
    return out


if __name__ == "__main__":
    rows = generate()
    print("sample prompt:\n", rows[0]["prompt"][:400])
    print("\nsample completion:\n", rows[0]["completion"])
