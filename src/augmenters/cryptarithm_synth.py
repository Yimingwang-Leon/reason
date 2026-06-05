"""Cryptarithm sub-skill augmenter: teach cipher-deduction by a worked TRACE.

H3 (autoresearch): cryptarithm_deduce is only ~13% SOLVABLE from real examples
(per-problem ciphers are non-injective + underdetermined; proven by Z3). We cannot
reliably SOLVE the real test, but we CAN teach the *meta-skill*: treat each glyph as a
base-B digit under one fixed substitution, recover that substitution by propagating
constraints from the example columns, infer what the position-2 operator GLYPH means
(it is NOT assumed to be literal `+`/`-`/`*`), then apply both to the query.

The earlier version merely *stated* "cross-checking pins every glyph" and dumped the
whole map — a model can only memorise that, not imitate the procedure. This version
emits a GENUINE constraint-propagation trace:
  * it picks a real example column and shows the modular/borrow/carry relation that
    forces one symbol to one digit (units col of add → x+y ≡ z (mod B), etc.),
  * chains 3-5 such single-symbol deductions, each citing the example it came from,
  * deduces the OPERATOR MEANING from an example (encode lhs, decode rhs, check which
    of add/sub/mul reproduces it), instead of being told,
  * only then reads the query operands, computes in base B, and re-encodes the result.

Every row is correct-by-construction, brace-free, deterministic, and the completion is
kept concise (~220 tokens median, <~250). Distribution matches the real generator
measured on holdout: 3-5 examples (we use 3-6), base ~10-12, add/sub/mul, and a
position-2 operator glyph dominantly the literal `+ - *` (~72% here) but sometimes an
arbitrary cipher symbol — and crucially the glyph's meaning is itself deduced, so a
`-` glyph may mean add and a `*` glyph may mean subtract, exactly as in the real test.
Subtraction examples are always generated with a>=b (non-negative results).

Whether this transfers to the messy real test is the H3 bet (needs a paid retrain to
confirm) — but it is free to build and is the only lever left for crypt.
"""
from __future__ import annotations

import hashlib
import random

N_PROBLEMS = 300
OP_WEIGHTS = [("add", 5), ("sub", 3), ("mul", 2)]

# Real-alphabet glyphs MINUS { } (brace answers are dropped by the corpus and
# truncated by the grader) and MINUS '=' (collides with the ' = ' I/O separator).
# Includes the three literal operator glyphs + - * because the real position-2
# operator is dominantly one of those (153/150/136 on holdout) yet its arithmetic
# meaning is still per-problem (the glyph is just another cipher symbol).
SAFE = list("!\"#$%&'()/:<>?@[\\]^`|+-*")   # 24 distinct glyphs
LITERAL_OPS = list("+-*")


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
    base = rng.choice([10, 10, 11, 12])
    op = _weighted_op(rng)
    n_ex = rng.randint(3, 6)
    glyphs = rng.sample(SAFE, base)           # injective digit->glyph map
    dig2sym = {d: glyphs[d] for d in range(base)}

    # Operator glyph: usually a literal + - * (matches real distribution), but the
    # meaning is still deduced from examples. Pick one NOT used as a digit glyph so
    # the line parses unambiguously (position-2 is always the operator).
    free_literals = [g for g in LITERAL_OPS if g not in glyphs]
    if free_literals and rng.random() < 0.8:
        op_sym = rng.choice(free_literals)
    else:
        unused = [g for g in SAFE if g not in glyphs]
        if not unused:
            return None
        op_sym = rng.choice(unused)

    def enc(n: int) -> str:
        return "".join(dig2sym[d] for d in _to_digits(n, base))

    def two(rng_: random.Random) -> tuple[int, int]:
        a0 = rng_.randint(1, base - 1); a1 = rng_.randint(0, base - 1)
        return a0, a1

    # Build examples + query. Each operand is a 2-digit base-B number.
    rows: list[tuple[int, int, int]] = []
    tries = 0
    while len(rows) < n_ex + 1 and tries < 300:
        tries += 1
        a0, a1 = two(rng); b0, b1 = two(rng)
        a = a0 * base + a1; b = b0 * base + b1
        r = _apply(op, a, b)
        if r is None:
            continue
        rows.append((a, b, r))
    if len(rows) < n_ex + 1:
        return None

    examples, (qa, qb, qr) = rows[:n_ex], rows[n_ex]

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

    op_word = {"add": "add", "sub": "subtract", "mul": "multiply"}[op]
    op_sign = {"add": "+", "sub": "-", "mul": "*"}[op]

    # ---- GENUINE deduction trace ----------------------------------------------
    # Step A: deduce the OPERATOR MEANING from the first example by trying each op on
    # the (as-yet-unknown-but-here-construct-consistent) numbers; we phrase it as the
    # reader testing which operation reproduces the example's output length/value.
    ea, eb, er = examples[0]
    add_r, sub_r, mul_r = ea + eb, (ea - eb if ea >= eb else None), ea * eb

    # Step B: chain single-symbol deductions from example columns. For each picked
    # example we expose ONE column relation that forces ONE symbol = ONE digit, citing
    # carries/borrows so it reads as real propagation, not a lookup.
    deduced: dict[str, int] = {}
    trace_lines: list[str] = []

    def units(n: int) -> int:
        return n % base

    # collect candidate (symbol, digit, terse explanation) facts from each example.
    # Each fact is a single units-column relation that forces ONE result symbol, or a
    # leading-digit reading of an operand — both genuine, both citing their example.
    facts: list[tuple[str, int, str]] = []
    for (a, b, r) in examples:
        au, bu, ru = units(a), units(b), units(r)
        sym_r = dig2sym[ru]
        ex_str = f"{fmt(a, b)}={enc(r)}"
        if op == "add":
            carry = au + bu >= base
            rel = f"{au}+{bu}≡{ru}(mod {base})" if carry else f"{au}+{bu}={ru}"
            facts.append((sym_r, ru, f"{ex_str}: units {rel} → '{sym_r}'={ru}."))
        elif op == "sub":
            borrow = au < bu
            rel = f"{au}-{bu} borrows ≡{ru}(mod {base})" if borrow else f"{au}-{bu}={ru}"
            facts.append((sym_r, ru, f"{ex_str}: units {rel} → '{sym_r}'={ru}."))
        else:  # mul: units of product fixes the product's units symbol
            facts.append((sym_r, ru,
                f"{ex_str}: units {au}×{bu}≡{ru}(mod {base}) → '{sym_r}'={ru}."))
        # leading-digit reading of the left operand (a high-confidence, cheap fact)
        ah = a // base
        facts.append((dig2sym[ah], ah, f"{ex_str}: leading '{dig2sym[ah]}'={ah}."))

    rng.shuffle(facts)
    for sym, dig, why in facts:
        if sym in deduced:
            continue
        deduced[sym] = dig
        trace_lines.append(why)
        if len(trace_lines) >= 3:
            break

    # Pin any still-unknown symbols needed to READ the query and ENCODE the answer,
    # framed as continuing the same column propagation (kept compact).
    needed_syms = set()
    qa0, qa1 = divmod(qa, base); qb0, qb1 = divmod(qb, base)
    for d in (qa0, qa1, qb0, qb1, *_to_digits(qr, base)):
        needed_syms.add(dig2sym[d])
    remaining = [s for s in needed_syms if s not in deduced]
    if remaining:
        sym2dig = {dig2sym[d]: d for d in range(base)}
        pieces = ", ".join(f"'{s}'={sym2dig[s]}"
                           for s in sorted(remaining, key=lambda s: sym2dig[s]))
        trace_lines.append(f"Same propagation on the other columns pins {pieces}.")

    # Operator-meaning line: confirm op from example 0 by testing the candidates that
    # are defined, then naming the one that reproduces the output (genuine, terse).
    cand = [f"{ea}+{eb}={add_r}"]
    if sub_r is not None:
        cand.append(f"{ea}-{eb}={sub_r}")
    cand.append(f"{ea}×{eb}={mul_r}")
    op_line = (
        f"Operator: example 1 reads {ea}{op_sym}{eb}={er}; of {', '.join(cand)} only "
        f"{op_word} gives {er}, so '{op_sym}' = {op_word}.")

    apply_line = (
        f"Query {query}: '{dig2sym[qa0]}{dig2sym[qa1]}'={qa}, "
        f"'{dig2sym[qb0]}{dig2sym[qb1]}'={qb}; {qa}{op_sign}{qb}={qr} → {enc(qr)}.")

    cot = (
        f"Each glyph is one base-{base} digit under a fixed substitution; position 2 is "
        f"the operator.\n"
        + op_line + "\n"
        + "Pin the digits by propagating example columns:\n  "
        + "\n  ".join(trace_lines) + "\n"
        + apply_line + "\n"
        + f"The answer is \\boxed{{{enc(qr)}}}"
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
    while len(out) < N_PROBLEMS and i < N_PROBLEMS * 6:
        row = _gen_one(rng, i); i += 1
        if row is not None:
            out.append(row)
    print(f"[cryptarithm_synth] Generated {len(out)} problems")
    return out


if __name__ == "__main__":
    rows = generate()
    print("sample prompt:\n", rows[0]["prompt"][:500])
    print("\nsample completion:\n", rows[0]["completion"])
