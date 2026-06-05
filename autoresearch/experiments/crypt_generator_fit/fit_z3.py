"""H1: Does cryptarithm_deduce fit a clean per-problem arithmetic generator?

Uses Z3 to definitively (SAT/UNSAT) test, for each holdout problem, whether
there EXISTS a per-problem symbol->digit map + (base, op, endian, reading)
consistent with ALL pairs INCLUDING the known query answer. This is the
generator-reverse-engineering measurement: if a high fraction fits, the
generator is characterizable and we can mass-generate synthetic training data
to teach the model the cipher-deduction prior (the path to crypt > 13%).

Run: python autoresearch/experiments/crypt_generator_fit/fit_z3.py
"""
from __future__ import annotations
import re, time
from collections import Counter
import pandas as pd
import z3

ROOT = "/Users/yimingwang/Kaggle/reason"
df = pd.read_csv(f"{ROOT}/data/holdout.csv")
sub = df[df["cat"] == "cryptarithm_deduce"]


def parse(prompt):
    lines = [l.strip() for l in prompt.splitlines() if l.strip()]
    exs, q = [], None
    for l in lines:
        if " = " in l and "wonderland" not in l.lower() and "below are" not in l.lower() and not l.lower().startswith("now"):
            a, b = l.split(" = ", 1); exs.append((a.strip(), b.strip()))
        m = re.search(r"determine the result for:\s*(.+)$", l)
        if m: q = m.group(1).strip()
    return exs, q


def num(syms_vars, B, endian):
    ds = list(syms_vars)
    if endian == "LE": ds = ds[::-1]
    v = 0
    for d in ds: v = v * B + d
    return v


def fit(pairs, B, endian, op, mode, injective, timeout_ms=2000):
    """pairs: list of (s0,s1,s3,s4,out_str). Returns True if SAT."""
    s = z3.Solver(); s.set("timeout", timeout_ms)
    syms = []
    for s0, s1, s3, s4, out in pairs:
        for c in [s0, s1, s3, s4] + list(out):
            if c not in syms: syms.append(c)
    var = {c: z3.Int(f"x_{i}") for i, c in enumerate(syms)}
    for c in syms:
        s.add(var[c] >= 0, var[c] < B)
    if injective:
        s.add(z3.Distinct(*[var[c] for c in syms]))
    for s0, s1, s3, s4, out in pairs:
        a = num([var[s0], var[s1]], B, endian)
        b = num([var[s3], var[s4]], B, endian)
        L = len(out)
        outv = num([var[c] for c in out], B, endian)
        if mode == "whole":
            if op == "add": s.add(a + b == outv, a + b >= 0)
            elif op == "sub": s.add(a - b == outv, a - b >= 0)
            elif op == "rsub": s.add(b - a == outv, b - a >= 0)
            elif op == "mul": s.add(a * b == outv)
            elif op == "absdiff": s.add(z3.If(a >= b, a - b, b - a) == outv)
            elif op == "concat":
                if L != 4: return False
                for vc, sym in zip(out, [s0, s1, s3, s4]): s.add(var[vc] == var[sym])
                continue
            elif op == "rconcat":
                if L != 4: return False
                for vc, sym in zip(out, [s3, s4, s0, s1]): s.add(var[vc] == var[sym])
                continue
            else: return False
        else:  # colmod: per-column op mod B, output length must be 1 or 2 (lead-zero drop)
            def col(x, y, o):
                if o == "add": return (x + y) % B
                if o == "sub": return (x - y) % B
                if o == "rsub": return (y - x) % B
                if o == "absdiff": return z3.If(x >= y, x - y, y - x) % B
                if o == "mul": return (x * y) % B
                return None
            c0 = col(var[s0], var[s3], op); c1 = col(var[s1], var[s4], op)
            if c0 is None: return False
            if L == 2:
                s.add(var[out[0]] == c0, var[out[1]] == c1)
            elif L == 1:
                s.add(c0 == 0, var[out[0]] == c1)
            else:
                return False
    return s.check() == z3.sat


OPS_WHOLE = ["add", "sub", "rsub", "absdiff", "mul", "concat", "rconcat"]
OPS_COL = ["add", "sub", "rsub", "absdiff", "mul"]


def main():
    fit_any = 0; tot = 0; how = Counter(); t0 = time.time()
    nofit = []
    for _, r in sub.iterrows():
        exs, q = parse(r["prompt"]); ans = str(r["answer"])
        if q is None or len(q) != 5 or any(len(i) != 5 for i, _ in exs): continue
        pairs = [(i[0], i[1], i[3], i[4], o) for i, o in (list(exs) + [(q, ans)])]
        tot += 1; found = None
        for inj in (True, False):
            for mode, ops in (("whole", OPS_WHOLE), ("colmod", OPS_COL)):
                for B in range(10, 17):
                    for endian in ("BE", "LE"):
                        for op in ops:
                            if fit(pairs, B, endian, op, mode, inj):
                                found = (mode, B, endian, op, "inj" if inj else "non"); break
                        if found: break
                    if found: break
                if found: break
            if found: break
        if found:
            fit_any += 1
            how[("op:" + found[3],)] += 1; how[("mode:" + found[0],)] += 1
            how[("base:%d" % found[1],)] += 1; how[("end:" + found[2],)] += 1; how[("inj:" + found[4],)] += 1
        else:
            nofit.append(r["id"])
    print(f"CRYPT generator-fit via Z3 (uses answers, full model space): {fit_any}/{tot} = {fit_any/tot*100:.1f}%  [{time.time()-t0:.0f}s]")
    for k, v in how.most_common(30): print("  ", k[0], v)
    print("no-fit sample:", nofit[:10])


if __name__ == "__main__":
    main()
