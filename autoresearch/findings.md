# Findings — Nemotron Reasoning Challenge: chasing 0.87 / 0.89

> **STATUS 2026-06-05: CONCLUDED (free research exhausted).** Honest ceiling ~0.85-0.86 (paid, unguaranteed). Guaranteed 0.86 = free public notebook. 0.89 unreachable. H2 closed (brace-cap +0.16pp). Cron stopped.

**Optimization target:** offline `oracle` (reasoner solve-rate on data/holdout.csv, 1899 problems) and projected LB = oracle × R.
**Current best:** LB 0.84 (run-005, 2ep=3ep). Offline oracle 89.0% (committed f083ffe). R = 0.84/0.877 = **0.958**.

## Current Understanding

- **LB = oracle × R.** oracle = what reasoners solve; R = how reliably the trained model reproduces the CoT at greedy decode.
- **oracle is maxed at 89.0%** and cannot meaningfully rise except via cryptarithm:
  | cat | oracle | ceiling reason |
  |---|---|---|
  | cipher / gravity / numeral / unit_conversion | 100% | deterministic-complete on holdout |
  | equation_numeric_deduce | 80.1% | 18/29 fails undecidable (query op absent from examples) |
  | bit_manipulation | 88.4% | GF(2)-affine test: only 2/37 fails recoverable; rest underdetermined / >3-var |
  | cryptarithm_deduce | 13.3% | **THE lever** — see below |
- **R = 0.958** (only ~4% reproduction loss). Concentrated in long-CoT cats (cipher ~1500 tok, bit_manip ~2100 tok) via autoregressive error-compounding. R levers (augmenters, shorter CoT, curriculum) are **unmeasurable offline** — need a paid retrain to validate.
- **Realistic ceiling ≈ 0.87** = oracle 0.89 × best-achievable R (~0.98). To beat 0.87 you MUST raise oracle > 0.89 → crack cryptarithm.

## The cryptarithm_deduce problem (the whole game)

- Format: examples `S0 S1 OP S3 S4 = OUT` then a query. Input always 5 chars; OUT 1-4 chars. 26-symbol alphabet (23 non-operator + `+ - *`). 3-5 examples/problem.
- Structure (reverse-engineered): per-problem symbol→digit cipher (base ~10-15), OP at position 2 is an operator whose arithmetic meaning is per-problem deduced from {add, sub, rsub, absdiff, mul, concat, rconcat}; negatives prefixed with op glyph as sign; two readings (whole-number base-B vs column-wise mod-B); BE/LE both occur.
- **Fundamental wall: underdetermined.** 3-5 examples can't pin a ~13-symbol per-problem cipher → unique query prediction for only ~2/165 from examples alone. Plus **24% (40/165) of answers contain `}` → truncated by `\boxed{}` extraction** (corpus also drops these rows).
- Best solvers: current committed = agentB (13.3%, 22/165). Union of all attempts = 16.4% (27/165).
- **Public field is stuck at 0.85-0.86** (huikang's public recipe ≈ ours: rank32, ~3ep, min-logprob). If anyone below 0.89 had cracked crypt it'd show; they haven't. So crypt is the universal wall, NOT a recipe gap.

## Lessons and Constraints (do NOT repeat)

- **2ep = 3ep = 0.84.** Epoch count is settled at 2. More epochs (4,5) will NOT help — corpus is memorized by epoch 1 (nll ~0.001). Don't spend $ on more epochs.
- **Crypt solver brute-force backtracking in Python is too slow** (no-fit problems exhaust). Use **Z3** (available, 4.16.0) for any cipher CSP — solves in ms.
- **lm_head**: our memory says vLLM eval errored on it (v3 submission ERROR). huikang TRAINS with lm_head in target_modules but may filter at submission. Unverified whether eval supports it now.
- Corpus build (src/corpus.py) already filters: wrong-answer rows (metric_correct), brace answers, completions > 7680 tokens. So bad CoT does NOT pollute training.
- Public 0.86 adapter (assiabenazzouz/adappter-v32-epoch-5) is **403-forbidden** to our Kaggle account (needs website dataset-consent). Can't download/submit from CLI.
- Tinker budget ~$10 left; 2ep retrain ~$20 (needs user approval — NEVER auto-spend).

## Open Questions (the research program)

- **H1 (running):** Does crypt fit a clean generator WITH answers (Z3, full model space)? If high → mass-generate synthetic data to teach the cipher-deduction prior → model may beat the 13% solver ceiling. If low → crypt is intrinsically capped.
- **H2:** Brace-cap workaround — does the real grader have a fallback extraction? Restructure crypt answers to dodge `\boxed{}` truncation (testable with 1 free submission). Potential to recover up to the solved∩braced subset.
- **H3:** Synthetic cipher-deduction corpus (teach the META-skill) — generate problems with KNOWN ciphers + clean deduction CoT; does training on these raise crypt at greedy? (paid validation).
- **H4:** Recipe deltas (lm_head if eval-supported, MoE-tie) — close the 0.84→0.85-0.86 gap vs public recipe (paid validation).
- **H5:** R levers — augmenter sub-skill drills for long-CoT cats (paid validation).
