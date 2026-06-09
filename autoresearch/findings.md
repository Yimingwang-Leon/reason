---
title: Findings — Nemotron Reasoning Challenge
type: report
status: active
updated: 2026-06-09
---

# Findings — Nemotron Reasoning Challenge

## Model that drives everything

**LB = oracle × R.**
- oracle = fraction of holdout (1899 problems) the reasoners can solve.
- R = how reliably the trained model reproduces the reasoner CoT at greedy decode (~0.96 measured).
- To raise LB you either raise oracle (crack a low-oracle category) or raise R (but R-tweaks are unmeasurable offline and need a paid retrain to validate).

Current best LB = 0.84 (run-005, 2ep). R ≈ 0.96.

## Per-category oracle + lever (canonical — see MEMORY.md grader/overview)

| cat | oracle | note |
|---|---|---|
| cipher / gravity / numeral / unit_conversion | 100% | deterministic-complete |
| bit_manipulation | 85% (94.4% after commit 71ce0a5 structural+simplicity-prior solver) | lever ≈ 2.5% of points |
| equation_numeric_deduce | 77% | some fails undecidable (query op absent from examples); lever ≈ 1.8% |
| cryptarithm_deduce | 17.6% | **BIGGEST LEVER** — loses ~7.1% of all points; format cap ~76% (}-truncation), NOT cracked yet |

(Earlier snapshots in this file quoted 13.3/88.4/80.1 — superseded; use the table above.)

## Proven-infeasible (each cost a real paid run — do NOT repeat)

- **bit_manip "structural / global-rule-assertion" CoT → R-collapse 0.95→0.2** (runs 006/007/008 = 0.71/0.73/0.72). The legacy column-wise CoT is the R local-optimum.
- **Beating run-005's 0.84 by redesigning CoT FORMAT is dead:** run-009 slimming = 0.82, run-010 locality-hardening = 0.82 (both directions hurt). 4 categories are already 100% oracle, so R-tweaking them is wasted effort.
- **lm_head LoRA (file-upload route) = measured eval no-op** (run-006 with == without == 0.71). lm_head is optional/no-op, not the missing delta.
- **Dropping MoE up/down LoRA → crippled 0.56.** MoE LoRA is essential.
- **2ep = 3ep = 0.84** — corpus is memorized by epoch 1 (nll ~0.001); more epochs don't help. (Flag: confirm this was a paid A/B before treating as load-bearing.)
- distillation route caps ~0.73 (community intel, not our experiment).

## Cryptarithm — OPEN LEADS (the biggest lever, NOT dead)

Current solver ~17.6%. Format cap ~76% (because ~24% of answers contain `}` and get truncated by `\boxed{}` extraction; corpus drops these rows). The remaining gap between 17.6% and ~76% is open territory, not a closed-form impossibility.

Structural findings (use as leads, not verdicts):
- Per-problem symbol→digit map; the map appears **non-injective** in some problems.
- **Column-wise arithmetic** reading (mod-base per column) coexists with whole-number base-B reading; BE/LE both occur.
- **Unknown base** (~10-15).
- OP at position 2 is a per-problem-deduced operator from {add, sub, rsub, absdiff, mul, concat, rconcat}; sign glyph prefixes negatives.
- **Zero-column omission** — short OUT (1-4 chars) suggests dropped/leading-zero columns.
- Use **Z3** (4.16.0 available) for any cipher CSP — solves in ms; Python brute-force backtracking is too slow.

Open directions (all need paid validation before trusting):
- Synthetic cipher-deduction corpus: generate problems with KNOWN ciphers + clean deduction CoT to teach the meta-skill; does it raise crypt at greedy?
- Brace-cap workaround: restructure crypt answers to dodge `\boxed{}` truncation; testable with 1 free submission (recovers up to the solved∩braced subset).

## Notes

- Corpus build (src/corpus.py) already filters wrong-answer rows, brace answers, and completions > 7680 tokens — bad CoT does not pollute training.
- Public 0.86 adapter (assiabenazzouz/adappter-v32-epoch-5) needs website dataset-consent; not downloadable from CLI. Public field recipe ≈ ours (rank32, ~3ep, min-logprob).
