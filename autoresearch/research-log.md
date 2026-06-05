# Research Log — nemotron-reasoning-087

## 2026-06-05 — Bootstrap + H1

**Init.** Goal: break 0.87 (chase 0.89). Baseline LB 0.84, offline oracle 89.0% (committed f083ffe). Only lever = cryptarithm (13.3%) + R (0.958). Public field stuck 0.85-0.86 → crypt is the universal wall (confirmed via huikang/mirzayasir public notebooks; recipe ≈ ours).

**H1 — crypt generator reverse-engineering (Z3).** Tested SAT of per-problem cipher+op+base+reading WITH known answers.
- Non-injective: 100% fit (50/50) — DEGENERATE (loose model, meaningless).
- Injective: ~2% fit in bases 11-15.
- Predict-from-examples (the real metric, = agentB solver): ~13%.
- **Conclusion: ciphers are non-injective + underdetermined; crypt cannot be SOLVED reliably (~13-16% cap). Signal: base~11, add-dominant whole-number.**
- Decision: PIVOT crypt effort from "solve" to H3 "teach the meta-skill via synthetic data."

**Next:** H3 (build synthetic cipher-deduction corpus, free offline) + H2 (brace-cap workaround, 1 free submission) + H4 recipe deltas (queued, paid). Paid experiments require user approval — never auto-spend.
