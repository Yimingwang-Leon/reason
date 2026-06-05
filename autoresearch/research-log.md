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

## 2026-06-05 — tick 2: H2 + corpus de-risk
**H2 (brace-cap).** Official demo doesn't expose grader. Brace-cap untestable for free; low value (~0.3% total). Open unknown: grader `\boxed{}` regex truncating vs greedy — if greedy, we slightly undercount crypt locally. Not worth a paid submission. Parked.
**Recipe note (H4):** official demo LoRA targets = `(in_proj|out_proj|up_proj|down_proj)` only — confirms these are eval-supported; no lm_head/attention in the minimal demo. Our 0.84 used q/k/v/o+in/out/up/down (also fine).
**Action:** running full corpus build to confirm H3 cryptarithm_synth integrates cleanly + get final corpus stats for run-006 (de-risks the paid shot).

## 2026-06-05 — tick 2 cont: corpus de-risk OK; run-006 NOT recommended
Corpus build OK: 9476 entries; H3 cryptarithm_synth = 300 rows integrated cleanly; all gates pass. Corpus ready.
**Decision (user rule "guarantee 0.86 or don't spend"):** run-006 projects only ~0.853 floor (0.89 oracle × 0.958 R); cannot guarantee 0.86 (R unmeasurable offline; H3 is an unvalidated bet). => DO NOT spend $20. Guaranteed 0.86 = free public notebook (user forks+submits on Kaggle). Reserve paid retrain until an offline projection clearly exceeds 0.86.

## 2026-06-05 — tick 3: H2 closed, research CONCLUDED
H2 brace-cap: even greedy `\boxed{}` extraction gains only +3 crypt problems (13.3->15.2%) = +0.16pp overall. Negligible. CLOSED.
**CONCLUSION:** Free offline research exhausted. Honest ceiling for our pipeline: ~0.85-0.86 (paid retrain, NOT guaranteed 0.86). Guaranteed 0.86 = free public notebook. 0.89 unreachable (crypt underdetermined, H1). Remaining hypotheses (H3 transfer, H4 deltas, H5 R) are unvalidatable without a paid retrain the user won't fund without a 0.86 guarantee. Stopping the cron loop — no productive free work remains. Restart only with a NEW direction or budget to spend.

## 2026-06-05 — tick 4: H4 — lm_head is the missing delta (REOPENED)
Downloaded a LIVE public 0.86 adapter (leegongman/0-86-adapter = kienngx tinker-adapter). Its target_modules INCLUDE **lm_head** (rank 32) and it scores 0.86 -> **lm_head IS eval-supported** (our memory's "vLLM errors on lm_head" is OUTDATED). Our pipeline sets train_unembed=False (skips lm_head to save $) — train_tinker.py:58 even admits "vLLM CAN accept lm_head LoRA". So **lm_head is a concrete recipe delta plausibly explaining our 0.84 vs field 0.86.**
**run-006 thesis (strongest shot at 0.87):** STACK our advantages (min-logprob curriculum that got 0.79->0.84, 89% oracle reasoners, H3 crypt augmenter) WITH the field's lm_head (--train-unembed). This combines levers the public 0.86 recipe likely lacks (curriculum) -> could EXCEED 0.86 toward 0.87. Build_submission must be verified to emit lm_head LoRA correctly (pre-run check).

## 2026-06-06 — lm_head submission test (DEFINITIVE)
Isolated test: run-005 adapter (no lm_head) submits fine via file-upload (0.84). run-005 + ZERO lm_head LoRA (only difference = 'lm_head' in target_modules) -> 400 Bad Request on CreateSubmission. CONCLUSION: Kaggle FILE-UPLOAD rejects lm_head adapters; NOTEBOOK route accepts them (public 0.86 has lm_head). So: run-006 with lm_head MUST be submitted via notebook (manual). Without lm_head -> file-upload works (our pipeline). Path A (bit_manip-98%, no lm_head, auto file-upload, ~0.87) preferred; Path B (+lm_head, manual notebook, ~0.87-0.88) optional upside.
