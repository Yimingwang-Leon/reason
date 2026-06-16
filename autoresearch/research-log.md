# Research Log — nemotron-reasoning-087

Bootstrap state (2026-06-05/06): baseline LB 0.84, offline oracle 89.0%. Distilled to current-state + open-leads only; day-by-day session narration and unproven "ceiling/concluded/give-up" verdicts stripped.

## Proven-infeasible (paid / measured)
- **lm_head LoRA**: file-upload submission = 400 reject (only diff vs working adapter was `lm_head` in target_modules); notebook route accepts it but eval is a measured NO-OP (run-006 with==without==0.71). Not a lever. (See also `project_lora_target_modules.md` — eval-supported targets only.)

## Open leads — crypt (BIGGEST lever, ~17.6% solved now, structural not solved)
Crypt is the largest single point sink — keep investigating; do NOT treat as closed.
- **Structure (H1, Z3):** ciphers are NON-injective symbol→digit maps; per-problem cipher+op+base+reading is underdetermined under a single global map. Signal: base ~11, add-dominant, whole-number results.
- **Column-wise arithmetic**, **unknown base**, **zero-column omission** are the unmodeled structural pieces to crack.
- **~76% format cap** from `}`-truncation in answers (brace/`\boxed{}` boundary) — a structural ceiling on extractable crypt, not a reason to abandon.

## Closed / negligible
- **H2 brace-cap workaround:** even greedy `\boxed{}` extraction nets +3 crypt problems (≈+0.16pp). Not worth a submission. The interesting residue is the ~76% truncation cap above.

## Notes
- Recipe/eval-supported LoRA targets: canonical in `project_lora_target_modules.md` (dup removed here).
- Field intel: public notebooks (~0.85-0.86) use a recipe ≈ ours; distillation route caps ~0.73 (community intel, not our experiment).
