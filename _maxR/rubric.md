The repo invariants confirm all three derivations are consistent with the implemented pipeline (self-consistent box via `extract_answer(cot)`, `GEN_LIMIT=7680` drop-not-truncate, brace-answer drop, augmenter sub-skill drills). I have everything I need to synthesize. No further exploration is required — the task is a synthesis, not a code change.

Here is the authoritative, deduplicated, ranked R-rubric.

---

# THE R-RUBRIC — Definitive checklist for a maximally greedy-reproducible CoT

**One-line model.** A LoRA (rank≤32, frozen base) memorizes `p(token | visible_prefix)` on completion tokens. Greedy temp=0 emits `argmax` at every position with NO recovery, so reproduction of an L-token trace ≈ `Π_t P(argmax_t = gold_t)`. R is therefore dominated by its single highest-residual-entropy token. **Every design rule below exists to drive that worst token's entropy to ~0 by making its value a deterministic function of tokens already printed in the same trace.**

---

## MECHANISMS (why the rules are true)

**M1 — What is trained vs. what is decoded.** SFT memorizes conditionals, not algorithms (repo converges to NLL≈3e-4 by epoch 1; 2ep==3ep). Greedy autoregression feeds its own output back as prefix, so one flipped token shifts the prefix and can cascade. R = product of per-token "stays-on-rail" events; it is dominated by the worst (highest-entropy) token, not the average.

**M2 — The local↔global split (THE CENTRAL LAW).** Three token regimes by residual entropy `H_t`:
- **(1) COPY** — gold is literally present earlier in the trace (restated input, transposed column, the boxed digits). The base induction/copy head fires; `H_t≈0` after light training; generalizes to ANY x because it reads the prefix, not x. Nearly free AND it seeds the local context for the next computed token.
- **(2) LOCAL-DERIVE** — gold is a fixed low-arity function of ≤2–3 literals within ~1–2 lines (`OR(0,NOT(0))=1`, one long-division digit). Memorizes to ≈0 entropy because the identical surface pattern recurs thousands of times.
- **(3) GLOBAL-INFER** — gold encodes a per-item rule the model must privately infer from x in one step (`the rule is XNOR(shift1,shift7)`). Across training, near-identical prefixes preceded MANY different gold spans ⇒ the memorized conditional is multimodal ⇒ greedy picks the train-modal (wrong) branch on a new item, and every downstream copy faithfully propagates the wrong commit. **This is the measured R-collapse 0.95→0.2, and it is independent of formatting/terseness** (paid: terse run-007 ≈ scaffolded run-008 ≈ 0.72).

**M3 — Why anchors raise R.** A consume step is `p(result | operands)`. If the operand sits hundreds of tokens up (or only in the masked, position-varying prompt), attention must do long-range retrieval and greedy mis-binds. Restating it 1–5 lines above converts a regime-(2/3) recompute into a regime-(1) copy. This is exactly the paid fact: **slimming restatements DROPPED the score; anchors HELP R.**

**M4 — Length is double-edged.** Copy tokens have P=1−ε; over thousands of tokens Π(1−ε) still erodes, and exceeding the 7680 generation cap means the box is never emitted ⇒ R=0 for that item (corpus.py DROPS, never truncates). So: minimize infer tokens, make the rest short-span copies, finish under the cap with margin — but never shorten by deleting anchors or merging micro-steps.

**M5 — Curriculum is a multiplier, not a cure.** min-logprob reweighting (`_branch_weight = min(1, |logprob|/branch_logprob)`) concentrates gradient on thin-margin tokens. It sharpens regime-(2) local slots (paid: 0.79→0.84) but cannot grant generalization to a regime-(3) token — it only memorizes the per-item rule harder (train NLL→0, test argmax still flips). Fix locality FIRST; then let curriculum sharpen the residual local slots over the ≤2 epochs you pay for.

**M6 — Offline R-proxy.** R is unmeasurable without a $33 run, and oracle (offline solve-rate) is NOT a proxy (oracle↑ has repeatedly meant R↓). The ONLY offline safeguard is the structural audit: (a) grep for global-rule assertions, (b) build a prefix map over corpus completion tokens and flag any shared prefix that diverges on a derived (non-copied) token. Minimize that flag count before paying.

---

## THE RANKED RUBRIC (apply in order; #1 is non-negotiable)

**R1 — LOCALITY LAW (master rule).** Every committing/derived token must be a deterministic function ONLY of tokens already printed verbatim earlier in the SAME completion — never of a per-item global rule inferred in-weights from x. **Forbid any line that asserts a generalization ("the rule/pattern/formula/operation/shift is X") that downstream lines consume.** Replace each with N explicit per-element local steps that re-read operands from a nearby anchor.
- *Why:* the exact mechanism of the 0.95→0.2 paid collapse (M2). #1 R-killer.
- *Apply:* token-walk the trace; for each NEW non-punctuation fact ask "is its value printed earlier in this completion?" If NO → infer token → either print its determining evidence immediately above, or expand into per-element commits. Compliant pattern: `i {expr} = {OP}({a},{b}) = {result}` with `{a},{b}` copied from the just-restated Input block, `{result}` a 1-bit lookup.

**R2 — ANCHOR / RESTATE-BEFORE-USE.** Immediately before any block that consumes a value, re-emit that value verbatim on its own line and consume it from there. Keep all per-element scaffold/echo rows (the `Input` block) even when they look redundant. Never optimize away a restatement to save tokens.
- *Why:* paid slimming regression; converts long-range retrieval into local copy (M3).
- *Apply:* for every operand in a compute line, confirm its nearest prior occurrence is within a few lines; if not, insert a verbatim restatement.

**R3 — ONE DETERMINISTIC MICRO-STEP PER LINE, FIXED TEMPLATE.** Emit exactly one element/bit/digit/column per line in a rigid repeated template (`<index> <expr> = <op>(<a>,<b>) = <result>`), one fact per line, full range 0..N−1 (no skipping). No prose inside the compute region; no line whose token length varies with reasoning depth. One helper formats every line byte-identically (separators/whitespace are behavioral, not cosmetic).
- *Why:* makes most of L tokens schema-locked (P≈1) so R is dominated by the few content slots; position becomes a deterministic key with no competitor; localizes any error to one line (M1).
- *Apply:* express the algorithm as a flat loop printing one fixed-format line per atomic unit; split any line that emits >1 derived digit/bit. Atomic arithmetic only — single-digit / single-bit ops; expand multi-digit math into long-mult/long-div one-digit-per-line.

**R4 — MECHANICAL, SPELLED-OUT SELECTION (deterministic generator).** Every argmax/best/winner/tie-break must be written as explicit pairwise comparisons of already-printed numbers, with a FIXED tie-break direction (e.g. always lowest index, always left-first), never a bare "Best: X". The generator must be byte-deterministic: no RNG, no unsorted set/dict iteration, no float== , no "pick the prettiest"; fixed section order and sort keys; two runs on identical input must be byte-identical.
- *Why:* a bare selection collapses a whole comparison's entropy into one regime-(3) token; hidden nondeterminism/unstated tie-breaks put two gold continuations after one prefix, splitting `p_theta` (M2).
- *Apply:* print the per-candidate counts immediately above, then the verdict (`Left longest: N / Right longest: M / winner: X`); assert determinism in a test.

**R5 — STABLE SCAFFOLD / FORMAT-IDENTITY (one category = one style).** Identical section headers, ordering, phrasing, decimal counts, delimiters, chat template, think delimiters, and box format across EVERY trace and augmenter in a category. No optional block that sometimes appears after the same context (guard it with an always-present literal cue). Never mix two CoT styles for one category; treat any separator/whitespace change as behavioral.
- *Why:* constant scaffold tokens get P≈1 (curriculum drives their weight ~0), spending margin budget on content; variable scaffolding creates spurious infer tokens and dilutes the rank-32 budget (consistent with the net-negative slim+aug+replay "dilution") (M1, M5).
- *Apply:* pin the completion template once (`{cot}\n</think>\n\boxed{{{ans}}}<|im_end|>`); diff several traces — only literal slot values may differ.

**R6 — SELF-CONSISTENT TERMINAL BOX.** `\boxed{}` must be a pure character-for-character copy of the per-element results just assembled on a visible line directly above it, in printed order. Never box ground truth if it differs from what the trace computed; never reformat between the last compute line and the box (leading zeros/truncation identical). Drop answers containing `{`/`}`.
- *Why:* boxing an unsupported value teaches a jump the model can't reproduce; the rel-tol(1e-2)-OR-exact grader accepts the self-consistent value anyway (M1). corpus.py already does `final_answer = extract_answer(cot)` — keep it.
- *Apply:* assemble the answer incrementally on visible lines, box that exact join; confirm each boxed char appeared verbatim above.

**R7 — LENGTH × FRAGILITY BUDGET (under the cap, minimal chaff).** Keep each category's worst-case completion comfortably under 7680 tokens (measure with the real tokenizer). Shorten ONLY by cutting genuinely never-re-read derived chaff (e.g. the dropped per-row bitsum hash); NEVER by removing anchors (R2) or merging micro-steps (R3). Redesign for length — never truncate (a stranded trace scores 0).
- *Why:* truncation ⇒ R=0; every non-load-bearing token only multiplies collapse risk (M4). corpus.py drops >GEN_LIMIT — keep that.
- *Apply:* trace data-flow; a line is removable only if no later line reads it AND it is not an anchor for a consumed value.

**R8 — DRILL EVERY MECHANICAL PRIMITIVE + TRAIN THE FULL PROCEDURE DISTRIBUTION.** For each atomic copy/transcription primitive the long trace relies on (char-by-char copy, 8-bit column copy, column matching, run extrapolation, concat/split, lstrip), ship an empty-`<think>` augmenter that drills exactly that primitive in the IDENTICAL surface format (matching.py extracts real sections from `reasoning/*.txt` to guarantee format identity). AND keep every faithfully-executed trace — including hard-tail / wrong-but-self-consistent ones — never survivorship-filter to only grader-correct outputs.
- *Why:* drills raise the per-token transcription reliability the `R^L` product depends on; including rare layouts widens argmax margins on exactly the prefixes greedy would otherwise drift on (M1).
- *Apply:* augmenters use `enable_thinking=False` and the same chat template; validate each emits rows (`mod_used>0`). Exclude a trace ONLY if it physically can't emit a valid box (over cap, or brace in answer).

**R9 — OFFLINE LOCALITY AUDIT BEFORE EVERY PAID RUN (single-variable steps).** Before any $33 run, (a) grep every reasoner's sample traces for global-rule assertions ("the rule is", "pattern is", "secret", a chosen word/shift/family) and verify each consumed value is printed earlier; (b) build a prefix/n-gram map over corpus completion tokens and flag shared prefixes that diverge on a derived token — treat the flag count as the offline R-risk score to minimize. Change exactly ONE structural variable per paid run; default to the legacy-local style.
- *Why:* oracle is not an R proxy; this structural audit is the only offline safeguard (M6).

---

## ANTIPATTERNS (each is a direct violation; listed by severity)

1. **GLOBAL-RULE ASSERTION THEN USE** ("the rule is XNOR(shift1,shift7)", "the cipher shift is +N", "the formula is k=…") — the #1 R-killer (0.95→0.2); model can't infer the right per-item rule, greedy emits the train-modal one, downstream cascades. (violates R1)
2. **TERSE / CLEVER ONE-SHOT** jump from inputs straight to `\boxed{}` (or merging micro-steps) — removes the copy scaffold; higher oracle, collapsed R: the oracle↑/R↓/LB↓ trap. (R1, R3)
3. **SLIMMING / DEDUPING ANCHORS & RESTATEMENTS** to save tokens — measured to DROP score; lengthens retrieval span, splits conditionals. (R2)
4. **BARE SELECTION** ("Best: AND47", "Left longest: 3") when the maxed counts aren't all printed just above — collapses an argmax into one high-entropy token. (R4)
5. **NONDETERMINISTIC / AMBIGUOUS-TIE-BREAK GENERATOR** (RNG, unsorted set/dict, float==, "pick the prettiest", direction-ambiguous ties) — two gold continuations after one prefix. (R4)
6. **MULTI-DIGIT / MULTI-BIT or VECTORIZED RESULT IN ONE TOKEN** (full product, full 8-bit output, parallel AND/OR/XOR) — bundles independent low-entropy events into one fragile joint argmax. (R3)
7. **PER-ITEM STRUCTURAL VARIATION / STYLE MIXING / SILENT SEPARATOR CHANGES** — multimodal continuations after identical prefixes; dilutes the rank-32 budget. (R5)
8. **MULTI-FACT PROSE SENTENCES** bundling several derived values — position no longer keys the next token; near-tie argmaxes. (R3)
9. **BOXING GROUND TRUTH** instead of the trace's own computed value — teaches an unsupported jump; greedy boxes what it actually computed → mismatch. (R6)
10. **OVER-LONG / TRUNCATED TRACES (> 7680)** — box never emitted → R=0; never truncate, redesign. (R7)
11. **SURVIVORSHIP FILTERING to only grader-correct traces** — strips rare hard-tail layouts → greedy drifts at test. (R8)
12. **ANSWERS CONTAINING `{` / `}`** — break `\boxed{}` extraction; a silent R=0 sink. (R6)
13. **RELYING ON CURRICULUM / EXTRA EPOCHS / lm_head / MORE DATA VOLUME to fix a non-local CoT** — curriculum only sharpens local tokens; lm_head measured no-op (0.71==0.71); epoch 3 buys nothing (2ep==2ep saturates); volume without R does nothing (ceiling ~0.84 was R-bound). Fix locality first. (R1, R5)
14. **USING OFFLINE ORACLE AS R EVIDENCE** — oracle proves the solver, not greedy reproduction on unseen x; audit via the context-divergent-gold proxy instead. (R9)

---

**Bottom line:** R1 (locality) is the entire game — it alone explains the only measured collapse. R2–R3 (anchor + one-fact-per-line) are the proven score-movers (slimming dropped it). R4–R6 close the remaining entropy leaks (selection, scaffold, box). R7–R9 are the guardrails (cap, drills, offline audit). Apply R1 first and never trade it for a higher offline oracle.