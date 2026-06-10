"""Tokenize prompt + CoT into (tokens, mask) jsonl for SFT.

mask=0 on prompt tokens (no loss), mask=1 on completion tokens (loss only here).
Skips problems without a successful CoT in reasoning/<id>.txt.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from tokenizers import Tokenizer  # type: ignore[import-untyped]
from transformers import AutoTokenizer  # type: ignore[import-untyped]

from src.problems import load_problems
from src.reasoning import extract_answer, metric_correct

ROOT = Path(__file__).resolve().parent.parent
TOKENIZER_PATH = ROOT / "src" / "tokenizer.json"
REASONING_DIR = ROOT / "reasoning"
CORPUS_DIR = ROOT / "corpus"
CORPUS_INDEX = ROOT / "corpus.jsonl"
BASE_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

# run-012: data/holdout.csv (1899 ids) is a CLEAN R-measurement surface. Main
# rows are restricted to train_split.csv (7601 ids) and any drill DERIVED from a
# holdout problem is excluded (leakage). Champion 'matching' drills are the only
# problem-tied drills (sections extracted from the champion's reasoning/<id>.txt);
# their opaque sha256("matching_{i}")[:8] ids were mapped back to source problem
# ids by re-running the champion's deterministic extraction+downsample
# (verified: 4515/4515 ids reproduce the augmentations dir exactly). The other
# drill kinds (splitting/concatenation/lstrip: seeded random symbols; spelling:
# tokenizer vocab; equation_*: synthetic generators) are NOT problem-tied.
TRAIN_SPLIT_CSV = ROOT / "data" / "train_split.csv"
HOLDOUT_CSV = ROOT / "data" / "holdout.csv"
DRILL_SOURCES_JSON = ROOT / "data" / "champion_drill_sources.json"


def _load_split_ids() -> tuple[set[str], set[str]]:
    import csv

    with open(TRAIN_SPLIT_CSV) as f:
        train_ids = {r["id"] for r in csv.DictReader(f)}
    with open(HOLDOUT_CSV) as f:
        holdout_ids = {r["id"] for r in csv.DictReader(f)}
    assert train_ids and holdout_ids and not (train_ids & holdout_ids), (
        f"train_split/holdout must be non-empty and disjoint "
        f"(got {len(train_ids)}/{len(holdout_ids)}, "
        f"overlap {len(train_ids & holdout_ids)})")
    return train_ids, holdout_ids

TOKEN_LIMIT = 8192   # max_model_len (prompt + completion)
GEN_LIMIT = 7680     # grader's GENERATION cap; completion must fit within this
PROMPT_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)

# Which augmenter categories actually get baked into the SFT corpus. The pipeline
# can consume every augmenter (orphaning bug fixed), but we only INCLUDE ones whose
# relevance to OUR reasoners is validated. The champion-copied augmenters
# (concatenation/splitting/spelling/lstrip/matching) target the champion's CoT
# formats / the abandoned cryptarithm skill and are wired-but-excluded until a paid
# run or the offline R-harness shows they help.
AUGMENTER_ALLOWLIST = {
    # run-011: champion sub-skill drills (champion_import.py: matching/splitting/
    # concatenation/spelling/lstrip) now tokenized in the SAME enable_thinking=True
    # regime as everything else (the run-009 enable_thinking=False mismatch is
    # fixed), plus our own equation drills (present in run-005 @ 0.84).
    "matching",
    "splitting",
    "concatenation",
    "spelling",
    "lstrip",
    "equation_reversal",
    "equation_arith",
    "equation_digitwise",
}

# Categories that must come ONLY from champion_import.py (pre-built drill files);
# the legacy standalone generators for the same categories are skipped to avoid
# double-injection (they emit identical category names with different content).
CHAMPION_PROVIDED = {"matching", "splitting", "concatenation", "spelling", "lstrip"}


def tokenize_prompt(prompt: str, chat_tok) -> list[int]:
    messages = [{"role": "user", "content": prompt + PROMPT_SUFFIX}]
    result = chat_tok.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=True,
    )
    # New transformers wraps in BatchEncoding when return_dict default-ish;
    # accept both list[int] and tokenizer encoding-like.
    if hasattr(result, "input_ids"):
        return list(result["input_ids"])
    return list(result)


def tokenize_augmenter_prompt(prompt: str, chat_tok) -> list[int]:
    """Augmenter sub-skill drills, tokenized in the SAME regime as inference
    (enable_thinking=True -> prompt ends "<think>\\n"). The drill work lives inside
    the think block; the completion closes it with "\\n</think>" (see main()).
    The previous enable_thinking=False ("<think></think>") regime NEVER occurs at
    inference -> 35-45% of trainable tokens trained an impossible regime (train/test
    mismatch, fixed for run-011)."""
    messages = [{"role": "user", "content": prompt}]
    result = chat_tok.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=True,
    )
    if hasattr(result, "input_ids"):
        return list(result["input_ids"])
    return list(result)


def _load_augmenters() -> list:
    """Import every augmenter module exposing generate(). Wiring these is what
    makes the 'reasoner 必带 augmenter' rule actually reach SFT — before this they
    were orphaned (nothing in the pipeline called generate())."""
    import importlib
    import pkgutil

    from src import augmenters as aug_pkg

    mods = []
    for info in pkgutil.iter_modules(aug_pkg.__path__):
        m = importlib.import_module(f"src.augmenters.{info.name}")
        if hasattr(m, "generate"):
            mods.append(m)
    return mods


MATH_REPLAY_PATH = ROOT / "data" / "replay" / "nemotron_math_1gb.jsonl"
MATH_REPLAY_TARGET_TOKENS = 2_000_000
MATH_REPLAY_COMP_LIMIT = 7400   # tighter than GEN_LIMIT: replay rows are cheap, skip long



def _balanced_boxed(text: str) -> str | None:
    """Last \\boxed{...} content with BALANCED braces (for replay rendering only).
    extract_answer truncates at the first '}' (grader-isomorphic), which mangles
    LaTeX like \\boxed{\\frac{1}{2}} into a fragment; replay boxes are never
    graded, so teach a well-formed box instead. None if unbalanced/absent."""
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j = i + len("\\boxed{")
    depth = 1
    out = []
    while j < len(text):
        c = text[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                ans = "".join(out).strip()
                return ans or None
        out.append(c)
        j += 1
    return None

def add_math_replay(chat_tok, tok) -> tuple[int, int]:
    """Append a general-math reasoning REPLAY mix (curb-forgetting regularizer; the
    public 0.85->0.86 lever, mohamedamr992), rendered in the SAME regime as every
    main trace (run-012 fix; the old renderer trained raw chat turns with NO think
    block, a regime that never occurs at inference):

        prompt     = chat_template(user msgs, add_generation_prompt=True,
                                   enable_thinking=True)        # ends '<think>\\n'
        completion = C + '\\n</think>\\n\\boxed{A}<|im_end|>'

    where C is the last assistant message content (reasoning, lives inside the
    think block) and A = last brace-BALANCED \\boxed{...} in C (skip if absent; replay boxes are never graded, so well-formedness beats grader-isomorphism here).
    Completions are encoded with the raw tokenizer (add_special_tokens=False)
    exactly like main traces; rows with completion > MATH_REPLAY_COMP_LIMIT or
    total > TOKEN_LIMIT are SKIPPED, never truncated. Accumulates until
    MATH_REPLAY_TARGET_TOKENS unmasked completion tokens. Category 'math_replay'.
    Gated by env ADD_MATH_REPLAY=1. Returns (examples, unmasked_tokens)."""
    if not MATH_REPLAY_PATH.exists():
        raise FileNotFoundError(
            f"[math_replay] ADD_MATH_REPLAY=1 but {MATH_REPLAY_PATH} missing - "
            "refusing to silently build a no-replay corpus (silent-skew trap)")

    kept = tot = 0
    n_no_ans = n_long_comp = n_long_total = 0
    with open(CORPUS_INDEX, "a") as idx_f, open(MATH_REPLAY_PATH) as fin:
        for line in fin:
            try:
                row = json.loads(line)
            except Exception:
                continue
            msgs = row.get("messages")
            if not msgs or len(msgs) < 2 or msgs[-1].get("role") != "assistant":
                continue
            content = str(msgs[-1].get("content", "")).rstrip("\n")
            ans = _balanced_boxed(content)
            # No box -> nothing to supervise; '}' in the answer can never round-trip
            # through \boxed{...} extraction (same rule as main traces).
            if not ans or "}" in ans:
                n_no_ans += 1
                continue
            completion = f"{content}\n</think>\n\\boxed{{{ans}}}<|im_end|>"
            comp_ids = tok.encode(completion, add_special_tokens=False).ids
            if len(comp_ids) > MATH_REPLAY_COMP_LIMIT:
                n_long_comp += 1
                continue
            r = chat_tok.apply_chat_template(
                msgs[:-1], tokenize=True, add_generation_prompt=True,
                enable_thinking=True,
            )
            prompt_ids = list(r["input_ids"]) if hasattr(r, "input_ids") else list(r)
            tokens = prompt_ids + comp_ids
            # Hard gates (skip, never truncate): a truncated row strands the box.
            assert len(comp_ids) <= MATH_REPLAY_COMP_LIMIT
            if len(tokens) > TOKEN_LIMIT:
                n_long_total += 1
                continue
            mask = [0] * len(prompt_ids) + [1] * len(comp_ids)
            unmasked = sum(mask)
            assert unmasked == len(comp_ids) > 0
            rid = f"replay_{kept:06d}"
            d = CORPUS_DIR / rid
            d.mkdir(parents=True, exist_ok=True)
            with open(d / "synthetic.jsonl", "w") as f:
                json.dump({"tokens": tokens, "mask": mask}, f)
                f.write("\n")
            idx_f.write(json.dumps({
                "problem_id": rid, "category": "math_replay",
                "token_count": len(tokens), "unmasked_token_count": unmasked,
                "answer": ans, "augmenter": True,
            }) + "\n")
            kept += 1
            tot += unmasked
            if tot >= MATH_REPLAY_TARGET_TOKENS:
                break
    print(f"[math_replay] appended {kept} examples, {tot:,} trainable tokens "
          f"| skipped: {n_no_ans} no/brace answer, {n_long_comp} completion>"
          f"{MATH_REPLAY_COMP_LIMIT}, {n_long_total} total>{TOKEN_LIMIT}")
    return kept, tot


def main() -> None:
    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    chat_tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    for old in CORPUS_DIR.glob("*"):
        if old.is_dir():
            import shutil
            shutil.rmtree(old)

    problems = load_problems()
    train_ids, holdout_ids = _load_split_ids()
    # Champion drill -> source problem id map (matching drills only; see header).
    with open(DRILL_SOURCES_JSON) as f:
        drill_sources: dict[str, str] = json.load(f)
    holdout_drill_ids = {d for d, src in drill_sources.items() if src in holdout_ids}

    n_used = n_skip = n_trunc = total_unmasked = 0
    n_overlong = n_brace_skip = 0
    n_holdout_main = n_holdout_drill = 0   # run-012 leakage exclusions
    n_self_consistent = 0   # traces whose own answer != exact truth but is grader-correct
    exact_box_mismatch: list[str] = []
    cat_counts: dict[str, int] = {}
    cat_tokens: dict[str, int] = {}

    with open(CORPUS_INDEX, "w") as idx_f:
        for p in problems:
            # run-012: train on train_split ONLY; holdout is the clean R surface.
            if p.id not in train_ids:
                assert p.id in holdout_ids, f"{p.id} in neither split"
                n_holdout_main += 1
                continue
            cot_path = REASONING_DIR / f"{p.id}.txt"
            if not cot_path.exists():
                n_skip += 1
                continue
            cot = cot_path.read_text().rstrip("\n")
            # Self-consistent supervision (no box rewrite). The grader accepts a
            # prediction that is string-exact OR within rel-tol 1e-2 (confirmed on the
            # competition Evaluation page), which is exactly the metric_correct filter
            # reasoning.py already applied — so the reasoner's OWN final answer is
            # grader-correct. Using it as the final box (instead of forcing exact truth)
            # makes the whole completion self-consistent: the CoT reasons to V and boxes
            # V, removing the CoT<->answer contradiction that previously hit ~40% of
            # traces (100% gravity/unit) and is the prime suspect for the R<1 gap.
            final_answer = extract_answer(cot)
            # huikang-faithful (2026-06-09): box the CoT's OWN extracted answer even
            # when it is NOT grader-correct (the 238 hard-tail bit_manip + cross-cat
            # traces the procedure got wrong but reasoned faithfully). Training the
            # full procedure distribution teaches deterministic execution -> higher R.
            # Was: drop non-correct (survivorship bias that stripped the hard tail).
            # exact_box_mismatch now counts the wrong-but-included traces (report only).
            if not metric_correct(p.answer, final_answer):
                exact_box_mismatch.append(p.id)
            elif final_answer != p.answer:
                n_self_consistent += 1   # grader-correct but not string-exact to truth
            # Drop only answers containing '}': \boxed{...} extraction truncates at
            # the first '}', so those can never be carried correctly. A '{'-only
            # answer round-trips through both our extractor and the grader's
            # (verified offline), so keep it (+15 crypt traces).
            if "}" in final_answer:
                n_brace_skip += 1
                continue
            completion = f"{cot}\n</think>\n\\boxed{{{final_answer}}}<|im_end|>"
            comp_ids = tok.encode(completion, add_special_tokens=False).ids
            # Generation budget gate: the grader caps GENERATION at GEN_LIMIT tokens.
            # A completion longer than that can never emit its final box at inference
            # (the model gets cut off mid-trace and scores 0), so it is a useless,
            # actively-harmful training target — skip it, do NOT truncate.
            if len(comp_ids) > GEN_LIMIT:
                n_overlong += 1
                continue
            prompt_ids = tokenize_prompt(p.prompt, chat_tok)
            tokens = prompt_ids + comp_ids
            mask = [0] * len(prompt_ids) + [1] * len(comp_ids)
            if len(tokens) > TOKEN_LIMIT:
                # prompt+completion over the model window: skip (truncating would also
                # strand the box). Rare; completion already <= GEN_LIMIT here.
                n_skip += 1
                continue
            unmasked = sum(mask)
            if unmasked == 0:
                n_skip += 1
                continue

            problem_dir = CORPUS_DIR / p.id
            problem_dir.mkdir(parents=True, exist_ok=True)
            with open(problem_dir / "synthetic.jsonl", "w") as f:
                json.dump({"tokens": tokens, "mask": mask}, f)
                f.write("\n")
            idx_f.write(json.dumps({
                "problem_id": p.id, "category": p.category,
                "token_count": len(tokens), "unmasked_token_count": unmasked,
                "answer": p.answer,
            }) + "\n")
            n_used += 1
            total_unmasked += unmasked
            cat_counts[p.category] = cat_counts.get(p.category, 0) + 1
            cat_tokens[p.category] = cat_tokens.get(p.category, 0) + unmasked

        # ---- Augmenter sub-skill drills (previously orphaned; now wired in) ----
        aug_counts: dict[str, int] = {}
        aug_empty: list[str] = []
        aug_excluded: list[str] = []
        for mod in _load_augmenters():
            mod_name = mod.__name__.split(".")[-1]
            rows = mod.generate()
            if not rows:
                # Augmenter produced no data (e.g. matching.py parses the champion's
                # bit_manip CoT format, which our reasoner does not emit). Not a wiring
                # bug — record and skip rather than hard-fail the build.
                aug_empty.append(mod_name)
                continue
            if rows[0]["category"] not in AUGMENTER_ALLOWLIST:
                # Wired (orphaning fixed) but not baked in pending validation.
                aug_excluded.append(rows[0]["category"])
                continue
            if (rows[0]["category"] in CHAMPION_PROVIDED
                    and mod_name != "champion_import"):
                # run-011: these five categories are sourced EXCLUSIVELY from the
                # champion's pre-built drill files (champion_import.py, exact
                # 4515/1500/1500/648/300). The legacy standalone generators emit
                # the same category names — letting both in would double-inject.
                aug_excluded.append(f"{rows[0]['category']}({mod_name}:legacy-dup)")
                continue
            mod_used = 0
            for row in rows:
                rid = row["id"]
                # run-012: drop drills DERIVED from holdout problems (champion
                # matching sections are extracted from per-problem reasoning
                # traces -> training on them leaks holdout structure).
                if rid in holdout_drill_ids:
                    n_holdout_drill += 1
                    continue
                completion_text = row.get("completion", row.get("answer", "")).rstrip("\n")
                if not completion_text:
                    continue
                # Close the think block the enable_thinking=True prompt opened, so
                # the drill teaches the exact think-open/think-close control flow
                # used at inference.
                comp_ids = tok.encode(completion_text + "\n</think><|im_end|>",
                                      add_special_tokens=False).ids
                prompt_ids = tokenize_augmenter_prompt(row["prompt"], chat_tok)
                tokens = prompt_ids + comp_ids
                mask = [0] * len(prompt_ids) + [1] * len(comp_ids)
                if len(tokens) > TOKEN_LIMIT:
                    tokens = tokens[:TOKEN_LIMIT]
                    mask = mask[:TOKEN_LIMIT]
                    n_trunc += 1
                unmasked = sum(mask)
                if unmasked == 0:
                    continue
                problem_dir = CORPUS_DIR / rid
                problem_dir.mkdir(parents=True, exist_ok=True)
                with open(problem_dir / "synthetic.jsonl", "w") as f:
                    json.dump({"tokens": tokens, "mask": mask}, f)
                    f.write("\n")
                idx_f.write(json.dumps({
                    "problem_id": rid, "category": row["category"],
                    "token_count": len(tokens), "unmasked_token_count": unmasked,
                    "answer": "", "augmenter": True,
                }) + "\n")
                mod_used += 1
                n_used += 1
                total_unmasked += unmasked
                cat_counts[row["category"]] = cat_counts.get(row["category"], 0) + 1
                cat_tokens[row["category"]] = cat_tokens.get(row["category"], 0) + unmasked
            cat = rows[0]["category"]
            aug_counts[cat] = mod_used
            # generate() returned data but none survived tokenization => real wiring bug.
            assert mod_used > 0, f"augmenter {mod_name} dropped ALL {len(rows)} rows"

    # P1.5 gate relaxed (run-009): augmenters intentionally removed for the clean
    # lm_head corpus, so no augmenter is required to reach the corpus.
    required_aug: set[str] = set()
    missing = required_aug - set(aug_counts)
    assert not missing, f"required equation augmenters missing from corpus: {missing}"

    print(f"Corpus: {n_used} entries used | {n_skip} skipped | {n_trunc} truncated "
          f"| {n_overlong} dropped(completion>{GEN_LIMIT}) | {n_brace_skip} dropped(brace answer)")
    print(f"run-012 holdout exclusions: {n_holdout_main} main problems "
          f"(of {len(holdout_ids)} holdout ids), {n_holdout_drill} champion drills "
          f"(of {len(holdout_drill_ids)} holdout-tied)")
    print(f"Self-consistent boxes (own answer, grader-correct but != exact truth): "
          f"{n_self_consistent}")
    # huikang-faithful inclusion policy (2026-06-09): we INTENTIONALLY keep
    # wrong-but-self-consistent traces (the procedure ran faithfully and boxed its
    # own result) to train the full hard-tail procedure distribution -> higher R.
    # The CoT is internally consistent (reasons to V, boxes V); the box just isn't
    # grader-correct. This is no longer an error; report the count.
    print(f"Wrong-but-self-consistent traces INCLUDED (inclusion policy): "
          f"{len(exact_box_mismatch)}")
    print(f"Unmasked tokens: {total_unmasked:,}")
    for cat in sorted(cat_counts):
        tag = "  [augmenter]" if cat in aug_counts else ""
        print(f"  {cat:<25} {cat_counts[cat]:>5}  ({cat_tokens[cat]:>9,} tokens){tag}")
    if aug_excluded:
        print(f"Augmenters wired but excluded (pending validation): {', '.join(aug_excluded)}")
    if aug_empty:
        print(f"Augmenters with no applicable data (skipped): {', '.join(aug_empty)}")
    print(f"\nIndex: {CORPUS_INDEX}")
    print(f"Per-problem segments: {CORPUS_DIR}/<id>/synthetic.jsonl")

    import os
    if os.environ.get("ADD_MATH_REPLAY") == "1":
        add_math_replay(chat_tok, tok)


if __name__ == "__main__":
    main()
