"""Tokenize prompt + CoT into (tokens, mask) jsonl for SFT.

mask=0 on prompt tokens (no loss), mask=1 on completion tokens (loss only here).
Skips problems without a successful CoT in reasoning/<id>.txt.
"""
from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer  # type: ignore[import-untyped]
from transformers import AutoTokenizer  # type: ignore[import-untyped]

from src.problems import load_problems

ROOT = Path(__file__).resolve().parent.parent
TOKENIZER_PATH = ROOT / "src" / "tokenizer.json"
REASONING_DIR = ROOT / "reasoning"
CORPUS_DIR = ROOT / "corpus"
CORPUS_INDEX = ROOT / "corpus.jsonl"
BASE_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

TOKEN_LIMIT = 8192
PROMPT_SUFFIX = (
    "\nPlease put your final answer inside `\\boxed{}`. "
    "For example: `\\boxed{your answer}`"
)


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


def main() -> None:
    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    chat_tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    for old in CORPUS_DIR.glob("*"):
        if old.is_dir():
            import shutil
            shutil.rmtree(old)

    problems = load_problems()
    n_used = n_skip = n_trunc = total_unmasked = 0
    cat_counts: dict[str, int] = {}
    cat_tokens: dict[str, int] = {}

    with open(CORPUS_INDEX, "w") as idx_f:
        for p in problems:
            cot_path = REASONING_DIR / f"{p.id}.txt"
            if not cot_path.exists():
                n_skip += 1
                continue
            cot = cot_path.read_text().rstrip("\n")
            completion = f"{cot}\n</think>\n\\boxed{{{p.answer}}}<|im_end|>"
            comp_ids = tok.encode(completion, add_special_tokens=False).ids
            prompt_ids = tokenize_prompt(p.prompt, chat_tok)
            tokens = prompt_ids + comp_ids
            mask = [0] * len(prompt_ids) + [1] * len(comp_ids)
            if len(tokens) > TOKEN_LIMIT:
                tokens = tokens[:TOKEN_LIMIT]
                mask = mask[:TOKEN_LIMIT]
                n_trunc += 1
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

    print(f"Corpus: {n_used} entries used | {n_skip} skipped | {n_trunc} truncated")
    print(f"Unmasked tokens: {total_unmasked:,}")
    for cat in sorted(cat_counts):
        print(f"  {cat:<25} {cat_counts[cat]:>5}  ({cat_tokens[cat]:>9,} tokens)")
    print(f"\nIndex: {CORPUS_INDEX}")
    print(f"Per-problem segments: {CORPUS_DIR}/<id>/synthetic.jsonl")


if __name__ == "__main__":
    main()
