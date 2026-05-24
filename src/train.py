"""Train LoRA on Nemotron-3-Nano-30B using transformers + peft + TRL on Modal H200.

unsloth was tried (commits c1-c8) but its compiled Nemotron_H cache hardcodes
the naive Mamba torch_forward, so the installed mamba-ssm fast path was never
used → OOM @ μ=1 with ~135 GB activations. Vanilla transformers 5.3+ has
native NemotronH with proper fast-path routing.

Run from local:
    modal run src/train.py --corpus-name run-001 --max-examples 100

Reads /work/corpus/<corpus-name>/corpus.jsonl (uploaded via push_corpus).
Writes /work/adapters/<corpus-name>/{adapter_config.json,adapter_model.safetensors}.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import modal

from src.modal_app import (
    ADAPTER_DIR, BASE_MODEL, CORPUS_DIR, MODEL_CACHE_PATH,
    VOLUMES, app, image, work_vol,
)


@app.function(image=image, volumes=VOLUMES, gpu="H200", timeout=4 * 3600)
def train(
    corpus_name: str = "run-001",
    lora_rank: int = 32,
    learning_rate: float = 2e-4,
    num_epochs: int = 1,
    batch_size: int = 1,           # per-device micro batch; naive Mamba is memory-heavy
    grad_accum_steps: int = 64,    # effective batch = 1 * 64 = 64
    max_seq_length: int = 2048,    # MINI: naive Mamba OOMs > ~3000; full run needs fast Mamba
    max_examples: int | None = None,  # set small (e.g. 100) for pipeline smoke test
):
    """SFT a LoRA on the tokenized corpus, save adapter to volume."""
    os.environ["HF_HOME"] = MODEL_CACHE_PATH
    os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE_PATH

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTConfig, SFTTrainer
    from datasets import Dataset

    # === DIAGNOSE mamba fast-path imports ============================
    # transformers/unsloth both do try/except on these 3 symbols. If any fails,
    # naive Mamba is used (OOMs at seq=8192). Print which one is missing.
    print("[diag] === probing mamba fast-path imports ===")
    for name, modpath in [
        ("causal_conv1d_fn", "causal_conv1d"),
        ("causal_conv1d_update", "causal_conv1d"),
        ("selective_state_update", "mamba_ssm.ops.triton.selective_state_update"),
        ("selective_state_update (alt)", "mamba_ssm.ops.selective_scan_interface"),
    ]:
        try:
            mod = __import__(modpath, fromlist=[name.split(" ")[0]])
            sym = getattr(mod, name.split(" ")[0], None)
            print(f"[diag]  ✓ {modpath}.{name.split(' ')[0]}: {sym}")
        except Exception as e:
            print(f"[diag]  ✗ {modpath}: {type(e).__name__}: {e}")
    print("[diag] ====================================")

    corpus_root = Path(CORPUS_DIR) / corpus_name
    index_path = corpus_root / "corpus.jsonl"
    if not index_path.exists():
        raise FileNotFoundError(f"No corpus at {index_path} — run push_corpus first")

    # Stream tokenized examples (each problem already has tokens + mask on disk)
    examples = []
    with open(index_path) as f:
        for line in f:
            meta = json.loads(line)
            seg_path = corpus_root / "corpus" / meta["problem_id"] / "synthetic.jsonl"
            with open(seg_path) as g:
                seg = json.loads(g.readline())
            examples.append({
                "input_ids": seg["tokens"],
                "attention_mask": [1] * len(seg["tokens"]),
                "labels": [
                    tok if m else -100 for tok, m in zip(seg["tokens"], seg["mask"])
                ],
                "category": meta["category"],
            })
    if max_examples is not None:
        # Stratified sample so each category is represented in the mini run
        from collections import defaultdict
        by_cat: dict[str, list[dict]] = defaultdict(list)
        for ex in examples:
            by_cat[ex["category"]].append(ex)
        per_cat = max(1, max_examples // max(1, len(by_cat)))
        examples = [e for cat in by_cat for e in by_cat[cat][:per_cat]]

    # Drop overlong examples — naive Mamba activations are O(L²); without
    # fast path, L > ~3000 OOMs on H200. Filter for the mini smoke test.
    examples = [e for e in examples if len(e["input_ids"]) <= max_seq_length]
    print(f"[train] loaded {len(examples)} examples (after seq_len filter ≤ {max_seq_length})")

    print(f"[train] loading base model {BASE_MODEL} (cached in {MODEL_CACHE_PATH})")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, cache_dir=MODEL_CACHE_PATH,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,             # use model repo's modeling_nemotron_h.py
        cache_dir=MODEL_CACHE_PATH,
        device_map="auto",
        attn_implementation="sdpa",
    )
    print(f"[train] base loaded in {time.time() - t0:.0f}s")

    # Discover MoE expert module names dynamically (NemotronH has 128 experts × 23 layers)
    moe_modules: list[str] = []
    for name, mod in model.named_modules():
        # MoE expert linear layers — exact names depend on transformers version
        if any(s in name for s in (".experts.", "expert_")):
            cls = mod.__class__.__name__
            if cls in ("Linear", "Linear8bitLt", "Linear4bit"):
                # Just store the suffix-class name; peft regex will catch all instances
                pass
    print(f"[train] model class: {model.__class__.__name__}")

    # LoRA target: attention + standard FFN + MoE experts (regex catches all)
    # Common NemotronH names: q_proj/k_proj/v_proj/o_proj (attention),
    # up_proj/down_proj/gate_proj (FFN), and MoE experts vary by impl.
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "up_proj", "down_proj", "gate_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    from transformers import DataCollatorForSeq2Seq
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, padding=True, label_pad_token_id=-100,
    )

    cfg = SFTConfig(
        output_dir=f"{ADAPTER_DIR}/{corpus_name}_tmp",
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        lr_scheduler_type="linear",
        warmup_ratio=0.03,
        logging_steps=5,
        save_strategy="no",
        bf16=True,
        max_length=max_seq_length,
        dataset_kwargs={"skip_prepare_dataset": True},  # we feed pre-tokenized
        remove_unused_columns=True,    # drop 'category' (str) before collator tensorizes
        report_to="none",
        seed=42,
        gradient_checkpointing=False,       # NemotronHForCausalLM doesn't support it
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=Dataset.from_list(examples),
        args=cfg,
        data_collator=collator,
    )
    print(f"[train] starting training: {len(examples)} ex × {num_epochs} epoch")
    t0 = time.time()
    trainer.train()
    print(f"[train] done in {(time.time() - t0) / 60:.1f} min")

    out_dir = Path(ADAPTER_DIR) / corpus_name
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))         # writes adapter_config.json + .safetensors
    tokenizer.save_pretrained(str(out_dir))     # for convenience
    work_vol.commit()
    print(f"[train] adapter saved to {out_dir}")


@app.function(image=image, volumes=VOLUMES, timeout=600)
def push_corpus(corpus_name: str, corpus_bytes: bytes):
    """Upload a packed corpus archive from local to the Modal volume.

    corpus_bytes: tar.gz of local ./corpus/ + ./corpus.jsonl
    """
    import io
    import tarfile

    target = Path(CORPUS_DIR) / corpus_name
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(corpus_bytes), mode="r:gz") as tar:
        tar.extractall(target)
    work_vol.commit()
    print(f"[push_corpus] extracted to {target}")
    n = sum(1 for _ in (target / "corpus.jsonl").open())
    print(f"[push_corpus] {n} entries in index")


@app.local_entrypoint()
def main(corpus_name: str = "run-001", max_examples: int = 0):
    """Local entry: package ./corpus + ./corpus.jsonl, push, train.

    max_examples=0 means full corpus; pass small int (e.g. 100) for smoke test.
    """
    import io
    import tarfile

    root = Path(__file__).resolve().parent.parent
    corpus_dir = root / "corpus"
    corpus_index = root / "corpus.jsonl"
    assert corpus_dir.exists() and corpus_index.exists(), \
        "run `python -m src.corpus` first to produce ./corpus/ + ./corpus.jsonl"

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(corpus_dir, arcname="corpus")
        tar.add(corpus_index, arcname="corpus.jsonl")
    print(f"[main] packed corpus: {buf.tell() / 1e6:.1f} MB")

    push_corpus.remote(corpus_name, buf.getvalue())
    train.remote(corpus_name=corpus_name, max_examples=max_examples or None)
    print(f"[main] training done. adapter at /work/adapters/{corpus_name}")
