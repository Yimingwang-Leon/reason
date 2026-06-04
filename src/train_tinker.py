"""Tinker-based SFT for Nemotron-3-Nano-30B LoRA.

Pivot from Modal after 14 attempts on CUDA env hell. Tinker is huikang's path
(0.85 LB published): managed fine-tuning API. We send pre-tokenized data +
LoRA config, get back checkpoint. No CUDA/torch/mamba/transformers conflicts.

Reads:  ./corpus.jsonl + ./corpus/<id>/synthetic.jsonl  (from src/corpus.py)
Writes: ./training/<run_name>/checkpoint info (tinker:// path, fetched via upload_adapter.py later)

Run:
    python -m src.train_tinker --run-name mini-001 --max-examples 100
    python -m src.train_tinker --run-name run-001    # full corpus
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
CORPUS_INDEX = ROOT / "corpus.jsonl"
CORPUS_DIR = ROOT / "corpus"
TRAINING_DIR = ROOT / "training"
ENV_PATH = ROOT / "env.json"

BASE_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"


@dataclasses.dataclass
class Cfg:
    run_name: str = "run-001"
    max_examples: int | None = None
    base_model: str = BASE_MODEL
    lora_rank: int = 32
    batch_size: int = 64           # effective batch (Tinker handles micro-batch internally)
    num_epochs: int = 1
    max_length: int = 8192
    learning_rate: float = 2e-4
    # Tinker LoRA scope. EMPIRICAL FINDINGS for NemotronH-3-Nano-30B-A3B-BF16:
    # - train_mlp=True   -> MoE only: experts.w1/w2/w3 (w3 empty) +
    #                       shared_experts.{up,down}_proj on the 23 MoE layers.
    #                       Does NOT touch Mamba or attention layers.
    # - train_attn=True  -> Mamba mixer.gate_proj (z portion of in_proj) +
    #                       Mamba mixer.out_proj + attention q/k/v/o_proj.
    #                       Covers the remaining 23 Mamba + 6 attention layers.
    # - train_unembed=False -> skip lm_head training ($$ wasted since vLLM
    #                          can accept lm_head LoRA but our build keeps it
    #                          optional).
    # Phase 4 default flips train_attn -> True so all 52 layers get LoRA
    # (Phase 2/3 left 29/52 layers naked, which capped LB at ~0.65 at best).
    train_mlp: bool = True
    train_attn: bool = True
    train_unembed: bool = False
    # AdamW
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip_norm: float = 1e9
    # Min-logprob curriculum (champion's R-lever). Off by default; enable for
    # multi-epoch runs. Each epoch >0 reweights tokens by branch_weight(prev_logprob)
    # so gradient concentrates on the still-hard tokens.
    curriculum: bool = False
    branch_logprob: float = 0.01
    first_cutoff_weight: float = 0.5
    save_every_epoch: bool = True   # late-checkpoint selection


def _load_env() -> None:
    env = json.loads(ENV_PATH.read_text())
    os.environ["TINKER_API_KEY"] = env["TINKER_API_KEY"]


def _load_examples(max_examples: int | None) -> list[dict]:
    """Load tokenized examples from local corpus."""
    if not CORPUS_INDEX.exists():
        raise FileNotFoundError(
            f"No {CORPUS_INDEX}. Run `python -m src.corpus` first."
        )
    examples = []
    with open(CORPUS_INDEX) as f:
        for line in f:
            meta = json.loads(line)
            seg_path = CORPUS_DIR / meta["problem_id"] / "synthetic.jsonl"
            with open(seg_path) as g:
                seg = json.loads(g.readline())
            examples.append({
                "tokens": seg["tokens"],
                "mask": seg["mask"],
                "category": meta["category"],
                "problem_id": meta["problem_id"],
            })
    if max_examples is not None:
        # Stratified by category
        from collections import defaultdict
        by_cat = defaultdict(list)
        for ex in examples:
            by_cat[ex["category"]].append(ex)
        per_cat = max(1, max_examples // max(1, len(by_cat)))
        examples = [e for cat in by_cat for e in by_cat[cat][:per_cat]]
    return examples


def _build_datum(tokens: list[int], mask: list[int], weights: list[float] | None = None):
    """Build a tinker.Datum from token list + train-mask.

    weights (optional, len==len(tokens)-1) overrides the per-target-token loss
    weights; used by the curriculum. Defaults to the binary completion mask."""
    import tinker
    if len(tokens) < 2:
        return None
    model_input = tinker.ModelInput(
        chunks=[tinker.EncodedTextChunk(tokens=tokens[:-1])]
    )
    target_tokens = tokens[1:]
    # weight=1.0 on completion tokens (mask==1), 0.0 on prompt (unless overridden)
    if weights is None:
        weights = [float(m) for m in mask[1:]]
    return tinker.Datum(
        model_input=model_input,
        loss_fn_inputs={
            "target_tokens": tinker.TensorData(
                data=target_tokens, dtype="int64", shape=[len(target_tokens)],
            ),
            "weights": tinker.TensorData(
                data=weights, dtype="float32", shape=[len(weights)],
            ),
        },
    )


def _branch_weight(logprob: float, branch_logprob: float) -> float:
    """min(1, |logprob|/branch_logprob). Tokens the model is already confident on
    (logprob ~ 0) get a small weight; hard/low-logprob tokens keep weight ~1, so the
    gradient concentrates on the hardest tokens (the min-logprob curriculum)."""
    return min(1.0, abs(logprob) / branch_logprob)


def _curriculum_weights(
    base_mask: list[float],
    prev_logprobs: list[float] | None,
    epoch: int,
    branch_logprob: float,
    first_cutoff_weight: float,
) -> list[float]:
    """Reweight completion-token weights for the min-logprob curriculum.

    new_w = base_mask * branch_weight(prev_logprob) * cutoff(epoch).
    Epoch 0 (no prev logprobs) returns base_mask * first_cutoff_weight (the gentle
    calibration pass). Pure -> unit-testable without Tinker.
    """
    cutoff = first_cutoff_weight if epoch == 0 else 1.0
    if prev_logprobs is None:
        return [m * cutoff for m in base_mask]
    return [
        m * _branch_weight(lp, branch_logprob) * cutoff
        for m, lp in zip(base_mask, prev_logprobs)
    ]


def _masked_nll(pairs: list[tuple[list[float], list[float]]]) -> float:
    """Mean per-example masked NLL from (logprobs, weights) pairs.

    Tinker returns per-target-token logprobs (<=0). NLL = -sum(lp*w)/sum(w) per
    example, masking to completion tokens via the weights, then averaged. Pure +
    side-effect-free so the telemetry math is unit-testable without a Tinker call.
    """
    per_ex: list[float] = []
    for lp, w in pairs:
        tot = sum(w)
        if tot > 0:
            per_ex.append(-sum(l * wi for l, wi in zip(lp, w)) / tot)
    return sum(per_ex) / len(per_ex) if per_ex else float("nan")


def _extract_nll_pairs(loss_fn_outputs, data) -> list[tuple[list[float], list[float]]]:
    """Pull (logprobs, weights) from a fwd_bwd result. loss_fn_outputs[i] is a dict
    {"logprobs": TensorData}; weights live on each Datum's loss_fn_inputs."""
    pairs = []
    for out, datum in zip(loss_fn_outputs, data):
        lp_td = out["logprobs"] if isinstance(out, dict) else getattr(out, "logprobs")
        lp = list(lp_td.data)
        w = list(datum.loss_fn_inputs["weights"].data)
        pairs.append((lp, w))
    return pairs


def _stratified_batches(
    examples: list[dict], batch_size: int, seed: int = 0
) -> list[list[int]]:
    """Indices grouped into equal-sized batches with categories evenly distributed."""
    rng = random.Random(seed)
    n_batches = math.ceil(len(examples) / batch_size)
    by_cat: dict[str, list[int]] = {}
    for i, ex in enumerate(examples):
        by_cat.setdefault(ex["category"], []).append(i)
    for v in by_cat.values():
        rng.shuffle(v)
    batches: list[list[int]] = [[] for _ in range(n_batches)]
    order = list(range(n_batches))
    rng.shuffle(order)
    assigned = 0
    for cat in sorted(by_cat.keys()):
        for idx in by_cat[cat]:
            batches[order[assigned % n_batches]].append(idx)
            assigned += 1
    return batches


async def main_async(cfg: Cfg) -> None:
    _load_env()
    import tinker

    log_path = TRAINING_DIR / cfg.run_name
    log_path.mkdir(parents=True, exist_ok=True)

    examples = _load_examples(cfg.max_examples)
    logger.info(f"Loaded {len(examples)} examples")

    n_batches = math.ceil(len(examples) / cfg.batch_size)
    total_steps = n_batches * cfg.num_epochs
    logger.info(f"Training: {n_batches} batches × {cfg.num_epochs} epochs = {total_steps} steps")

    # Save config snapshot
    with open(log_path / "config.json", "w") as f:
        json.dump({**dataclasses.asdict(cfg),
                   "time": datetime.now().isoformat(),
                   "n_examples": len(examples)}, f, indent=2)

    sc = tinker.ServiceClient()
    training_client = await sc.create_lora_training_client_async(
        base_model=cfg.base_model,
        rank=cfg.lora_rank,
        train_mlp=cfg.train_mlp,
        train_attn=cfg.train_attn,
        train_unembed=cfg.train_unembed,
    )
    logger.info(f"Created Tinker LoRA training client: {cfg.base_model} rank={cfg.lora_rank}")

    metrics_f = open(log_path / "metrics.jsonl", "w")
    step = 0
    prev_lp: dict[str, list[float]] = {}   # problem_id -> last epoch's per-target logprobs
    epoch_ckpts: dict[int, str] = {}
    for epoch in range(cfg.num_epochs):
        epoch_t0 = time.time()
        batches = _stratified_batches(examples, cfg.batch_size, seed=epoch)
        new_lp: dict[str, list[float]] = {}
        cat_nll_sum: dict[str, float] = {}
        cat_nll_n: dict[str, int] = {}
        for batch_indices in batches:
            data, batch_ids, batch_base = [], [], []
            for i in batch_indices:
                ex = examples[i]
                tokens = ex["tokens"][:cfg.max_length]
                mask = ex["mask"][:cfg.max_length]
                base = [float(m) for m in mask[1:]]
                weights = None
                if cfg.curriculum:
                    weights = _curriculum_weights(
                        base, prev_lp.get(ex["problem_id"]), epoch,
                        cfg.branch_logprob, cfg.first_cutoff_weight)
                d = _build_datum(tokens, mask, weights)
                if d is not None:
                    data.append(d)
                    batch_ids.append(ex["problem_id"])
                    batch_base.append((ex["category"], base))
            if not data:
                continue

            # Step-linear LR decay
            lr = cfg.learning_rate * (1.0 - step / max(1, total_steps))

            t0 = time.time()
            fwd_bwd_future = await training_client.forward_backward_async(
                data, loss_fn="cross_entropy",
            )
            optim_future = await training_client.optim_step_async(
                tinker.AdamParams(
                    learning_rate=lr,
                    beta1=cfg.beta1, beta2=cfg.beta2,
                    eps=cfg.eps, weight_decay=cfg.weight_decay,
                    grad_clip_norm=cfg.grad_clip_norm,
                )
            )
            fwd_bwd_result = await fwd_bwd_future.result_async()
            await optim_future.result_async()
            elapsed = time.time() - t0

            # Telemetry uses the BASE mask (pure per-token NLL), independent of the
            # curriculum gradient weights. Also stash logprobs for next epoch.
            try:
                logprobs = [
                    list((o["logprobs"] if isinstance(o, dict) else o.logprobs).data)
                    for o in fwd_bwd_result.loss_fn_outputs
                ]
                tel_pairs = [(lp, base) for lp, (_, base) in zip(logprobs, batch_base)]
                mean_nll = _masked_nll(tel_pairs)
                for pid, lp in zip(batch_ids, logprobs):
                    new_lp[pid] = lp
                for (cat, base), lp in zip(batch_base, logprobs):
                    one = _masked_nll([(lp, base)])
                    if one == one:  # not nan
                        cat_nll_sum[cat] = cat_nll_sum.get(cat, 0.0) + one
                        cat_nll_n[cat] = cat_nll_n.get(cat, 0) + 1
            except Exception as e:
                logger.warning(f"nll telemetry read failed: {e}")
                mean_nll = float("nan")

            logger.info(
                f"epoch={epoch} step={step}/{total_steps} lr={lr:.2e} "
                f"n={len(data)} nll={mean_nll:.4f} t={elapsed:.1f}s"
            )
            metrics_f.write(json.dumps({
                "epoch": epoch, "step": step, "lr": lr, "n": len(data),
                "nll_per_token": mean_nll, "elapsed_s": elapsed,
                "time": datetime.now().isoformat(),
            }) + "\n")
            metrics_f.flush()
            step += 1

        # Per-category telemetry for the epoch (find weak sub-skills).
        per_cat = {c: round(cat_nll_sum[c] / cat_nll_n[c], 5)
                   for c in sorted(cat_nll_sum) if cat_nll_n.get(c)}
        metrics_f.write(json.dumps({"epoch_summary": epoch, "per_cat_nll": per_cat}) + "\n")
        metrics_f.flush()
        logger.info(f"Epoch {epoch} per-cat nll: {per_cat}")
        prev_lp = new_lp   # this epoch's logprobs drive next epoch's curriculum weights

        # Late-checkpoint selection: save each epoch so the best can be picked.
        if cfg.save_every_epoch and epoch < cfg.num_epochs - 1:
            sf = await training_client.save_weights_for_sampler_async(name=f"epoch{epoch}")
            sr = await sf.result_async()
            epoch_ckpts[epoch] = sr.path if hasattr(sr, "path") else str(sr)
            logger.info(f"Saved epoch{epoch} checkpoint: {epoch_ckpts[epoch]}")
        logger.info(f"Epoch {epoch} done in {time.time() - epoch_t0:.1f}s")
    metrics_f.close()

    # Final checkpoint. save_weights_for_sampler returns the tinker:// path that
    # upload_adapter.py / build_submission.py download.
    save_future = await training_client.save_weights_for_sampler_async(name="final")
    save_resp = await save_future.result_async()
    tinker_path = save_resp.path if hasattr(save_resp, "path") else str(save_resp)
    logger.info(f"Saved weights: {tinker_path}")
    (log_path / "tinker_path.txt").write_text(tinker_path + "\n")
    if epoch_ckpts:
        (log_path / "epoch_checkpoints.json").write_text(json.dumps(epoch_ckpts, indent=2))
    print(f"[train_tinker] DONE. tinker path: {tinker_path}")
    print(f"           (saved to {log_path}/tinker_path.txt)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="run-001")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--no-attn", action="store_true",
                        help="disable train_attn (Phase 2/3 used this; default is on)")
    parser.add_argument("--no-mlp", action="store_true")
    parser.add_argument("--train-unembed", action="store_true")
    parser.add_argument("--curriculum", action="store_true",
                        help="min-logprob curriculum (reweight hard tokens each epoch)")
    parser.add_argument("--branch-logprob", type=float, default=0.01)
    parser.add_argument("--first-cutoff", type=float, default=0.5)
    args = parser.parse_args()

    cfg = Cfg(
        run_name=args.run_name,
        max_examples=args.max_examples,
        lora_rank=args.lora_rank,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.lr,
        train_attn=not args.no_attn,
        train_mlp=not args.no_mlp,
        train_unembed=args.train_unembed,
        curriculum=args.curriculum,
        branch_logprob=args.branch_logprob,
        first_cutoff_weight=args.first_cutoff,
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    asyncio.run(main_async(cfg))


if __name__ == "__main__":
    main()
