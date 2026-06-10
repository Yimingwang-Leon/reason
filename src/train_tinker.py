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

import numpy as np
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
    test_resume_after_epoch0: bool = False  # smoke: validate resume round-trip on real state
    # Hard spend ceiling. Clean 3-epoch ~= $30 (432 steps) + ~$0.4 save surcharges, so
    # 33 lets a clean run finish while capping resume-redos; real $ stays <~$35 (<$40
    # wallet). usd_per_step from the run-004 anchor ($10/144 step, same per-step compute).
    budget_usd: float = 33.0
    usd_per_step: float = 10.0 / 144.0


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
    # numpy-vectorized: the per-token math runs in C and RELEASES the GIL, so the
    # Tinker SDK's background heartbeat thread is not starved. The pure-Python
    # per-token version caused ~0.2s of GIL-held work per example which (over a
    # 64-example batch) stalled the heartbeat and got run-005 v1's session killed.
    cutoff = first_cutoff_weight if epoch == 0 else 1.0
    base = np.asarray(base_mask, dtype=np.float64)
    if prev_logprobs is None:
        return (base * cutoff).tolist()
    bw = np.minimum(1.0, np.abs(np.asarray(prev_logprobs, dtype=np.float64)) / branch_logprob)
    return (base * bw * cutoff).tolist()


def _masked_nll(pairs: list[tuple[list[float], list[float]]]) -> float:
    """Mean per-example masked NLL from (logprobs, weights) pairs.

    Tinker returns per-target-token logprobs (<=0). NLL = -sum(lp*w)/sum(w) per
    example, masking to completion tokens via the weights, then averaged. numpy
    inner math releases the GIL (heartbeat-safe). Pure + side-effect-free so the
    telemetry math is unit-testable without a Tinker call.
    """
    per_ex: list[float] = []
    for lp, w in pairs:
        wa = np.asarray(w, dtype=np.float64)
        tot = wa.sum()
        if tot > 0:
            la = np.asarray(lp, dtype=np.float64)
            per_ex.append(float(-(la * wa).sum() / tot))
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
    prev_lp: dict[str, list[float]] = {}   # problem_id -> last epoch's per-target logprobs
    epoch_ckpts: dict[int, str] = {}
    state_path: str | None = None          # last saved TRAINING state (incl. optimizer)
    # Session-death / stall errors -> resume from state. In tinker 0.22.3 the run-005
    # death class (SidecarDiedError / InternalServerError / RequestFailedError / stall)
    # are ALL subclasses of tinker.TinkerError, so catch that broadly (+ Timeout types).
    RESUME_ERRORS = (tinker.TinkerError, tinker.Timeout, TimeoutError, ConnectionError)
    # ...but these are permanent (config/auth) and must fail fast, not retry-spin:
    NON_RETRYABLE = (tinker.AuthenticationError, tinker.BadRequestError, tinker.NotFoundError,
                     tinker.PermissionDeniedError, tinker.UnprocessableEntityError, tinker.ConflictError)
    MAX_RESUMES = 8                         # secondary anti-spin guard; the $ budget below is the real cap
    # HARD CEILING: total estimated spend (incl. resume-redos AND save/create ops) cannot
    # exceed cfg.budget_usd. spent is estimated per executed step from the run-004 anchor
    # ($10/144 step, train_attn=True; curriculum is client-side so per-step compute matches).
    spent_usd = 0.0
    budget_hit = False

    async def _resume_client():
        """Rebuild the training client after a session death: from the last saved state
        (with optimizer) if we have one, else fresh from base (epoch 0). Retries with
        growing backoff because the network is flakiest right after a death; raises only
        if every attempt fails."""
        last_err = None
        for attempt in range(5):
            try:
                await asyncio.sleep(min(5 * (attempt + 1), 30))
                if state_path is not None:
                    logger.info(f"RESUME: rebuilding from state {state_path} (attempt {attempt+1}/5)")
                    return await sc.create_training_client_from_state_with_optimizer_async(state_path)
                logger.info(f"RESUME: no state yet (epoch 0) -> fresh client (attempt {attempt+1}/5)")
                return await sc.create_lora_training_client_async(
                    base_model=cfg.base_model, rank=cfg.lora_rank, train_mlp=cfg.train_mlp,
                    train_attn=cfg.train_attn, train_unembed=cfg.train_unembed)
            except Exception as e:    # reconnect itself can fail when the network is flaky
                last_err = e
                logger.warning(f"RESUME reconnect attempt {attempt+1}/5 failed: {e}")
        raise RuntimeError(f"could not reconnect after 5 attempts") from last_err

    for epoch in range(cfg.num_epochs):
        resumes = 0
        while True:    # retry this epoch from the last clean state on session death
            epoch_t0 = time.time()
            step = epoch * n_batches       # deterministic step (LR reproducible on resume)
            batches = _stratified_batches(examples, cfg.batch_size, seed=epoch)
            new_lp: dict[str, list[float]] = {}
            cat_nll_sum: dict[str, float] = {}
            cat_nll_n: dict[str, int] = {}
            try:
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

                    lr = cfg.learning_rate * (1.0 - step / max(1, total_steps))
                    t0 = time.time()
                    fwd_bwd_future = await training_client.forward_backward_async(
                        data, loss_fn="cross_entropy")
                    optim_future = await training_client.optim_step_async(
                        tinker.AdamParams(
                            learning_rate=lr, beta1=cfg.beta1, beta2=cfg.beta2,
                            eps=cfg.eps, weight_decay=cfg.weight_decay,
                            grad_clip_norm=cfg.grad_clip_norm))
                    fwd_bwd_result = await fwd_bwd_future.result_async()
                    await optim_future.result_async()
                    elapsed = time.time() - t0

                    # Telemetry on BASE mask (pure per-token NLL); stash logprobs for
                    # next epoch's curriculum. numpy-vectorized -> heartbeat-safe.
                    try:
                        # float32 numpy arrays (single objects, ~7x less memory than
                        # Python float lists, no per-element GC) -> prev_lp/new_lp held
                        # across epochs for the curriculum no longer trigger GC
                        # stop-the-world pauses that starve the SDK heartbeat thread
                        # (the real cause of run-005 v1's session death).
                        logprobs = [
                            np.asarray((o["logprobs"] if isinstance(o, dict) else o.logprobs).data,
                                       dtype=np.float32)
                            for o in fwd_bwd_result.loss_fn_outputs]
                        mean_nll = _masked_nll(
                            [(lp, base) for lp, (_, base) in zip(logprobs, batch_base)])
                        for pid, lp in zip(batch_ids, logprobs):
                            new_lp[pid] = lp
                        for (cat, base), lp in zip(batch_base, logprobs):
                            one = _masked_nll([(lp, base)])
                            if one == one:
                                cat_nll_sum[cat] = cat_nll_sum.get(cat, 0.0) + one
                                cat_nll_n[cat] = cat_nll_n.get(cat, 0) + 1
                    except Exception as e:
                        logger.warning(f"nll telemetry read failed: {e}")
                        mean_nll = float("nan")

                    logger.info(
                        f"epoch={epoch} step={step}/{total_steps} lr={lr:.2e} "
                        f"n={len(data)} nll={mean_nll:.4f} t={elapsed:.1f}s")
                    metrics_f.write(json.dumps({
                        "epoch": epoch, "step": step, "lr": lr, "n": len(data),
                        "nll_per_token": mean_nll, "elapsed_s": elapsed,
                        "time": datetime.now().isoformat()}) + "\n")
                    metrics_f.flush()
                    step += 1
                    spent_usd += cfg.usd_per_step
                    if spent_usd >= cfg.budget_usd:
                        budget_hit = True
                        break    # break batch loop -> stop everything below
                break    # epoch completed (or budget hit) without a session death

            except RESUME_ERRORS as e:
                if isinstance(e, NON_RETRYABLE):
                    metrics_f.close()    # config/auth error -> retrying won't help, fail fast
                    raise
                resumes += 1
                spent_usd += cfg.usd_per_step   # the dead+reconnect round-trip costs something
                logger.warning(f"epoch {epoch}: session died ~step {step} "
                               f"({type(e).__name__}: {e}). resume {resumes}/{MAX_RESUMES}")
                if resumes > MAX_RESUMES or spent_usd >= cfg.budget_usd:
                    metrics_f.close()
                    raise RuntimeError(
                        f"epoch {epoch} aborted (resumes={resumes}, ~${spent_usd:.1f}); "
                        f"recover from last state: {state_path}, epoch ckpts: {epoch_ckpts}") from e
                training_client = await _resume_client()   # retry the epoch from clean state

        if budget_hit:
            logger.warning(f"BUDGET ${cfg.budget_usd:.0f} reached (~${spent_usd:.1f} est) during "
                           f"epoch {epoch}; stopping early and saving current weights as final.")
            break

        # ---- epoch succeeded ----
        per_cat = {c: round(cat_nll_sum[c] / cat_nll_n[c], 5)
                   for c in sorted(cat_nll_sum) if cat_nll_n.get(c)}
        metrics_f.write(json.dumps({"epoch_summary": epoch, "per_cat_nll": per_cat}) + "\n")
        metrics_f.flush()
        logger.info(f"Epoch {epoch} per-cat nll: {per_cat}")
        prev_lp = new_lp

        # Save TRAINING state (with optimizer) so a later death resumes from here.
        try:
            ssf = await training_client.save_state_async(name=f"state_ep{epoch}", overwrite=True)
            ssr = await ssf.result_async()
            state_path = ssr.path if hasattr(ssr, "path") else str(ssr)
            spent_usd += cfg.usd_per_step    # surcharge: save ops are billed too
            logger.info(f"saved training state ep{epoch}: {state_path}")
        except Exception as e:
            logger.warning(f"save_state ep{epoch} failed: {e} (resume falls back to prior state)")

        # Sampler-weights checkpoint per epoch (the LB-submittable adapters).
        # Wrapped: a transient save blip must not crash the run (state_ep{epoch} is
        # already saved above, so the epoch's work is not lost regardless).
        if cfg.save_every_epoch and epoch < cfg.num_epochs - 1:
            try:
                sf = await training_client.save_weights_for_sampler_async(name=f"epoch{epoch}")
                sr = await sf.result_async()
                epoch_ckpts[epoch] = sr.path if hasattr(sr, "path") else str(sr)
                spent_usd += cfg.usd_per_step    # surcharge
                # persist immediately so a later crash can't strand recoverable paths
                (log_path / "epoch_checkpoints.json").write_text(json.dumps(epoch_ckpts, indent=2))
                logger.info(f"Saved epoch{epoch} checkpoint: {epoch_ckpts[epoch]}")
            except Exception as e:
                logger.warning(f"save_weights epoch{epoch} failed: {e} (state_ep{epoch} kept)")

        # Smoke validation hook: prove the resume round-trip works on real state.
        if cfg.test_resume_after_epoch0 and epoch == 0 and state_path is not None:
            logger.info("[test-resume] rebuilding client from saved state to validate resume...")
            training_client = await sc.create_training_client_from_state_with_optimizer_async(state_path)
            logger.info("[test-resume] resume round-trip OK")

        logger.info(f"Epoch {epoch} done in {time.time() - epoch_t0:.1f}s")
    metrics_f.close()

    # Final checkpoint, with retry-from-state so a blip at the finish line (the exact
    # run-005 failure class, most likely after a long session) doesn't lose the whole
    # run. epoch_checkpoints.json is already persisted incrementally above.
    if epoch_ckpts:
        (log_path / "epoch_checkpoints.json").write_text(json.dumps(epoch_ckpts, indent=2))
    tinker_path = None
    for attempt in range(4):
        try:
            sf = await training_client.save_weights_for_sampler_async(name="final")
            sr = await sf.result_async()
            tinker_path = sr.path if hasattr(sr, "path") else str(sr)
            break
        except Exception as e:
            logger.warning(f"final save attempt {attempt+1}/4 failed: {e}")
            if attempt < 3:
                try:
                    training_client = await _resume_client()   # rebuild from last state, retry
                except Exception as e2:
                    logger.warning(f"reconnect for final save failed: {e2}")
    if tinker_path:
        (log_path / "tinker_path.txt").write_text(tinker_path + "\n")
        logger.info(f"Saved weights: {tinker_path}")
        print(f"[train_tinker] DONE. tinker path: {tinker_path}")
        print(f"           (saved to {log_path}/tinker_path.txt)")
    else:
        logger.error(f"FINAL SAVE FAILED after retries. Recover: resume from {state_path} "
                     f"and save_weights, or use epoch checkpoints {epoch_ckpts}.")
        print(f"[train_tinker] FINAL SAVE FAILED. last state={state_path} epochs={epoch_ckpts}")


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
    parser.add_argument("--test-resume", action="store_true",
                        help="smoke: deliberately resume from state after epoch 0 to validate it")
    parser.add_argument("--usd-per-step", type=float, default=10.0 / 144.0,
                        help="per-step $ estimate for the budget guard; recalibrate "
                             "token-proportionally per corpus (run-004 anchor = $0.0694 "
                             "at ~326k tok/step; run-011 drill-heavy mix = ~165k tok/step "
                             "-> 0.0351, else the guard truncates mid-epoch)")
    parser.add_argument("--budget-usd", type=float, default=33.0,
                        help="hard spend ceiling; training stops + saves before exceeding this")
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
        test_resume_after_epoch0=args.test_resume,
        budget_usd=args.budget_usd,
        usd_per_step=args.usd_per_step,
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    asyncio.run(main_async(cfg))


if __name__ == "__main__":
    main()
