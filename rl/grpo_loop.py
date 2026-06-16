"""grpo_loop.py — R3 条件武器:50 步单腿 GRPO(PLAN §1.2-④ / §2.3 红线)。

规格【硬编码,不许命令行越过】:
  - cohort ≤1000 题 × K≤8 × ≤2 pass(总 rollout 硬顶 16000;顶配 K16×3pass 禁止)
  - crypt(cryptarithm_deduce)rollout cap 4096,其余 7680
  - A = r − mean(组);全同奖励(零方差)组整组丢弃,不进 forward_backward
  - loss = importance_sampling;--cispo 切 cispo(ratio 界 0.0/4.0,官方约定)
  - KL 子采样锚 step-0 默认开:coef 0.05、15% 子采样,锚=run-011 final sampler
  - probe→pass→save 顺序写死:只有过 anchor_probe 的 state 才 save_state 成为
    合法回滚点;常驻 2 个(grpo_rb_a/grpo_rb_b 轮换);探针熔断 → 回滚最近合法点并停腿
  - journal jsonl 每步落盘;rollout 边采边落盘(tokens/logprobs/reward)
  - LR 2e-5 恒定,grad_clip 1.0;判停只认 holdout 翻转,不认 train reward 曲线
  - 前 5 步实测 >12 min/步 → 步数砍到 40(时钟预登记)
  - --live 前置:余额(--balance-usd,console 人工读数,SDK 无余额端点)≥1.3×--budget-usd,
    否则拒绝发车;预算守卫按 $0.0029/rollout(crypt@4096 $0.002)+ --usd-per-step 记账

默认 mock(零网络):回放 data/probes/mine_results.jsonl 合成 rollout,
走 datum 构造/advantage/零方差丢弃/KL 锚/熔断/journal/回滚全路径。
  python rl/grpo_loop.py --steps 12 --probe-every 5            # dry-run
  python rl/grpo_loop.py --steps 12 --probe-every 5 --mock-trip # 演练熔断回滚
解锁线(发车前人工核对):G2 覆盖高+翻转低 + 激进档授权 + 最佳挑战者 LB ≥0.86。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np  # advantage/reward 全 numpy(心跳饿死教训,run-005 v1)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import rl.anchor_probe as anchor_probe  # noqa: E402

# ---- 硬编码规格(预登记,改动 = 重新预登记) ----
HARD_COHORT_MAX = 1000
HARD_K_MAX = 8
HARD_PASS_MAX = 2
HARD_ROLLOUT_MAX = HARD_COHORT_MAX * HARD_K_MAX * HARD_PASS_MAX  # 16000
CAP_DEFAULT = 7680
CAP_CRYPT = 4096          # 0/1 reward 下与 7680 等价,省 80% 截断垃圾钱
LR = 2e-5                  # 恒定,无退火
GRAD_CLIP = 1.0
KL_COEF = 0.05
KL_SUBSAMPLE = 0.15        # 15% 子采样锚
RESIDENT_STATES = ("grpo_rb_a", "grpo_rb_b")  # 常驻 2 个合法回滚点,轮换覆盖
USD_PER_ROLLOUT = 0.0029
USD_PER_ROLLOUT_CRYPT = 0.002
SLOW_STEP_SEC = 12 * 60    # 前 5 步均值超此值 → 步数砍到 40
RUN011_SAMPLER = "tinker://cad6ab5c-35bc-516c-90bd-804b0a6166f5:train:0/sampler_weights/final"

MINE_RESULTS = ROOT / "data" / "probes" / "mine_results.jsonl"
DECODE_BASE = ROOT / "data" / "run012_holdout_decode.jsonl"


# ---------------------------------------------------------------- rollout 形态
@dataclass
class Rollout:
    pid: str
    cat: str
    prompt_tokens: list[int]
    tokens: list[int]          # 生成段
    logprobs: list[float]      # 采样时刻逐生成 token logprob(永不回填)
    reward: float
    stop_reason: str = "stop"


@dataclass
class LegState:
    step: int = 0
    est_cost: float = 0.0
    n_rollouts: int = 0
    good_states: list[str] = field(default_factory=list)  # 合法回滚点(≤2)
    step_secs: list[float] = field(default_factory=list)


def cap_for(cat: str) -> int:
    return CAP_CRYPT if cat == "cryptarithm_deduce" else CAP_DEFAULT


def rollout_price(cat: str) -> float:
    return USD_PER_ROLLOUT_CRYPT if cat == "cryptarithm_deduce" else USD_PER_ROLLOUT


# ------------------------------------------------------------ datum 构造(纯算)
def build_datum_dict(ro: Rollout, adv: float) -> dict:
    """cookbook rl_loop.py:203-227 拼法,白名单键 only(datum.py _KEY_TO_TYPE):
    model_input = prompt + sampled[:-1]
    target/logprobs/advantages = prompt 段(P-1)补 0 + 生成段。
    多余键(mask 等)一律不进 —— 后端白名单外即报错。"""
    p, s = ro.prompt_tokens, ro.tokens
    assert len(p) >= 1 and len(s) >= 1, ro.pid
    model_input = p + s[:-1]
    pad = len(p) - 1
    target = [0] * pad + s
    logprobs = [0.0] * pad + list(ro.logprobs)
    advantages = [0.0] * pad + [float(adv)] * len(s)
    assert len(target) == len(model_input) == len(logprobs) == len(advantages)
    return {"model_input": model_input,
            "loss_fn_inputs": {"target_tokens": target,
                               "logprobs": logprobs,
                               "advantages": advantages}}


def to_tinker_datum(dd: dict):
    """live only:dict → tinker.Datum(键集与 dtype 按 SDK 白名单)。"""
    import tinker
    mi = tinker.ModelInput(chunks=[tinker.EncodedTextChunk(tokens=dd["model_input"])])
    n = len(dd["model_input"])
    lfi = {"target_tokens": tinker.TensorData(data=dd["loss_fn_inputs"]["target_tokens"],
                                              dtype="int64", shape=[n]),
           "logprobs": tinker.TensorData(data=dd["loss_fn_inputs"]["logprobs"],
                                         dtype="float32", shape=[n]),
           "advantages": tinker.TensorData(data=dd["loss_fn_inputs"]["advantages"],
                                           dtype="float32", shape=[n])}
    return tinker.Datum(model_input=mi, loss_fn_inputs=lfi)


# ------------------------------------------------------- advantage / KL 锚
def group_advantages(groups: list[list[Rollout]]) -> tuple[list[tuple[Rollout, float]], int]:
    """A_i = r_i - mean(组),Dr.GRPO 风格不除 std;零方差组丢弃(只花采样钱)。"""
    kept, dropped = [], 0
    for g in groups:
        r = np.asarray([x.reward for x in g], dtype=np.float32)
        if float(r.std()) == 0.0:
            dropped += 1
            continue
        a = r - float(r.mean())
        kept.extend(zip(g, a.tolist()))
    return kept, dropped


def kl_anchor_adjust(datums: list[dict], rollouts: list[Rollout],
                     ref_logprob_fn, rng: random.Random,
                     coef: float = KL_COEF, frac: float = KL_SUBSAMPLE) -> dict:
    """cookbook metrics.py:124 incorporate_kl_penalty 的子采样版:
    对 frac 的 rollout 用锚策略(run-011)算 ref logprob,
    advantages += coef·(avg_kl − per_token_kl),原地改 datum。"""
    idx = [i for i in range(len(datums)) if rng.random() < frac]
    if not idx:
        return {"n_anchored": 0, "avg_kl": None}
    kls = []
    for i in idx:
        ro = rollouts[i]
        ref = np.asarray(ref_logprob_fn(ro), dtype=np.float32)      # 生成段
        samp = np.asarray(ro.logprobs, dtype=np.float32)
        per_tok_kl = samp - ref                                      # E_p[logp-logq]
        avg_kl = float(per_tok_kl.mean())
        adv = np.asarray(datums[i]["loss_fn_inputs"]["advantages"], dtype=np.float32)
        pad = len(adv) - len(per_tok_kl)
        adv[pad:] += coef * (avg_kl - per_tok_kl)
        datums[i]["loss_fn_inputs"]["advantages"] = adv.tolist()
        kls.append(avg_kl)
    return {"n_anchored": len(idx), "avg_kl": round(float(np.mean(kls)), 6)}


# ------------------------------------------------------------------ mock 采样
def _pseudo_tokens(seed: str, n: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(7, 50000) for _ in range(n)]


def load_mock_groups(path: Path) -> list[list[Rollout]]:
    """回放 mine_results.jsonl(pid/category/n_correct/K/trace;挖矿没存 tokens/logprobs,
    这里确定性合成,只为走通形态路径——live 永远存真值)。"""
    groups = []
    for line in open(path):
        d = json.loads(line)
        pid, cat, K = d["pid"], d["category"], int(d["K"])
        cap = cap_for(cat)
        n_gen = min(max(len(d["trace"]) // 3, 16), cap)   # 字符→粗 token 数,撞 cap 截断
        h = hashlib.sha1(pid.encode()).hexdigest()
        p_len = 600 + int(h[:4], 16) % 900                # 600-1500 prompt tok
        g = []
        for k in range(K):
            rng = random.Random(f"{pid}:{k}")
            g.append(Rollout(
                pid=pid, cat=cat,
                prompt_tokens=_pseudo_tokens(pid, p_len),
                tokens=_pseudo_tokens(f"{pid}:{k}", n_gen),
                logprobs=[-abs(rng.gauss(0.4, 0.3)) for _ in range(n_gen)],
                reward=1.0 if k < int(d["n_correct"]) else 0.0,
                stop_reason="length" if n_gen == cap else "stop"))
        groups.append(g)
    return groups


def mock_ref_logprob(ro: Rollout) -> list[float]:
    rng = random.Random(f"ref:{ro.pid}")
    return [lp - rng.gauss(0.0, 0.05) for lp in ro.logprobs]


# ------------------------------------------------------------------- journal
class Journal:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a")

    def write(self, event: str, **kw):
        rec = {"ts": round(time.time(), 1), "event": event, **kw}
        self.f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.f.flush()


# ------------------------------------------------------- probe→pass→save(写死)
def probe_pass_save(st: LegState, journal: Journal, live: bool, training_client,
                    decode_rows: dict, mock_trip: bool) -> bool:
    """顺序不可重排:① 跑探针 ② PASS 才 ③ save_state 成为合法回滚点。
    返回 False = 熔断(调用方必须回滚+停腿)。"""
    if mock_trip:
        rep = {"fuse": {"tripped": True, "reasons": ["mock-trip 注入(演练)"]},
               "net_drawdown_pct": 99.0}
    else:
        rep = anchor_probe.run_probe(decode_rows)
    journal.write("probe", step=st.step, tripped=rep["fuse"]["tripped"],
                  net_drawdown_pct=rep.get("net_drawdown_pct"),
                  reasons=rep["fuse"]["reasons"])
    if rep["fuse"]["tripped"]:
        print(f"FUSE_TRIPPED step={st.step}: {rep['fuse']['reasons']}")
        return False
    name = RESIDENT_STATES[(len(st.good_states)) % len(RESIDENT_STATES)]
    if live:
        ssf = training_client.save_state_async(name=name, overwrite=True)
        path = ssf.result().path if hasattr(ssf, "result") else None
    else:
        path = f"mock://{name}"
    st.good_states.append(name)
    if len(st.good_states) > len(RESIDENT_STATES):   # 只留台账尾部 2 个名字
        st.good_states = st.good_states[-len(RESIDENT_STATES):]
    journal.write("save", step=st.step, name=name, path=path)
    return True


def rollback(st: LegState, journal: Journal, live: bool, sc):
    last = st.good_states[-1] if st.good_states else "step-0 初始 state(无合法点)"
    journal.write("rollback", step=st.step, to=last)
    print(f"[rollback] 回滚到 {last};本腿终止(判停只认 holdout 翻转,不续跑)")
    if live and st.good_states:
        # 真回滚 = 用最近合法 state 重建 training client(weights+RL Adam 矩)
        # sc.create_training_client_from_state_with_optimizer(f"...weights/{last}")
        pass


# ---------------------------------------------------------------------- 主腿
def run_leg(args) -> int:
    # ---- 硬顶钳制(命令行不可越过) ----
    cohort_n = min(args.cohort_size, HARD_COHORT_MAX)
    K = min(args.k, HARD_K_MAX)
    passes = min(args.passes, HARD_PASS_MAX)
    steps = args.steps
    total_rollout_budget = min(cohort_n * K * passes, HARD_ROLLOUT_MAX)

    journal = Journal(args.journal)
    rollout_log = args.journal.with_name(args.journal.stem + "_rollouts.jsonl")
    journal.write("leg_start", live=args.live, steps=steps, cohort=cohort_n, K=K,
                  passes=passes, rollout_budget=total_rollout_budget,
                  loss="cispo" if args.cispo else "importance_sampling",
                  kl_anchor=not args.no_kl_anchor, lr=LR, grad_clip=GRAD_CLIP,
                  caps={"crypt": CAP_CRYPT, "default": CAP_DEFAULT})

    # ---- live 前置闸 ----
    sc = training_client = sampler = None
    if args.live:
        if args.balance_usd is None or args.balance_usd < 1.3 * args.budget_usd:
            print(f"REFUSED: 余额 {args.balance_usd} < 1.3×预算 {args.budget_usd}"
                  "(--balance-usd 取 Tinker console 实读数;SDK 无余额端点)")
            return 3
        import tinker
        from src.train_tinker import _load_env
        _load_env()
        sc = tinker.ServiceClient()
        # 权重 = run-011 final 训练态,优化器全新(fresh Adam,不带 SFT 末期矩)
        training_client = sc.create_training_client_from_state(args.init_state)
        sampler = training_client.save_weights_and_get_sampling_client(name="grpo_step0")
        # TODO(live 首发):list_checkpoints 确认 init_state 存在;tinker.Timeout 不是异常类,
        # 任何 except 元组沿用 train_tinker.py:286 的 issubclass 过滤。

    # ---- 数据源 ----
    if args.live:
        raise NotImplementedError(
            "live cohort 采样:按 recon_tinker §1.3 每题 sample(num_samples=K) future 群发,"
            "回收后立刻落盘 rollout_log,再进 group_advantages。解锁线未到,故意不实装。")
    groups_pool = load_mock_groups(args.mock_replay)
    decode_rows = anchor_probe._load_decode(DECODE_BASE)   # mock 探针解码 = 基线重放

    st = LegState()
    rng = random.Random(11)
    loss_fn = "cispo" if args.cispo else "importance_sampling"
    loss_cfg = {"clip_low_threshold": 0.0, "clip_high_threshold": 4.0} if args.cispo else None
    pool_i = 0

    with open(rollout_log, "a") as rlog:
        for step in range(1, steps + 1):
            t0 = time.time()
            st.step = step
            # 取本步题组(mock:循环回放池)
            batch = []
            for _ in range(args.batch_problems):
                batch.append(groups_pool[pool_i % len(groups_pool)])
                pool_i += 1
            n_ro = sum(len(g) for g in batch)
            if st.n_rollouts + n_ro > total_rollout_budget:
                journal.write("halt", step=step, reason="rollout 硬顶",
                              used=st.n_rollouts, budget=total_rollout_budget)
                print(f"[halt] rollout 预算耗尽 {st.n_rollouts}/{total_rollout_budget}")
                break
            st.n_rollouts += n_ro
            # rollout 边采边落盘(session 死亡采样钱不丢)
            for g in batch:
                for ro in g:
                    rlog.write(json.dumps({
                        "pid": ro.pid, "cat": ro.cat, "reward": ro.reward,
                        "n_prompt": len(ro.prompt_tokens), "n_gen": len(ro.tokens),
                        "stop": ro.stop_reason, "step": step,
                        # live 必须存全量 tokens/logprobs;mock 合成值只存形态摘要
                        "tokens": None if not args.live else ro.tokens,
                        "logprobs": None if not args.live else ro.logprobs,
                    }) + "\n")
            rlog.flush()

            # advantage + 零方差丢弃
            kept, dropped = group_advantages(batch)
            rewards = np.asarray([ro.reward for g in batch for ro in g], dtype=np.float32)
            datums, kept_ros = [], []
            for ro, adv in kept:
                datums.append(build_datum_dict(ro, adv))
                kept_ros.append(ro)

            # KL 子采样锚(step-0 默认开,不接受"熔断二次触发才开")
            kl_info = {"n_anchored": 0, "avg_kl": None}
            if not args.no_kl_anchor and datums:
                ref_fn = mock_ref_logprob if not args.live else None  # live: 锚 sampler compute_logprobs
                kl_info = kl_anchor_adjust(datums, kept_ros, ref_fn, rng)

            # 更新
            if args.live and datums:
                td = [to_tinker_datum(d) for d in datums]
                import tinker
                fb = training_client.forward_backward_async(td, loss_fn=loss_fn,
                                                            loss_fn_config=loss_cfg)
                op = training_client.optim_step_async(tinker.AdamParams(
                    learning_rate=LR, grad_clip_norm=GRAD_CLIP))
                fb_out = fb.result(); op.result()
                metrics = {k: v for k, v in (fb_out.metrics or {}).items()
                           if k.startswith("e_")}        # MoE 路由健康度趋势
                sampler = training_client.save_weights_and_get_sampling_client(
                    name=f"grpo_step{step}")
            else:
                # mock:loss 数值仅形态校验(ratio=1 ⇒ L_IS = -mean(A))
                adv_all = np.concatenate([
                    np.asarray(d["loss_fn_inputs"]["advantages"], dtype=np.float32)
                    for d in datums]) if datums else np.zeros(1, dtype=np.float32)
                metrics = {"mock_loss": round(float(-adv_all.mean()), 6)}

            # 记账 + journal
            step_cost = sum(rollout_price(ro.cat) for g in batch for ro in g) \
                + (args.usd_per_step if datums else 0.0)
            st.est_cost += step_cost
            dt = time.time() - t0
            st.step_secs.append(dt)
            journal.write("step", step=step, n_problems=len(batch), n_rollouts=n_ro,
                          n_groups_kept=len(batch) - dropped, n_groups_dropped=dropped,
                          n_datums=len(datums), mean_reward=round(float(rewards.mean()), 4),
                          kl=kl_info, loss_fn=loss_fn, metrics=metrics,
                          est_cost_usd=round(st.est_cost, 2), sec=round(dt, 2))

            # 预算守卫
            if st.est_cost > args.budget_usd:
                journal.write("halt", step=step, reason="预算超限",
                              est_cost=round(st.est_cost, 2), budget=args.budget_usd)
                print(f"[halt] 预算超限 ${st.est_cost:.2f} > ${args.budget_usd}")
                break

            # 时钟预登记:前 5 步均 >12min → 砍到 40 步
            if step == 5 and steps > 40 and np.mean(st.step_secs) > SLOW_STEP_SEC:
                steps = 40
                journal.write("clock_cut", step=step, new_steps=40,
                              mean_step_sec=round(float(np.mean(st.step_secs)), 1))

            # probe→pass→save(写死顺序;熔断 = 回滚 + 停腿)
            if step % args.probe_every == 0 or step == steps:
                trip_now = args.mock_trip and step >= args.mock_trip_step
                if args.live:
                    raise NotImplementedError("live 探针:当前 sampler 贪心解码 anchor_set "
                                              "(~$0.6)→ rows 进 run_probe;未实装。")
                ok = probe_pass_save(st, journal, args.live, training_client,
                                     decode_rows, mock_trip=trip_now)
                if not ok:
                    rollback(st, journal, args.live, sc)
                    journal.write("leg_end", step=step, status="FUSED",
                                  est_cost=round(st.est_cost, 2),
                                  good_states=st.good_states)
                    return 2

    journal.write("leg_end", step=st.step, status="OK",
                  est_cost=round(st.est_cost, 2), n_rollouts=st.n_rollouts,
                  good_states=st.good_states)
    print(f"[leg_end] OK steps={st.step} rollouts={st.n_rollouts} "
          f"est=${st.est_cost:.2f} 回滚点={st.good_states}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cohort-size", type=int, default=HARD_COHORT_MAX)
    ap.add_argument("--k", type=int, default=HARD_K_MAX)
    ap.add_argument("--passes", type=int, default=HARD_PASS_MAX)
    ap.add_argument("--batch-problems", type=int, default=40,
                    help="每步题数(1000×2pass/50步=40)")
    ap.add_argument("--cispo", action="store_true", help="importance_sampling → cispo")
    ap.add_argument("--no-kl-anchor", action="store_true",
                    help="关 KL 子采样锚(默认开,coef 0.05/15%%)")
    ap.add_argument("--probe-every", type=int, default=10)
    ap.add_argument("--journal", type=Path,
                    default=ROOT / "data" / "replay" / "grpo_journal.jsonl")
    ap.add_argument("--mock-replay", type=Path, default=MINE_RESULTS)
    ap.add_argument("--mock-trip", action="store_true", help="演练:注入探针熔断")
    ap.add_argument("--mock-trip-step", type=int, default=10)
    ap.add_argument("--live", action="store_true", help="付费路径(默认 off,纯离线)")
    ap.add_argument("--budget-usd", type=float, default=100.0)
    ap.add_argument("--balance-usd", type=float, default=None,
                    help="Tinker console 实读余额;--live 需 ≥1.3×budget")
    ap.add_argument("--usd-per-step", type=float, default=1.5,
                    help="RL 步更新单价(红线:按 $0.75-2.4 重标,勿沿用 SFT 0.04)")
    ap.add_argument("--init-state", default="tinker://cad6ab5c-35bc-516c-90bd-"
                    "804b0a6166f5:train:0/weights/state_ep1")
    args = ap.parse_args()
    return run_leg(args)


if __name__ == "__main__":
    sys.exit(main())
