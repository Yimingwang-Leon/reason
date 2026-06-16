"""RFT rollout 采集(PLAN §1.2-② 02 主攻线,弹药环节)。

从指定 tinker checkpoint 对题集采 K 条 @T,边采边落盘 jsonl,断点续采。
- crypt(cryptarithm_*)max_tokens=4096(营救 trace 实测 913-1628 字符,7680 只产截断垃圾);
  其余类 7680(与 grader 上限一致)。
- reward 用 eval_gate 同源判分链(src.reasoning.extract_answer/metric_correct,即
  eval_gate→run_one 的同一对原语;grader 同构 rel_tol=1e-2)。选样硬化在 rft_filter 做。
- 默认 MOCK(零网络零花费):回放 --mock-rollouts(mine_results.jsonl 格式)合成
  全 schema 记录,reward 仍真打分。--live 才碰 tinker。

落盘 schema(每行一条 rollout):
  {id, cat, sample_idx, tokens, text, logprobs, reward, stop_reason, gen_len, temperature}
prompt tokens 单独存 <out>.prompts.jsonl(每题一行 {id, prompt_tokens}),省 K 倍重复。

Run:
    python rl/rft_harvest.py --out /tmp/rft_dryrun/harvest.jsonl            # mock
    python rl/rft_harvest.py --live --checkpoint tinker://.../sampler_weights/final \
        --ids data/prey_train_candidates.json --out data/rft/harvest_r1.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.reasoning import extract_answer, metric_correct  # eval_gate 同源判分链

MAX_TOKENS_DEFAULT = 7680
MAX_TOKENS_CRYPT = 4096


def is_crypt(cat: str) -> bool:
    return cat.startswith("cryptarithm")


def max_tokens_for(cat: str) -> int:
    return MAX_TOKENS_CRYPT if is_crypt(cat) else MAX_TOKENS_DEFAULT


def load_truth(csv_path: Path) -> dict[str, dict]:
    """id -> {cat, answer, prompt}。主 csv 缺 id 时回退 data/train.csv(cat 现场 classify)。"""
    import pandas as pd
    out: dict[str, dict] = {}
    df = pd.read_csv(csv_path)
    has_cat = "cat" in df.columns
    for _, r in df.iterrows():
        cat = r["cat"] if has_cat else None
        out[str(r["id"])] = {"cat": cat, "answer": str(r["answer"]), "prompt": str(r["prompt"])}
    return out


def lookup(truth: dict[str, dict], pid: str) -> dict:
    rec = truth.get(pid)
    if rec is None:
        import pandas as pd
        df = pd.read_csv(ROOT / "data" / "train.csv")
        row = df[df["id"].astype(str) == pid]
        if row.empty:
            raise KeyError(f"problem id {pid} not found in csv nor data/train.csv")
        rec = {"cat": None, "answer": str(row.iloc[0]["answer"]), "prompt": str(row.iloc[0]["prompt"])}
        truth[pid] = rec
    if rec["cat"] is None:
        from src.problems import classify
        rec["cat"] = classify(rec["prompt"])
    return rec


def load_ids(path: Path) -> list[str]:
    """猎物名单:.json 的 id 列表(或含 id/pid 字段的对象列表),或 jsonl。"""
    text = path.read_text().strip()
    items = json.loads(text) if text.startswith("[") else [json.loads(l) for l in text.splitlines() if l.strip()]
    ids = []
    for it in items:
        ids.append(it if isinstance(it, str) else str(it.get("id") or it.get("pid")))
    return ids


def existing_counts(out_path: Path) -> dict[str, int]:
    """断点续采:统计已落盘 rollout 数(坏行跳过不致命)。"""
    counts: dict[str, int] = defaultdict(int)
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    counts[json.loads(line)["id"]] += 1
                except Exception:
                    continue
    return counts


def grade(answer: str, text: str) -> int:
    return int(metric_correct(answer, extract_answer(text)))


def _write(f, rec: dict) -> None:
    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    f.flush()


# ---------------------------------------------------------------- mock 路径

def _pseudo_tokens(seed_text: str, n: int) -> list[int]:
    """确定性伪 token 序列(mock 专用,长度≈len(text)/4 模拟真实 token 数)。"""
    h = hash(seed_text) & 0x7FFFFFFF
    return [(h + j * 2654435761) % 131072 for j in range(n)]


def _corrupt_box(text: str) -> str:
    """把最后一个 \\boxed{...} 内容污染掉,制造 reward=0 的负样本(mock 专用)。"""
    i = text.rfind("\\boxed{")
    if i < 0:
        return text + "\\boxed{__bad__}"
    j = text.find("}", i)
    return text[: i + len("\\boxed{")] + "__bad__" + text[j if j >= 0 else len(text):]


def run_mock(args, truth: dict[str, dict]) -> None:
    rows = [json.loads(l) for l in open(args.mock_rollouts) if l.strip()]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    have = existing_counts(out_path)
    prompts_path = Path(str(out_path) + ".prompts.jsonl")
    have_prompts = set(existing_counts(prompts_path))
    n_new = 0
    with open(out_path, "a") as f, open(prompts_path, "a") as pf:
        for row in rows:
            pid, k = row["pid"], int(row["K"])
            rec = lookup(truth, pid)
            cat = rec["cat"]
            need = k - have.get(pid, 0)
            if need <= 0:
                continue
            if pid not in have_prompts:
                _write(pf, {"id": pid, "prompt_tokens": _pseudo_tokens(rec["prompt"], 1024)})
                have_prompts.add(pid)
            for s in range(k - need, k):
                pos = s < int(row["n_correct"])
                text = row["trace"] if pos else _corrupt_box(row["trace"])
                n_tok = min(max(8, len(text) // 4), max_tokens_for(cat))
                # 最后一条负样本标 length,锻炼 filter 的 stop-only 闸
                stop = "length" if (not pos and s == k - 1) else "stop"
                _write(f, {
                    "id": pid, "cat": cat, "sample_idx": s,
                    "tokens": _pseudo_tokens(text, n_tok),
                    "text": text,
                    "logprobs": [round(-0.3 - 0.001 * (j % 7), 5) for j in range(n_tok)],
                    "reward": grade(rec["answer"], text),
                    "stop_reason": stop, "gen_len": n_tok,
                    "temperature": args.temperature,
                })
                n_new += 1
    print(f"[mock] {len(rows)} problems replayed, {n_new} new rollouts -> {out_path}")


# ---------------------------------------------------------------- live 路径

def run_live(args, truth: dict[str, dict], ids: list[str]) -> None:
    import tinker
    from transformers import AutoTokenizer
    from src.corpus import BASE_MODEL, tokenize_prompt
    from src.train_tinker import _load_env

    _load_env()
    chat_tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    import time as _time

    def _retry(fn, what, n=5):
        """网络抖动(ConnectError/RST)重试;最后一次失败才抛。"""
        for i in range(n):
            try:
                return fn()
            except Exception as e:
                if i == n - 1:
                    raise
                print(f"  [retry {i+1}/{n}] {what}: {type(e).__name__}: {str(e)[:80]}")
                _time.sleep(min(15 * (i + 1), 60))

    sc = _retry(lambda: tinker.ServiceClient(), "ServiceClient")
    sampler = _retry(lambda: sc.create_sampling_client(
        base_model=BASE_MODEL, model_path=args.checkpoint), "create_sampling_client")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    have = existing_counts(out_path)
    prompts_path = Path(str(out_path) + ".prompts.jsonl")
    have_prompts = set(existing_counts(prompts_path))

    todo = [(pid, args.k - have.get(pid, 0)) for pid in ids if args.k - have.get(pid, 0) > 0]
    print(f"[live] {len(todo)}/{len(ids)} problems need sampling (K={args.k} T={args.temperature})")

    with open(out_path, "a") as f, open(prompts_path, "a") as pf:
        for w0 in range(0, len(todo), args.wave):
            wave = todo[w0:w0 + args.wave]
            futures = []
            for pid, need in wave:
                rec = lookup(truth, pid)
                p_ids = tokenize_prompt(rec["prompt"], chat_tok)
                if pid not in have_prompts:
                    _write(pf, {"id": pid, "prompt_tokens": p_ids})
                    have_prompts.add(pid)
                sp = tinker.SamplingParams(
                    max_tokens=max_tokens_for(rec["cat"]),
                    temperature=args.temperature, top_p=args.top_p)
                # 教科书写法(cookbook rl_loop):先发全部 future,再逐个收割
                # 提交本身也会被网络 RST 打断 → 同样要重试,不能裸抛(06-12 实测教训)
                try:
                    fut = _retry(lambda p=p_ids, s=sp, n=need: sampler.sample(
                        prompt=tinker.ModelInput.from_ints(p), num_samples=n,
                        sampling_params=s), f"submit {pid}", n=4)
                except Exception as e:
                    print(f"  [warn] {pid} submit failed for good: {type(e).__name__}: {e}")
                    continue
                futures.append((pid, rec, fut))
            for pid, rec, fut in futures:
                try:
                    resp = fut.result()
                except Exception as e:  # 单题失败不拖死整场;断点续采会补
                    print(f"  [warn] {pid} sample failed: {type(e).__name__}: {e}")
                    continue
                base_idx = have.get(pid, 0)
                for s, seq in enumerate(resp.sequences):
                    text = chat_tok.decode(seq.tokens)
                    _write(f, {
                        "id": pid, "cat": rec["cat"], "sample_idx": base_idx + s,
                        "tokens": list(seq.tokens),
                        "text": text,
                        # float32 logprobs 圆到 5 位:IS ratio 误差 <1e-4,体积省一半
                        "logprobs": [round(float(x), 5) for x in seq.logprobs],
                        "reward": grade(rec["answer"], text),
                        "stop_reason": str(seq.stop_reason),
                        "gen_len": len(seq.tokens), "temperature": args.temperature,
                    })
                have[pid] = base_idx + len(resp.sequences)
            print(f"  wave {w0 // args.wave + 1}: {min(w0 + args.wave, len(todo))}/{len(todo)} done")
    print(f"[live] harvest complete -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", default=None, help="tinker://.../sampler_weights/... (--live 必填)")
    ap.add_argument("--csv", default=str(ROOT / "data" / "train_split.csv"))
    ap.add_argument("--ids", default=None, help="猎物名单 json/jsonl;缺省 = csv 全部 id(live 慎用)")
    ap.add_argument("--out", required=True, help="rollout jsonl 输出(append,断点续采)")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--wave", type=int, default=32, help="并发 future 波大小")
    ap.add_argument("--live", action="store_true", help="真调 tinker(花钱);默认 mock 离线")
    ap.add_argument("--mock-rollouts", default=str(ROOT / "data" / "probes" / "mine_results.jsonl"))
    args = ap.parse_args()

    truth = load_truth(Path(args.csv))
    if not args.live:
        run_mock(args, truth)
        return
    if not args.checkpoint:
        ap.error("--live requires --checkpoint")
    ids = load_ids(Path(args.ids)) if args.ids else list(truth)
    run_live(args, truth, ids)


if __name__ == "__main__":
    main()
