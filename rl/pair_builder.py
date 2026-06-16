"""pair_builder.py — DPO 附属臂构对(PLAN 附属臂;数据构造层 ONLY,训练侧见文末 TODO)。

从 rft_harvest.py 落盘构 preference pair:
  chosen   = 同题 reward=1 且 stop 完整的 rollout(短者优先,与 rft_filter 同口径);
  rejected = 同题贪心错解(--rejected decode jsonl,行含 id/text[/tokens/ok]);
             离线 dry-run 可 --rejected-source harvest 用同题 reward=0 采样错解顶替
             (语义占位:正式构对必须换贪心 decode,贪心错才是 LB 上真实的失败模式)。

构造规则(预登记):
  - 公共前缀置零:prompt 段 + chosen/rejected 完成段的最长公共 token 前缀 mask=0,
    DPO 梯度只落在分歧点之后(公共前缀两边 logprob 差恒为 0,纯噪声省 token)。
  - crypt(cryptarithm_*)完成段预算 3072:chosen 超长直接弃对(截断的"对"不完整,
    box 都没收尾,不能当 chosen);rejected 超长截到 3072(rejected 本就不要求完整)。
  - 每题 ≤--max-pairs-per-problem(默认 1)对;chosen 取最短合格,稳定可复现。
  - rejected 无 tokens 时用 --tokenizer 现场编码(本地缓存,无网络);
    无 tokenizer 又无 tokens → 跳过并计数,绝不静默。

输出 jsonl(每行一对):
  {problem_id, category, prompt_tokens,
   chosen:  {tokens, mask, text},      # tokens=完成段;mask 同长,公共前缀=0
   rejected:{tokens, mask, text},
   divergence_idx,                      # 完成段分歧起点(=公共前缀长)
   meta: {chosen_gen, rejected_gen, rejected_truncated}}

Run(离线 dry-run):
    python rl/pair_builder.py --harvest /tmp/rft_dryrun/harvest.jsonl \
        --rejected-source harvest --out /tmp/rft_dryrun/pairs.jsonl
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

CRYPT_COMPLETION_BUDGET = 3072


def is_crypt(cat: str) -> bool:
    return cat.startswith("cryptarithm")


def common_prefix_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def load_truth(csv_path: Path) -> dict[str, str]:
    import pandas as pd
    df = pd.read_csv(csv_path)
    return {str(r["id"]): str(r["answer"]) for _, r in df.iterrows()}


def pick_chosen(rolls: list[dict]) -> dict | None:
    """同题合格 chosen:reward=1 且 stop;最短优先(与 rft_filter 同口径)。"""
    ok = [r for r in rolls if r.get("reward") and r.get("stop_reason") == "stop"]
    ok.sort(key=lambda r: (r.get("gen_len", len(r.get("tokens", []))), r.get("sample_idx", 0)))
    return ok[0] if ok else None


def pick_rejected_harvest(rolls: list[dict]) -> dict | None:
    """dry-run 占位:同题 reward=0 采样错解,最短优先。"""
    bad = [r for r in rolls if not r.get("reward")]
    bad.sort(key=lambda r: (r.get("gen_len", len(r.get("tokens", []))), r.get("sample_idx", 0)))
    return bad[0] if bad else None


def rejected_from_decode(row: dict, truth: str | None, tok) -> dict | None:
    """贪心 decode 行 → rejected 候选;答对/无法判定错 → None(不构对)。"""
    text = row.get("text") or ""
    if truth is None or not text:
        return None
    if metric_correct(truth, extract_answer(text)):
        return None  # 贪心已对,没有 rejected
    tokens = row.get("tokens")
    if tokens is None:
        if tok is None:
            return None  # 调用方计数 skip_no_tokens
        tokens = tok.encode(text, add_special_tokens=False)
    return {"tokens": list(tokens), "text": text,
            "gen_len": len(tokens), "stop_reason": str(row.get("stop_reason", "stop"))}


def build_pair(pid: str, cat: str, prompt_tokens: list[int],
               chosen: dict, rejected: dict) -> dict | None:
    c_tok, r_tok = list(chosen["tokens"]), list(rejected["tokens"])
    truncated = False
    if is_crypt(cat):
        if len(c_tok) > CRYPT_COMPLETION_BUDGET:
            return None  # 截断的 chosen 不完整,弃对
        if len(r_tok) > CRYPT_COMPLETION_BUDGET:
            r_tok = r_tok[:CRYPT_COMPLETION_BUDGET]
            truncated = True
    div = common_prefix_len(c_tok, r_tok)
    if div >= len(c_tok) or div >= len(r_tok):
        return None  # 一方是另一方前缀,无分歧段可学
    c_mask = [0] * div + [1] * (len(c_tok) - div)
    r_mask = [0] * div + [1] * (len(r_tok) - div)
    return {
        "problem_id": pid, "category": cat,
        "prompt_tokens": prompt_tokens,
        "chosen": {"tokens": c_tok, "mask": c_mask, "text": chosen.get("text", "")},
        "rejected": {"tokens": r_tok, "mask": r_mask, "text": rejected.get("text", "")},
        "divergence_idx": div,
        "meta": {"chosen_gen": len(c_tok), "rejected_gen": len(r_tok),
                 "rejected_truncated": truncated},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--harvest", required=True, help="rft_harvest rollout jsonl(chosen 源)")
    ap.add_argument("--prompts", default=None, help="侧车,默认 <harvest>.prompts.jsonl")
    ap.add_argument("--rejected-source", choices=("decode", "harvest"), default="decode",
                    help="decode=贪心错解 jsonl(正式);harvest=同题采样错解(dry-run 占位)")
    ap.add_argument("--rejected", default=None, help="贪心 decode jsonl(--rejected-source decode 必填)")
    ap.add_argument("--csv", default=str(ROOT / "data" / "train_split.csv"))
    ap.add_argument("--tokenizer", default=None,
                    help="decode 行缺 tokens 时现场编码用(如 src.corpus.BASE_MODEL);缺省不编码只跳过")
    ap.add_argument("--max-pairs-per-problem", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.rejected_source == "decode" and not args.rejected:
        ap.error("--rejected-source decode 需要 --rejected")

    prompts_path = Path(args.prompts or (args.harvest + ".prompts.jsonl"))
    prompt_tok = {d["id"]: d["prompt_tokens"] for d in load_jsonl(prompts_path)}
    by_pid: dict[str, list[dict]] = defaultdict(list)
    for r in load_jsonl(Path(args.harvest)):
        by_pid[r["id"]].append(r)

    truth = load_truth(Path(args.csv)) if args.rejected_source == "decode" else {}
    decode_rows = ({d["id"]: d for d in load_jsonl(Path(args.rejected))}
                   if args.rejected_source == "decode" else {})
    tok = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    n = defaultdict(int)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for pid in sorted(by_pid):
            rolls = by_pid[pid]
            cat = rolls[0]["cat"]
            chosen = pick_chosen(rolls)
            if chosen is None:
                n["skip_no_chosen"] += 1
                continue
            if args.rejected_source == "harvest":
                rejected = pick_rejected_harvest(rolls)
                why = "skip_no_rejected"
            else:
                row = decode_rows.get(pid)
                rejected = rejected_from_decode(row, truth.get(pid), tok) if row else None
                why = ("skip_no_tokens" if row and rejected is None and tok is None
                       and row.get("tokens") is None and row.get("text")
                       and not metric_correct(truth.get(pid, ""), extract_answer(row.get("text") or ""))
                       else "skip_greedy_correct_or_missing")
            if rejected is None:
                n[why] += 1
                continue
            pair = build_pair(pid, cat, prompt_tok.get(pid, []), chosen, rejected)
            if pair is None:
                n["skip_degenerate_or_crypt_long"] += 1
                continue
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            n["pairs"] += 1
            n[f"pairs_{cat}"] += 1

    print(json.dumps(dict(n), indent=1, ensure_ascii=False))
    print(f"-> {out_path}")


# ---------------------------------------------------------------------------
# TODO(训练侧,本骨架故意不实装 —— 解锁线/预算拍板后再做):
#  1. DPO loss 在 tinker 上的落法:无原生 dpo loss_fn → 用 importance_sampling 伪装
#     (chosen advantage=+beta、rejected=-beta,公共前缀 mask=0 已在数据层做掉),或
#     forward 两次取 logprob 差自算梯度;ref policy = run-011 final(compute_logprobs)。
#  2. beta 扫描预算、与 RFT 主臂的先后次序(pair 数 < 500 不值得开臂)。
#  3. rejected 必须换真贪心 decode(--rejected-source decode);采样错解 ≠ 贪心失败模式。
#  4. anchor_probe 熔断同样适用:每 N 步重放探针集,规格同 grpo_loop。
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
