"""anchor_probe.py — RL 反遗忘熔断探针(PLAN §1.2-④ C2/C3)。

探针集构成【写死,不许改】:
  四健康类(cipher/gravity/numeral/unit_conversion)各 30
  + equation_numeric_deduce 已对 40(eq spot)
  + bit_manipulation 已对 40(bit spot)
  = 200 题,全部从 data/run012_holdout_decode.jsonl 的 ok=true 行按 id 排序确定性选取。
id 列表固化到 data/probes/anchor_set.json:首次运行生成,此后只读;要重建必须手动删文件。

熔断判定(任一触发 → 打印 FUSE_TRIPPED + exit 2):
  - 净回撤 ≥3%(200 题基线全对 → 当前 ≤194 对)
  - eq spot 回撤 ≥3 题(40 → ≤37)
附带预警指标(不熔断,只记账):平均生成长度漂移、复读 8-gram 率。

默认离线重放(--decode 指向任意解码 jsonl,行含 id/text[/gen/ok]);
--live 才走 tinker 贪心采样(~200×$0.0029≈$0.6),需 --adapter。

用法:
  python rl/anchor_probe.py                              # 自重放基线(自检,必 PASS)
  python rl/anchor_probe.py --decode data/replay/x.jsonl # 评一份新解码
  python rl/anchor_probe.py --live --adapter tinker://...
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.reasoning import extract_answer, metric_correct  # noqa: E402

DECODE_BASE = ROOT / "data" / "run012_holdout_decode.jsonl"
ANCHOR_SET = ROOT / "data" / "probes" / "anchor_set.json"

# ---- 构成写死(预登记,不许事后挪) ----
HEALTHY = {"cipher": 30, "gravity": 30, "numeral": 30, "unit_conversion": 30}
SPOT = {"equation_numeric_deduce": 40, "bit_manipulation": 40}
NET_DRAWDOWN_FUSE = 0.03   # 净回撤 ≥3% 熔断
EQ_SPOT_FUSE = 3           # eq 已对回撤 ≥3 题熔断(与净回撤同级)
NGRAM_N = 8                # 复读检测 n-gram 字窗
NGRAM_WARN = 0.30          # 单条复读率预警线(同 RFT C5 弃样线)


def _load_decode(path: Path) -> dict[str, dict]:
    rows = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rows[d["id"]] = d
    return rows


def build_or_load_anchor_set() -> dict:
    """固化探针集。存在即只读加载;不存在则从基线 decode 确定性生成。"""
    if ANCHOR_SET.exists():
        return json.loads(ANCHOR_SET.read_text())
    base = _load_decode(DECODE_BASE)
    want = dict(HEALTHY) | dict(SPOT)
    by_cat: dict[str, list[dict]] = {c: [] for c in want}
    for d in sorted(base.values(), key=lambda r: r["id"]):
        if d.get("ok") and d["cat"] in by_cat:
            by_cat[d["cat"]].append(d)
    ids, short = [], []
    for cat, n in want.items():
        pool = by_cat[cat]
        if len(pool) < n:
            short.append(f"{cat}: {len(pool)}<{n}")
        for d in pool[:n]:
            ids.append({"id": d["id"], "cat": cat, "truth": str(d["truth"]),
                        "base_gen": int(d.get("gen") or 0)})
    if short:
        raise SystemExit(f"探针池不足,违反预登记构成: {short}")
    spec = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": str(DECODE_BASE.relative_to(ROOT)),
        "composition": want,
        "fuse": {"net_drawdown_pct": NET_DRAWDOWN_FUSE * 100, "eq_spot_questions": EQ_SPOT_FUSE},
        "n": len(ids),
        "ids": ids,
    }
    ANCHOR_SET.parent.mkdir(parents=True, exist_ok=True)
    ANCHOR_SET.write_text(json.dumps(spec, ensure_ascii=False, indent=1))
    print(f"[anchor_probe] 固化探针集 {len(ids)} 题 -> {ANCHOR_SET}")
    return spec


def _repeat_ngram_rate(text: str, n: int = NGRAM_N) -> float:
    words = text.split()
    if len(words) < n + 1:
        return 0.0
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def score_decode(spec: dict, cur: dict[str, dict]) -> dict:
    """对一份解码评探针。cur 行至少含 id;优先 text 重打分,缺 text 退回 ok 字段。"""
    per_cat: dict[str, dict] = {}
    missing, rep_rates, rep_warn = [], [], 0
    gen_base_sum = gen_cur_sum = gen_n = 0
    n_correct = 0
    for a in spec["ids"]:
        cat = a["cat"]
        pc = per_cat.setdefault(cat, {"n": 0, "correct": 0})
        pc["n"] += 1
        row = cur.get(a["id"])
        if row is None:
            missing.append(a["id"])
            continue
        if "text" in row and row["text"]:
            ok = metric_correct(a["truth"], extract_answer(row["text"]))
            r = _repeat_ngram_rate(row["text"])
            rep_rates.append(r)
            rep_warn += r > NGRAM_WARN
        else:
            ok = bool(row.get("ok"))
        if ok:
            pc["correct"] += 1
            n_correct += 1
        g = row.get("gen")
        if g and a["base_gen"]:
            gen_base_sum += a["base_gen"]
            gen_cur_sum += int(g)
            gen_n += 1

    n = spec["n"]
    drawdown = (n - n_correct) / n  # 基线全对(构造即 ok=true)
    eq = per_cat.get("equation_numeric_deduce", {"n": 0, "correct": 0})
    eq_dd = eq["n"] - eq["correct"]
    bit = per_cat.get("bit_manipulation", {"n": 0, "correct": 0})
    reasons = []
    if drawdown >= NET_DRAWDOWN_FUSE:
        reasons.append(f"净回撤 {drawdown:.1%} ≥ {NET_DRAWDOWN_FUSE:.0%} ({n_correct}/{n})")
    if eq_dd >= EQ_SPOT_FUSE:
        reasons.append(f"eq spot 回撤 {eq_dd} 题 ≥ {EQ_SPOT_FUSE} ({eq['correct']}/{eq['n']})")
    return {
        "n": n, "correct": n_correct, "base_correct": n,
        "net_drawdown_pct": round(drawdown * 100, 2),
        "per_cat": per_cat,
        "eq_spot": {"base": eq["n"], "cur": eq["correct"], "drawdown": eq_dd},
        "bit_spot": {"base": bit["n"], "cur": bit["correct"], "drawdown": bit["n"] - bit["correct"]},
        "len_drift_pct": round((gen_cur_sum / gen_base_sum - 1) * 100, 2) if gen_base_sum else None,
        "repeat_ngram": {
            "mean": round(sum(rep_rates) / len(rep_rates), 4) if rep_rates else None,
            "n_gt_warn": rep_warn,
        },
        "missing": missing,
        "fuse": {"tripped": bool(reasons), "reasons": reasons},
    }


def live_decode(spec: dict, adapter: str, out_path: Path, max_tokens: int) -> dict[str, dict]:
    """--live:对探针集贪心采样并落盘解码行。付费(~$0.6),默认不走。"""
    import pandas as pd
    import tinker
    from transformers import AutoTokenizer
    from src.corpus import BASE_MODEL, tokenize_prompt
    from src.train_tinker import _load_env
    _load_env()
    df = pd.read_csv(ROOT / "data" / "holdout.csv").set_index("id")
    chat_tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    sc = tinker.ServiceClient()
    sampler = sc.create_sampling_client(base_model=BASE_MODEL, model_path=adapter)
    sp = tinker.SamplingParams(max_tokens=max_tokens, temperature=0.0, top_p=1.0)
    futs = []
    for a in spec["ids"]:
        ids = tokenize_prompt(str(df.loc[a["id"], "prompt"]), chat_tok)
        futs.append((a, sampler.sample(prompt=tinker.ModelInput.from_ints(ids),
                                       num_samples=1, sampling_params=sp)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = {}
    with open(out_path, "w") as f:  # 边收边落盘:session 死亡不丢钱
        for i, (a, fut) in enumerate(futs):
            seq = fut.result().sequences[0]
            text = chat_tok.decode(seq.tokens)
            row = {"id": a["id"], "cat": a["cat"], "truth": a["truth"],
                   "gen": len(seq.tokens), "text": text}
            rows[a["id"]] = row
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if (i + 1) % 20 == 0:
                print(f"  probe sampled {i + 1}/{len(futs)}")
    return rows


def run_probe(decode_rows: dict[str, dict]) -> dict:
    """供 grpo_loop 进程内调用:给解码行,返回报告(含 fuse 字段)。"""
    return score_decode(build_or_load_anchor_set(), decode_rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--decode", type=Path, default=DECODE_BASE,
                    help="待评解码 jsonl(默认=基线自重放,自检用)")
    ap.add_argument("--out", type=Path, default=None, help="报告 json 输出路径")
    ap.add_argument("--live", action="store_true", help="付费:tinker 贪心采样探针集")
    ap.add_argument("--adapter", default=None, help="--live 必填:tinker://... sampler 路径")
    ap.add_argument("--max-tokens", type=int, default=7680)
    args = ap.parse_args()

    spec = build_or_load_anchor_set()
    if args.live:
        if not args.adapter:
            ap.error("--live 需要 --adapter")
        out_dec = ROOT / "data" / "replay" / f"anchor_live_{int(time.time())}.jsonl"
        cur = live_decode(spec, args.adapter, out_dec, args.max_tokens)
        print(f"[anchor_probe] live 解码落盘 {out_dec}")
    else:
        cur = _load_decode(args.decode)

    rep = score_decode(spec, cur)
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    if rep["fuse"]["tripped"]:
        print("FUSE_TRIPPED")
        return 2
    print("PROBE_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
