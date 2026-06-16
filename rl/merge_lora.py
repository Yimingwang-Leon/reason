#!/usr/bin/env python3
"""LoRA ΔW 空间合并 + 五重诊断闸(提案 04 §2.3/§5/§10)。

路径:
  低秩精确(默认): QR([s1·B1,√λ·s2·B2]) × QR([A1;√λ·A2]ᵀ) → 64×64 核 SVD → 截 32,免稠密化。
  稠密(--dare/--ties): 逐模块流式 fp32 ΔW,DARE p 置零×1/(1-p) / TIES sign-election 后 disjoint-mean,
                        svd_lowrank(q=96,niter=6) 截 32(ρ 分母用精确 ‖D‖²_F)。

五闸(任一 FAIL → exit 1 不打包):
  G1 键审计: 宿主键 100% 保留(含全部 MoE up/down),多余键 0
  G2 dtype : 逐张量与宿主一致
  G3 范数  : 全局+逐模块 ‖ΔW‖_F ∈ 宿主±5%(合并时超闸模块已重标回宿主范数)
  G4 能量  : 全局 ρ=Σ‖ΔW_r32‖²/Σ‖D‖² ≥ 0.90,逐模块直方图落盘 merge_diag_*.json
  G5 NaN/shape: 输出全有限,shape 与宿主一致
方向诊断(只记录): 双亲逐模块 ΔW 余弦;TIES 符号冲突率。

用法: python rl/merge_lora.py --host DIR --inject DIR --lam 0.25 [--dare 0.9] [--ties] --out DIR
"""
import argparse, json, math, os, shutil, sys, time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SUF_A, SUF_B = ".lora_A.weight", ".lora_B.weight"
NORM_TOL = 0.05   # 范数闸 ±5%
RHO_MIN = 0.90    # 能量闸
RANK = 32


def load_cfg(d):
    with open(os.path.join(d, "adapter_config.json")) as f:
        return json.load(f)


def lr_norm(B, A, s=1.0):
    """‖s·B@A‖_F via 32×32 trace trick(免稠密)。"""
    return s * math.sqrt(max(torch.sum((B.T @ B) * (A @ A.T)).item(), 0.0))


def lr_dot(B1, A1, B2, A2):
    """<B1A1, B2A2>_F via 小核 trace trick。"""
    return torch.sum((B1.T @ B2) * (A1 @ A2.T)).item()


def merge_module(B1, A1, B2, A2, s1, s2, args, gen):
    """返回 (B_new, A_new, diag dict)。B/A 为 fp32,ΔW 尺度 = s·B@A。"""
    d = {}
    n1sq = max(lr_dot(B1, A1, B1, A1), 0.0) * s1 * s1
    n2sq = max(lr_dot(B2, A2, B2, A2), 0.0) * s2 * s2
    dot12 = lr_dot(B1, A1, B2, A2) * s1 * s2
    d["cos"] = dot12 / math.sqrt(max(n1sq * n2sq, 1e-30)) if n1sq > 0 and n2sq > 0 else 0.0
    d["norm_host"] = math.sqrt(n1sq)

    dense = args.dare is not None or args.ties
    if dense:
        v1 = s1 * (B1 @ A1)
        v2 = args.lam * s2 * (B2 @ A2)
        nz = (v1 != 0) & (v2 != 0)
        d["sign_conflict"] = ((torch.sign(v1) != torch.sign(v2)) & nz).sum().item() / max(nz.sum().item(), 1)
        if args.dare is not None:
            keep = (torch.rand(v2.shape, generator=gen) >= args.dare).to(v2.dtype)
            v2 = v2 * keep / (1.0 - args.dare)
        if args.ties:  # sign-election 后 disjoint-mean(lam 已并入 v2,lam=1.0 即 50/50 等权)
            gamma = torch.sign(v1 + v2)
            m1 = (torch.sign(v1) == gamma) & (v1 != 0)
            m2 = (torch.sign(v2) == gamma) & (v2 != 0)
            cnt = (m1.to(v1.dtype) + m2.to(v1.dtype)).clamp(min=1.0)
            D = (v1 * m1 + v2 * m2) / cnt
        else:
            D = v1 + v2
        den = D.pow(2).sum().item()
        q = min(96, min(D.shape))
        U, S, V = torch.svd_lowrank(D, q=q, niter=6)
        B_new = U[:, :RANK] * S[:RANK]
        A_new = V[:, :RANK].T.contiguous()
        num = (S[:RANK] ** 2).sum().item()
        d["rho"] = min(num / max(den, 1e-30), 1.0)
    else:  # 低秩精确路径
        rl = math.sqrt(args.lam)
        QB, RB = torch.linalg.qr(torch.cat([s1 * B1, rl * s2 * B2], dim=1))
        QA, RA = torch.linalg.qr(torch.cat([A1, rl * A2], dim=0).T)
        U, S, Vh = torch.linalg.svd(RB @ RA.T)
        B_new = QB @ (U[:, :RANK] * S[:RANK])
        A_new = (QA @ Vh.T[:, :RANK]).T.contiguous()
        den = (S ** 2).sum().item()
        d["rho"] = min((S[:RANK] ** 2).sum().item() / max(den, 1e-30), 1.0)

    # 范数闸: 超 ±5% 的模块重标回宿主范数(g 乘在 B 上)
    nm = lr_norm(B_new, A_new, s1)
    d["norm_merged"] = nm
    nh = d["norm_host"]
    if nh > 0 and (nm < (1 - NORM_TOL) * nh or nm > (1 + NORM_TOL) * nh):
        B_new = B_new * (nh / max(nm, 1e-30))
        d["rescaled"] = True
    else:
        d["rescaled"] = False
    d["norm_final"] = lr_norm(B_new, A_new, s1)
    return B_new, A_new, d


def build(args):
    cfg_h, cfg_i = load_cfg(args.host), load_cfg(args.inject)
    s1 = cfg_h["lora_alpha"] / cfg_h["r"]
    s2 = cfg_i["lora_alpha"] / cfg_i["r"]
    gen = torch.Generator().manual_seed(args.seed)
    fh = safe_open(os.path.join(args.host, "adapter_model.safetensors"), framework="pt")
    fi = safe_open(os.path.join(args.inject, "adapter_model.safetensors"), framework="pt")
    hkeys = list(fh.keys())
    # 注入键翻译:--inject-key-map "inject_prefix=host_prefix",把 inject 键改写到宿主命名空间
    kmap = {}
    if args.inject_key_map:
        old, new = args.inject_key_map.split("=", 1)
        kmap = {k.replace(old, new, 1): k for k in fi.keys()}
    else:
        kmap = {k: k for k in fi.keys()}
    ikeys = set(kmap)
    mods = [k[: -len(SUF_A)] for k in hkeys if k.endswith(SUF_A)]
    print(f"[build] host={args.host} s1={s1} | inject={args.inject} s2={s2} | "
          f"{len(hkeys)} keys = {len(mods)} pairs | lam={args.lam} dare={args.dare} ties={args.ties}")

    out, per_mod, passthrough = {}, {}, 0
    t0 = time.time()
    for i, m in enumerate(mods):
        kA, kB = m + SUF_A, m + SUF_B
        tA, tB = fh.get_tensor(kA), fh.get_tensor(kB)
        if kA not in ikeys or kB not in ikeys:  # 宿主独有键原样保留
            out[kA], out[kB] = tA, tB
            passthrough += 1
            continue
        B1, A1 = tB.float(), tA.float()
        B2, A2 = fi.get_tensor(kmap[kB]).float(), fi.get_tensor(kmap[kA]).float()
        B_new, A_new, d = merge_module(B1, A1, B2, A2, s1, s2, args, gen)
        out[kB] = (B_new / s1).to(tB.dtype)   # 写回宿主尺度约定(ΔW = s1·B·A)
        out[kA] = A_new.to(tA.dtype)
        per_mod[m] = d
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(mods)} ({time.time()-t0:.0f}s)")

    os.makedirs(args.out, exist_ok=True)
    save_file({k: out[k] for k in hkeys}, os.path.join(args.out, "adapter_model.safetensors"))
    shutil.copyfile(os.path.join(args.host, "adapter_config.json"),
                    os.path.join(args.out, "adapter_config.json"))  # 宿主 config 字节不动
    print(f"[build] written {args.out} ({len(hkeys)} keys, {passthrough} passthrough) in {time.time()-t0:.0f}s")
    return per_mod


def run_gates(args, per_mod):
    """对已写盘产物跑五闸 + 落盘诊断 JSON。返回 True=全 PASS。"""
    cfg_h = load_cfg(args.host)
    s1 = cfg_h["lora_alpha"] / cfg_h["r"]
    fh = safe_open(os.path.join(args.host, "adapter_model.safetensors"), framework="pt")
    fo = safe_open(os.path.join(args.out, "adapter_model.safetensors"), framework="pt")
    hk, ok = set(fh.keys()), set(fo.keys())
    gates, fails = {}, []

    # G1 键审计
    missing, extra = sorted(hk - ok), sorted(ok - hk)
    moe = lambda ks: sum(1 for k in ks if ".experts." in k and ("up_proj" in k or "down_proj" in k))
    g1 = {"missing": len(missing), "extra": len(extra), "moe_host": moe(hk), "moe_out": moe(ok)}
    g1["pass"] = not missing and not extra and g1["moe_out"] == g1["moe_host"]
    gates["G1_keys"] = g1
    if not g1["pass"]:
        fails.append(f"G1 keys: missing={missing[:3]} extra={extra[:3]} moe {g1['moe_out']}/{g1['moe_host']}")

    # G2 dtype + G5 NaN/shape(同一遍流式扫描)
    bad_dt, bad_shape, bad_nan = [], [], []
    for k in sorted(hk & ok):
        sh, so = fh.get_slice(k), fo.get_slice(k)
        if so.get_dtype() != sh.get_dtype():
            bad_dt.append(k)
        if so.get_shape() != sh.get_shape():
            bad_shape.append(k)
        if not torch.isfinite(fo.get_tensor(k)).all():
            bad_nan.append(k)
    gates["G2_dtype"] = {"bad": len(bad_dt), "pass": not bad_dt}
    gates["G5_nan_shape"] = {"bad_shape": len(bad_shape), "bad_nan": len(bad_nan),
                             "pass": not bad_shape and not bad_nan}
    if bad_dt:
        fails.append(f"G2 dtype: {bad_dt[:3]}")
    if bad_shape or bad_nan:
        fails.append(f"G5: shape={bad_shape[:3]} nan={bad_nan[:3]}")

    # G3 范数(对写盘后的 bf16 产物重算,逐模块+全局)
    viol, hsq, osq = [], 0.0, 0.0
    mods = [k[: -len(SUF_A)] for k in sorted(hk) if k.endswith(SUF_A)]
    for m in mods:
        kA, kB = m + SUF_A, m + SUF_B
        nh = lr_norm(fh.get_tensor(kB).float(), fh.get_tensor(kA).float(), s1)
        no = lr_norm(fo.get_tensor(kB).float(), fo.get_tensor(kA).float(), s1)
        hsq += nh * nh
        osq += no * no
        if nh > 0 and abs(no / nh - 1.0) > NORM_TOL:
            viol.append((m, nh, no))
        if m in per_mod:
            per_mod[m]["norm_written"] = no
    gn_h, gn_o = math.sqrt(hsq), math.sqrt(osq)
    g3 = {"global_host": gn_h, "global_out": gn_o, "global_ratio": gn_o / gn_h,
          "module_violations": len(viol),
          "pass": abs(gn_o / gn_h - 1.0) <= NORM_TOL and not viol}
    gates["G3_norm"] = g3
    if not g3["pass"]:
        fails.append(f"G3 norm: global_ratio={g3['global_ratio']:.4f} viol={len(viol)} e.g. {viol[:2]}")

    # G4 能量闸: 全局 ρ(能量加权)≥ 0.90;逐模块直方图
    if per_mod:
        # ρ_m = ‖r32‖²/‖D‖² ⇒ ‖D‖² = norm_merged²/ρ_m(norm_merged = rescale 前的 ‖r32‖)
        num = sum(d["norm_merged"] ** 2 for d in per_mod.values())
        den = sum(d["norm_merged"] ** 2 / max(d["rho"], 1e-12) for d in per_mod.values())
        rho_g = num / max(den, 1e-30)
        rhos = [d["rho"] for d in per_mod.values()]
        hist = [0] * 12  # [<0.5, 0.5-0.9 step .05 ×8, 0.90-0.95, 0.95-0.99, ≥0.99] → 简化 12 桶
        edges = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.99, 0.999]
        for r in rhos:
            hist[sum(1 for e in edges if r >= e)] += 1
        g4 = {"rho_global": rho_g, "rho_min": min(rhos), "rho_mean": sum(rhos) / len(rhos),
              "hist_edges": edges, "hist": hist, "pass": rho_g >= RHO_MIN}
    else:
        g4 = {"rho_global": 1.0, "pass": True, "note": "no merged modules"}
    gates["G4_energy"] = g4
    if not g4["pass"]:
        fails.append(f"G4 energy: rho_global={g4['rho_global']:.4f} < {RHO_MIN}")

    # 方向诊断(只记录)
    if per_mod:
        coss = [d["cos"] for d in per_mod.values()]
        gates["direction"] = {
            "cos_mean": sum(coss) / len(coss), "cos_min": min(coss), "cos_max": max(coss),
            "cos_neg_frac": sum(1 for c in coss if c < 0) / len(coss),
            "sign_conflict_mean": (sum(d.get("sign_conflict", 0.0) for d in per_mod.values()) / len(per_mod))
            if args.ties or args.dare is not None else None,
            "rescaled_modules": sum(1 for d in per_mod.values() if d["rescaled"]),
        }

    tag = os.path.basename(os.path.normpath(args.out))
    diag_path = os.path.join(args.diag_dir, f"merge_diag_{tag}.json")
    with open(diag_path, "w") as f:
        json.dump({"args": vars(args), "gates": gates, "per_module": per_mod}, f, indent=1)
    print(f"[diag] {diag_path}")
    for name in ["G1_keys", "G2_dtype", "G3_norm", "G4_energy", "G5_nan_shape"]:
        print(f"  {name}: {'PASS' if gates[name]['pass'] else 'FAIL'} "
              f"{ {k: v for k, v in gates[name].items() if k not in ('pass', 'hist', 'hist_edges')} }")
    if "direction" in gates:
        print(f"  direction: {gates['direction']}")
    if fails:
        print("[GATES FAIL]")
        for x in fails:
            print("  " + x)
        return False
    print("[GATES PASS]")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--inject", required=True)
    ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--dare", type=float, default=None, help="DARE 置零概率 p")
    ap.add_argument("--ties", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inject-key-map", default=None, help="inject_prefix=host_prefix")
    ap.add_argument("--diag-dir", default=str(__import__("pathlib").Path(__file__).resolve().parent.parent / "data" / "probes"))
    args = ap.parse_args()
    per_mod = build(args)
    if not run_gates(args, per_mod):
        sys.exit(1)


if __name__ == "__main__":
    main()
