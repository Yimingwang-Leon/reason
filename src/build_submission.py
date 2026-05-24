"""Build submission.zip from a Tinker run.

Pipeline (huikang's post-processing path):
  1. Download adapter tarball from Tinker.
  2. Rename keys: base_model.model.model -> base_model.model.backbone
     (vLLM expects 'backbone' for NemotronH; without this, LoRA loads as no-op.)
  3. MoE expert unfuse: .experts.w1 -> per-expert .experts.{i}.up_proj;
     .experts.w2 -> per-expert .experts.{i}.down_proj.
  4. Mamba in_proj reconstruction:
     - If a layer has BOTH mixer.gate_proj AND mixer.x_proj: SVD merge
       (combined rank-2r matrix -> compressed back to rank-r in_proj).
     - If a Mamba layer has ONLY mixer.gate_proj (typical for current
       NemotronH Tinker output — Tinker splits z out but leaves x packed):
       lift gate LoRA into in_proj by zero-padding rows [4096:10304]
       (the x/B/C/dt portions).
  5. Skip empty .w3 experts and anything else that has no usable weight.
  6. Write adapter_config.json with the exact target_modules present in the
     post-processed safetensors.
  7. Zip as /tmp/submission.zip (Kaggle requires this filename) and optionally
     submit via the kaggle CLI.

Reference: external/nemotron-huikang/notebook_tinker.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tarfile
import tempfile
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = ROOT / "training"
ENV_PATH = ROOT / "env.json"
BASE_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
# NemotronH Mamba-2 packed in_proj output dim — confirmed via HF metadata
# (shape (10304, 2688) for all 23 Mamba layers in 30B-A3B-BF16).
# Layout: [z(4096) | x(4096) | B(1024) | C(1024) | dt(64)] = 10304.
MAMBA_IN_PROJ_DIM = 10304
MAMBA_Z_DIM = 4096  # rows [0:4096] in packed in_proj (the gate / z portion)
MAMBA_X_DIM = 4096  # rows [4096:8192]


def download_from_tinker(tinker_path: str, out_dir: Path) -> None:
    env = json.loads(ENV_PATH.read_text())
    os.environ["TINKER_API_KEY"] = env["TINKER_API_KEY"]
    import tinker
    print(f"[download] tinker SDK {tinker.__version__}")
    if not tinker.__version__.startswith("0.16"):
        print("[download] WARN: tinker 0.22+ has a download URL regression; use 0.16.1")
    sc = tinker.ServiceClient()
    rest = sc.create_rest_client()
    print(f"[download] requesting {tinker_path}")
    url = rest.get_checkpoint_archive_url_from_tinker_path(tinker_path).result().url

    with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp:
        tar_path = tmp.name
    urllib.request.urlretrieve(url, tar_path)
    print(f"[download] {os.path.getsize(tar_path) / 1e6:.1f} MB")
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tar:
        tar.extractall(out_dir)
    os.unlink(tar_path)
    contents = list(out_dir.iterdir())
    if len(contents) == 1 and contents[0].is_dir():
        inner = contents[0]
        for f in inner.iterdir():
            f.rename(out_dir / f.name)
        inner.rmdir()
    (out_dir / "checkpoint_complete").unlink(missing_ok=True)
    print(f"[download] {sorted(p.name for p in out_dir.iterdir())}")


def _rename(key: str) -> str:
    return key.replace("base_model.model.model", "base_model.model.backbone")


def post_process(adapter_dir: Path, rank: int) -> set[str]:
    """In-place post-process safetensors. Returns set of module suffixes present."""
    src = adapter_dir / "adapter_model.safetensors"
    if not src.exists():
        raise FileNotFoundError(src)

    # Load all tensors
    adapter: dict[str, torch.Tensor] = {}
    with safe_open(src, framework="pt", device="cpu") as f:
        for k in f.keys():
            adapter[k] = f.get_tensor(k)
    print(f"[post] loaded {len(adapter)} input tensors")

    # Group by base (strip .lora_A.weight / .lora_B.weight)
    bases: set[str] = set()
    for k in adapter:
        bases.add(re.sub(r"\.lora_[AB]\.weight$", "", k))
    print(f"[post] {len(bases)} unique (A,B) pairs")

    # Identify Mamba mixer projections (gate_proj=z, x_proj=x). Use a STRICT
    # match — endswith(".mixer.gate_proj") — to avoid catching MoE
    # shared_experts.gate_proj if that ever appears (different module entirely).
    mamba_projs: dict[str, dict[str, str]] = {}  # layer_base -> {"gate_proj"|"x_proj": base}
    for b in bases:
        for proj in ("gate_proj", "x_proj"):
            if b.endswith(f".mixer.{proj}"):
                layer = b[: -(len(proj) + 1)]  # strip ".gate_proj" / ".x_proj"
                mamba_projs.setdefault(layer, {})[proj] = b
    # Layers needing FULL SVD merge (both gate AND x present, huikang's path).
    mamba_merge = {l: p for l, p in mamba_projs.items() if "gate_proj" in p and "x_proj" in p}
    # Layers where ONLY gate (z) was LoRA'd — lift into in_proj with zero pad.
    mamba_gate_only = {l: p["gate_proj"] for l, p in mamba_projs.items()
                       if "gate_proj" in p and "x_proj" not in p}
    # All Mamba bases handled by special path (skip in default loop).
    merge_bases: set[str] = set()
    for p in mamba_merge.values():
        merge_bases.update(p.values())
    merge_bases.update(mamba_gate_only.values())
    print(f"[post] Mamba SVD-merge layers (gate+x): {len(mamba_merge)}; "
          f"gate-only lift layers: {len(mamba_gate_only)}")

    out: dict[str, torch.Tensor] = {}

    for base in sorted(bases):
        ka, kb = f"{base}.lora_A.weight", f"{base}.lora_B.weight"
        if ka not in adapter or kb not in adapter:
            continue
        lora_A = adapter[ka]
        lora_B = adapter[kb]

        # Skip empty .w3 experts (NemotronH MoE has w1/w2 only, w3 sometimes empty)
        if ".experts.w3" in base and lora_A.numel() == 0:
            continue

        # Skip gate_proj / x_proj — handled in Mamba merge below
        if base in merge_bases:
            continue

        renamed = _rename(base)

        # MoE expert unfusing: .experts.w1 -> per-expert .experts.{i}.up_proj
        #                      .experts.w2 -> per-expert .experts.{i}.down_proj
        if ".experts.w1" in base or ".experts.w2" in base:
            if lora_A.shape[0] == 1:
                lora_A = lora_A.expand(lora_B.shape[0], -1, -1).contiguous()
            elif lora_B.shape[0] == 1:
                lora_B = lora_B.expand(lora_A.shape[0], -1, -1).contiguous()
            num_experts = lora_A.shape[0]
            proj = "up_proj" if ".w1" in base else "down_proj"
            for i in range(num_experts):
                exp = re.sub(r"\.experts\.w[12]", f".experts.{i}.{proj}", renamed)
                out[f"{exp}.lora_A.weight"] = lora_A[i].contiguous()
                out[f"{exp}.lora_B.weight"] = lora_B[i].contiguous()
            continue

        # Default: rename and pass through
        out[f"{renamed}.lora_A.weight"] = lora_A
        out[f"{renamed}.lora_B.weight"] = lora_B

    # Path A: gate+x SVD merge (huikang's combined rank-2r -> rank-r compress)
    for layer, projs in sorted(mamba_merge.items()):
        in_proj_base = f"{_rename(layer)}.in_proj"

        gate_A = adapter[f"{projs['gate_proj']}.lora_A.weight"].float()
        gate_B = adapter[f"{projs['gate_proj']}.lora_B.weight"].float()
        x_A = adapter[f"{projs['x_proj']}.lora_A.weight"].float()
        x_B = adapter[f"{projs['x_proj']}.lora_B.weight"].float()
        r = gate_A.shape[0]

        # Stack: A as (2r, in_dim), B as (in_proj_dim, 2r) with gate at top, x below.
        A_cat = torch.cat([gate_A, x_A], dim=0)
        B_block = torch.zeros(MAMBA_IN_PROJ_DIM, 2 * r)
        B_block[: gate_B.shape[0], :r] = gate_B
        B_block[gate_B.shape[0] : gate_B.shape[0] + x_B.shape[0], r:] = x_B

        Q_B, R_B = torch.linalg.qr(B_block)
        Q_A, R_A = torch.linalg.qr(A_cat.T)
        core = R_B @ R_A.T
        U, S, Vh = torch.linalg.svd(core, full_matrices=False)

        new_B = (Q_B @ U[:, :r]) * S[:r].unsqueeze(0)
        new_A = Vh[:r, :] @ Q_A.T

        kept, total = S[:r].sum().item(), S.sum().item()
        pct = (kept / total * 100) if total > 0 else 0.0
        print(f"[post] SVD {layer.split('.layers.')[-1]}: kept {pct:.0f}% of singular mass")

        out[f"{in_proj_base}.lora_A.weight"] = new_A
        out[f"{in_proj_base}.lora_B.weight"] = new_B

    # Path B: gate-only lift — gate LoRA targets z portion (rows [0:Z_DIM]) of
    # the packed in_proj; pad B rows [Z_DIM:IN_PROJ_DIM] with zeros so the
    # x/B/C/dt outputs are unchanged from the base model.
    # Math: full in_proj LoRA = padded_B @ A where A == gate_A (rank-r unchanged).
    for layer, gate_base in sorted(mamba_gate_only.items()):
        in_proj_base = f"{_rename(layer)}.in_proj"
        gate_A = adapter[f"{gate_base}.lora_A.weight"]   # (r, 2688)
        gate_B = adapter[f"{gate_base}.lora_B.weight"]   # (Z_DIM=4096, r)
        r = gate_A.shape[0]
        assert gate_B.shape[0] == MAMBA_Z_DIM, (
            f"gate_proj B has {gate_B.shape[0]} rows, expected {MAMBA_Z_DIM}"
        )
        new_B = torch.zeros(MAMBA_IN_PROJ_DIM, r, dtype=gate_B.dtype)
        new_B[:MAMBA_Z_DIM, :] = gate_B
        out[f"{in_proj_base}.lora_A.weight"] = gate_A
        out[f"{in_proj_base}.lora_B.weight"] = new_B
        print(f"[post] gate-only -> in_proj {layer.split('.layers.')[-1]}: "
              f"rank={r}, padded B from ({MAMBA_Z_DIM},{r}) to ({MAMBA_IN_PROJ_DIM},{r})")

    # Recompute target_modules from what we actually produced.
    suffixes: Counter[str] = Counter()
    for k in out:
        parts = k.split(".")
        for i, p in enumerate(parts):
            if p in ("lora_A", "lora_B") and i > 0:
                suffixes[parts[i - 1]] += 1
                break
    target_modules = sorted(suffixes.keys())

    # Tinker emits fp32 LoRA; downcast to bf16 to halve submission size.
    # Loss of precision is negligible for inference (vLLM also runs in bf16).
    bytes_before = sum(t.numel() * t.element_size() for t in out.values())
    out = {k: v.to(torch.bfloat16) for k, v in out.items()}
    bytes_after = sum(t.numel() * t.element_size() for t in out.values())
    print(f"[post] wrote {len(out)} tensors; target_modules={target_modules}; "
          f"suffix counts={dict(suffixes)}")
    print(f"[post] dtype -> bf16: {bytes_before/1e9:.2f}GB -> {bytes_after/1e9:.2f}GB")

    save_file(out, str(src))

    cfg = {
        "base_model_name_or_path": BASE_MODEL,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "lora_alpha": rank,
        "lora_dropout": 0.0,
        "peft_type": "LORA",
        "r": rank,
        "target_modules": target_modules,
        "task_type": "CAUSAL_LM",
    }
    (adapter_dir / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
    return set(target_modules)


def make_zip(adapter_dir: Path, out_zip: Path) -> None:
    # Write to a temp path then atomic-rename so any concurrent upload sees
    # either the old complete file or the new complete file — never a partial.
    if out_zip.exists():
        out_zip.unlink()
    tmp = out_zip.with_suffix(out_zip.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    import zipfile
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(adapter_dir / "adapter_config.json", "adapter_config.json")
        z.write(adapter_dir / "adapter_model.safetensors", "adapter_model.safetensors")
    tmp.rename(out_zip)
    print(f"[zip] {out_zip} ({out_zip.stat().st_size / 1e6:.1f} MB)")


def submit_to_kaggle(zip_path: Path, message: str) -> None:
    cmd = [
        "kaggle", "competitions", "submit",
        "-c", "nvidia-nemotron-model-reasoning-challenge",
        "-f", str(zip_path), "-m", message,
    ]
    print(f"[submit] {' '.join(cmd[:6])} ...")
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-500:])
    if r.returncode != 0:
        print("STDERR:", r.stderr[-500:])
        raise RuntimeError("submission failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--message", default="")
    parser.add_argument("--skip-download", action="store_true",
                        help="reuse adapter already in /tmp/sub_<run-name>")
    args = parser.parse_args()

    work = Path(f"/tmp/sub_{args.run_name}")

    if not args.skip_download:
        tinker_path_file = TRAINING_DIR / args.run_name / "tinker_path.txt"
        if not tinker_path_file.exists():
            raise FileNotFoundError(tinker_path_file)
        tinker_path = tinker_path_file.read_text().strip()
        if work.exists():
            import shutil
            shutil.rmtree(work)
        download_from_tinker(tinker_path, work)

    post_process(work, rank=args.rank)

    zip_path = Path("/tmp/submission.zip")
    make_zip(work, zip_path)

    if args.submit:
        if not args.message:
            raise ValueError("--message required when --submit")
        submit_to_kaggle(zip_path, args.message)
    else:
        print(f"[main] zip ready: {zip_path}")


if __name__ == "__main__":
    main()
