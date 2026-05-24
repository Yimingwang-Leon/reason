"""Shared Modal App/Image/Volume definitions for training + adapter upload."""
from __future__ import annotations

import modal

APP_NAME = "nemotron-reason"
BASE_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
GPU = "H200"  # H100 80GB OOMs at bf16 base; H200 141GB gives ~58GB headroom

# ----- Image -----
# Base: PyTorch 2.6 + CUDA 12.4 + cuDNN 9 (-devel includes nvcc, required by
# causal-conv1d / mamba-ssm source builds). debian_slim w/o CUDA toolkit fails
# their setup.py at the "bare_metal_version" CUDA detection step.
#
# Pin transformers >= 5.3 (KV cache fix). Drop trust_remote_code at load time.
# unsloth >= 2026.3 has native Nemotron-3 patching.
image = (
    modal.Image.from_registry("pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel")
    .apt_install("git", "build-essential")
    # Stage 1: build prereqs + einops
    .pip_install("packaging>=24.0", "wheel", "ninja", "setuptools>=70", "einops")
    # Stage 1b: REPLACE conda's torch (pre-cxx11 ABI) with pip torch (cxx11 ABI).
    # causal-conv1d's prebuilt symbol is `c10::detail::torchCheckFail` mangled
    # with __cxx11 namespace — only present in pip torch wheels, not conda's.
    # Without this, the .so fails to import at runtime with undefined symbol error.
    .pip_install(
        "torch==2.6.0",
        extra_options="--upgrade --force-reinstall --index-url https://download.pytorch.org/whl/cu124",
    )
    # Stage 2: Mamba fast-path kernels — compiled against pip torch 2.6+cu124.
    # --no-deps prevents pulling in nvidia-cu13/cuda-toolkit which would conflict.
    # Without these, naive Mamba OOMs at ~135 GB / 140 GB H200 even with μ=1.
    .pip_install("causal-conv1d", extra_options="--no-build-isolation --no-deps")
    .pip_install("mamba-ssm==2.2.4", extra_options="--no-build-isolation --no-deps")
    # Stage 3: huikang's stable stack (transformers 4.57.6 + trust_remote_code=True).
    # Rationale (after 12 failed iterations on bleeding-edge path):
    # - transformers 5.x integrations/moe.py mismatches peft fp32 LoRA vs bf16 MoE weights
    # - mamba-ssm 2.2 references GreedySearchDecoderOnlyOutput (removed in 5.x)
    # - mamba-ssm 2.3 pulls cu13/cuda-toolkit, ABI-conflict with our torch
    # huikang's 4.57.6 + trust_remote_code=True uses model repo's own modeling_nemotron_h.py
    # which handles dtype + has the fast-path check. KV cache bug irrelevant for SFT.
    .pip_install(
        "transformers==4.57.6",
        "peft>=0.13,<0.14",
        "trl>=0.12,<0.13",
        "accelerate>=1.0",
        "datasets>=3.0",
        "safetensors>=0.4",
        "huggingface_hub>=0.26",
        "hf-transfer",
        "kaggle>=1.6",
        "tokenizers>=0.20",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false"})
    # Mount our local src/ package into the container so train.py + upload_adapter.py
    # can do `from src.modal_app import ...` inside Modal.
    .add_local_python_source("src")
)

# ----- Volumes -----
# model_cache: persistent HF download of base model (~64GB) so we don't re-pull.
# work: corpus inputs, adapter outputs, training logs.
model_cache_vol = modal.Volume.from_name("nemotron-model-cache", create_if_missing=True)
work_vol = modal.Volume.from_name("nemotron-work", create_if_missing=True)

MODEL_CACHE_PATH = "/cache/hf"
WORK_PATH = "/work"
ADAPTER_DIR = f"{WORK_PATH}/adapters"
CORPUS_DIR = f"{WORK_PATH}/corpus"

VOLUMES = {MODEL_CACHE_PATH: model_cache_vol, WORK_PATH: work_vol}

# ----- App -----
app = modal.App(APP_NAME)
