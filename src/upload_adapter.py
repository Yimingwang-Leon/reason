"""Download Tinker LoRA checkpoint and upload to user's Kaggle Model instance.

After src/train_tinker.py finishes, it writes training/<run>/tinker_path.txt.
This script reads it, downloads from Tinker, extracts, and pushes to Kaggle.

Usage:
    python -m src.upload_adapter --run-name mini-tinker-002

Result: Kaggle Model `<KAGGLE_USERNAME>/nemotron-adapter/Transformers/default`
        gets a new version. Then submit notebook can pull it.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = ROOT / "training"
ENV_PATH = ROOT / "env.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="mini-tinker-002")
    parser.add_argument("--instance-suffix", default="default",
                        help="Kaggle model instance slug")
    args = parser.parse_args()

    env = json.loads(ENV_PATH.read_text())
    os.environ["TINKER_API_KEY"] = env["TINKER_API_KEY"]
    kaggle_user = env["KAGGLE_USERNAME"]

    tinker_path_file = TRAINING_DIR / args.run_name / "tinker_path.txt"
    if not tinker_path_file.exists():
        raise FileNotFoundError(
            f"No {tinker_path_file}. Run src/train_tinker.py first."
        )
    tinker_path = tinker_path_file.read_text().strip()
    print(f"[upload] tinker path: {tinker_path}")

    # === 1. Get download URL from Tinker REST API ===
    import tinker
    sc = tinker.ServiceClient()
    rest = sc.create_rest_client()
    url_fut = rest.get_checkpoint_archive_url_from_tinker_path(tinker_path)
    url = url_fut.result().url
    print(f"[upload] download URL obtained (len {len(url)})")

    # === 2. Download tar & extract to temp dir ===
    with tempfile.TemporaryDirectory() as tmpdir:
        tar_path = Path(tmpdir) / "adapter.tar"
        adapter_dir = Path(tmpdir) / "weights"
        adapter_dir.mkdir()

        print("[upload] downloading...")
        urllib.request.urlretrieve(url, tar_path)
        size_mb = tar_path.stat().st_size / 1e6
        print(f"[upload] downloaded {size_mb:.1f} MB")

        with tarfile.open(tar_path) as tar:
            tar.extractall(adapter_dir)

        # If extracted into a subdirectory, flatten
        contents = list(adapter_dir.iterdir())
        if len(contents) == 1 and contents[0].is_dir():
            inner = contents[0]
            for f in inner.iterdir():
                shutil.move(str(f), str(adapter_dir / f.name))
            inner.rmdir()

        files = sorted(adapter_dir.iterdir())
        print("[upload] extracted files:")
        for f in files:
            sz = f.stat().st_size
            print(f"  {f.name}: {sz / 1e6:.2f} MB")

        # Sanity: must have adapter_config.json + adapter_model.safetensors
        names = {f.name for f in files}
        assert "adapter_config.json" in names, f"missing adapter_config.json in {names}"
        assert any(n.endswith(".safetensors") for n in names), \
            f"missing safetensors in {names}"

        # === 3. Upload to Kaggle ===
        # Use kaggle CLI directly — simpler than Python API for this purpose
        instance = f"{kaggle_user}/nemotron-adapter/Transformers/{args.instance_suffix}"
        print(f"[upload] target Kaggle instance: {instance}")

        # Set kaggle creds from env.json (CLI auto-detects but we want to be sure)
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_dir.mkdir(exist_ok=True)
        # Already exists usually; CLI uses access_token

        # Create model instance if needed, else upload new version.
        # Using kaggle CLI for robustness across SDK versions.
        import subprocess

        # Write metadata.json
        meta = {
            "ownerSlug": kaggle_user,
            "modelSlug": "nemotron-adapter",
            "instanceSlug": args.instance_suffix,
            "framework": "Transformers",
            "licenseName": "Apache 2.0",
            "overview": f"Nemotron-3-Nano-30B LoRA adapter (run: {args.run_name})",
            "trainingData": [],
            "modelInstanceType": "Unspecified",
        }
        (adapter_dir / "model-instance-metadata.json").write_text(
            json.dumps(meta, indent=2)
        )

        # Check whether instance exists by trying to list versions
        check = subprocess.run(
            ["kaggle", "models", "instances", "get", instance],
            capture_output=True, text=True,
        )
        exists = "404" not in check.stderr and check.returncode == 0

        if not exists:
            print(f"[upload] creating new model instance {instance}...")
            r = subprocess.run(
                ["kaggle", "models", "instances", "create", "-p", str(adapter_dir)],
                capture_output=True, text=True,
            )
            print(r.stdout.strip())
            if r.returncode != 0:
                raise RuntimeError(f"create failed: {r.stderr}")
        else:
            print(f"[upload] instance exists, uploading new version...")
            # version create doesn't need the metadata file
            (adapter_dir / "model-instance-metadata.json").unlink(missing_ok=True)
            r = subprocess.run(
                ["kaggle", "models", "instances", "versions", "create",
                 "-p", str(adapter_dir), instance,
                 "-n", f"Run {args.run_name}"],
                capture_output=True, text=True,
            )
            print(r.stdout.strip())
            if r.returncode != 0:
                raise RuntimeError(f"version create failed: {r.stderr}")

        print(f"\n[upload] DONE. instance: {instance}")
        print(f"        URL: https://www.kaggle.com/models/{kaggle_user}/nemotron-adapter")


if __name__ == "__main__":
    main()
