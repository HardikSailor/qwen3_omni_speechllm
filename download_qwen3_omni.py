"""Download Qwen3-Omni-30B-A3B-Instruct weights into the shared HF cache.

Download only (no model load), so it is safe to run on the login node:
    python download_qwen3_omni.py [--model-id Qwen/Qwen3-Omni-30B-A3B-Thinking]
"""
import argparse
import os

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

from huggingface_hub import snapshot_download

CACHE_DIR = "/scratch/prj0000000234/sailorhb/hf_models/"
MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--cache-dir", default=CACHE_DIR)
    ap.add_argument("--max-workers", type=int, default=8)
    args = ap.parse_args()

    print(f"Downloading {args.model_id} -> {args.cache_dir}", flush=True)
    path = snapshot_download(
        repo_id=args.model_id,
        cache_dir=args.cache_dir,
        max_workers=args.max_workers,
    )
    n = sum(f.endswith(".safetensors") for f in os.listdir(path))
    print(f"Download complete: {path} ({n} safetensors shards)", flush=True)


if __name__ == "__main__":
    main()
