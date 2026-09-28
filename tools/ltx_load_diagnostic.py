"""Stage-isolated LTX checkpoint diagnostics for Windows/native crashes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "models" / "ltx" / "ltxv-2b-0.9.6-distilled-04-25.safetensors"

STAGES = ("metadata", "vae", "transformer", "transformer_bf16", "transformer_fp16")

CHILD = r"""
import json
import sys
from pathlib import Path
import psutil
import torch
from safetensors import safe_open

stage = sys.argv[1]
checkpoint = Path(sys.argv[2])

def mark(message):
    ram = psutil.virtual_memory()
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        gpu = f" cuda_free={free/(1024**3):.2f}/{total/(1024**3):.2f}GiB"
    else:
        gpu = ""
    print(f"[{stage}] {message} ram_available={ram.available/(1024**3):.2f}GiB{gpu}", flush=True)

mark("START")

if stage == "metadata":
    with safe_open(checkpoint, framework="pt") as f:
        metadata = f.metadata() or {}
        config = json.loads(metadata.get("config", "{}"))
        print("metadata_keys:", sorted(metadata.keys()), flush=True)
        print("config_keys:", sorted(config.keys()), flush=True)
    mark("PASS")

elif stage == "vae":
    from ltx_video.models.autoencoders.causal_video_autoencoder import CausalVideoAutoencoder
    mark("before from_pretrained")
    model = CausalVideoAutoencoder.from_pretrained(checkpoint)
    mark("after from_pretrained")
    del model
    mark("PASS")

elif stage in ("transformer", "transformer_bf16", "transformer_fp16"):
    from ltx_video.models.transformers.transformer3d import Transformer3DModel
    mark("before from_pretrained")
    model = Transformer3DModel.from_pretrained(checkpoint)
    mark("after from_pretrained")
    if stage == "transformer_bf16":
        mark("before to(bfloat16)")
        model = model.to(torch.bfloat16)
        mark("after to(bfloat16)")
    elif stage == "transformer_fp16":
        mark("before to(float16)")
        model = model.to(torch.float16)
        mark("after to(float16)")
    del model
    mark("PASS")
else:
    raise ValueError(stage)
"""


def run_stage(stage: str) -> int:
    print(f"\n=== {stage} ===", flush=True)
    proc = subprocess.run(
        [sys.executable, "-c", CHILD, stage, str(CHECKPOINT)],
        text=True,
    )
    unsigned = proc.returncode & 0xFFFFFFFF
    print(f"[parent] EXIT_CODE={proc.returncode} HEX=0x{unsigned:08X}", flush=True)
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("all",) + STAGES, default="all")
    args = parser.parse_args()

    if not CHECKPOINT.exists():
        print(f"Checkpoint not found: {CHECKPOINT}")
        return 2

    stages = STAGES if args.stage == "all" else (args.stage,)
    for stage in stages:
        code = run_stage(stage)
        if code != 0:
            print(f"STOP: {stage} failed; later stages skipped.", flush=True)
            return 1
    print("\nSTATUS: ALL REQUESTED STAGES PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
