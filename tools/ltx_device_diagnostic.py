"""Isolate LTX text-encoder and CUDA-placement crash boundaries."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "models" / "ltx" / "ltxv-2b-0.9.6-distilled-04-25.safetensors"
TEXT_ENCODER = "PixArt-alpha/PixArt-XL-2-1024-MS"

CHILD = r"""
import sys
from pathlib import Path
import psutil
import torch

stage = sys.argv[1]
checkpoint = Path(sys.argv[2])
text_encoder_id = sys.argv[3]

def mark(msg):
    ram = psutil.virtual_memory()
    free, total = torch.cuda.mem_get_info()
    allocated = torch.cuda.memory_allocated()
    print(
        f"[{stage}] {msg} ram_available={ram.available/(1024**3):.2f}GiB "
        f"cuda_free={free/(1024**3):.2f}/{total/(1024**3):.2f}GiB "
        f"cuda_allocated={allocated/(1024**3):.2f}GiB",
        flush=True,
    )

mark("START")

if stage == "t5":
    from transformers import T5EncoderModel
    mark("before T5 from_pretrained")
    model = T5EncoderModel.from_pretrained(text_encoder_id, subfolder="text_encoder")
    mark("after T5 from_pretrained")
    model = model.to(torch.bfloat16)
    mark("after T5 to(bfloat16)")
    del model
    mark("PASS")

elif stage in ("transformer_cuda_bf16", "transformer_cuda_fp16"):
    from ltx_video.models.transformers.transformer3d import Transformer3DModel
    dtype = torch.bfloat16 if stage.endswith("bf16") else torch.float16
    mark("before Transformer from_pretrained")
    model = Transformer3DModel.from_pretrained(checkpoint)
    mark("after Transformer from_pretrained")
    model = model.to(dtype)
    mark(f"after Transformer to({dtype})")
    model = model.to("cuda")
    torch.cuda.synchronize()
    mark("after Transformer to(cuda)")
    del model
    torch.cuda.empty_cache()
    mark("PASS")

elif stage == "vae_cuda_bf16":
    from ltx_video.models.autoencoders.causal_video_autoencoder import CausalVideoAutoencoder
    mark("before VAE from_pretrained")
    model = CausalVideoAutoencoder.from_pretrained(checkpoint)
    mark("after VAE from_pretrained")
    model = model.to(torch.bfloat16)
    mark("after VAE to(bfloat16)")
    model = model.to("cuda")
    torch.cuda.synchronize()
    mark("after VAE to(cuda)")
    del model
    torch.cuda.empty_cache()
    mark("PASS")

else:
    raise ValueError(stage)
"""

STAGES = ("t5", "transformer_cuda_bf16", "transformer_cuda_fp16", "vae_cuda_bf16")

def run(stage: str) -> int:
    print(f"\n=== {stage} ===", flush=True)
    proc = subprocess.run([sys.executable, "-c", CHILD, stage, str(CHECKPOINT), TEXT_ENCODER], text=True)
    unsigned = proc.returncode & 0xFFFFFFFF
    print(f"[parent] EXIT_CODE={proc.returncode} HEX=0x{unsigned:08X}", flush=True)
    return proc.returncode

def main() -> int:
    if not CHECKPOINT.exists():
        print(f"Checkpoint not found: {CHECKPOINT}")
        return 2
    for stage in STAGES:
        code = run(stage)
        if code != 0:
            print(f"STOP: {stage} failed; later stages skipped.", flush=True)
            return 1
    print("\nSTATUS: ALL DEVICE STAGES PASS", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
