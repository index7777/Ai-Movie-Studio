"""Measure cumulative LTX CUDA placement in one isolated child process."""

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

checkpoint = Path(sys.argv[1])
text_encoder_id = sys.argv[2]

def mark(msg):
    ram = psutil.virtual_memory()
    free, total = torch.cuda.mem_get_info()
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    print(
        f"{msg} ram_available={ram.available/(1024**3):.2f}GiB "
        f"cuda_free={free/(1024**3):.2f}/{total/(1024**3):.2f}GiB "
        f"allocated={allocated/(1024**3):.2f}GiB reserved={reserved/(1024**3):.2f}GiB",
        flush=True,
    )

from ltx_video.models.transformers.transformer3d import Transformer3DModel
from ltx_video.models.autoencoders.causal_video_autoencoder import CausalVideoAutoencoder
from transformers import T5EncoderModel

mark("[0] START")

mark("[1] before Transformer load")
transformer = Transformer3DModel.from_pretrained(checkpoint)
transformer = transformer.to(torch.bfloat16)
mark("[1] before Transformer CUDA")
transformer = transformer.to("cuda")
torch.cuda.synchronize()
mark("[1] after Transformer CUDA")

mark("[2] before VAE load")
vae = CausalVideoAutoencoder.from_pretrained(checkpoint)
vae = vae.to(torch.bfloat16)
mark("[2] before VAE CUDA")
vae = vae.to("cuda")
torch.cuda.synchronize()
mark("[2] after VAE CUDA")

mark("[3] before T5 load")
text_encoder = T5EncoderModel.from_pretrained(text_encoder_id, subfolder="text_encoder")
text_encoder = text_encoder.to(torch.bfloat16)
mark("[3] before T5 CUDA")
text_encoder = text_encoder.to("cuda")
torch.cuda.synchronize()
mark("[3] after T5 CUDA")

print("PASS: cumulative Transformer + VAE + T5 CUDA placement", flush=True)
"""

def main() -> int:
    if not CHECKPOINT.exists():
        print(f"Checkpoint not found: {CHECKPOINT}")
        return 2
    print("Running cumulative placement in isolated subprocess...", flush=True)
    proc = subprocess.run([sys.executable, "-c", CHILD, str(CHECKPOINT), TEXT_ENCODER], text=True)
    unsigned = proc.returncode & 0xFFFFFFFF
    print(f"[parent] EXIT_CODE={proc.returncode} HEX=0x{unsigned:08X}", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
