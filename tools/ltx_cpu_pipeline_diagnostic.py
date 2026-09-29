"""Prototype an LTX pipeline that remains CPU-resident during construction.

This is a local/manual diagnostic, not CI. It deliberately reproduces the
upstream create_ltx_video_pipeline() components without the eager .to(cuda)
calls that overflow 6 GB GPUs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import psutil
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "models" / "ltx" / "ltxv-2b-0.9.6-distilled-04-25.safetensors"
TEXT_ENCODER = "PixArt-alpha/PixArt-XL-2-1024-MS"


def mark(message: str) -> None:
    ram = psutil.virtual_memory()
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        cuda = (
            f" cuda_free={free/(1024**3):.2f}/{total/(1024**3):.2f}GiB"
            f" allocated={allocated/(1024**3):.2f}GiB"
            f" reserved={reserved/(1024**3):.2f}GiB"
        )
    else:
        cuda = ""
    print(
        f"{message} ram_available={ram.available/(1024**3):.2f}GiB{cuda}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()

    if not checkpoint.exists():
        print(f"Checkpoint not found: {checkpoint}")
        print("Copy/download the LTX 2B distilled checkpoint first.")
        return 2

    from transformers import T5EncoderModel, T5Tokenizer
    from ltx_video.models.autoencoders.causal_video_autoencoder import CausalVideoAutoencoder
    from ltx_video.models.transformers.transformer3d import Transformer3DModel
    from ltx_video.pipelines.pipeline_ltx_video import LTXVideoPipeline
    from ltx_video.schedulers.rf import RectifiedFlowScheduler
    from ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier

    mark("[0] START")

    with safe_open(checkpoint, framework="pt") as f:
        metadata = f.metadata() or {}
        configs = json.loads(metadata.get("config", "{}"))
        allowed_steps = configs.get("allowed_inference_steps")
    mark("[1] metadata")

    vae = CausalVideoAutoencoder.from_pretrained(checkpoint).to(torch.bfloat16)
    mark("[2] VAE CPU BF16")

    transformer = Transformer3DModel.from_pretrained(checkpoint).to(torch.bfloat16)
    mark("[3] Transformer CPU BF16")

    scheduler = RectifiedFlowScheduler.from_pretrained(checkpoint)
    mark("[4] scheduler")

    text_encoder = T5EncoderModel.from_pretrained(
        TEXT_ENCODER, subfolder="text_encoder"
    ).to(torch.bfloat16)
    tokenizer = T5Tokenizer.from_pretrained(TEXT_ENCODER, subfolder="tokenizer")
    mark("[5] T5/tokenizer CPU BF16")

    pipeline = LTXVideoPipeline(
        transformer=transformer,
        patchifier=SymmetricPatchifier(patch_size=1),
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
        vae=vae,
        prompt_enhancer_image_caption_model=None,
        prompt_enhancer_image_caption_processor=None,
        prompt_enhancer_llm_model=None,
        prompt_enhancer_llm_tokenizer=None,
        allowed_inference_steps=allowed_steps,
    )
    mark("[6] pipeline constructed; still CPU-resident")

    params = [
        ("transformer", next(pipeline.transformer.parameters()).device),
        ("vae", next(pipeline.vae.parameters()).device),
        ("text_encoder", next(pipeline.text_encoder.parameters()).device),
    ]
    for name, device in params:
        print(f"{name}_device={device}", flush=True)

    if any(device.type != "cpu" for _, device in params):
        print("STATUS: FAIL - unexpected eager CUDA residency", flush=True)
        return 3

    mark("[7] PASS")
    print("STATUS: CPU-RESIDENT PIPELINE CONSTRUCTION PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
