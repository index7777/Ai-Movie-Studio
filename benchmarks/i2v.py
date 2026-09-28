"""Hardware/profile checkpoint and real I2V backend benchmark."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.base import GenerationRequest
from engine.hardware import detect_hardware
from engine.profiles import select_runtime_profile


def hardware_check() -> int:
    hw = detect_hardware()
    profile = select_runtime_profile(hw)

    print("=== Ai-Movie-Studio I2V Benchmark Bootstrap ===")
    print("Hardware")
    for key, value in asdict(hw).items():
        print(f"  {key}: {value}")
    print("Runtime profile")
    for key, value in asdict(profile).items():
        print(f"  {key}: {value}")

    if not hw.cuda_available:
        print("STATUS: FAIL - CUDA GPU unavailable")
        return 2
    if hw.free_vram_gib < 3.0:
        print("STATUS: WARN - free VRAM below 3 GiB; close GPU-heavy desktop apps before inference")
        return 1
    print("STATUS: READY FOR BACKEND BENCHMARK")
    return 0


def run_ltx(args: argparse.Namespace) -> int:
    from engine.ltx import LTXVideoEngine

    image = Path(args.image)
    if not image.exists():
        print(f"STATUS: FAIL - input image not found: {image}")
        return 2

    checkpoint = ROOT / "models" / "ltx" / "ltxv-2b-0.9.6-distilled-04-25.safetensors"
    if not checkpoint.exists():
        print(f"STATUS: FAIL - checkpoint not found: {checkpoint}")
        return 2

    config = ROOT / "configs" / "ltxv-2b-0.9.6-distilled-rtx2060.yaml"
    output_dir = ROOT / "output" / "ltx"

    # Upstream config uses a relative checkpoint path. Run from the checkpoint
    # directory so it resolves without duplicating a 6.34 GB model file.
    import os
    old_cwd = Path.cwd()
    os.chdir(checkpoint.parent)
    try:
        engine = LTXVideoEngine(config)
        print("Loading LTX backend...")
        engine.load()
        print("Starting generation...")
        result = engine.generate(
            GenerationRequest(
                image_path=image.resolve(),
                prompt=args.prompt,
                output_path=output_dir,
                seed=args.seed,
                width=args.width,
                height=args.height,
                options={"num_frames": args.frames},
            )
        )
    finally:
        os.chdir(old_cwd)

    print("=== RESULT ===")
    print(f"backend: {result.backend}")
    print(f"output: {result.output_path}")
    print(f"elapsed_seconds: {result.elapsed_seconds:.2f}")
    for key, value in result.metadata.items():
        print(f"{key}: {value}")
    print("STATUS: GENERATION PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["check", "ltx"], default="check")
    parser.add_argument("--image")
    parser.add_argument("--prompt", default="The camera slowly pushes forward. Subtle natural motion.")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--frames", type=int, default=25)
    args = parser.parse_args()

    if args.backend == "check":
        return hardware_check()
    if not args.image:
        parser.error("--image is required for --backend ltx")
    return run_ltx(args)


if __name__ == "__main__":
    raise SystemExit(main())
