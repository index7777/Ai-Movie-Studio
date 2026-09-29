"""Ai-Movie-Studio local environment preflight.

Run from the project virtual environment:
    python tools/preflight.py

This script intentionally performs local/manual checks only.
It is not intended for CI or GitHub Actions.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys


def run(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        return (result.stdout or result.stderr).strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def bytes_to_gib(value: int) -> str:
    return f"{value / (1024 ** 3):.2f} GiB"


def main() -> int:
    print("=== Ai-Movie-Studio Preflight ===")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Executable: {sys.executable}")
    print(f"OS: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")

    try:
        import psutil
        ram = psutil.virtual_memory()
        print(f"System RAM: {bytes_to_gib(ram.total)} total / {bytes_to_gib(ram.available)} available")
    except ImportError:
        print("System RAM: psutil not installed")

    ffmpeg = shutil.which("ffmpeg")
    print(f"FFmpeg: {ffmpeg or 'NOT FOUND'}")
    if ffmpeg:
        first_line = run([ffmpeg, "-version"]).splitlines()
        if first_line:
            print(f"FFmpeg version: {first_line[0]}")

    nvidia_smi = shutil.which("nvidia-smi")
    print(f"nvidia-smi: {nvidia_smi or 'NOT FOUND'}")
    if nvidia_smi:
        query = run([
            nvidia_smi,
            "--query-gpu=name,driver_version,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ])
        print(f"NVIDIA GPU: {query}")

    try:
        import torch
    except ImportError:
        print("PyTorch: NOT INSTALLED")
        print("STATUS: BASE ENV OK; CUDA TEST PENDING")
        return 2

    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA runtime: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("STATUS: FAIL - PyTorch cannot access CUDA")
        return 3

    index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    major, minor = torch.cuda.get_device_capability(index)

    print(f"CUDA device: {torch.cuda.get_device_name(index)}")
    print(f"Compute capability: {major}.{minor}")
    print(f"VRAM reported by PyTorch: {bytes_to_gib(props.total_memory)}")

    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        print(f"CUDA free VRAM: {bytes_to_gib(free_bytes)} / {bytes_to_gib(total_bytes)}")
    except Exception as exc:
        print(f"CUDA free VRAM: unavailable ({exc})")

    # FP16 arithmetic can work on Pascal, but fast Tensor Core FP16 paths
    # start with Volta/Turing-class hardware. Keep capability and smoke-test
    # results separate so "expected usable" is not misread as compatibility.
    fp16_fast_tensor_core_expected = major >= 7
    bf16_supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    print(f"FP16 fast Tensor Core path expected: {fp16_fast_tensor_core_expected}")
    print(f"BF16 reported supported: {bf16_supported}")

    try:
        x = torch.randn((1024, 1024), device="cuda", dtype=torch.float16)
        y = x @ x
        torch.cuda.synchronize()
        del x, y
        torch.cuda.empty_cache()
        print("FP16 smoke test: PASS")
    except Exception as exc:
        print(f"FP16 smoke test: FAIL ({type(exc).__name__}: {exc})")
        return 4

    print("STATUS: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
