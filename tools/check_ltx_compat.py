"""Non-destructive compatibility check before installing the LTX backend."""

from __future__ import annotations

import importlib.util
import platform
import sys

import torch


def main() -> int:
    print("=== LTX-Video Compatibility Check ===")
    print(f"Python: {platform.python_version()}")
    print(f"Torch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if sys.version_info < (3, 10):
        print("STATUS: FAIL - LTX-Video requires Python >= 3.10")
        return 2

    if not torch.cuda.is_available():
        print("STATUS: FAIL - CUDA unavailable")
        return 2

    major, minor = torch.cuda.get_device_capability()
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Compute capability: {major}.{minor}")
    print("dtype policy: float16")
    print("FP8 kernels: disabled (Ada+ optimization; not for RTX 2060/Turing)")

    installed = importlib.util.find_spec("ltx_video") is not None
    print(f"ltx_video installed: {installed}")
    print("STATUS: BASE COMPATIBLE; LTX PACKAGE " + ("FOUND" if installed else "NOT INSTALLED"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
