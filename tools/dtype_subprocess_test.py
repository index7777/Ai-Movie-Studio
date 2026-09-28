"""Isolate CUDA dtype tests in a subprocess.

A native CUDA/PyTorch crash must not terminate the parent diagnostic process.
"""

from __future__ import annotations

import subprocess
import sys


TEST = r"""
import torch
dtype = getattr(torch, sys.argv[1])
device = torch.device("cuda")
print("GPU:", torch.cuda.get_device_name(), flush=True)
print("dtype:", dtype, flush=True)
a = torch.randn((1024, 1024), device=device, dtype=dtype)
b = torch.randn((1024, 1024), device=device, dtype=dtype)
c = a @ b
torch.cuda.synchronize()
print("PASS", c.dtype, float(c[0, 0]), flush=True)
"""


def run(name: str) -> int:
    proc = subprocess.run(
        [sys.executable, "-c", "import sys;" + TEST, name],
        text=True,
        capture_output=True,
    )
    print(f"=== {name} ===")
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    print(f"EXIT_CODE={proc.returncode}")
    return proc.returncode


def main() -> int:
    fp16 = run("float16")
    bf16 = run("bfloat16")
    print(f"SUMMARY fp16={fp16} bf16={bf16}")
    return 0 if fp16 == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
