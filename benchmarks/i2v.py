"""Hardware/profile checkpoint before attaching a real I2V backend."""

from __future__ import annotations

from dataclasses import asdict

from engine.hardware import detect_hardware
from engine.profiles import select_runtime_profile


def main() -> int:
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


if __name__ == "__main__":
    raise SystemExit(main())
