"""Runtime profiles. Policy is based on actual hardware, not model UI defaults."""

from __future__ import annotations

from dataclasses import dataclass

from .hardware import HardwareInfo


@dataclass(frozen=True, slots=True)
class RuntimeProfile:
    name: str
    dtype: str
    cpu_offload: bool
    batch_size: int
    target_width: int
    target_height: int
    target_duration_seconds: float
    max_frames_hint: int


def select_runtime_profile(hw: HardwareInfo) -> RuntimeProfile:
    # MVP baseline: <= 6.5 GiB cards, including RTX 2060 6GB.
    if hw.cuda_available and hw.total_vram_gib <= 6.5:
        return RuntimeProfile(
            name="low_vram",
            dtype="float16",
            cpu_offload=True,
            batch_size=1,
            target_width=512,
            target_height=288,
            target_duration_seconds=2.0,
            max_frames_hint=49,
        )

    if hw.cuda_available and hw.total_vram_gib <= 12.5:
        return RuntimeProfile("medium_vram", "float16", True, 1, 640, 360, 3.0, 73)

    if hw.cuda_available:
        return RuntimeProfile("high_vram", "float16", False, 1, 768, 432, 4.0, 97)

    return RuntimeProfile("cpu_only", "float32", False, 1, 384, 216, 1.0, 25)
