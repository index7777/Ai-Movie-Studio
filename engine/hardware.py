"""Hardware detection used by backend-independent runtime policy."""

from __future__ import annotations

from dataclasses import dataclass

import psutil
import torch


@dataclass(frozen=True, slots=True)
class HardwareInfo:
    cuda_available: bool
    gpu_name: str | None
    compute_capability: tuple[int, int] | None
    total_vram_gib: float
    free_vram_gib: float
    system_ram_gib: float
    available_ram_gib: float


def _gib(value: int) -> float:
    return value / (1024 ** 3)


def detect_hardware() -> HardwareInfo:
    ram = psutil.virtual_memory()
    if not torch.cuda.is_available():
        return HardwareInfo(False, None, None, 0.0, 0.0, _gib(ram.total), _gib(ram.available))

    device = torch.cuda.current_device()
    free_vram, total_vram = torch.cuda.mem_get_info(device)
    return HardwareInfo(
        True,
        torch.cuda.get_device_name(device),
        torch.cuda.get_device_capability(device),
        _gib(total_vram),
        _gib(free_vram),
        _gib(ram.total),
        _gib(ram.available),
    )
