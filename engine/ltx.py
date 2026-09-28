"""LTX-Video backend adapter.

The adapter is intentionally lazy: importing Ai-Movie-Studio does not require
LTX-Video to be installed. The official package is loaded only when the backend
is selected.
"""

from __future__ import annotations

from pathlib import Path
import time

import torch

from .base import GenerationRequest, GenerationResult, VideoEngine
from .hardware import detect_hardware
from .profiles import select_runtime_profile


class LTXVideoEngine(VideoEngine):
    name = "ltx-video-2b-distilled"

    def __init__(self, pipeline_config: Path) -> None:
        self.pipeline_config = Path(pipeline_config)
        self._infer = None
        self._config_cls = None

    def load(self) -> None:
        try:
            from ltx_video.inference import InferenceConfig, infer
        except ImportError as exc:
            raise RuntimeError(
                "LTX-Video backend is not installed. Install the official "
                "LTX-Video package with its inference extras first."
            ) from exc

        if not self.pipeline_config.exists():
            raise FileNotFoundError(f"LTX pipeline config not found: {self.pipeline_config}")

        hw = detect_hardware()
        profile = select_runtime_profile(hw)
        if profile.dtype != "float16":
            raise RuntimeError(f"Unsupported LTX MVP dtype policy: {profile.dtype}")

        self._infer = infer
        self._config_cls = InferenceConfig

    def generate(self, request: GenerationRequest) -> GenerationResult:
        if self._infer is None or self._config_cls is None:
            raise RuntimeError("Engine is not loaded")

        hw = detect_hardware()
        profile = select_runtime_profile(hw)
        width = request.width or profile.target_width
        height = request.height or profile.target_height

        # LTX temporal shape convention: 8n + 1 frames.
        requested_frames = int(request.options.get("num_frames", profile.max_frames_hint))
        num_frames = max(9, requested_frames - ((requested_frames - 1) % 8))

        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()

        try:
            config = self._config_cls(
                pipeline_config=str(self.pipeline_config),
                prompt=request.prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                seed=request.seed,
                output_path=str(request.output_path),
                conditioning_media_paths=[str(request.image_path)],
                conditioning_start_frames=[0],
            )
            self._infer(config)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise RuntimeError(
                "LTX-Video CUDA OOM. Retry with fewer frames/lower resolution, "
                "then record the failed profile in docs/PITFALLS.md."
            ) from None

        elapsed = time.perf_counter() - started
        peak_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
        return GenerationResult(
            output_path=request.output_path,
            elapsed_seconds=elapsed,
            backend=self.name,
            metadata={
                "width": width,
                "height": height,
                "num_frames": num_frames,
                "peak_cuda_allocated_gib": round(peak_gib, 3),
                "profile": profile.name,
            },
        )

    def unload(self) -> None:
        self._infer = None
        self._config_cls = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
