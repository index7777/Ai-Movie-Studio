"""Backend-neutral contracts for image-to-video engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class GenerationRequest:
    image_path: Path
    prompt: str
    output_path: Path
    duration_seconds: float = 2.0
    seed: int = 12345
    width: int | None = None
    height: int | None = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GenerationResult:
    output_path: Path
    elapsed_seconds: float
    backend: str
    metadata: dict[str, Any] = field(default_factory=dict)


class VideoEngine(ABC):
    name = "base"

    @abstractmethod
    def load(self) -> None:
        """Load model resources."""

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate a video from one conditioning image."""

    def unload(self) -> None:
        """Release backend resources when supported."""
