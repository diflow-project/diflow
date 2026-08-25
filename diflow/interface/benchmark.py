"""Workflow-owned inputs and sweep settings for automatic benchmarking."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Tuple

DEFAULT_BENCHMARK_RESOLUTIONS = ((256, 256), (512, 512), (1024, 1024))
DEFAULT_BENCHMARK_BATCH_SIZES = (1, 2, 4, 8)


@dataclass(frozen=True)
class BenchmarkSpec:
    """Inputs and shape sweep used before a workflow server becomes ready.

    ``height`` and ``width`` are injected by the profiler. ``inputs`` should
    contain every other request value needed to expand and execute the graph.
    """

    inputs: Mapping[str, Any]
    resolutions: Tuple[Tuple[int, int], ...] = field(
        default_factory=lambda: DEFAULT_BENCHMARK_RESOLUTIONS
    )
    batch_sizes: Tuple[int, ...] = field(
        default_factory=lambda: DEFAULT_BENCHMARK_BATCH_SIZES
    )
    warmup: int = 2
    repeats: int = 5
    profile_steps: int = 2
    offload_idle_models: bool = False

    def __post_init__(self) -> None:
        inputs = copy.deepcopy(dict(self.inputs))
        resolutions = tuple(
            (int(height), int(width)) for height, width in self.resolutions
        )
        batch_sizes = tuple(int(batch_size) for batch_size in self.batch_sizes)

        if not resolutions or any(
            height <= 0 or width <= 0 for height, width in resolutions
        ):
            raise ValueError("BenchmarkSpec.resolutions must contain positive sizes")
        if not batch_sizes or any(batch_size <= 0 for batch_size in batch_sizes):
            raise ValueError("BenchmarkSpec.batch_sizes must contain positive values")
        if self.warmup < 0:
            raise ValueError("BenchmarkSpec.warmup must be non-negative")
        if self.repeats <= 0:
            raise ValueError("BenchmarkSpec.repeats must be positive")
        if self.profile_steps <= 0:
            raise ValueError("BenchmarkSpec.profile_steps must be positive")

        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "resolutions", resolutions)
        object.__setattr__(self, "batch_sizes", batch_sizes)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "inputs": copy.deepcopy(dict(self.inputs)),
            "resolutions": [list(resolution) for resolution in self.resolutions],
            "batch_sizes": list(self.batch_sizes),
            "warmup": self.warmup,
            "repeats": self.repeats,
            "profile_steps": self.profile_steps,
            "offload_idle_models": self.offload_idle_models,
        }
