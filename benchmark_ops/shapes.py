"""Shape sweeps for op latency profiling.

A "shape" is the part of a request that changes how long an operator takes to run:
the batch size the worker groups requests into, and the output resolution.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_BATCH_SIZES = [1, 2, 4, 8]
DEFAULT_RESOLUTIONS = [(256, 256), (512, 512), (1024, 1024)]


@dataclass(frozen=True)
class Shape:
    batch_size: int
    height: int
    width: int

    def to_dict(self) -> Dict[str, int]:
        return {
            "batch_size": self.batch_size,
            "height": self.height,
            "width": self.width,
        }

    @classmethod
    def from_dict(cls, shape_dict: Dict[str, Any]) -> "Shape":
        return cls(
            batch_size=int(shape_dict["batch_size"]),
            height=int(shape_dict["height"]),
            width=int(shape_dict["width"]),
        )

    def __str__(self) -> str:
        return f"bs{self.batch_size}_{self.height}x{self.width}"


@dataclass(frozen=True)
class ShapeSweep:
    batch_sizes: Tuple[int, ...]
    resolutions: Tuple[Tuple[int, int], ...]

    @classmethod
    def default(cls) -> "ShapeSweep":
        return cls(
            batch_sizes=tuple(DEFAULT_BATCH_SIZES),
            resolutions=tuple(DEFAULT_RESOLUTIONS),
        )

    @classmethod
    def from_dict(cls, sweep_dict: Optional[Dict[str, Any]]) -> "ShapeSweep":
        if not sweep_dict:
            return cls.default()

        batch_sizes = sweep_dict.get("batch_sizes") or DEFAULT_BATCH_SIZES
        resolutions = sweep_dict.get("resolutions") or DEFAULT_RESOLUTIONS
        return cls(
            batch_sizes=tuple(int(batch_size) for batch_size in batch_sizes),
            resolutions=tuple(
                (int(height), int(width)) for height, width in resolutions
            ),
        )

    def expand(self) -> List[Shape]:
        """Enumerate every (resolution, batch size) combination.

        Resolution is the outer loop: all batch sizes for one resolution are profiled
        from a single captured graph, so grouping this way avoids re-running capture.
        """
        shapes = []
        for height, width in self.resolutions:
            for batch_size in self.batch_sizes:
                shapes.append(Shape(batch_size=batch_size, height=height, width=width))
        return shapes

    def reference_shape(self) -> Shape:
        """The canonical shape recorded in benchmark result metadata."""
        height, width = self.resolutions[0]
        return Shape(batch_size=min(self.batch_sizes), height=height, width=width)


def parse_batch_sizes(raw_batch_sizes: str) -> Tuple[int, ...]:
    """Parse a `--batch-sizes 1,2,4` CLI value."""
    batch_sizes = []
    for token in raw_batch_sizes.split(","):
        token = token.strip()
        if not token:
            continue
        batch_size = int(token)
        if batch_size < 1:
            raise ValueError(f"Batch size must be >= 1, got {batch_size}")
        batch_sizes.append(batch_size)

    if not batch_sizes:
        raise ValueError(f"No batch sizes parsed from {raw_batch_sizes!r}")
    return tuple(batch_sizes)


def parse_resolutions(raw_resolutions: str) -> Tuple[Tuple[int, int], ...]:
    """Parse a `--resolutions 512x512,1024x1024` CLI value as (height, width) pairs."""
    resolutions = []
    for token in raw_resolutions.split(","):
        token = token.strip()
        if not token:
            continue
        if "x" not in token:
            raise ValueError(
                f"Resolution {token!r} must be formatted as <height>x<width>"
            )
        raw_height, raw_width = token.split("x", 1)
        resolutions.append((int(raw_height), int(raw_width)))

    if not resolutions:
        raise ValueError(f"No resolutions parsed from {raw_resolutions!r}")
    return tuple(resolutions)


def group_by_resolution(shapes: Sequence[Shape]) -> Dict[Tuple[int, int], List[int]]:
    """Map (height, width) -> batch sizes, preserving sweep order."""
    grouped: Dict[Tuple[int, int], List[int]] = {}
    for shape in shapes:
        grouped.setdefault((shape.height, shape.width), []).append(shape.batch_size)
    return grouped
