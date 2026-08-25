"""Schema and persistence for shape-aware automatic benchmark results."""

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from benchmark_ops.shapes import Shape
from diflow.profiling.gpu import get_gpu_info, normalize_gpu_name

DEFAULT_RESULTS_DIR = "benchmark_ops/results"
RESULT_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class LatencyStats:
    median: float
    mean: float
    p50: float
    p99: float
    min: float
    max: float
    samples: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "median": self.median,
            "mean": self.mean,
            "p50": self.p50,
            "p99": self.p99,
            "min": self.min,
            "max": self.max,
            "samples": self.samples,
        }


def _percentile(sorted_samples: Sequence[float], percentile: float) -> float:
    """Nearest-rank percentile, which keeps p99 meaningful for small sample counts."""
    if not sorted_samples:
        raise ValueError("Cannot take a percentile of zero samples")

    rank = math.ceil(percentile / 100 * len(sorted_samples))
    return sorted_samples[max(0, min(len(sorted_samples) - 1, rank - 1))]


def summarize_latencies(samples: Sequence[float]) -> LatencyStats:
    if not samples:
        raise ValueError("Cannot summarize zero latency samples")

    sorted_samples = sorted(samples)
    count = len(sorted_samples)
    middle = count // 2
    if count % 2 == 1:
        median = sorted_samples[middle]
    else:
        median = (sorted_samples[middle - 1] + sorted_samples[middle]) / 2

    return LatencyStats(
        median=median,
        mean=sum(sorted_samples) / count,
        p50=_percentile(sorted_samples, 50),
        p99=_percentile(sorted_samples, 99),
        min=sorted_samples[0],
        max=sorted_samples[-1],
        samples=count,
    )


@dataclass
class OpLatencyRecord:
    """Latency of one operator, in one execution mode, at one shape."""

    op_id: str
    mode: str
    shape: Shape
    model_path: Optional[str] = None
    patches: List[str] = field(default_factory=list)
    # How often this op runs per request: constant + per_step * num_inference_steps.
    occurrences: Dict[str, int] = field(default_factory=dict)
    input_shapes: Dict[str, List[int]] = field(default_factory=dict)
    latency: Optional[LatencyStats] = None
    gpu_memory_used: Optional[int] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        record = {
            "op_id": self.op_id,
            "mode": self.mode,
            "model_path": self.model_path,
            "patches": list(self.patches),
            "occurrences": dict(self.occurrences),
            "shape": self.shape.to_dict(),
            "input_shapes": self.input_shapes,
        }
        if self.latency is not None:
            record["latency"] = self.latency.to_dict()
        if self.gpu_memory_used is not None:
            record["gpu_memory_used"] = self.gpu_memory_used
        if self.error is not None:
            record["error"] = self.error
        return record

    def occurrences_at(self, num_inference_steps: int) -> int:
        return (
            self.occurrences.get("constant", 0)
            + self.occurrences.get("per_step", 0) * num_inference_steps
        )


def build_result(
    case_name: str,
    workflow_name: str,
    suite: str,
    records: Sequence[OpLatencyRecord],
    reference_shape: Shape,
    warmup: int,
    repeats: int,
    profiled_num_inference_steps: int,
    device: str,
    gpu_info: Optional[Tuple[str, int, int]] = None,
    model_load_profiles: Optional[Sequence[Dict[str, Any]]] = None,
    profile_errors: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    gpu_type, gpu_memory, gpu_count = gpu_info if gpu_info else get_gpu_info()
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "gpu_type": gpu_type,
        "gpu_type_normalized": normalize_gpu_name(gpu_type),
        "gpu_memory_total": gpu_memory,
        "gpu_count": gpu_count,
        # `case` is the registry name and names the file; `workflow` is the name the
        # graph registers with the backend as its service id. Two cases can share a
        # workflow name, e.g. a real-weights case and its dummy-weights twin.
        "case": case_name,
        "workflow": workflow_name,
        "suite": suite,
        "device": device,
        "warmup": warmup,
        "repeats": repeats,
        "profiled_num_inference_steps": profiled_num_inference_steps,
        "reference_shape": reference_shape.to_dict(),
        "ops": [record.to_dict() for record in records],
        "model_load_profiles": list(model_load_profiles or []),
        "profile_errors": list(profile_errors or []),
    }


def get_gpu_results_dir(
    results_dir: str = DEFAULT_RESULTS_DIR, gpu_type: Optional[str] = None
) -> str:
    """Latency depends on the GPU, so results are namespaced by normalized GPU name."""
    if gpu_type is None:
        gpu_type, _, _ = get_gpu_info()
    return os.path.join(results_dir, normalize_gpu_name(gpu_type))


def get_result_path(
    case_name: str,
    results_dir: str = DEFAULT_RESULTS_DIR,
    gpu_type: Optional[str] = None,
) -> str:
    return os.path.join(get_gpu_results_dir(results_dir, gpu_type), f"{case_name}.json")


def load_existing_result(
    case_name: str,
    results_dir: str = DEFAULT_RESULTS_DIR,
    gpu_type: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    result_path = get_result_path(case_name, results_dir, gpu_type)
    if not os.path.exists(result_path):
        return None

    try:
        with open(result_path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Ignoring invalid op latency result at {result_path}")
        return None


def result_matches_current_gpu(result: Dict[str, Any]) -> bool:
    if result.get("schema_version") != RESULT_SCHEMA_VERSION:
        return False
    gpu_type, gpu_memory, _ = get_gpu_info()
    result_gpu_type = result.get("gpu_type_normalized", result.get("gpu_type", ""))

    try:
        result_gpu_memory = int(result.get("gpu_memory_total", -1))
    except (TypeError, ValueError):
        return False

    return normalize_gpu_name(str(result_gpu_type)) == normalize_gpu_name(
        gpu_type
    ) and result_gpu_memory == int(gpu_memory)


def save_result(
    result: Dict[str, Any],
    case_name: str,
    results_dir: str = DEFAULT_RESULTS_DIR,
) -> str:
    gpu_results_dir = get_gpu_results_dir(results_dir, result.get("gpu_type"))
    os.makedirs(gpu_results_dir, exist_ok=True)
    result_path = os.path.join(gpu_results_dir, f"{case_name}.json")
    print(f"Saving op latency result to {result_path}")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=4)
    return result_path
