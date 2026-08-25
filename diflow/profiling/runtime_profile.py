"""Strongly typed runtime profile produced by automatic workflow benchmarking."""

from __future__ import annotations

import importlib.resources
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, Mapping, Optional, Tuple

if TYPE_CHECKING:
    from diflow.interface.workflow import Workflow

RUNTIME_PROFILE_SCHEMA_VERSION = 2
ExecutionKey = Tuple[str, int, int, int]
PROFILE_FREE_OPERATOR_IDS = {"IndexedTensor", "GuidanceTensor"}


class RuntimeProfileError(ValueError):
    """Base class for unusable runtime profile data."""


class MissingProfileError(RuntimeProfileError):
    """Raised when a requested operator or shape was never benchmarked."""


class UnsupportedProfileError(RuntimeProfileError):
    """Raised when benchmarking explicitly marked a shape unsupported."""


@dataclass(frozen=True)
class ExecutionRuntimeProfile:
    latency_seconds: Optional[float]
    activation_memory_bytes: Optional[int]
    error: Optional[str] = None

    @property
    def successful(self) -> bool:
        return (
            self.error is None
            and self.latency_seconds is not None
            and self.activation_memory_bytes is not None
        )


@dataclass
class OperatorRuntimeProfile:
    disk_to_host_latency_seconds: float = 0.0
    host_to_gpu_latency_seconds: float = 0.0
    model_memory_bytes: int = 0
    executions: Dict[ExecutionKey, ExecutionRuntimeProfile] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class TransferRuntimeProfile:
    block_sizes: Tuple[int, ...]
    fetch_overheads_us: Tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.block_sizes) != len(self.fetch_overheads_us):
            raise RuntimeProfileError(
                "transfer block sizes and overheads must have equal length"
            )
        if tuple(sorted(self.block_sizes)) != self.block_sizes:
            raise RuntimeProfileError("transfer block sizes must be sorted")


TransferProfilePair = Tuple[TransferRuntimeProfile, TransferRuntimeProfile]


@dataclass
class RuntimeProfile:
    schema_version: int
    gpu_type: str
    gpu_memory_total: int
    gpu_count: int
    operators: Dict[str, OperatorRuntimeProfile]
    failed_resolutions: Dict[Tuple[int, int], str]
    intra_transfer: TransferRuntimeProfile
    inter_transfer: TransferRuntimeProfile

    @classmethod
    def from_file(cls, path: str | Path) -> "RuntimeProfile":
        profile_path = Path(path).expanduser()
        try:
            payload = json.loads(profile_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeProfileError(
                f"Cannot read runtime profile {profile_path}: {exc}"
            ) from exc
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RuntimeProfile":
        schema_version = int(payload.get("schema_version", 0))
        if schema_version != RUNTIME_PROFILE_SCHEMA_VERSION:
            raise RuntimeProfileError(
                "Unsupported runtime profile schema "
                f"{schema_version}; expected {RUNTIME_PROFILE_SCHEMA_VERSION}"
            )

        operators: Dict[str, OperatorRuntimeProfile] = {}
        loaded_operator_ids = set()
        for raw in payload.get("model_load_profiles", ()):
            op_id = str(raw["op_id"])
            loaded_operator_ids.add(op_id)
            profile = operators.setdefault(op_id, OperatorRuntimeProfile())
            profile.disk_to_host_latency_seconds = max(
                profile.disk_to_host_latency_seconds,
                float(raw["disk_to_host_latency_seconds"]),
            )
            profile.host_to_gpu_latency_seconds = max(
                profile.host_to_gpu_latency_seconds,
                float(raw["host_to_gpu_latency_seconds"]),
            )
            profile.model_memory_bytes = max(
                profile.model_memory_bytes, int(raw["model_memory_bytes"])
            )

        for raw in payload.get("ops", ()):
            op_id = str(raw["op_id"])
            shape = raw["shape"]
            key: ExecutionKey = (
                str(raw.get("mode", "default")),
                int(shape["batch_size"]),
                int(shape["height"]),
                int(shape["width"]),
            )
            latency = raw.get("latency")
            execution = ExecutionRuntimeProfile(
                latency_seconds=(None if latency is None else float(latency["median"])),
                activation_memory_bytes=(
                    None
                    if raw.get("gpu_memory_used") is None
                    else max(0, int(raw["gpu_memory_used"]))
                ),
                error=(None if raw.get("error") is None else str(raw["error"])),
            )
            op_profile = operators.setdefault(op_id, OperatorRuntimeProfile())
            existing = op_profile.executions.get(key)
            op_profile.executions[key] = cls._merge_execution(existing, execution)

        if not operators:
            raise RuntimeProfileError("Runtime profile contains no operator data")
        missing_load_profiles = sorted(
            set(operators) - loaded_operator_ids - PROFILE_FREE_OPERATOR_IDS
        )
        if missing_load_profiles:
            raise RuntimeProfileError(
                "Runtime profile is missing model load data for operators: "
                + ", ".join(missing_load_profiles)
            )

        failed_resolutions: Dict[Tuple[int, int], str] = {}
        for raw in payload.get("profile_errors", ()):
            resolution = (int(raw["height"]), int(raw["width"]))
            error = str(raw["error"])
            existing_error = failed_resolutions.get(resolution)
            if existing_error is not None and error not in existing_error:
                error = f"{existing_error}; {error}"
            failed_resolutions[resolution] = error

        intra, inter = cls._load_transfer_profiles()
        return cls(
            schema_version=schema_version,
            gpu_type=str(payload.get("gpu_type", "unknown")),
            gpu_memory_total=int(payload.get("gpu_memory_total", 0)),
            gpu_count=int(payload.get("gpu_count", 0)),
            operators=operators,
            failed_resolutions=failed_resolutions,
            intra_transfer=intra,
            inter_transfer=inter,
        )

    @staticmethod
    def _merge_execution(
        existing: Optional[ExecutionRuntimeProfile],
        candidate: ExecutionRuntimeProfile,
    ) -> ExecutionRuntimeProfile:
        if existing is None:
            return candidate
        entries = (existing, candidate)
        failures = [item for item in entries if not item.successful]
        if failures:
            errors = tuple(
                dict.fromkeys(
                    item.error or "incomplete benchmark result" for item in failures
                )
            )
            return ExecutionRuntimeProfile(None, None, "; ".join(errors))
        return ExecutionRuntimeProfile(
            latency_seconds=max(float(item.latency_seconds) for item in entries),
            activation_memory_bytes=max(
                int(item.activation_memory_bytes) for item in entries
            ),
        )

    @staticmethod
    def _load_transfer_profiles() -> TransferProfilePair:
        resource = importlib.resources.files("diflow").joinpath(
            "configs", "transfer_profiles.json"
        )
        try:
            payload = json.loads(resource.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeProfileError(
                f"Cannot read packaged transfer profiles: {exc}"
            ) from exc

        def parse(name: str) -> TransferRuntimeProfile:
            raw = payload[name]
            return TransferRuntimeProfile(
                block_sizes=tuple(int(value) for value in raw["block_sizes"]),
                fetch_overheads_us=tuple(
                    float(value) for value in raw["fetch_overheads_us"]
                ),
            )

        return parse("intra_node"), parse("inter_node")

    @staticmethod
    def _adjust_batch_size(batch_size: int) -> int:
        if batch_size < 1:
            raise ValueError("batch size must be positive")
        return 1 << (batch_size - 1).bit_length()

    def _operator(self, op_id: str) -> OperatorRuntimeProfile:
        try:
            return self.operators[op_id]
        except KeyError as exc:
            raise MissingProfileError(
                f"Operator {op_id!r} was not benchmarked at server startup"
            ) from exc

    def execution(
        self,
        op_id: str,
        mode: str,
        batch_size: int,
        height: int,
        width: int,
    ) -> ExecutionRuntimeProfile:
        profile = self._operator(op_id)
        adjusted_batch = self._adjust_batch_size(batch_size)
        exact_key = (mode, adjusted_batch, height, width)
        exact = profile.executions.get(exact_key)
        if exact is not None:
            if exact.successful:
                return exact
            raise UnsupportedProfileError(
                f"Operator {op_id!r} mode={mode!r} at batch={adjusted_batch}, "
                f"shape={height}x{width} is unsupported: "
                f"{exact.error or 'incomplete benchmark result'}"
            )

        resolution_error = self.failed_resolutions.get((height, width))
        if resolution_error is not None:
            raise UnsupportedProfileError(
                f"Resolution {height}x{width} is unsupported because benchmark "
                f"capture failed: {resolution_error}"
            )

        candidates = [
            (abs(key[2] - height) + abs(key[3] - width), key, execution)
            for key, execution in profile.executions.items()
            if key[0] == mode and key[1] == adjusted_batch and execution.successful
        ]
        if not candidates:
            raise MissingProfileError(
                f"No runtime profile for operator {op_id!r} mode={mode!r}, "
                f"batch={adjusted_batch}, shape={height}x{width}"
            )
        return min(candidates, key=lambda item: (item[0], item[1]))[2]

    def execution_latency(
        self, op_id: str, mode: str, batch_size: int, height: int, width: int
    ) -> float:
        latency = self.execution(op_id, mode, batch_size, height, width).latency_seconds
        assert latency is not None
        return latency

    def activation_memory(
        self, op_id: str, mode: str, batch_size: int, height: int, width: int
    ) -> int:
        memory = self.execution(
            op_id, mode, batch_size, height, width
        ).activation_memory_bytes
        assert memory is not None
        return memory

    def model_memory(self, op_id: str) -> int:
        return self._operator(op_id).model_memory_bytes

    def loading_latency(self, op_id: str) -> float:
        return self._operator(op_id).host_to_gpu_latency_seconds

    def max_peak_total_memory(self, op_id: str) -> int:
        profile = self._operator(op_id)
        activation_memories = [
            int(execution.activation_memory_bytes)
            for execution in profile.executions.values()
            if execution.successful
        ]
        if not activation_memories:
            raise MissingProfileError(
                f"Operator {op_id!r} has no successful execution profile"
            )
        return profile.model_memory_bytes + max(activation_memories)

    def peak_total_memory(
        self, op_id: str, mode: str, batch_size: int, height: int, width: int
    ) -> int:
        return self.model_memory(op_id) + self.activation_memory(
            op_id, mode, batch_size, height, width
        )

    def validate_execution(
        self, op_id: str, mode: str, batch_size: int, height: int, width: int
    ) -> None:
        if op_id in PROFILE_FREE_OPERATOR_IDS:
            return
        self.execution(op_id, mode, batch_size, height, width)

    @staticmethod
    def _workflow_nodes(workflow: "Workflow") -> Iterable[Any]:
        yield from workflow.workflow_nodes
        for region in workflow.regions:
            for program in region.subprograms():
                yield from program.iter_nodes()

    def validate_workflow(self, workflow: "Workflow") -> None:
        missing = sorted(
            {
                node.op.id
                for node in self._workflow_nodes(workflow)
                if node.op.id not in PROFILE_FREE_OPERATOR_IDS
                and node.op.id not in self.operators
            }
        )
        if missing:
            raise MissingProfileError(
                "Workflow contains operators not covered by the startup benchmark: "
                + ", ".join(missing)
            )

    def to_scheduling_profiles(self) -> Dict[str, Any]:
        from diflow.backend.scheduler.scheduling_core import ModelProfile

        return {
            op_id: ModelProfile(
                loading_latency=profile.host_to_gpu_latency_seconds,
                execution_latencies={
                    key: float(execution.latency_seconds)
                    for key, execution in profile.executions.items()
                    if execution.successful
                },
            )
            for op_id, profile in self.operators.items()
        }
