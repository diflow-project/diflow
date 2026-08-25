"""Stateful scheduling core backends for dynamic scheduling."""

from __future__ import annotations

import logging
import math
import os
from abc import ABC, abstractmethod
from bisect import bisect_left
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TransferProfile:
    block_sizes: Tuple[int, ...]
    fetch_overheads_us: Tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.block_sizes) != len(self.fetch_overheads_us):
            raise ValueError(
                "block sizes and transfer overheads must have equal length"
            )
        if tuple(sorted(self.block_sizes)) != self.block_sizes:
            raise ValueError("transfer profile block sizes must be sorted")


@dataclass(frozen=True)
class ModelProfile:
    loading_latency: float
    execution_latencies: Mapping[Tuple[object, ...], float]


def lookup_execution_latency(
    profile: ModelProfile,
    mode: str,
    batch_size: int,
    height: int = 0,
    width: int = 0,
) -> float:
    """Use an exact shape, then same-batch nearest-resolution fallback."""
    adjusted_batch_size = _next_power_of_2(batch_size)
    if height > 0 and width > 0:
        exact = profile.execution_latencies.get(
            (mode, adjusted_batch_size, height, width)
        )
        if exact is not None:
            return exact

    candidates = [
        (abs(key[2] - height) + abs(key[3] - width), latency)
        for key, latency in profile.execution_latencies.items()
        if len(key) == 4
        and key[0] == mode
        and key[1] == adjusted_batch_size
        and key[2] > 0
        and key[3] > 0
    ]
    return min(candidates, default=(math.inf, math.inf))[1]


def max_profiled_batch_size(
    profile: Optional[ModelProfile], mode: str, height: int, width: int
) -> Optional[int]:
    """Largest successful batch at the exact or nearest profiled resolution."""
    if profile is None:
        return None
    shaped = [
        (key, latency)
        for key, latency in profile.execution_latencies.items()
        if len(key) == 4 and key[0] == mode and math.isfinite(latency)
    ]
    if shaped:
        exact = [key for key, _ in shaped if key[2:] == (height, width)]
        if exact:
            return max(int(key[1]) for key in exact)
        nearest_resolution = min(
            {(int(key[2]), int(key[3])) for key, _ in shaped},
            key=lambda resolution: abs(resolution[0] - height)
            + abs(resolution[1] - width),
        )
        return max(int(key[1]) for key, _ in shaped if key[2:] == nearest_resolution)

    return None


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    model_name: str
    mode: str
    batch_size: int
    uses_model_profile: bool
    tensor_offsets: Tuple[int, ...]
    source_worker_ranks: Tuple[int, ...]
    source_host_ids: Tuple[int, ...]
    source_sizes_bytes: Tuple[int, ...]
    height: int = 0
    width: int = 0

    def __post_init__(self) -> None:
        source_count = len(self.source_worker_ranks)
        if (
            len(self.source_host_ids) != source_count
            or len(self.source_sizes_bytes) != source_count
        ):
            raise ValueError("tensor source arrays must have equal length")
        if not self.tensor_offsets or self.tensor_offsets[0] != 0:
            raise ValueError("tensor offsets must start at zero")
        if self.tensor_offsets[-1] != source_count:
            raise ValueError("last tensor offset must equal the source count")
        if any(
            left > right
            for left, right in zip(self.tensor_offsets, self.tensor_offsets[1:])
        ):
            raise ValueError("tensor offsets must be non-decreasing")


@dataclass(frozen=True)
class CostBreakdown:
    queue: float
    transfer: float
    loading: float
    execution: float
    total: float

    @property
    def reserved_latency(self) -> float:
        return self.transfer + self.loading + self.execution


@dataclass(frozen=True)
class SelectionResult:
    worker_rank: int
    cost: CostBreakdown
    backend: str


@dataclass(frozen=True)
class Reservation:
    worker_rank: int
    cost: CostBreakdown


class SchedulingCoreBackend(ABC):
    """Backend-neutral state and operations used by DynamicScheduler."""

    name: str

    @abstractmethod
    def select_and_reserve(self, task: TaskSpec) -> Optional[SelectionResult]:
        pass

    @abstractmethod
    def reserve_on_worker(self, task: TaskSpec, worker_rank: int) -> SelectionResult:
        pass

    @abstractmethod
    def complete(self, task_id: str) -> bool:
        pass

    @abstractmethod
    def update_active_models(
        self, worker_rank: int, active_models: Iterable[str]
    ) -> None:
        pass

    @abstractmethod
    def available_worker_count(self) -> int:
        pass

    @abstractmethod
    def snapshot(self) -> Dict[int, Dict[str, object]]:
        pass


class PythonSchedulingCore(SchedulingCoreBackend):
    """Reference implementation for correctness tests and safe fallback."""

    name = "python"

    def __init__(
        self,
        worker_host_ids: Mapping[int, int],
        active_models: Mapping[int, Iterable[str]],
        model_profiles: Mapping[str, ModelProfile],
        intra_profile: TransferProfile,
        inter_profile: TransferProfile,
        worker_latency_threshold: float,
    ) -> None:
        if not worker_host_ids:
            raise ValueError("at least one worker is required")
        self._worker_ranks = tuple(worker_host_ids)
        self._worker_host_ids = dict(worker_host_ids)
        self._active_models = {
            rank: set(active_models.get(rank, ())) for rank in self._worker_ranks
        }
        self._model_profiles = dict(model_profiles)
        self._intra_profile = intra_profile
        self._inter_profile = inter_profile
        self._worker_latency_threshold = worker_latency_threshold
        self._queue_latencies = {rank: 0.0 for rank in self._worker_ranks}
        self._reservations: Dict[str, Reservation] = {}

    @staticmethod
    def _lookup_transfer_latency(size_bytes: int, profile: TransferProfile) -> float:
        if not profile.block_sizes:
            return math.inf
        index = bisect_left(profile.block_sizes, size_bytes)
        if index == len(profile.block_sizes):
            index -= 1
        return profile.fetch_overheads_us[index] / 1e6

    def _transfer_latency(self, task: TaskSpec, worker_rank: int) -> float:
        dst_host_id = self._worker_host_ids[worker_rank]
        latency = 0.0
        for tensor_index in range(len(task.tensor_offsets) - 1):
            begin = task.tensor_offsets[tensor_index]
            end = task.tensor_offsets[tensor_index + 1]
            if begin == end:
                continue

            selected = begin
            for source_index in range(begin, end):
                if task.source_host_ids[source_index] == dst_host_id:
                    selected = source_index
                    break

            if task.source_worker_ranks[selected] == worker_rank:
                continue
            intra_node = task.source_host_ids[selected] == dst_host_id
            profile = self._intra_profile if intra_node else self._inter_profile
            latency += self._lookup_transfer_latency(
                task.source_sizes_bytes[selected], profile
            )
        return latency

    def _cost(self, task: TaskSpec, worker_rank: int) -> CostBreakdown:
        queue_latency = self._queue_latencies[worker_rank]
        transfer_latency = self._transfer_latency(task, worker_rank)
        loading_latency = 0.0
        execution_latency = 0.0

        if task.uses_model_profile:
            profile = self._model_profiles.get(task.model_name)
            if profile is None:
                loading_latency = math.inf
                execution_latency = math.inf
            else:
                if task.model_name not in self._active_models[worker_rank]:
                    loading_latency = profile.loading_latency
                execution_latency = lookup_execution_latency(
                    profile,
                    task.mode,
                    task.batch_size,
                    task.height,
                    task.width,
                )

        reserved_latency = transfer_latency + loading_latency + execution_latency
        return CostBreakdown(
            queue=queue_latency,
            transfer=transfer_latency,
            loading=loading_latency,
            execution=execution_latency,
            total=queue_latency + reserved_latency,
        )

    def _result_for_reservation(self, reservation: Reservation) -> SelectionResult:
        return SelectionResult(reservation.worker_rank, reservation.cost, self.name)

    def _record_reservation(
        self, task_id: str, worker_rank: int, cost: CostBreakdown
    ) -> SelectionResult:
        reservation = Reservation(worker_rank=worker_rank, cost=cost)
        self._reservations[task_id] = reservation
        self._queue_latencies[worker_rank] += cost.reserved_latency
        return self._result_for_reservation(reservation)

    def select_and_reserve(self, task: TaskSpec) -> Optional[SelectionResult]:
        existing = self._reservations.get(task.task_id)
        if existing is not None:
            return self._result_for_reservation(existing)

        best_rank = None
        best_cost = None
        for worker_rank in self._worker_ranks:
            if self._queue_latencies[worker_rank] > self._worker_latency_threshold:
                continue
            cost = self._cost(task, worker_rank)
            if best_cost is None or cost.total < best_cost.total:
                best_rank = worker_rank
                best_cost = cost

        if best_rank is None or best_cost is None or math.isinf(best_cost.total):
            return None
        return self._record_reservation(task.task_id, best_rank, best_cost)

    def reserve_on_worker(self, task: TaskSpec, worker_rank: int) -> SelectionResult:
        if worker_rank not in self._worker_host_ids:
            raise ValueError(f"unknown worker rank: {worker_rank}")
        existing = self._reservations.get(task.task_id)
        if existing is not None:
            if existing.worker_rank != worker_rank:
                raise ValueError(
                    f"task {task.task_id} is already reserved on worker "
                    f"{existing.worker_rank}"
                )
            return self._result_for_reservation(existing)
        return self._record_reservation(
            task.task_id, worker_rank, self._cost(task, worker_rank)
        )

    def complete(self, task_id: str) -> bool:
        reservation = self._reservations.pop(task_id, None)
        if reservation is None:
            return False
        worker_rank = reservation.worker_rank
        if math.isinf(reservation.cost.reserved_latency):
            self._queue_latencies[worker_rank] = sum(
                item.cost.reserved_latency
                for item in self._reservations.values()
                if item.worker_rank == worker_rank
            )
        else:
            self._queue_latencies[worker_rank] = max(
                0.0,
                self._queue_latencies[worker_rank] - reservation.cost.reserved_latency,
            )
        return True

    def update_active_models(
        self, worker_rank: int, active_models: Iterable[str]
    ) -> None:
        if worker_rank not in self._active_models:
            raise ValueError(f"unknown worker rank: {worker_rank}")
        self._active_models[worker_rank] = set(active_models)

    def available_worker_count(self) -> int:
        return sum(
            latency <= self._worker_latency_threshold
            for latency in self._queue_latencies.values()
        )

    def snapshot(self) -> Dict[int, Dict[str, object]]:
        reservations_by_worker: Dict[int, Dict[str, float]] = {
            rank: {} for rank in self._worker_ranks
        }
        for task_id, reservation in self._reservations.items():
            reservations_by_worker[reservation.worker_rank][
                task_id
            ] = reservation.cost.reserved_latency
        return {
            rank: {
                "queue_latency": self._queue_latencies[rank],
                "reservations": reservations_by_worker[rank],
            }
            for rank in self._worker_ranks
        }


class CppSchedulingCore(SchedulingCoreBackend):
    """Thin adapter around the AOT-compiled stateful C++ core."""

    name = "cpp"

    def __init__(
        self,
        worker_host_ids: Mapping[int, int],
        active_models: Mapping[int, Iterable[str]],
        model_profiles: Mapping[str, ModelProfile],
        intra_profile: TransferProfile,
        inter_profile: TransferProfile,
        worker_latency_threshold: float,
    ) -> None:
        from . import _scheduling_core

        worker_ranks = list(worker_host_ids)
        model_names = list(model_profiles)
        loading_latencies = [
            model_profiles[name].loading_latency for name in model_names
        ]
        execution_model_names = []
        execution_modes = []
        execution_batch_sizes = []
        execution_heights = []
        execution_widths = []
        execution_latencies = []
        for model_name, profile in model_profiles.items():
            for key, latency in profile.execution_latencies.items():
                mode, batch_size = key[:2]
                height, width = key[2:] if len(key) == 4 else (0, 0)
                execution_model_names.append(model_name)
                execution_modes.append(mode)
                execution_batch_sizes.append(batch_size)
                execution_heights.append(height)
                execution_widths.append(width)
                execution_latencies.append(latency)

        self._core = _scheduling_core.SchedulingCore(
            worker_ranks,
            [worker_host_ids[rank] for rank in worker_ranks],
            [list(active_models.get(rank, ())) for rank in worker_ranks],
            model_names,
            loading_latencies,
            execution_model_names,
            execution_modes,
            execution_batch_sizes,
            execution_heights,
            execution_widths,
            execution_latencies,
            list(intra_profile.block_sizes),
            list(intra_profile.fetch_overheads_us),
            list(inter_profile.block_sizes),
            list(inter_profile.fetch_overheads_us),
            worker_latency_threshold,
        )

    @staticmethod
    def _task_arguments(task: TaskSpec) -> tuple:
        return (
            task.task_id,
            task.model_name,
            task.mode,
            task.batch_size,
            task.height,
            task.width,
            task.uses_model_profile,
            list(task.tensor_offsets),
            list(task.source_worker_ranks),
            list(task.source_host_ids),
            list(task.source_sizes_bytes),
        )

    def _result(self, payload: Mapping[str, object]) -> SelectionResult:
        cost_payload = payload["cost"]
        cost = CostBreakdown(
            queue=float(cost_payload["queue"]),
            transfer=float(cost_payload["transfer"]),
            loading=float(cost_payload["loading"]),
            execution=float(cost_payload["execution"]),
            total=float(cost_payload["total"]),
        )
        return SelectionResult(int(payload["worker_rank"]), cost, self.name)

    def select_and_reserve(self, task: TaskSpec) -> Optional[SelectionResult]:
        payload = self._core.select_and_reserve(*self._task_arguments(task))
        return None if payload is None else self._result(payload)

    def reserve_on_worker(self, task: TaskSpec, worker_rank: int) -> SelectionResult:
        payload = self._core.reserve_on_worker(*self._task_arguments(task), worker_rank)
        return self._result(payload)

    def complete(self, task_id: str) -> bool:
        return bool(self._core.complete(task_id))

    def update_active_models(
        self, worker_rank: int, active_models: Iterable[str]
    ) -> None:
        self._core.update_active_models(worker_rank, list(active_models))

    def available_worker_count(self) -> int:
        return int(self._core.available_worker_count())

    def snapshot(self) -> Dict[int, Dict[str, object]]:
        return dict(self._core.snapshot())


def create_scheduling_core(
    worker_host_ids: Mapping[int, int],
    active_models: Mapping[int, Iterable[str]],
    model_profiles: Mapping[str, ModelProfile],
    intra_profile: TransferProfile,
    inter_profile: TransferProfile,
    worker_latency_threshold: float,
    backend: Optional[str] = None,
) -> SchedulingCoreBackend:
    """Create one authoritative backend; fallback is initialization-only."""
    selected_backend = (
        backend or os.environ.get("DIFLOW_SCHEDULING_CORE", "auto")
    ).lower()
    if selected_backend not in {"auto", "cpp", "python"}:
        raise ValueError("DIFLOW_SCHEDULING_CORE must be one of: auto, cpp, python")

    arguments = (
        worker_host_ids,
        active_models,
        model_profiles,
        intra_profile,
        inter_profile,
        worker_latency_threshold,
    )
    if selected_backend in {"auto", "cpp"}:
        try:
            return CppSchedulingCore(*arguments)
        except (ImportError, OSError) as exc:
            if selected_backend == "cpp":
                raise RuntimeError("C++ SchedulingCore is unavailable") from exc
            logger.warning(
                "C++ SchedulingCore is unavailable; using Python backend: %s", exc
            )
    return PythonSchedulingCore(*arguments)


def _next_power_of_2(value: int) -> int:
    if value < 1:
        raise ValueError("batch size must be positive")
    return 1 << (value - 1).bit_length()
