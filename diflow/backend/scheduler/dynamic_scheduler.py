"""Dynamic scheduler backed by a stateful scheduling core."""

import logging
import math
import random
import time
from typing import Any, Dict, List, Tuple

from diflow.operators.schedulers.base_scheduler import BaseScheduler
from diflow.profiling.runtime_profile import RuntimeProfile

from .base_scheduler import Scheduler
from .scheduling_core import (
    TaskSpec,
    TransferProfile,
    create_scheduling_core,
    max_profiled_batch_size,
)
from .task import Task, get_task_id
from .utils import SchedulingPolicy, have_same_patches

logger = logging.getLogger(__name__)

DTYPE_BYTES_MAP = {
    "torch.float32": 4,
    "torch.float16": 2,
    "torch.bfloat16": 2,
    "torch.half": 2,
    "torch.int64": 8,
    "torch.int32": 4,
    "torch.int16": 2,
    "torch.int8": 1,
    "torch.uint8": 1,
    "torch.bool": 1,
}

PROFILE_FREE_MODEL_NAMES = {"IndexedTensor", "GuidanceTensor"}


class DynamicScheduler(Scheduler):
    """Scheduler implementation for the DYNAMIC scheduling policy."""

    def __init__(
        self,
        all_workers_info: Dict[int, Dict[str, Any]],
        runtime_profile: RuntimeProfile,
    ):
        self.runtime_profile = runtime_profile
        super().__init__(SchedulingPolicy.DYNAMIC, all_workers_info)

        self.worker_host_ids = self._build_worker_host_ids()

        # Avoid selecting a worker whose queued work exceeds 10 ms.
        self.worker_latency_threshold = 0.01
        self.model_profiles = runtime_profile.to_scheduling_profiles()
        self.scheduling_core = create_scheduling_core(
            worker_host_ids=self.worker_host_ids,
            active_models={
                rank: status["active_models"]
                for rank, status in self.worker_status.items()
            },
            model_profiles=self.model_profiles,
            intra_profile=TransferProfile(
                runtime_profile.intra_transfer.block_sizes,
                runtime_profile.intra_transfer.fetch_overheads_us,
            ),
            inter_profile=TransferProfile(
                runtime_profile.inter_transfer.block_sizes,
                runtime_profile.inter_transfer.fetch_overheads_us,
            ),
            worker_latency_threshold=self.worker_latency_threshold,
        )
        logger.info(
            "Dynamic scheduler initialized scheduling_core=%s worker_count=%d",
            self.scheduling_core.name,
            len(self.all_workers_info),
        )

    def _initialize_custom_worker_status(self):
        """Scheduling queue state lives exclusively in SchedulingCore."""
        pass

    def _build_worker_host_ids(self) -> Dict[int, int]:
        hostname_to_id = {}
        worker_host_ids = {}
        for worker_rank, worker_info in self.all_workers_info.items():
            hostname = worker_info["hostname"]
            if hostname not in hostname_to_id:
                hostname_to_id[hostname] = len(hostname_to_id)
            worker_host_ids[worker_rank] = hostname_to_id[hostname]
        return worker_host_ids

    @staticmethod
    def _get_tensor_size_bytes(tensor_info: Dict[str, Any]) -> int:
        dtype_str = str(tensor_info["dtype"]).lower()
        if dtype_str not in DTYPE_BYTES_MAP:
            raise ValueError(f"Unknown dtype: {dtype_str}")
        num_elements = math.prod(tensor_info["size"])
        return num_elements * DTYPE_BYTES_MAP[dtype_str]

    @staticmethod
    def _task_resolution(task: Task) -> Tuple[int, int]:
        inputs = task.inputs or {}
        try:
            return int(inputs.get("height", 0)), int(inputs.get("width", 0))
        except (TypeError, ValueError):
            return 0, 0

    def _build_task_spec(self, task_group: List[Task]) -> TaskSpec:
        workflow_node = task_group[0].workflow_node
        tensor_offsets = [0]
        source_worker_ranks = []
        source_host_ids = []
        source_sizes_bytes = []

        for task in task_group:
            for worker_tensorinfo_dict in (task.node_input_locations or {}).values():
                for source_worker_rank, tensor_info in worker_tensorinfo_dict.items():
                    source_worker_ranks.append(source_worker_rank)
                    source_host_ids.append(self.worker_host_ids[source_worker_rank])
                    source_sizes_bytes.append(self._get_tensor_size_bytes(tensor_info))
                tensor_offsets.append(len(source_worker_ranks))

        height, width = self._task_resolution(task_group[0])
        is_scheduler_op = isinstance(workflow_node.op, BaseScheduler)
        return TaskSpec(
            task_id=get_task_id(task_group),
            model_name=workflow_node.op.id,
            mode=workflow_node.mode,
            batch_size=len(task_group),
            height=height,
            width=width,
            uses_model_profile=(
                not is_scheduler_op
                and workflow_node.op.id not in PROFILE_FREE_MODEL_NAMES
            ),
            tensor_offsets=tuple(tensor_offsets),
            source_worker_ranks=tuple(source_worker_ranks),
            source_host_ids=tuple(source_host_ids),
            source_sizes_bytes=tuple(source_sizes_bytes),
        )

    async def check_worker_availability(self, task: Task) -> bool:
        async with self.worker_status_lock:
            return self.scheduling_core.available_worker_count() > 0

    async def select_worker_for_task_group(self, task_group: List[Task]) -> int:
        """Atomically select a worker and reserve its estimated queue latency."""
        selection_start_time = time.perf_counter()
        task_spec = self._build_task_spec(task_group)
        async with self.worker_status_lock:
            result = self.scheduling_core.select_and_reserve(task_spec)
            if result is None:
                logger.warning(
                    "No worker passed the queue threshold; selecting a random worker"
                )
                worker_rank = random.choice(list(self.all_workers_info))
                result = self.scheduling_core.reserve_on_worker(task_spec, worker_rank)

        selection_latency_ms = (time.perf_counter() - selection_start_time) * 1000
        logger.info(
            "select_worker_for_task_group backend=%s latency_ms=%.3f "
            "worker_count=%d task_group_size=%d selected_worker=%s task_id=%s "
            "queue=%.6f transfer=%.6f loading=%.6f execution=%.6f total=%.6f",
            result.backend,
            selection_latency_ms,
            len(self.all_workers_info),
            len(task_group),
            result.worker_rank,
            task_spec.task_id,
            result.cost.queue,
            result.cost.transfer,
            result.cost.loading,
            result.cost.execution,
            result.cost.total,
        )
        return result.worker_rank

    async def reserve_worker_for_task_group(
        self, worker_rank: int, task_group: List[Task]
    ):
        """Account for coordinator-pinned denoise scheduler tasks."""
        task_spec = self._build_task_spec(task_group)
        async with self.worker_status_lock:
            self.scheduling_core.reserve_on_worker(task_spec, worker_rank)

    async def update_worker_status_after_completion(
        self,
        worker_rank: int,
        active_models: List[str],
        gpu_memory_info: Dict[str, Any],
        task_id: str = None,
    ):
        """Update reporting fields and SchedulingCore in one critical section."""
        async with self.worker_status_lock:
            status = self.worker_status[worker_rank]
            status["last_ping"] = time.time()
            status["active_models"] = active_models or []
            status["gpu_memory_info"] = gpu_memory_info or {}
            self.scheduling_core.update_active_models(
                worker_rank, status["active_models"]
            )
            if task_id is not None:
                status["task_group"].discard(task_id)
                if not self.scheduling_core.complete(task_id):
                    logger.warning(
                        "Completion received for unknown scheduling reservation "
                        "worker=%s task_id=%s",
                        worker_rank,
                        task_id,
                    )

    async def update_custom_worker_status_after_completion(
        self,
        worker_rank: int,
        task_id: str,
    ):
        """Compatibility hook for callers using the base completion flow."""
        async with self.worker_status_lock:
            if not self.scheduling_core.complete(task_id):
                logger.warning(
                    "Completion received for unknown scheduling reservation "
                    "worker=%s task_id=%s",
                    worker_rank,
                    task_id,
                )

    async def cleanup_request(self, request_id: str):
        """No request-scoped SchedulingCore state exists beyond reservations."""
        pass

    def get_worker_status(self, worker_rank: int) -> Dict[str, Any]:
        """Merge reporting fields with an authoritative scheduling snapshot."""
        status = dict(super().get_worker_status(worker_rank))
        core_status = self.scheduling_core.snapshot()[worker_rank]
        status["estimated_latency"] = dict(core_status["reservations"])
        status["estimated_queue_latency"] = core_status["queue_latency"]
        return status

    def can_group_tasks(self, task1: Task, task2: Task) -> bool:
        if task1.workflow_node.op.id != task2.workflow_node.op.id:
            return False
        if task1.workflow_node.mode != task2.workflow_node.mode:
            return False
        if self._task_resolution(task1) != self._task_resolution(task2):
            return False
        if not have_same_patches(task1.workflow_node.op, task2.workflow_node.op):
            return False

        if task1.workflow_node.op.id == "IndexedTensor":
            if task1.request_id != task2.request_id:
                return False
            return task1.input_map["tensor"] == task2.input_map["tensor"]

        if len(task1.lazy_inputs) > 0 or len(task2.lazy_inputs) > 0:
            return False
        return True

    async def select_tasks_for_grouping(
        self, ready_tasks: List[Task], model_batch_configs: Dict[str, Dict[str, int]]
    ) -> List[Task]:
        task_group = []
        first_task = ready_tasks.pop(0)
        task_group.append(first_task)
        batch_size = 1
        batch_mode = "throughput_mode"

        async with self.worker_status_lock:
            num_available_workers = self.scheduling_core.available_worker_count()

        num_required_workers_latency_mode = 0
        task_count_dict = {}
        for task in ready_tasks:
            task_count_dict.setdefault(task.workflow_node.op.id, 0)
            task_count_dict[task.workflow_node.op.id] += 1
        for op_id, count in task_count_dict.items():
            if op_id not in model_batch_configs:
                num_required_workers_latency_mode += count
            else:
                num_required_workers_latency_mode += math.ceil(
                    count / model_batch_configs[op_id]["latency_mode"]
                )

        if num_required_workers_latency_mode <= num_available_workers:
            batch_mode = "latency_mode"

        logger.debug("batch_mode for %s: %s", first_task.workflow_node.name, batch_mode)

        if first_task.workflow_node.op.id in model_batch_configs:
            batch_size = model_batch_configs[first_task.workflow_node.op.id][batch_mode]
        elif first_task.workflow_node.op.id == "IndexedTensor":
            batch_size = float("inf")

        height, width = self._task_resolution(first_task)
        profiled_limit = max_profiled_batch_size(
            self.model_profiles.get(first_task.workflow_node.op.id),
            first_task.workflow_node.mode,
            height,
            width,
        )
        if profiled_limit is not None:
            batch_size = min(batch_size, profiled_limit)

        logger.debug("batch_size for %s: %s", first_task.workflow_node.name, batch_size)

        index = 0
        while len(task_group) < batch_size and index < len(ready_tasks):
            candidate_task = ready_tasks[index]
            if self.can_group_tasks(task_group[0], candidate_task):
                task_group.append(ready_tasks.pop(index))
            else:
                index += 1
        return task_group
