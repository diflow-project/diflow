"""
Exclusive scheduler implementation for EXCLUSIVE scheduling policy.

This scheduler ensures that each request is assigned to a dedicated worker,
providing exclusive access to worker resources for each request.
"""

from typing import Any, Dict, List

from .base_scheduler import Scheduler
from .task import Task
from .utils import SchedulingPolicy, have_same_patches


class ExclusiveScheduler(Scheduler):
    """Scheduler implementation for EXCLUSIVE scheduling policy."""

    def __init__(
        self,
        all_workers_info: Dict[int, Dict[str, Any]],
    ):
        super().__init__(SchedulingPolicy.EXCLUSIVE, all_workers_info)

        # map request_id -> worker_rank
        self.request_worker_map: Dict[str, str] = {}

    def _initialize_custom_worker_status(self):
        """Initialize worker status for exclusive scheduling."""
        for worker_rank, _ in self.all_workers_info.items():
            # map worker_rank -> request_id
            self.worker_status[worker_rank]["request_id"] = None

    async def check_worker_availability(self, task: Task) -> bool:
        """Check if any worker is available for exclusive scheduling."""
        async with self.worker_status_lock:
            request_id = task.request_id
            if request_id in self.request_worker_map:
                return True

            for worker_rank, _ in self.all_workers_info.items():
                if self.worker_status[worker_rank]["request_id"] is None:
                    return True
            return False

    async def select_worker_for_task_group(
        self,
        task_group: List[Task],
    ) -> int:
        """Select a worker for exclusive scheduling."""
        async with self.worker_status_lock:
            request_id = task_group[0].request_id
            if request_id in self.request_worker_map:
                selected_worker_rank = self.request_worker_map[request_id]
            else:
                for worker_rank, _ in self.all_workers_info.items():
                    if self.worker_status[worker_rank]["request_id"] is None:
                        selected_worker_rank = worker_rank
                        self.request_worker_map[request_id] = worker_rank
                        self.worker_status[worker_rank]["request_id"] = request_id
                        break

            assert (
                selected_worker_rank is not None
            ), "No worker available for exclusive scheduling"
            return selected_worker_rank

    async def update_custom_worker_status_after_completion(
        self,
        worker_rank: int,
        task_id: str,
    ):
        """Update worker status after task completion for exclusive scheduling."""
        pass

    async def cleanup_request(self, request_id: str):
        """Clean up request-specific data for exclusive scheduling."""
        async with self.worker_status_lock:
            if request_id in self.request_worker_map:
                worker_rank = self.request_worker_map[request_id]
                self.worker_status[worker_rank]["request_id"] = None
                del self.request_worker_map[request_id]

    def can_group_tasks(self, task1: Task, task2: Task) -> bool:
        """Check if two tasks can be grouped together for exclusive scheduling."""
        if task1.workflow_node.op.id != task2.workflow_node.op.id:
            return False

        if task1.workflow_node.mode != task2.workflow_node.mode:
            return False

        # Check if patches are the same
        if not have_same_patches(task1.workflow_node.op, task2.workflow_node.op):
            return False

        # For exclusive scheduling, we can be more restrictive
        # Only allow grouping within the same request since each request gets its own worker
        if task1.request_id != task2.request_id:
            return False

        if task1.workflow_node.op.id == "IndexedTensor":
            if task1.request_id != task2.request_id:
                return False

            if task1.input_map["tensor"] != task2.input_map["tensor"]:
                return False

            return True

        # If one of the tasks has lazy inputs, they cannot be grouped together.
        if len(task1.lazy_inputs) > 0 or len(task2.lazy_inputs) > 0:
            return False

        return True

    async def select_tasks_for_grouping(
        self, ready_tasks: List[Task], model_batch_configs: Dict[str, Dict[str, int]]
    ) -> List[Task]:
        """Select tasks for grouping using exclusive scheduling logic."""
        task_group = []

        # Pop the first task from ready_tasks
        first_task = ready_tasks.pop(0)
        task_group.append(first_task)
        batch_size = 1

        # For exclusive scheduling, we can be more aggressive with batching
        # since each request gets its own worker
        if first_task.workflow_node.op.id in model_batch_configs:
            # Use throughput mode for exclusive scheduling to maximize batching
            batch_size = model_batch_configs[first_task.workflow_node.op.id][
                "throughput_mode"
            ]
        elif first_task.workflow_node.op.id == "IndexedTensor":
            batch_size = float("inf")

        # Iterate over a copy of ready_tasks to allow removal
        i = 0
        while len(task_group) < batch_size and i < len(ready_tasks):
            candidate_task = ready_tasks[i]
            if self.can_group_tasks(task_group[0], candidate_task):
                task_group.append(ready_tasks.pop(i))
                # Do not increment i, as the list has shifted
            else:
                i += 1

        return task_group
