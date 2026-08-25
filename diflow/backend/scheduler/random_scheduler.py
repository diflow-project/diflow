"""
Random scheduler implementation for RANDOM scheduling policy.

This scheduler randomly selects workers for task groups,
providing a simple load balancing approach.
"""

import random
from typing import Any, Dict, List

from .base_scheduler import Scheduler
from .task import Task
from .utils import SchedulingPolicy, have_same_patches


class RandomScheduler(Scheduler):
    """Scheduler implementation for RANDOM scheduling policy."""

    def __init__(
        self,
        all_workers_info: Dict[int, Dict[str, Any]],
    ):
        super().__init__(SchedulingPolicy.RANDOM, all_workers_info)

    def _initialize_custom_worker_status(self):
        """Initialize worker status for random scheduling."""
        pass

    async def check_worker_availability(self, task: Task) -> bool:
        """Check if any worker is available for random scheduling."""
        return True  # Always available, will randomly select a worker

    async def select_worker_for_task_group(
        self,
        task_group: List[Task],
    ) -> int:
        """Select a worker randomly for random scheduling."""
        async with self.worker_status_lock:
            selected_worker_rank = random.choice(list(self.all_workers_info.keys()))

            return selected_worker_rank

    async def update_custom_worker_status_after_completion(
        self,
        worker_rank: int,
        task_id: str,
    ):
        """Update worker status after task completion for random scheduling."""
        pass

    async def cleanup_request(self, request_id: str):
        """Clean up request-specific data for random scheduling."""
        # No request-specific cleanup needed for random scheduling
        pass

    def can_group_tasks(self, task1: Task, task2: Task) -> bool:
        """Check if two tasks can be grouped together for random scheduling."""
        # Basic compatibility checks
        if task1.workflow_node.op.id != task2.workflow_node.op.id:
            return False

        if task1.workflow_node.mode != task2.workflow_node.mode:
            return False

        # Check if patches are the same
        if not have_same_patches(task1.workflow_node.op, task2.workflow_node.op):
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

        # For random scheduling, we use the same basic grouping rules
        # but allow more flexibility since worker selection is random
        return True

    async def select_tasks_for_grouping(
        self,
        ready_tasks: List[Task],
        request_arrival_times: Dict[str, float],
        node_depth: Dict[str, Dict[str, int]],
        model_batch_configs: Dict[str, Dict[str, int]],
    ) -> List[Task]:
        """Select tasks for grouping using random scheduling logic."""
        # For random scheduling, we use a simple approach:
        task_group = []

        # Pop the first task from ready_tasks
        first_task = ready_tasks.pop(0)
        task_group.append(first_task)
        batch_size = 1

        # Use throughput mode for random scheduling (simpler approach)
        if first_task.workflow_node.op.id in model_batch_configs:
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
