"""
Base scheduler class for task scheduling operations.

This module provides an abstract base class for different scheduling policies,
allowing each scheduler to implement scheduling functions independently.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from .task import Task, get_task_id
from .utils import SchedulingPolicy


class Scheduler(ABC):
    """
    Abstract base class for task schedulers.

    Each scheduler implementation handles different scheduling policies
    and provides a consistent interface for the coordinator.
    """

    def __init__(
        self,
        scheduling_policy: SchedulingPolicy,
        all_workers_info: Dict[int, Dict[str, Any]],
    ):
        self.scheduling_policy = scheduling_policy
        self.all_workers_info = all_workers_info
        self.worker_status_lock = asyncio.Lock()
        self.worker_status: Dict[int, Dict[str, Any]] = {}
        self._initialize_worker_status()

    def _get_default_worker_status(self) -> Dict[str, Any]:
        """Get default worker status fields that all schedulers should have."""
        return {
            "last_ping": 0.0,
            "task_group": set(),
            "active_models": [],
            "gpu_memory_info": {},
        }

    def _initialize_worker_status(self):
        """Initialize worker status."""
        for worker_rank, _ in self.all_workers_info.items():
            self.worker_status[worker_rank] = self._get_default_worker_status()
        self._initialize_custom_worker_status()

    @abstractmethod
    def _initialize_custom_worker_status(self):
        """Initialize custom worker status based on scheduling policy."""
        pass

    @abstractmethod
    async def check_worker_availability(self, task: Task) -> bool:
        """
        Check if any worker is available for scheduling.

        Args:
            task: Task to be scheduled

        Returns:
            bool: True if at least one worker is available, False otherwise.
        """
        pass

    @abstractmethod
    async def select_worker_for_task_group(
        self,
        task_group: List[Task],
    ) -> int:
        """
        Select a worker for a task group.

        Args:
            task_group: List of tasks to be scheduled together

        Returns:
            int: Selected worker rank
        """
        pass

    async def reserve_worker_for_task_group(
        self,
        worker_rank: int,
        task_group: List[Task],
    ):
        """Reserve work when the coordinator pins a task to a worker.

        Stateless schedulers do not need additional accounting. Stateful
        schedulers override this hook so pinned and normally selected tasks use
        the same reservation lifecycle.
        """
        return None

    async def update_worker_status_after_scheduling(
        self,
        worker_rank: int,
        task_group: List[Task],
    ):
        """
        Update worker status after scheduling a task group."""
        async with self.worker_status_lock:
            status = self.worker_status[worker_rank]
            task_id = get_task_id(task_group)
            status["task_group"].add(task_id)

    async def _update_default_worker_status_fields(
        self,
        worker_rank: int,
        active_models: List[str] = None,
        gpu_memory_info: Dict[str, Any] = None,
        task_id: str = None,
    ):
        """
        Update default worker status fields that all schedulers should update.

        Args:
            worker_rank: Worker that completed the task
            active_models: List of active models on the worker
            gpu_memory_info: GPU memory information
            task_id: Task ID, optional; by default, it is the node name of the first task in the task group
        """
        async with self.worker_status_lock:
            status = self.worker_status[worker_rank]
            status["last_ping"] = time.time()
            status["active_models"] = active_models
            status["gpu_memory_info"] = gpu_memory_info
            if task_id is not None:
                status["task_group"].discard(task_id)

    async def update_worker_status_after_completion(
        self,
        worker_rank: int,
        active_models: List[str],
        gpu_memory_info: Dict[str, Any],
        task_id: str = None,
    ):
        """
        Update worker status after task completion.

        Args:
            worker_rank: Worker that completed the task
            active_models: List of active models on the worker
            gpu_memory_info: GPU memory information
        """
        await self._update_default_worker_status_fields(
            worker_rank, active_models, gpu_memory_info, task_id
        )
        if task_id is not None:
            await self.update_custom_worker_status_after_completion(
                worker_rank, task_id
            )

    @abstractmethod
    async def update_custom_worker_status_after_completion(
        self,
        worker_rank: int,
        task_id: str = None,
    ):
        """
        Update custom worker status after task completion.

        Args:
            worker_rank: Worker that completed the task
            task_id: Task ID, optional; by default, it is the node name of the first task in the task group
        """
        pass

    @abstractmethod
    async def cleanup_request(self, request_id: str):
        """
        Clean up request-specific data when a request is completed.

        Args:
            request_id: Request ID to clean up
        """
        pass

    @abstractmethod
    def can_group_tasks(self, task1: Task, task2: Task) -> bool:
        """
        Check if two tasks can be grouped together based on scheduler-specific logic.

        Args:
            task1: First task to check
            task2: Second task to check

        Returns:
            bool: True if tasks can be grouped, False otherwise
        """
        pass

    @abstractmethod
    async def select_tasks_for_grouping(
        self, ready_tasks: List[Task], model_batch_configs: Dict[str, Dict[str, int]]
    ) -> List[Task]:
        """
        Select tasks for grouping based on scheduler-specific logic.

        Args:
            ready_tasks: List of tasks ready for scheduling
            model_batch_configs: Configuration for model batching

        Returns:
            List[Task]: Selected task group
        """
        pass

    def get_worker_status(self, worker_rank: int) -> Dict[str, Any]:
        """Get current worker status for a specific worker."""
        return self.worker_status[worker_rank]
