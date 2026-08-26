from __future__ import annotations

import logging
import sys
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Any, Dict, List

import torch


@dataclass
class FreeingTask:
    tensor: torch.Tensor


@dataclass
class FetchingTask:
    id: str
    tensor_info: Dict[str, Any]
    size: List[int]
    dtype: torch.dtype
    remote_worker_rank: int


class BaseDataEngine(ABC):
    """Asynchronous intermediate-tensor transfer backend."""

    backend_name: str

    def __init__(self, *, device_id: int, worker_id: int) -> None:
        self.device_id = device_id
        self.worker_id = worker_id
        self.running = False
        self.received_tensors: Dict[str, torch.Tensor] = {}
        self.fetch_errors: Dict[str, BaseException] = {}
        self.tensor_arrival: Dict[str, threading.Event] = {}
        self.freeing_task_queue: Queue[FreeingTask] = Queue()
        self.fetching_task_queue: Queue[FetchingTask] = Queue()
        self.freeing_thread = threading.Thread(
            target=self._freeing_loop,
            name=f"{self.backend_name}-free-{worker_id}",
        )
        self.fetching_thread = threading.Thread(
            target=self._fetching_loop,
            name=f"{self.backend_name}-fetch-{worker_id}",
        )

        self.logger = logging.getLogger(
            f"{type(self).__name__}-{worker_id}(device: {device_id})"
        )
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
            self.logger.addHandler(handler)

    def _freeing_loop(self) -> None:
        while self.running or not self.freeing_task_queue.empty():
            try:
                task = self.freeing_task_queue.get(timeout=0.1)
            except Empty:
                continue
            try:
                self._free_tensor(task.tensor)
            except Exception:
                self.logger.exception("Failed to free an intermediate tensor")

    def _fetching_loop(self) -> None:
        while self.running or not self.fetching_task_queue.empty():
            try:
                task = self.fetching_task_queue.get(timeout=0.1)
            except Empty:
                continue
            try:
                tensor = self._fetch_tensor(task)
            except BaseException as error:
                self.fetch_errors[task.id] = error
                self._signal_arrival(task.id)
                continue
            self.received_tensors[task.id] = tensor
            self._signal_arrival(task.id)

    def _signal_arrival(self, tensor_id: str) -> None:
        arrival = self.tensor_arrival.setdefault(tensor_id, threading.Event())
        arrival.set()

    def start(self) -> None:
        self.running = True
        self.freeing_thread.start()
        self.fetching_thread.start()

    def stop(self) -> None:
        if not self.running:
            return
        self.running = False
        self.freeing_thread.join()
        self.fetching_thread.join()
        self._shutdown()
        self.logger.info("Data engine stopped")

    def __enter__(self) -> BaseDataEngine:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    def submit_fetch_task(self, task: FetchingTask) -> None:
        if not self.running:
            raise RuntimeError("Data engine is not running")
        self.fetching_task_queue.put(task)

    def submit_free_task(self, task: FreeingTask) -> None:
        if not self.running:
            raise RuntimeError("Data engine is not running")
        self.freeing_task_queue.put(task)

    def get(self, tensor_id: str, timeout: float = 60.0) -> torch.Tensor:
        arrival = self.tensor_arrival.setdefault(tensor_id, threading.Event())
        while not arrival.wait(timeout=timeout):
            self.logger.debug(
                "Timeout waiting for tensor %s; fetch queue size=%s",
                tensor_id,
                self.fetching_task_queue.qsize(),
            )

        self.tensor_arrival.pop(tensor_id, None)
        error = self.fetch_errors.pop(tensor_id, None)
        if error is not None:
            raise RuntimeError(f"Failed to fetch tensor {tensor_id}") from error
        return self.received_tensors.pop(tensor_id)

    @abstractmethod
    def store_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Make a tensor remotely transferable and return its local GPU tensor."""

    @abstractmethod
    def get_tensor_handle(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """Return backend-specific JSON metadata for a stored tensor."""

    @abstractmethod
    def _fetch_tensor(self, task: FetchingTask) -> torch.Tensor:
        pass

    @abstractmethod
    def _free_tensor(self, tensor: torch.Tensor) -> None:
        pass

    def _shutdown(self) -> None:
        pass
