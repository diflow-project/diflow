from __future__ import annotations

import logging
import sys
import threading
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Dict, List, Optional, Sequence, Tuple

import torch

try:
    from diflow.backend.data_engine._data_engine import NvshmemDataEngineBackend
except ImportError as error:
    NvshmemDataEngineBackend = None
    _NATIVE_IMPORT_ERROR: Optional[ImportError] = error
else:
    _NATIVE_IMPORT_ERROR = None


@dataclass
class FreeingTask:
    tensor: torch.Tensor


@dataclass
class FetchingTask:
    id: str
    remote_address: int
    size: List[int]
    dtype: torch.dtype
    remote_nvshmem_pe: int


class SendingThread(threading.Thread):
    def __init__(self, engine: NvshmemDataEngine):
        super().__init__()
        self.engine = engine
        self.logger = logging.getLogger(f"SendingThread-{engine.worker_id}")

    def run(self):
        while self.engine.running or self.engine.freeing_task_queue.qsize() > 0:
            try:
                task = self.engine.freeing_task_queue.get(block=True, timeout=30)
                self.logger.debug(f"Processing freeing task")
                self.engine.backend.free_tensor(task.tensor)
                self.logger.debug("Successfully freed tensor")
            except Empty:
                pass


class FetchingThread(threading.Thread):
    def __init__(self, engine: NvshmemDataEngine):
        super().__init__()
        self.engine = engine
        self.logger = logging.getLogger(f"FetchingThread-{engine.worker_id}")

    def run(self):
        while self.engine.running or self.engine.fetching_task_queue.qsize() > 0:
            try:
                task = self.engine.fetching_task_queue.get(block=True, timeout=30)
                self.logger.debug(f"Processing fetch task with id {task.id}")
                tensor = self.engine.backend.fetch_tensor(
                    task.remote_address, task.size, task.dtype, task.remote_nvshmem_pe
                )
                self.engine.arrive(tensor, task.id)
                self.logger.debug(f"Successfully fetched tensor with id {task.id}")
            except Empty:
                pass


class NvshmemDataEngine:
    backend: NvshmemDataEngineBackend
    device_id: int
    worker_id: int
    nvshmem_pe: int
    running: bool

    def __init__(
        self,
        arena_size,
        # page_size, num_pages, soa_buffer_size, soa_threshold,
        device_id,
        worker_id,
    ):
        if NvshmemDataEngineBackend is None:
            raise RuntimeError(
                "The DiFlow NVSHMEM extension is unavailable. Install a native "
                "DiFlow build with CUDA, MPI, and NVSHMEM before starting workers. "
                "See docs/installation.md."
            ) from _NATIVE_IMPORT_ERROR

        self.device_id = device_id
        self.worker_id = worker_id
        self.running = False

        self.backend = NvshmemDataEngineBackend(
            arena_size,
            # page_size, num_pages, soa_buffer_size, soa_threshold,
            device_id,
            worker_id,
        )
        self.nvshmem_pe = self.backend.nvshmem_pe()

        self.received_tensors: Dict[str, torch.Tensor] = {}
        self.tensor_arrival: Dict[str, threading.Event] = {}

        self.freeing_task_queue: Queue[FreeingTask] = Queue()
        self.fetching_task_queue: Queue[FetchingTask] = Queue()
        self.freeing_thread = SendingThread(self)
        self.fetching_thread = FetchingThread(self)

        # Setup logging
        self.logger = logging.getLogger(
            f"NvshmemDataEngine-{worker_id}(device: {device_id}, nvshmem_pe: {self.nvshmem_pe})"
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

    def _start_async_sending_thread(self):
        self.freeing_thread.start()

    def _start_async_receiving_thread(self):
        self.fetching_thread.start()

    def start(self):
        self.running = True
        self._start_async_sending_thread()
        self._start_async_receiving_thread()

    def stop(self):
        self.running = False
        self.freeing_thread.join()
        self.fetching_thread.join()
        # Release the native backend (and its NVSHMEM arena) now, while MPI is
        # still initialized. If we leave it to interpreter-exit GC, it runs
        # after mpi4py's atexit MPI_Finalize and nvshmem_free segfaults on
        # NVSHMEM >= 3.x.
        self.backend = None
        print("DataEngine stopped")

    def __enter__(self) -> NvshmemDataEngine:
        """Context manager entry point"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point"""
        self.stop()

    def create_tensor(self, size: Sequence[int], *, dtype: torch.dtype) -> torch.Tensor:
        self.logger.debug(f"Creating tensor with size {size} and dtype {dtype}")
        tensor = self.backend.create_tensor(list(size), dtype)
        self.logger.debug(f"Successfully created tensor")
        return tensor

    def submit_fetch_task(self, task: FetchingTask):
        if not self.running:
            raise RuntimeError("DataEngine is not running")
        self.logger.debug(f"Submitting fetch task with id {task.id}")
        self.fetching_task_queue.put(task)
        self.logger.debug("Fetch task submitted successfully")

    def submit_free_task(self, task: FreeingTask):
        if not self.running:
            raise RuntimeError("DataEngine is not running")
        self.logger.debug(f"Submitting free task")
        self.freeing_task_queue.put(task)
        self.logger.debug("Free task submitted successfully")

    def get(self, tensor_id: str, timeout: float = 60.0) -> torch.Tensor:
        key = tensor_id
        arrival = self.tensor_arrival.setdefault(key, threading.Event())

        # Wait for the tensor to be available
        while not arrival.wait(timeout=timeout):
            self.logger.debug(
                f"Timeout waiting for tensor with id {tensor_id}. "
                f"Current queue size: {self.fetching_task_queue.qsize()}"
            )

        tensor = self.received_tensors.pop(key)
        self.tensor_arrival.pop(key)
        return tensor

    def arrive(self, tensor: torch.Tensor, tensor_id: str):
        key = tensor_id
        self.received_tensors[key] = tensor
        arrival = self.tensor_arrival.setdefault(key, threading.Event())
        arrival.set()
