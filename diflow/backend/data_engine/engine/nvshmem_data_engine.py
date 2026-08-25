from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from .base_data_engine import BaseDataEngine, FetchingTask

try:
    from diflow.backend.data_engine._data_engine import NvshmemDataEngineBackend
except ImportError as error:
    NvshmemDataEngineBackend = None
    _NATIVE_IMPORT_ERROR: Optional[ImportError] = error
else:
    _NATIVE_IMPORT_ERROR = None


def nvshmem_is_available() -> bool:
    return NvshmemDataEngineBackend is not None


class NvshmemDataEngine(BaseDataEngine):
    backend_name = "nvshmem"

    def __init__(self, *, arena_size: int, device_id: int, worker_id: int) -> None:
        if NvshmemDataEngineBackend is None:
            raise RuntimeError(
                "The DiFlow NVSHMEM extension is unavailable. Install DiFlow with "
                "the optional NVSHMEM build requirements or use "
                "--transfer-backend host. See docs/installation.md."
            ) from _NATIVE_IMPORT_ERROR

        super().__init__(device_id=device_id, worker_id=worker_id)
        self.backend = NvshmemDataEngineBackend(
            arena_size,
            device_id,
            worker_id,
        )
        self.nvshmem_pe = self.backend.nvshmem_pe()

    def store_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        allocated = self.backend.create_tensor(list(tensor.size()), tensor.dtype)
        allocated.copy_(tensor, non_blocking=True)
        return allocated

    def get_tensor_handle(self, tensor: torch.Tensor) -> Dict[str, Any]:
        if not self.backend.owns_tensor(tensor):
            raise ValueError("Tensor is not managed by the NVSHMEM data engine")
        return {
            "backend": self.backend_name,
            "ptr": tensor.data_ptr(),
        }

    def _fetch_tensor(self, task: FetchingTask) -> torch.Tensor:
        backend = task.tensor_info.get("backend", self.backend_name)
        if backend != self.backend_name:
            raise ValueError(
                f"NVSHMEM data engine cannot fetch tensor from backend {backend!r}"
            )
        return self.backend.fetch_tensor(
            int(task.tensor_info["ptr"]),
            task.size,
            task.dtype,
            task.remote_worker_rank,
        )

    def _free_tensor(self, tensor: torch.Tensor) -> None:
        self.backend.free_tensor(tensor)

    def _shutdown(self) -> None:
        # Release NVSHMEM while MPI is still initialized. Leaving this to
        # interpreter-exit GC can run after mpi4py finalizes MPI.
        self.backend = None
