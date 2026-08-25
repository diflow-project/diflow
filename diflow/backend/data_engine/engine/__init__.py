from __future__ import annotations

from typing import Literal

from .base_data_engine import BaseDataEngine, FetchingTask, FreeingTask
from .host_memory_data_engine import HostMemoryDataEngine
from .nvshmem_data_engine import NvshmemDataEngine, nvshmem_is_available

TransferBackend = Literal["auto", "nvshmem", "host"]


def resolve_transfer_backend(backend: TransferBackend) -> str:
    if backend == "auto":
        return "nvshmem" if nvshmem_is_available() else "host"
    if backend == "nvshmem" and not nvshmem_is_available():
        raise RuntimeError(
            "NVSHMEM transfer was requested, but the DiFlow NVSHMEM extension "
            "is unavailable. Install the optional NVSHMEM build requirements or "
            "use --transfer-backend host."
        )
    if backend not in {"nvshmem", "host"}:
        raise ValueError(f"Unknown transfer backend: {backend}")
    return backend


def create_data_engine(
    *,
    backend: TransferBackend,
    arena_size: int,
    device_id: int,
    worker_id: int,
    world_size: int,
    host_transfer_dir: str,
    transfer_session_id: str,
) -> BaseDataEngine:
    resolved = resolve_transfer_backend(backend)
    if resolved == "nvshmem":
        return NvshmemDataEngine(
            arena_size=arena_size,
            device_id=device_id,
            worker_id=worker_id,
        )
    return HostMemoryDataEngine(
        device_id=device_id,
        worker_id=worker_id,
        world_size=world_size,
        transfer_dir=host_transfer_dir,
        session_id=transfer_session_id,
    )


__all__ = [
    "BaseDataEngine",
    "FetchingTask",
    "FreeingTask",
    "HostMemoryDataEngine",
    "NvshmemDataEngine",
    "TransferBackend",
    "create_data_engine",
    "nvshmem_is_available",
    "resolve_transfer_backend",
]
