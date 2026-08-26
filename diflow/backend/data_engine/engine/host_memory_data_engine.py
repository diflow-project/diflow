from __future__ import annotations

import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .base_data_engine import BaseDataEngine, FetchingTask


@dataclass
class _HostAllocation:
    path: Optional[Path]
    owner_rank: int
    mapping: Optional[torch.Tensor]


class HostMemoryDataEngine(BaseDataEngine):
    """Single-node tensor transfer through file-backed shared host memory."""

    backend_name = "host"

    def __init__(
        self,
        *,
        device_id: int,
        worker_id: int,
        world_size: int,
        transfer_dir: str,
        session_id: str,
        device: Optional[str] = None,
    ) -> None:
        self.world_size = world_size
        self.device = torch.device(device or f"cuda:{device_id}")
        self.transfer_root = Path(transfer_dir).expanduser().resolve()
        self.session_id = session_id
        self.session_dir = self.transfer_root / session_id
        self.worker_dir = self.session_dir / f"worker-{worker_id}"
        self._validate_session_id(session_id)
        self._prepare_directory()
        self._allocations: Dict[int, _HostAllocation] = {}
        super().__init__(device_id=device_id, worker_id=worker_id)

    @staticmethod
    def _validate_session_id(session_id: str) -> None:
        try:
            parsed = uuid.UUID(session_id)
        except ValueError as error:
            raise ValueError(
                f"Invalid host transfer session ID: {session_id}"
            ) from error
        if parsed.hex != session_id.replace("-", "").lower():
            raise ValueError(f"Invalid host transfer session ID: {session_id}")

    def _prepare_directory(self) -> None:
        try:
            self.worker_dir.mkdir(parents=True, exist_ok=False)
        except OSError as error:
            raise RuntimeError(
                f"Unable to create host transfer directory {self.worker_dir}: {error}"
            ) from error
        if not os.access(self.worker_dir, os.W_OK):
            raise RuntimeError(
                f"Host transfer directory is not writable: {self.worker_dir}"
            )

    def store_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(tensor).__name__}")
        local_tensor = tensor.detach()
        path: Optional[Path] = None
        mapping: Optional[torch.Tensor] = None

        if self.world_size > 1:
            path = self.worker_dir / f"{uuid.uuid4().hex}.tensor"
            required_bytes = local_tensor.numel() * local_tensor.element_size()
            available_bytes = shutil.disk_usage(self.worker_dir).free
            if required_bytes > available_bytes:
                raise RuntimeError(
                    "Insufficient host transfer space: "
                    f"tensor requires {required_bytes} bytes but only "
                    f"{available_bytes} bytes are available in {self.worker_dir}"
                )
            try:
                mapping = torch.from_file(
                    str(path),
                    shared=True,
                    size=local_tensor.numel(),
                    dtype=local_tensor.dtype,
                ).reshape(tuple(local_tensor.shape))
                mapping.copy_(local_tensor.contiguous(), non_blocking=False)
            except BaseException:
                path.unlink(missing_ok=True)
                raise

        self._allocations[id(local_tensor)] = _HostAllocation(
            path=path,
            owner_rank=self.worker_id,
            mapping=mapping,
        )
        return local_tensor

    def get_tensor_handle(self, tensor: torch.Tensor) -> Dict[str, Any]:
        try:
            allocation = self._allocations[id(tensor)]
        except KeyError as error:
            raise ValueError("Tensor is not managed by the host data engine") from error
        return {
            "backend": self.backend_name,
            "path": str(allocation.path) if allocation.path is not None else None,
            "owner_rank": allocation.owner_rank,
        }

    def _fetch_tensor(self, task: FetchingTask) -> torch.Tensor:
        backend = task.tensor_info.get("backend", self.backend_name)
        if backend != self.backend_name:
            raise ValueError(
                f"Host data engine cannot fetch tensor from backend {backend!r}"
            )
        raw_path = task.tensor_info.get("path")
        if not raw_path:
            raise ValueError(
                "A local-only host tensor cannot be fetched by another worker"
            )
        path = Path(raw_path).resolve()
        try:
            path.relative_to(self.session_dir)
        except ValueError as error:
            raise ValueError(
                f"Host tensor path is outside session directory: {path}"
            ) from error

        numel = 1
        for dimension in task.size:
            numel *= dimension
        expected_bytes = torch.empty((), dtype=task.dtype).element_size() * numel
        try:
            actual_bytes = path.stat().st_size
        except OSError as error:
            raise RuntimeError(f"Host tensor is unavailable: {path}") from error
        if actual_bytes < expected_bytes:
            raise RuntimeError(
                f"Host tensor {path} is truncated: expected {expected_bytes} bytes, "
                f"found {actual_bytes}"
            )

        mapping = torch.from_file(
            str(path),
            shared=True,
            size=numel,
            dtype=task.dtype,
        ).reshape(tuple(task.size))
        if self.device.type == "cpu":
            tensor = mapping.clone()
        else:
            tensor = mapping.to(self.device, non_blocking=False)
        self._allocations[id(tensor)] = _HostAllocation(
            path=path,
            owner_rank=int(task.tensor_info["owner_rank"]),
            mapping=None,
        )
        return tensor

    def _free_tensor(self, tensor: torch.Tensor) -> None:
        allocation = self._allocations.pop(id(tensor), None)
        if allocation is None:
            return
        if allocation.owner_rank == self.worker_id and allocation.path is not None:
            allocation.path.unlink(missing_ok=True)

    def _shutdown(self) -> None:
        for tensor_id, allocation in tuple(self._allocations.items()):
            if allocation.owner_rank == self.worker_id and allocation.path is not None:
                allocation.path.unlink(missing_ok=True)
            self._allocations.pop(tensor_id, None)
        try:
            self.worker_dir.rmdir()
        except OSError:
            pass
