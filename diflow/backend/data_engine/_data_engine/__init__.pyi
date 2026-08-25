from typing import List

import torch

class NvshmemDataEngineBackend:
    def __init__(
        self,
        arena_size: int,
        # page_size: int,
        # num_pages: int,
        # soa_buffer_size: int,
        # soa_threshold: int,
        device_id: int,
        worker_id: int,
    ) -> None: ...
    def nvshmem_pe(self) -> int: ...
    def create_tensor(self, size: List[int], dtype: torch.dtype) -> torch.Tensor: ...
    def free_tensor(self, tensor: torch.Tensor) -> None: ...
    def fetch_tensor(
        self,
        remote_src: int,
        size: List[int],
        dtype: torch.dtype,
        remote_device_id: int,
    ) -> torch.Tensor: ...
    def owns_tensor(self, tensor: torch.Tensor) -> bool: ...
