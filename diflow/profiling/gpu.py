"""Small GPU helpers used by both runtime profiling and benchmark tools."""

import re
from typing import Tuple

import torch


def normalize_gpu_name(gpu_name: str) -> str:
    """Convert vendor-specific GPU names into stable path-safe identifiers."""
    normalized_name = gpu_name.strip().lower()
    vendor_aliases = {
        "nvidia corporation": "nvidia",
        "advanced micro devices, inc.": "amd",
        "advanced micro devices": "amd",
        "amd/ati": "amd",
        "intel corporation": "intel",
        "intel(r)": "intel",
    }
    for vendor_name, canonical_name in vendor_aliases.items():
        normalized_name = normalized_name.replace(vendor_name, canonical_name)
    normalized_name = normalized_name.replace("(r)", "").replace("(tm)", "")
    normalized_name = normalized_name.replace("\u00ae", "").replace("\u2122", "")
    normalized_name = re.sub(r"[^a-z0-9]+", "_", normalized_name).strip("_")
    return normalized_name or "unknown_gpu"


def get_gpu_info() -> Tuple[str, int, int]:
    """Return the primary GPU name, total memory in bytes, and device count."""
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    return (
        torch.cuda.get_device_name(),
        int(torch.cuda.get_device_properties(0).total_memory),
        torch.cuda.device_count(),
    )
