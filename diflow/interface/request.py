from typing import Any, Dict, Optional

from pydantic import BaseModel


class InferenceRequest(BaseModel):
    """Request model for workflow inference

    Args:
        inputs: Input data dictionary for the workflow
        timeout: Optional timeout in seconds for the inference request
    """

    inputs: Dict[str, Any]
    timeout: Optional[float] = None  # Maximum execution time in seconds (optional)
    profiled_latency: Optional[float] = None  # Profiled latency in seconds
