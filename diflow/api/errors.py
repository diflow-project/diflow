from __future__ import annotations

from typing import Any, Dict, Optional


class WorkflowAPIError(ValueError):
    """A stable, machine-readable public API error."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        path: Optional[str] = None,
        status_code: int = 422,
        retryable: bool = False,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.path = path
        self.status_code = status_code
        self.retryable = retryable
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        error: Dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
        }
        if self.path is not None:
            error["path"] = self.path
        if self.details:
            error["details"] = self.details
        return {"error": error}
