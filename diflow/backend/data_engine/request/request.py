from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional


@dataclass
class Request:
    request_type: Literal["fetch"]
    request_id: str
