"""Private third-party packages used by DiFlow.

Release wheels contain ``diflow._vendor.diffusers`` directly. In a source
checkout, the private namespace also searches the pinned Git submodule so that
editable installs use the same code without installing a top-level
``diffusers`` distribution.
"""

from __future__ import annotations

import os
from pathlib import Path


def _source_diffusers_root() -> Path:
    configured = os.getenv("DIFLOW_DIFFUSERS_SOURCE")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "submodules" / "diffusers" / "src"


_source_root = _source_diffusers_root()
if (_source_root / "diffusers" / "__init__.py").is_file():
    __path__.append(str(_source_root))
