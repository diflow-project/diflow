"""Compatibility module for starting a DiFlow workflow server."""

from __future__ import annotations

from typing import Optional, Sequence

from diflow.cli.serve import main as serve_main


def main(argv: Optional[Sequence[str]] = None) -> int:
    return serve_main(argv, prog="python -m diflow.launch_server")


if __name__ == "__main__":
    raise SystemExit(main())
