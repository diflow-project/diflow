from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from diflow.cli.serve import main as serve_main


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="diflow",
        description="Serve diffusion workflows with DiFlow.",
    )
    parser.add_argument("command", nargs="?", choices=["serve"], help="Command to run.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        _parser().print_help()
        return 0
    if arguments[0] == "serve":
        return serve_main(arguments[1:], prog="diflow serve")
    _parser().error(f"unknown command: {arguments[0]}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
