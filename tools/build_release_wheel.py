#!/usr/bin/env python3
"""Build and audit one dual-backend DiFlow wheel for GitHub Releases."""

from __future__ import annotations

import argparse
import ast
import hashlib
import html
import importlib.util
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable, Sequence

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from build_support.vendored_diffusers import validate_diffusers_checkout

REPOSITORY = "diflow-project/diflow"
SUPPORTED_PYTHON = {(3, 10), (3, 11), (3, 12)}


def _run(command: Sequence[str], *, cwd: Path = ROOT, env=None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(ROOT), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _validate_platform() -> None:
    if sys.platform != "linux" or platform.machine() not in {"x86_64", "amd64"}:
        raise RuntimeError("Release wheels currently require Linux x86_64")
    if sys.version_info[:2] not in SUPPORTED_PYTHON:
        versions = ", ".join(
            ".".join(map(str, item)) for item in sorted(SUPPORTED_PYTHON)
        )
        raise RuntimeError(
            f"Release wheels support Python {versions}; found {platform.python_version()}"
        )


def _validate_repository(allow_dirty: bool) -> None:
    validate_diffusers_checkout(ROOT)
    if not allow_dirty:
        status = _git("status", "--porcelain")
        if status:
            raise RuntimeError(
                "The DiFlow checkout has tracked modifications. Commit them before "
                "building a release wheel, or use --allow-dirty for local validation."
            )


def audit_wheel(wheel: Path) -> None:
    """Reject incomplete wheels and accidental source-checkout contents."""
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())

    required = {
        "benchmark_ops/__init__.py",
        "diflow/_vendor/diffusers/__init__.py",
        "diflow/_vendor/diffusers/LICENSE",
        "diflow/_vendor/diffusers/_diflow_vendor.py",
        "workflow_hub/__init__.py",
    }
    missing = sorted(required - names)
    if missing:
        raise RuntimeError(f"Wheel is missing vendored files: {', '.join(missing)}")
    if not any(
        name.startswith("diflow/backend/scheduler/_scheduling_core")
        and name.endswith(".so")
        for name in names
    ):
        raise RuntimeError("Wheel is missing the native SchedulingCore extension")
    if not any(
        name.startswith("diflow/backend/data_engine/_data_engine")
        and name.endswith(".so")
        for name in names
    ):
        raise RuntimeError("Wheel is missing the native NVSHMEM data-engine extension")

    forbidden = sorted(
        name
        for name in names
        if name.startswith(("diffusers/", "submodules/", "tests/"))
        or "/.git" in name
        or "__pycache__" in name
        or name.endswith((".pyc", ".pyo"))
    )
    if forbidden:
        raise RuntimeError(
            "Wheel contains forbidden paths: " + ", ".join(forbidden[:20])
        )

    absolute_imports = []
    with zipfile.ZipFile(wheel) as archive:
        for name in sorted(item for item in names if item.endswith(".py")):
            tree = ast.parse(archive.read(name), filename=name)
            for node in ast.walk(tree):
                imported = []
                if isinstance(node, ast.Import):
                    imported = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported = [node.module]
                elif (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "import_module"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                ):
                    imported = [node.args[0].value]
                if any(
                    module == "diffusers" or module.startswith("diffusers.")
                    for module in imported
                ):
                    absolute_imports.append(f"{name}:{node.lineno}")
    if absolute_imports:
        raise RuntimeError(
            "Wheel contains absolute Diffusers imports: "
            + ", ".join(absolute_imports[:20])
        )

    metadata_name = next(
        (name for name in names if name.endswith(".dist-info/METADATA")), None
    )
    if metadata_name is None:
        raise RuntimeError("Wheel is missing distribution metadata")
    with zipfile.ZipFile(wheel) as archive:
        metadata = archive.read(metadata_name).decode("utf-8")
    if re.search(
        r"^Requires-Dist:\s*diffusers(?:[\s\[<>=!~;(]|$)",
        metadata,
        re.MULTILINE,
    ):
        raise RuntimeError("Wheel must not depend on a top-level Diffusers package")

    nvshmem_requirements = []
    for line in metadata.splitlines():
        if not line.startswith("Requires-Dist:"):
            continue
        requirement = Requirement(line.partition(":")[2].strip())
        if requirement.name.lower().replace("_", "-") == "nvidia-nvshmem-cu12":
            nvshmem_requirements.append(requirement)
    if not nvshmem_requirements:
        raise RuntimeError(
            "Wheel metadata must provide the optional nvshmem runtime extra"
        )
    if any(
        requirement.marker is None
        or 'extra == "nvshmem"' not in str(requirement.marker)
        for requirement in nvshmem_requirements
    ):
        raise RuntimeError(
            "The NVSHMEM runtime must remain an optional wheel dependency"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_release_index(
    output_dir: Path,
    release_tag: str,
    wheels: Iterable[Path],
    repository: str = REPOSITORY,
) -> Path:
    """Write a pip-compatible find-links page for one GitHub release."""
    links = []
    checksums = []
    base = f"https://github.com/{repository}/releases/download/{release_tag}"
    for wheel in sorted(wheels, key=lambda item: item.name):
        checksum = _sha256(wheel)
        url = f"{base}/{wheel.name}#sha256={checksum}"
        links.append(
            f'  <a href="{html.escape(url)}">{html.escape(wheel.name)}</a><br>'
        )
        checksums.append(f"{checksum}  {wheel.name}\n")
    if not links:
        raise RuntimeError("No wheels were found while generating the release index")

    index = output_dir / "index.html"
    index.write_text(
        '<!doctype html>\n<html><head><meta charset="utf-8">'
        "<title>DiFlow wheels</title></head><body>\n"
        + "\n".join(links)
        + "\n</body></html>\n",
        encoding="utf-8",
    )
    (output_dir / "SHA256SUMS").write_text("".join(checksums), encoding="utf-8")
    return index


def build(args: argparse.Namespace) -> list[Path]:
    _validate_platform()
    _validate_repository(args.allow_dirty)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    build_dir = ROOT / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)

    environment = os.environ.copy()
    environment.pop("DIFLOW_SKIP_DATA_ENGINE", None)
    environment["DIFLOW_REQUIRE_NVSHMEM"] = "1"
    with tempfile.TemporaryDirectory(
        prefix=".wheel-build-", dir=output_dir
    ) as temporary:
        temporary_output = Path(temporary)
        _run(
            [
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--no-isolation",
                "--outdir",
                str(temporary_output),
            ],
            env=environment,
        )
        built = list(temporary_output.glob("diflow-*.whl"))
        if len(built) != 1:
            raise RuntimeError(f"Expected one newly built wheel, found {len(built)}")
        wheel = output_dir / built[0].name
        shutil.move(str(built[0]), wheel)

    audit_wheel(wheel)
    _run([sys.executable, "-m", "twine", "check", str(wheel)])
    if not args.skip_auditwheel:
        if importlib.util.find_spec("auditwheel") is None:
            raise RuntimeError(
                "auditwheel is required for a release build; install "
                "requirements/build.txt or pass --skip-auditwheel for local testing"
            )
        report = output_dir / f"{wheel.name}.auditwheel.txt"
        completed = subprocess.run(
            [sys.executable, "-m", "auditwheel", "show", str(wheel)],
            check=True,
            capture_output=True,
            text=True,
        )
        report.write_text(completed.stdout, encoding="utf-8")

    write_release_index(
        output_dir,
        release_tag=args.release_tag,
        wheels=output_dir.glob("diflow-*.whl"),
        repository=args.repository,
    )
    return [wheel]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist")
    parser.add_argument("--release-tag", required=True)
    parser.add_argument("--repository", default=REPOSITORY)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--skip-auditwheel", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    wheels = build(args)
    for wheel in wheels:
        print(f"Built {wheel}")
    print(f"Release index: {args.output_dir.resolve() / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
