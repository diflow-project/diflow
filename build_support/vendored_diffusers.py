"""Materialize the pinned Diffusers fork under DiFlow's private namespace."""

from __future__ import annotations

import ast
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PRIVATE_PACKAGE = "diflow._vendor.diffusers"
SUBMODULE_PATH = Path("submodules/diffusers")
_IGNORED_NAMES = {".git", "__pycache__", ".pytest_cache", ".ruff_cache"}
_FROM_IMPORT = re.compile(
    r"^(?P<indent>\s*)from\s+diffusers(?P<suffix>(?:\.[\w.]+)?\s+import\s+)",
    re.MULTILINE,
)
_PLAIN_IMPORT = re.compile(
    r"^(?P<indent>\s*)import\s+diffusers"
    r"(?P<alias>\s+as\s+[A-Za-z_]\w*)?"
    r"(?P<comment>\s*(?:#.*)?)$",
    re.MULTILINE,
)


class VendoringError(RuntimeError):
    """Raised when the pinned Diffusers source cannot be safely vendored."""


@dataclass(frozen=True)
class DiffusersSource:
    root: Path
    package: Path
    commit: str
    version: str


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _diffusers_version(package: Path) -> str:
    source = (package / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', source, re.MULTILINE)
    if match is None:
        raise VendoringError("Unable to determine the vendored Diffusers version")
    return match.group(1)


def validate_diffusers_checkout(repository_root: Path) -> DiffusersSource:
    """Validate that the checked-out submodule matches the main repository gitlink."""
    repository_root = repository_root.resolve()
    submodule_root = repository_root / SUBMODULE_PATH
    package = submodule_root / "src" / "diffusers"
    if not (package / "__init__.py").is_file():
        raise VendoringError(
            "The Diffusers submodule is not initialized. Run "
            "`git submodule update --init --recursive`."
        )

    try:
        expected = _git(repository_root, "rev-parse", f"HEAD:{SUBMODULE_PATH}")
        actual = _git(submodule_root, "rev-parse", "HEAD")
        dirty = _git(submodule_root, "status", "--porcelain")
    except (OSError, subprocess.CalledProcessError) as error:
        raise VendoringError(
            "Release wheels must be built from a Git checkout with an initialized "
            "Diffusers submodule"
        ) from error

    if actual != expected:
        raise VendoringError(
            "The Diffusers checkout does not match the commit pinned by DiFlow: "
            f"expected {expected}, found {actual}. Run "
            "`git submodule update --init --recursive`."
        )
    if dirty:
        raise VendoringError(
            "The Diffusers submodule has local modifications. Release wheels must "
            "use the exact pinned source."
        )
    if not (submodule_root / "LICENSE").is_file():
        raise VendoringError("The Diffusers license file is missing")

    return DiffusersSource(
        root=submodule_root,
        package=package,
        commit=actual,
        version=_diffusers_version(package),
    )


def rewrite_private_imports(source: str) -> str:
    """Relocate absolute self-imports used by the upstream package."""

    def replace_from(match: re.Match[str]) -> str:
        return (
            f"{match.group('indent')}from {PRIVATE_PACKAGE}" f"{match.group('suffix')}"
        )

    def replace_import(match: re.Match[str]) -> str:
        alias = match.group("alias") or ""
        return (
            f"{match.group('indent')}from diflow._vendor import diffusers"
            f"{alias}{match.group('comment')}"
        )

    rewritten = _FROM_IMPORT.sub(replace_from, source)
    rewritten = _PLAIN_IMPORT.sub(replace_import, rewritten)
    for quote in ('"', "'"):
        rewritten = rewritten.replace(
            f"import_module({quote}diffusers{quote})",
            f"import_module({quote}{PRIVATE_PACKAGE}{quote})",
        )
        rewritten = rewritten.replace(
            f"f{quote}diffusers.", f"f{quote}{PRIVATE_PACKAGE}."
        )
        rewritten = rewritten.replace(
            f"{quote}diffusers.", f"{quote}{PRIVATE_PACKAGE}."
        )
    return rewritten


def _copy_ignore(_: str, names: Iterable[str]) -> list[str]:
    return [
        name
        for name in names
        if name in _IGNORED_NAMES or name.endswith((".pyc", ".pyo"))
    ]


def _absolute_diffusers_imports(path: Path) -> list[str]:
    violations = []
    for python_file in path.rglob("*.py"):
        tree = ast.parse(
            python_file.read_text(encoding="utf-8"), filename=str(python_file)
        )
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and (node.module == "diffusers" or node.module.startswith("diffusers."))
            ):
                violations.append(f"{python_file}:{node.lineno}")
            elif isinstance(node, ast.Import) and any(
                alias.name == "diffusers" or alias.name.startswith("diffusers.")
                for alias in node.names
            ):
                violations.append(f"{python_file}:{node.lineno}")
    return violations


def materialize_vendored_diffusers(
    repository_root: Path, build_lib: Path
) -> DiffusersSource:
    """Copy the exact submodule source into a wheel build directory."""
    source = validate_diffusers_checkout(repository_root)
    target = Path(build_lib) / "diflow" / "_vendor" / "diffusers"
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source.package, target, ignore=_copy_ignore)

    for python_file in target.rglob("*.py"):
        original = python_file.read_text(encoding="utf-8")
        rewritten = rewrite_private_imports(original)
        if rewritten != original:
            python_file.write_text(rewritten, encoding="utf-8")

    shutil.copy2(source.root / "LICENSE", target / "LICENSE")
    (target / "_diflow_vendor.py").write_text(
        "# Generated during the DiFlow wheel build.\n"
        f"DIFFUSERS_COMMIT = {source.commit!r}\n"
        f"DIFFUSERS_VERSION = {source.version!r}\n",
        encoding="utf-8",
    )

    violations = _absolute_diffusers_imports(target)
    if violations:
        raise VendoringError(
            "Vendored Diffusers still contains top-level self-imports: "
            + ", ".join(violations[:10])
        )
    return source
