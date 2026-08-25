"""Native extension build for DiFlow.

Source and editable installs must use the active Torch/CUDA environment:

    python -m pip install . --no-build-isolation

The metadata and sdist paths remain CPU-safe. A wheel or editable install never
silently falls back to a pure-Python distribution.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import sysconfig
from pathlib import Path
from typing import Iterable, Optional

import setuptools

ROOT = Path(__file__).resolve().parent
SKIP_NATIVE = os.getenv("DIFLOW_SKIP_NATIVE", "0") == "1"
SKIP_DATA_ENGINE = os.getenv("DIFLOW_SKIP_DATA_ENGINE", "0") == "1"
NATIVE_BUILD_COMMANDS = {
    "bdist_wheel",
    "build_ext",
    "develop",
    "editable_wheel",
    "install",
}


def _native_build_requested() -> bool:
    return any(argument in NATIVE_BUILD_COMMANDS for argument in sys.argv)


def _first_valid_directory(
    environment_name: str,
    candidates: Iterable[Optional[Path]],
    required_paths: Iterable[str],
) -> Path:
    configured = os.getenv(environment_name)
    checked = []
    values = [Path(configured).expanduser()] if configured else []
    values.extend(candidate for candidate in candidates if candidate is not None)
    for candidate in values:
        resolved = candidate.resolve()
        checked.append(str(resolved))
        if all((resolved / relative).exists() for relative in required_paths):
            return resolved
    locations = ", ".join(checked) if checked else "none"
    raise RuntimeError(
        f"Unable to locate {environment_name}. Checked: {locations}. "
        f"Set {environment_name} to a directory containing "
        f"{', '.join(required_paths)}."
    )


def _nvshmem_package_directory() -> Optional[Path]:
    try:
        spec = importlib.util.find_spec("nvidia.nvshmem")
    except (ImportError, ModuleNotFoundError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(next(iter(spec.submodule_search_locations)))


def _site_nvshmem_directory() -> Path:
    return Path(sysconfig.get_paths()["purelib"]) / "nvidia" / "nvshmem"


def _first_existing_file(
    directory: Path,
    filenames: Iterable[str],
    description: str,
) -> Path:
    candidates = [directory / filename for filename in filenames]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError(
        f"Unable to locate {description}. Checked: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


ext_modules = []
cmdclass = {}

if SKIP_NATIVE and "bdist_wheel" in sys.argv:
    raise RuntimeError(
        "DIFLOW_SKIP_NATIVE cannot be used to build a release wheel. "
        "Build on Linux with Torch and the documented native prerequisites."
    )

if SKIP_DATA_ENGINE and "bdist_wheel" in sys.argv:
    raise RuntimeError(
        "DIFLOW_SKIP_DATA_ENGINE cannot be used to build a release wheel. "
        "It is limited to scheduling-core development builds."
    )

if not SKIP_NATIVE:
    try:
        from torch.utils.cpp_extension import (
            BuildExtension,
            CppExtension,
            CUDAExtension,
        )
    except ImportError as exc:
        if _native_build_requested():
            raise RuntimeError(
                "PyTorch is unavailable in the build environment. Install the "
                "build requirements first, then run pip with "
                "--no-build-isolation. See docs/installation.md."
            ) from exc
    else:
        cmdclass = {"build_ext": BuildExtension}
        ext_modules.append(
            CppExtension(
                name="diflow.backend.scheduler._scheduling_core",
                sources=["csrc/scheduling/scheduling_core.cpp"],
                extra_compile_args=["-O3", "-std=c++17"],
            )
        )

        if not SKIP_DATA_ENGINE:
            nvshmem_dir = _first_valid_directory(
                "NVSHMEM_DIR",
                (
                    _nvshmem_package_directory(),
                    _site_nvshmem_directory(),
                    Path("/usr/local/nvshmem"),
                ),
                ("include/nvshmem.h", "lib"),
            )
            mpi_dir = _first_valid_directory(
                "MPI_DIR",
                (
                    Path("/usr/lib/x86_64-linux-gnu/openmpi"),
                    Path("/usr/local/mpi"),
                ),
                ("include/mpi.h", "lib"),
            )
            print(f"NVSHMEM directory: {nvshmem_dir}")
            print(f"MPI directory: {mpi_dir}")
            nvshmem_lib_dir = nvshmem_dir / "lib"
            nvshmem_host_library = _first_existing_file(
                nvshmem_lib_dir,
                ("libnvshmem_host.so", "libnvshmem_host.so.3"),
                "NVSHMEM host library",
            )
            nvshmem_device_library = _first_existing_file(
                nvshmem_lib_dir, ("libnvshmem_device.a",), "NVSHMEM device library"
            )
            nvshmem_bootstrap_library = _first_existing_file(
                nvshmem_lib_dir,
                ("nvshmem_bootstrap_mpi.so", "nvshmem_bootstrap_mpi.so.3"),
                "NVSHMEM MPI bootstrap library",
            )
            nvshmem_rpaths = ["-Wl,-rpath,$ORIGIN/../../../nvidia/nvshmem/lib"]
            if any(
                arg in {"build_ext", "develop", "editable_wheel"} for arg in sys.argv
            ):
                nvshmem_rpaths.append(f"-Wl,-rpath,{nvshmem_lib_dir}")
            configured_rpath = os.getenv("DIFLOW_NVSHMEM_RPATH")
            if configured_rpath:
                nvshmem_rpaths.append(f"-Wl,-rpath,{configured_rpath}")

            os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0;8.9;9.0+PTX")
            cxx_flags = [
                "-O3",
                "-DOMPI_SKIP_MPICXX=1",
                "-Wno-deprecated-declarations",
                "-Wno-unused-variable",
                "-Wno-sign-compare",
                "-Wno-reorder",
                "-Wno-attributes",
            ]
            nvcc_flags = [
                "-O3",
                "-DOMPI_SKIP_MPICXX=1",
                "-Xcompiler",
                "-O3",
                "-rdc=true",
                "--ptxas-options=--register-usage-level=10",
            ]
            include_dirs = [
                str(ROOT / "csrc"),
                str(nvshmem_dir / "include"),
                str(mpi_dir / "include"),
            ]
            sources = [
                "csrc/data_engine/allocator/buddy_allocator.cpp",
                "csrc/data_engine/allocator/paged_allocator.cpp",
                "csrc/data_engine/allocator/small_object_allocator.cpp",
                "csrc/data_engine/allocator/complicated_allocator.cpp",
                "csrc/data_engine/engine/backends/nvshmem/nvshmem_backend.cpp",
            ]
            library_dirs = [str(nvshmem_dir / "lib"), str(mpi_dir / "lib")]
            nvcc_dlink = [
                "-dlink",
                f"-L{nvshmem_lib_dir}",
                f"-l:{nvshmem_host_library.name}",
                "-lnvshmem_device",
                f"-L{mpi_dir / 'lib'}",
                "-lmpi",
            ]
            extra_link_args = [
                str(nvshmem_host_library),
                str(nvshmem_device_library),
                str(nvshmem_bootstrap_library),
                *nvshmem_rpaths,
                "-l:libmpi.so",
                f"-Wl,-rpath,{mpi_dir / 'lib'}",
            ]
            ext_modules.append(
                CUDAExtension(
                    name="diflow.backend.data_engine._data_engine",
                    include_dirs=include_dirs,
                    library_dirs=library_dirs,
                    sources=sources,
                    extra_compile_args={
                        "cxx": cxx_flags,
                        "nvcc": nvcc_flags,
                        "nvcc_dlink": nvcc_dlink,
                    },
                    extra_link_args=extra_link_args,
                )
            )

setuptools.setup(ext_modules=ext_modules, cmdclass=cmdclass)
