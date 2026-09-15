# Installation

DiFlow is a Linux/CUDA project with a native scheduling extension. Prebuilt
wheels are the fastest installation path on supported systems. Docker provides
a reproducible full environment, while source installation is intended for
development and custom native builds.

## Validated environment

| Component | Validated configuration |
|---|---|
| Operating system | Ubuntu 22.04 |
| Python | 3.10 |
| CUDA toolkit | 12.4 |
| GPU | NVIDIA H20 |
| Open MPI | 4.1 |
| NVSHMEM | 3.7.1 (optional) |
| Torch | 2.6.0 (CUDA 12.4) |
| Transformers | 5.15.1 |
| Hugging Face Hub | 1.23.0 |
| Diffusers | 0.40.0.dev0 + DiFlow patches (pinned submodule) |

The default native build includes `sm_80`, `sm_89`, and `sm_90` plus PTX,
covering A100, RTX 4090, and Hopper-class compilation targets. These targets are
not a substitute for release testing on each physical GPU. Multi-node serving
requires a working NVSHMEM transport and MPI configuration. Single-node workers
can instead transfer intermediate tensors through shared host memory.

## Prebuilt wheel

Release wheels support Linux x86_64 with glibc 2.34 or newer and Python 3.10,
3.11, or 3.12. They contain the native C++ SchedulingCore, the NVSHMEM
data-engine extension, and the single-node host-memory transfer backend.
Install from the release's hash-pinned wheel index so pip selects the matching
Python ABI:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  --find-links https://github.com/diflow-project/diflow/releases/download/v0.1.0/index.html \
  diflow==0.1.0
```

The wheel includes DiFlow's pinned Diffusers fork as
`diflow._vendor.diffusers`. It neither installs nor replaces the top-level
`diffusers` package, so applications may keep a separate upstream Diffusers
installation. Objects from the two package namespaces are intentionally not
interchangeable.

The wheel does not bundle the NVIDIA driver, PyTorch/CUDA runtime, or the MPI
runtime. Pip installs the declared Python packages, including PyTorch 2.6.0.
The NVSHMEM extension is present but its runtime package remains optional, so
the base installation works on machines without NVSHMEM and `auto` selects the
host-memory backend. Install the matching runtime to activate NVSHMEM:

```bash
python -m pip install "diflow[nvshmem]"
```

The host must provide a compatible NVIDIA driver and Open MPI runtime for
NVSHMEM. Verify the result before serving:

```bash
python -m pip check
python -c "import diflow.backend.scheduler._scheduling_core"
python -c "import diflow._vendor.diffusers as d; print(d.__version__)"
diflow --help
```

The common wheel intentionally uses a `linux_x86_64` platform tag instead of
claiming broader manylinux compatibility: SchedulingCore links against the
PyTorch C++ runtime installed in the target environment, while the data-engine
extension links against compatible NVSHMEM and Open MPI runtimes. If those
optional libraries are unavailable, importing the NVSHMEM extension is skipped
and `--transfer-backend auto` selects host memory. Use
`--transfer-backend nvshmem` to require NVSHMEM explicitly.

## Docker

Clone with the pinned diffusers submodule and build from a clean checkout:

```bash
git clone --recurse-submodules https://github.com/diflow-project/diflow.git
cd DiFlow
docker build -f docker/Dockerfile -t diflow:latest .
```

Download the example models on the host. The `hf` command is provided by
`huggingface-hub`:

```bash
hf auth login
./scripts/download_models.sh /path/to/diflow-models
```

Start the built-in FLUX.1-schnell workflow:

```bash
docker run --gpus all --rm \
  --shm-size=16g \
  -p 8000:8000 \
  -v /path/to/diflow-models/FLUX.1-schnell:/models/FLUX.1-schnell:ro \
  diflow:latest \
  diflow serve \
    --host 0.0.0.0 \
    --workflow flux-schnell \
    --model-path /models/FLUX.1-schnell
```

The container runs as an unprivileged user and does not start SSH. Bind the API
only to trusted networks or put it behind authentication, TLS, and access
controls.

## Source installation

Install system prerequisites on Ubuntu 22.04:

```bash
sudo apt-get update
sudo apt-get install -y \
  git libopenmpi-dev ninja-build openmpi-bin \
  python-is-python3 python3-dev python3-pip python3-venv
```

Create an isolated environment:

```bash
git clone --recurse-submodules https://github.com/diflow-project/diflow.git
cd DiFlow
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the tested CUDA 12.4 dependency set:

```bash
python -m pip install \
  -r requirements/runtime.txt \
  -r requirements/models.txt \
  -r requirements/build.txt \
  -c requirements/constraints-cu124.txt

python -m pip install --no-cache-dir --force-reinstall --no-binary=mpi4py mpi4py==4.1.2
```

Build DiFlow. Without NVSHMEM installed, this builds the scheduling core and
uses the host-memory transfer backend. The editable installation loads the
pinned Diffusers source from the initialized submodule under DiFlow's private
namespace:

```bash
export TORCH_CUDA_ARCH_LIST=9.0
python -m pip install -e . --no-build-isolation
```

Use `8.0` for A100, `8.9` for RTX 4090, or
`8.0;8.9;9.0+PTX` for a release wheel.

For the optional NVSHMEM backend, install its dependency before building:

```bash
python -m pip install \
  -r requirements/nvshmem.txt \
  -c requirements/constraints-cu124.txt
```

DiFlow then detects the NVSHMEM wheel and standard Ubuntu Open MPI path. Set
these variables explicitly when using non-standard locations:

```bash
export NVSHMEM_DIR="$(python -c 'import importlib.util; print(next(iter(importlib.util.find_spec("nvidia.nvshmem").submodule_search_locations)))')"
export MPI_DIR=/usr/lib/x86_64-linux-gnu/openmpi
```

Require the optional extension to be present in strict NVSHMEM builds:

```bash
export TORCH_CUDA_ARCH_LIST=9.0
DIFLOW_REQUIRE_NVSHMEM=1 python -m pip install -e . --no-build-isolation
```

When upgrading an existing checkout, run `git submodule update --init --recursive`
and follow the dependency and build steps above in a fresh environment. The
updated diffusers fork needs Hugging Face Hub 1.x and the matching Transformers
5.x stack; the previous Torch 2.5.1 / Transformers 4.56.2 constraints are not
compatible with this release. Rebuild DiFlow's native extensions after changing
Torch instead of reusing extensions compiled for the old environment.

Verify the installation:

```bash
python -c "import diflow"
python -c "import diflow.backend.scheduler._scheduling_core"
python -m pip check
diflow --help
```

For an NVSHMEM build, additionally verify:

```bash
python -c "import diflow.backend.data_engine._data_engine"
```

## Transfer backends

DiFlow selects the intermediate-tensor transport at startup:

```bash
diflow serve --transfer-backend auto ...
diflow serve --transfer-backend host --num-workers 2 ...
diflow serve --transfer-backend nvshmem ...
```

`auto` prefers the NVSHMEM extension when it is installed and otherwise uses
host memory. The host backend supports local single-node workers only and uses
`/dev/shm/diflow` by default. Override that root with
`--host-transfer-dir PATH`. Ensure `/dev/shm` is large enough for in-flight
intermediate tensors; for Docker, set an appropriate `--shm-size`.

## Building release wheels

A host-capable release wheel must be built on Linux x86_64 with Torch and the
pinned Diffusers submodule available. Install the release tooling and invoke
the audited builder from a clean checkout:

```bash
python -m pip install -r requirements/build.txt \
  -r requirements/nvshmem.txt \
  -c requirements/constraints-cu124.txt
python tools/build_release_wheel.py --release-tag v0.1.0
```

Run the command once in each Python 3.10, 3.11, and 3.12 build environment.
The builder verifies the main checkout and submodule, embeds the Diffusers
commit provenance and license, includes the NVSHMEM extension, checks the wheel
contents, runs `twine check` and `auditwheel show`, and generates these release
assets in `dist/`:

- ABI-specific `diflow-*.whl` files
- `index.html` with absolute, hash-pinned GitHub Release links
- `SHA256SUMS`
- an `auditwheel` report for each wheel

Upload all generated assets to the matching GitHub Release. The builder rejects
a dirty or mismatched Diffusers submodule. `--allow-dirty` and
`--skip-auditwheel` exist only for local development validation.

The common release wheel always contains SchedulingCore, the NVSHMEM data
engine, and host-memory transfer support. Building it requires the pinned
NVSHMEM package and a compatible Open MPI development environment. Installing
the NVSHMEM runtime remains optional. `DIFLOW_SKIP_NATIVE=1` remains limited to
editable CPU test environments and cannot produce a release wheel.

## Troubleshooting

### PyTorch is unavailable in the build environment

Install build requirements first and use `--no-build-isolation`. For release
wheels, use `tools/build_release_wheel.py`, which selects the supported build
mode. The native extension build must use the same Torch ABI as the final
runtime.

### Unable to locate NVSHMEM_DIR or MPI_DIR

This message is informational unless `DIFLOW_REQUIRE_NVSHMEM=1` is set; DiFlow
will still build with host-memory transfer. For NVSHMEM builds, set both
variables to prefixes containing `include/` and `lib/`. On Ubuntu,
`MPI_DIR=/usr/lib/x86_64-linux-gnu/openmpi` is the usual Open MPI location.
Release wheels look for the NVSHMEM dependency installed alongside DiFlow. When
building a wheel against a system-wide custom NVSHMEM installation, also set
`DIFLOW_NVSHMEM_RPATH=$NVSHMEM_DIR/lib` so the deployed linker can locate that
installation.

### NVSHMEM device enumeration failed

Use `--transfer-backend host` for local single-node workers, or fix the NVSHMEM
transport before using multiple nodes. Do not attempt to fall back after a
partially initialized NVSHMEM runtime.

### Host transfer directory is full or unavailable

Use a writable tmpfs with enough capacity and pass it through
`--host-transfer-dir`. In containers, increase `--shm-size`. The host backend
does not support workers distributed across multiple machines.

### Automatic benchmark runs out of memory

OOM profile points are marked unsupported and the rest of the sweep continues.
Use smaller startup sweeps as described in
[Runtime profiles](runtime-profiles.md).
