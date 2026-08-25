# Installation

DiFlow is a Linux/CUDA project with C++ and CUDA extensions. Use the official
Docker path when possible. Source installation is intended for development and
platform-specific wheel builds.

## Validated environment

| Component | Validated configuration |
|---|---|
| Operating system | Ubuntu 22.04 |
| Python | 3.10; CPU tests also cover 3.11 |
| CUDA toolkit | 12.4 |
| GPU | NVIDIA H20 |
| Open MPI | 4.1 |
| NVSHMEM | 3.7.1 (optional) |
| Torch | 2.5.1 |

The default native build includes `sm_80`, `sm_89`, and `sm_90` plus PTX,
covering A100, RTX 4090, and Hopper-class compilation targets. These targets are
not a substitute for release testing on each physical GPU. Multi-node serving
requires a working NVSHMEM transport and MPI configuration. Single-node workers
can instead transfer intermediate tensors through shared host memory.

## Docker

Clone with the pinned diffusers submodule and build from a clean checkout:

```bash
git clone --recurse-submodules https://github.com/diflow-project/DiFlow.git
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
git clone --recurse-submodules https://github.com/diflow-project/DiFlow.git
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
python -m pip install -e submodules/diffusers
```

Build DiFlow. Without NVSHMEM installed, this builds the scheduling core and
uses the host-memory transfer backend:

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

Verify the installation:

```bash
python -c "import diflow"
python -c "import diflow.backend.scheduler._scheduling_core"
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

## Building a native wheel

A host-capable release wheel must be built on Linux with Torch available. Add
the optional NVSHMEM requirements and `DIFLOW_REQUIRE_NVSHMEM=1` for a wheel
that includes the NVSHMEM extension:

```bash
python -m pip install -r requirements/build.txt \
  -c requirements/constraints-cu124.txt
TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0+PTX" \
  python -m build --wheel --no-isolation
python -m twine check dist/*.whl
```

DiFlow refuses to produce a release wheel when Torch or the scheduling-core
prerequisites are absent. `DIFLOW_SKIP_NATIVE=1` is limited to editable CPU
test environments and source-distribution metadata.

## Troubleshooting

### PyTorch is unavailable in the build environment

Install build requirements first and use `--no-build-isolation`. The native
extension build must use the same Torch/CUDA environment as the final runtime.

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
