# Installation

DiFlow is a Linux/CUDA project with C++ and CUDA extensions. Use the official
Docker path when possible. Source installation is intended for development and
platform-specific wheel builds.

## Validated environment

| Component | Validated configuration |
|---|---|
| Operating system | Ubuntu 22.04 |
| Python | 3.10 |
| CUDA toolkit | 12.4 |
| GPU | NVIDIA H20 |
| Open MPI | 4.1 |
| NVSHMEM | 3.7.1 |
| Torch | 2.6.0 (CUDA 12.4) |
| Transformers | 5.15.1 |
| Hugging Face Hub | 1.23.0 |
| Diffusers | 0.40.0.dev0 + DiFlow patches (pinned submodule) |

The default native build includes `sm_80`, `sm_89`, and `sm_90` plus PTX,
covering A100, RTX 4090, and Hopper-class compilation targets. These targets are
not a substitute for release testing on each physical GPU. Multi-node serving
also requires a working NVSHMEM transport and MPI configuration.

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
python -m pip install -e submodules/diffusers -c requirements/constraints-cu124.txt
```

DiFlow detects the installed NVSHMEM wheel and standard Ubuntu Open MPI path.
Set these variables explicitly when using non-standard locations:

```bash
export NVSHMEM_DIR="$(python -c 'import importlib.util; print(next(iter(importlib.util.find_spec("nvidia.nvshmem").submodule_search_locations)))')"
export MPI_DIR=/usr/lib/x86_64-linux-gnu/openmpi
```

Build for the GPU architectures you will deploy. A single architecture makes
local development builds faster:

```bash
export TORCH_CUDA_ARCH_LIST=9.0
python -m pip install -e . --no-build-isolation
```

Use `8.0` for A100, `8.9` for RTX 4090, or
`8.0;8.9;9.0+PTX` for a release wheel.

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
python -c "import diflow.backend.data_engine._data_engine"
python -m pip check
diflow --help
```

## Building a native wheel

A release wheel must be built on Linux with all native prerequisites available:

```bash
python -m pip install -r requirements/build.txt \
  -c requirements/constraints-cu124.txt
TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0+PTX" \
  python -m build --wheel --no-isolation
python -m twine check dist/*.whl
```

DiFlow deliberately refuses to produce a release wheel when Torch or native
prerequisites are absent. `DIFLOW_SKIP_NATIVE=1` is limited to editable CPU
test environments and source-distribution metadata.

## Troubleshooting

### PyTorch is unavailable in the build environment

Install build requirements first and use `--no-build-isolation`. The native
extension build must use the same Torch/CUDA environment as the final runtime.

### Unable to locate NVSHMEM_DIR or MPI_DIR

Set both variables to prefixes containing `include/` and `lib/`. On Ubuntu,
`MPI_DIR=/usr/lib/x86_64-linux-gnu/openmpi` is the usual Open MPI location.
Release wheels look for the NVSHMEM dependency installed alongside DiFlow. When
building a wheel against a system-wide custom NVSHMEM installation, also set
`DIFLOW_NVSHMEM_RPATH=$NVSHMEM_DIR/lib` so the deployed linker can locate that
installation.

### NVSHMEM device enumeration failed

Single-node execution may still have another usable transport, but do not assume
that this warning is harmless for cross-node serving. Verify the NVSHMEM
transport on every release environment before advertising multi-node support.

### Automatic benchmark runs out of memory

OOM profile points are marked unsupported and the rest of the sweep continues.
Use smaller startup sweeps as described in
[Runtime profiles](runtime-profiles.md).
