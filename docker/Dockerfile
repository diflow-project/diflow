# DiFlow release image for CUDA 12.4.
#
# Build from a clean checkout:
#   docker build -f docker/Dockerfile -t diflow:latest .
#
# Run:
#   docker run --gpus all --rm -p 8000:8000 \
#     -v /absolute/model/path:/models/FLUX.1-schnell:ro \
#     diflow:latest diflow serve --host 0.0.0.0 \
#       --workflow flux-schnell --model-path /models/FLUX.1-schnell

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ARG DIFFUSERS_COMMIT=1c3482238417884e430bbb23a44a508b9d1461c2
ARG TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0+PTX"

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libgl1 \
        libglib2.0-0 \
        libopenmpi-dev \
        ninja-build \
        openmpi-bin \
        python-is-python3 \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/DiFlow

COPY requirements/ requirements/
RUN python -m pip install --upgrade "pip>=25,<27" \
    && python -m pip install \
        -r requirements/runtime.txt \
        -r requirements/models.txt \
        -r requirements/build.txt \
        -r requirements/nvshmem.txt \
        -c requirements/constraints-cu124.txt \
    && python -m pip install --no-cache-dir --force-reinstall --no-binary=mpi4py "mpi4py==4.1.2"

ENV NVSHMEM_DIR=/usr/local/lib/python3.10/dist-packages/nvidia/nvshmem \
    MPI_DIR=/usr/lib/x86_64-linux-gnu/openmpi

COPY docker/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod 0755 /usr/local/bin/docker-entrypoint.sh


COPY . .

# Source archives do not contain initialized submodules. Fetch the pinned public
# commit over HTTPS when necessary.
RUN if [ ! -f submodules/diffusers/setup.py ]; then \
        rm -rf submodules/diffusers \
        && mkdir -p submodules/diffusers \
        && git -C submodules/diffusers init \
        && git -C submodules/diffusers remote add origin \
            https://github.com/diflow-project/diffusers.git \
        && git -C submodules/diffusers -c http.version=HTTP/1.1 fetch \
            --depth 1 origin "${DIFFUSERS_COMMIT}" \
        && git -C submodules/diffusers checkout --detach FETCH_HEAD \
        && rm -rf submodules/diffusers/.git; \
    fi \
    && python -m pip install -e submodules/diffusers \
    && DIFLOW_REQUIRE_NVSHMEM=1 TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
        python -m pip install . --no-build-isolation \
    && python -c "import diflow; import diflow.backend.scheduler._scheduling_core; import diflow.backend.data_engine._data_engine"

RUN useradd --create-home --uid 10001 --shell /bin/bash diflow \
    && mkdir -p /models /home/diflow/.cache/diflow \
    && chown -R diflow:diflow /workspace/DiFlow /models /home/diflow

ENV HOME=/home/diflow \
    HF_HOME=/home/diflow/.cache/huggingface

USER diflow

EXPOSE 8000

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["diflow", "--help"]
