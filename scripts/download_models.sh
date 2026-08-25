#!/usr/bin/env bash
set -euo pipefail

model_root="${1:-${DIFLOW_MODEL_DIR:-./models}}"
mkdir -p "$model_root"

download_if_missing() {
    local repo_id="$1"
    local local_dir="$2"

    if [ -d "$local_dir" ]; then
        echo "Skipping $repo_id: $local_dir already exists"
        return
    fi

    hf download "$repo_id" --local-dir "$local_dir"
}

download_if_missing \
    black-forest-labs/FLUX.1-schnell \
    "$model_root/FLUX.1-schnell"
download_if_missing \
    black-forest-labs/FLUX.1-dev \
    "$model_root/FLUX.1-dev"
download_if_missing \
    XLabs-AI/flux-controlnet-canny-diffusers \
    "$model_root/Xlabs-AI--flux-controlnet-canny-diffusers"
download_if_missing \
    XLabs-AI/flux-controlnet-depth-diffusers \
    "$model_root/Xlabs-AI--flux-controlnet-depth-diffusers"
