#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/run_bw_test.sh [OPTIONS] [-- BW_TEST_OPTIONS...]

Run the DiFlow NVSHMEM bandwidth benchmark with exactly two MPI processes.

Options:
  --hostfile FILE  Open MPI hostfile for multi-node execution.
  --log-dir DIR    Directory for benchmark logs and CSV output.
  -h, --help       Show this help message.

Environment:
  DIFLOW_MPIRUN    mpirun executable to use (default: mpirun).
  DIFLOW_PYTHON    Python executable to use (default: python3).
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mpirun_bin="${DIFLOW_MPIRUN:-mpirun}"
python_bin="${DIFLOW_PYTHON:-python3}"
hostfile=""
log_dir="$repo_root/logs/bw-test-$(date +%Y%m%d-%H%M%S)-$$"
benchmark_args=()

while (($# > 0)); do
    case "$1" in
        --hostfile)
            if (($# < 2)); then
                echo "Error: --hostfile requires a file path." >&2
                exit 2
            fi
            hostfile="$2"
            shift 2
            ;;
        --log-dir)
            if (($# < 2)); then
                echo "Error: --log-dir requires a directory path." >&2
                exit 2
            fi
            log_dir="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            benchmark_args+=("$@")
            break
            ;;
        *)
            benchmark_args+=("$1")
            shift
            ;;
    esac
done

if [[ -n "$hostfile" && ! -f "$hostfile" ]]; then
    echo "Error: hostfile does not exist: $hostfile" >&2
    exit 2
fi
if ! command -v "$mpirun_bin" >/dev/null 2>&1; then
    echo "Error: mpirun executable not found: $mpirun_bin" >&2
    exit 1
fi
if ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "Error: Python executable not found: $python_bin" >&2
    exit 1
fi

mkdir -p "$log_dir"
export NVSHMEM_MPI_SUPPORT="${NVSHMEM_MPI_SUPPORT:-1}"

command=("$mpirun_bin" -n 2)
if [[ -n "$hostfile" ]]; then
    command+=(--hostfile "$hostfile")
fi
command+=(
    "$python_bin"
    "$repo_root/scripts/bw_test.py"
    --log-dir "$log_dir"
)
command+=("${benchmark_args[@]}")

exec "${command[@]}"
