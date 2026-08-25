import argparse
import os
import signal
import sys
import time

import pytest

from diflow.cli.serve import (
    WorkerProcess,
    _pid_is_running,
    build_worker_command,
    read_worker_hostnames,
    resolve_worker_layout,
    validate_transfer_layout,
)


def test_local_layout_defaults_to_one_worker():
    assert resolve_worker_layout(None, None) == (["localhost"], 1)


def test_hostfile_expands_slots(tmp_path):
    hostfile = tmp_path / "hosts"
    hostfile.write_text("node-a slots=2\nnode-b\n")

    assert read_worker_hostnames(str(hostfile)) == ["node-a", "node-a", "node-b"]
    assert resolve_worker_layout(str(hostfile), None) == (
        ["node-a", "node-a", "node-b"],
        3,
    )


def test_explicit_worker_count_must_match_hostfile(tmp_path):
    hostfile = tmp_path / "hosts"
    hostfile.write_text("node-a\nnode-b\n")

    with pytest.raises(ValueError, match="must match"):
        resolve_worker_layout(str(hostfile), 1)


def test_worker_command_uses_module_entrypoint_and_configs():
    args = argparse.Namespace(
        hostfile=None,
        base_port=14000,
        prefetch_models_config="prefetch.yaml",
        transfer_backend="host",
        host_transfer_dir="/dev/shm/diflow",
        transfer_session_id="session-id",
    )

    assert build_worker_command(args, 2) == [
        "mpirun",
        "-n",
        "2",
        sys.executable,
        "-m",
        "diflow.backend.worker",
        "--base-port",
        "14000",
        "--prefetch-models-config",
        "prefetch.yaml",
        "--transfer-backend",
        "host",
        "--host-transfer-dir",
        "/dev/shm/diflow",
        "--transfer-session-id",
        "session-id",
    ]


def test_host_transfer_rejects_hostfile():
    with pytest.raises(ValueError, match="single-node"):
        validate_transfer_layout("host", "hosts")


def test_nvshmem_transfer_allows_hostfile():
    validate_transfer_layout("nvshmem", "hosts")


@pytest.mark.skipif(sys.platform != "linux", reason="requires /proc")
def test_worker_stop_cleans_descendant_in_own_session(tmp_path):
    child_pid_path = tmp_path / "child.pid"
    child_code = "import time; time.sleep(60)"
    parent_code = (
        "import pathlib, subprocess, sys, time; "
        f"child = subprocess.Popen([sys.executable, '-c', {child_code!r}], "
        "start_new_session=True); "
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(child.pid)); "
        "time.sleep(60)"
    )
    worker = WorkerProcess([sys.executable, "-c", parent_code])
    child_pid = None

    try:
        worker.start()
        deadline = time.monotonic() + 3.0
        while not child_pid_path.is_file() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert child_pid_path.is_file()

        child_pid = int(child_pid_path.read_text())
        assert _pid_is_running(child_pid)
        worker.stop(timeout=0.5)

        deadline = time.monotonic() + 1.0
        while _pid_is_running(child_pid) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not _pid_is_running(child_pid)
    finally:
        worker.stop(timeout=0.1)
        if child_pid is not None and _pid_is_running(child_pid):
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
