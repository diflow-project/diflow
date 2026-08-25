import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "scripts" / "run_bw_test.sh"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def _launcher_environment(tmp_path: Path) -> tuple[dict[str, str], Path, Path]:
    capture_path = tmp_path / "mpirun-arguments.txt"
    fake_mpirun = tmp_path / "mpirun"
    fake_python = tmp_path / "python3"
    _write_executable(
        fake_mpirun,
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$DIFLOW_CAPTURE"\n',
    )
    _write_executable(fake_python, "#!/usr/bin/env bash\nexit 0\n")

    environment = os.environ.copy()
    environment.update(
        {
            "DIFLOW_CAPTURE": str(capture_path),
            "DIFLOW_MPIRUN": str(fake_mpirun),
            "DIFLOW_PYTHON": str(fake_python),
        }
    )
    return environment, capture_path, fake_python


def test_local_launcher_uses_two_mpi_processes_and_forwards_options(tmp_path):
    environment, capture_path, fake_python = _launcher_environment(tmp_path)
    log_dir = tmp_path / "logs"

    subprocess.run(
        [
            str(LAUNCHER),
            "--log-dir",
            str(log_dir),
            "--",
            "--min-block-size",
            "12",
        ],
        check=True,
        env=environment,
    )

    assert capture_path.read_text().splitlines() == [
        "-n",
        "2",
        str(fake_python),
        str(REPO_ROOT / "scripts" / "bw_test.py"),
        "--log-dir",
        str(log_dir),
        "--min-block-size",
        "12",
    ]


def test_launcher_passes_hostfile_only_to_mpirun(tmp_path):
    environment, capture_path, fake_python = _launcher_environment(tmp_path)
    hostfile = tmp_path / "hosts"
    hostfile.write_text("node-a\nnode-b\n")

    subprocess.run(
        [
            str(LAUNCHER),
            "--hostfile",
            str(hostfile),
            "--log-dir",
            str(tmp_path / "logs"),
            "--num-blocks",
            "4",
        ],
        check=True,
        env=environment,
    )

    arguments = capture_path.read_text().splitlines()
    assert arguments[:4] == ["-n", "2", "--hostfile", str(hostfile)]
    assert arguments[4:7] == [
        str(fake_python),
        str(REPO_ROOT / "scripts" / "bw_test.py"),
        "--log-dir",
    ]
    assert arguments[-2:] == ["--num-blocks", "4"]


def test_launcher_rejects_missing_hostfile(tmp_path):
    result = subprocess.run(
        [str(LAUNCHER), "--hostfile", str(tmp_path / "missing")],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "hostfile does not exist" in result.stderr


def test_launcher_help_does_not_require_runtime_commands():
    result = subprocess.run(
        [str(LAUNCHER), "--help"],
        capture_output=True,
        check=True,
        text=True,
    )

    assert "exactly two MPI processes" in result.stdout
