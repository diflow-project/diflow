from __future__ import annotations

import argparse
import importlib.resources
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple

from benchmark_ops.shapes import parse_batch_sizes, parse_resolutions
from diflow.backend.data_engine.engine import resolve_transfer_backend
from diflow.cli.auto_benchmark import (
    DEFAULT_AUTO_BENCHMARK_CACHE_DIR,
    run_auto_benchmark,
)
from diflow.cli.workflow_loader import (
    LoadedWorkflow,
    WorkflowLoadError,
    add_workflow_arguments,
    load_workflow,
    workflow_kwargs,
)
from diflow.interface.benchmark import BenchmarkSpec
from diflow.profiling.runtime_profile import RuntimeProfile


def default_config_path(filename: str) -> str:
    return str(importlib.resources.files("diflow").joinpath("configs", filename))


DEFAULT_HOST_TRANSFER_DIR = "/dev/shm/diflow"


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--workflow",
        required=True,
        help="Built-in workflow name or path to a Python workflow file.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--base-port", type=int, default=14000)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of workers. Defaults to 1 locally or the hostfile entry count.",
    )
    parser.add_argument(
        "--hostfile",
        help="MPI hostfile with one host per worker. Omit for local serving.",
    )
    parser.add_argument(
        "--transfer-backend",
        choices=["auto", "nvshmem", "host"],
        default="auto",
        help="Intermediate-tensor transfer backend. Auto prefers NVSHMEM when available.",
    )
    parser.add_argument(
        "--host-transfer-dir",
        default=DEFAULT_HOST_TRANSFER_DIR,
        help="Shared-memory directory used by the single-node host backend.",
    )
    parser.add_argument(
        "--scheduling-policy",
        default="dynamic",
        choices=["exclusive", "random", "dynamic"],
    )
    parser.add_argument(
        "--preload-models-config",
        default=default_config_path("preload_models.yaml"),
    )
    parser.add_argument(
        "--prefetch-models-config",
        default=default_config_path("prefetch_models.yaml"),
    )
    parser.add_argument(
        "--model-batch-config",
        default=default_config_path("model_batch.json"),
    )
    parser.add_argument("--enable-early-abort", action="store_true")
    parser.add_argument(
        "--runtime-profile",
        help="Use an existing runtime profile and skip automatic benchmarking.",
    )
    parser.add_argument(
        "--no-auto-benchmark",
        action="store_true",
        help="Disable startup profiling; requires --runtime-profile.",
    )
    parser.add_argument(
        "--auto-benchmark-cache-dir",
        default=DEFAULT_AUTO_BENCHMARK_CACHE_DIR,
        help="Cache directory for startup-time shape profiles.",
    )
    parser.add_argument(
        "--force-auto-benchmark",
        action="store_true",
        help="Ignore a matching cache entry and profile again.",
    )
    parser.add_argument(
        "--auto-benchmark-batch-sizes",
        help="Override benchmark batch sizes, for example 1,2,4.",
    )
    parser.add_argument(
        "--auto-benchmark-resolutions",
        help="Override benchmark resolutions, for example 512x512,1024x1024.",
    )
    parser.add_argument(
        "--strict-auto-benchmark",
        action="store_false",
        dest="auto_benchmark_best_effort",
        help="Fail startup on the first non-OOM profiling error.",
    )
    parser.add_argument(
        "--auto-benchmark-best-effort",
        action="store_true",
        dest="auto_benchmark_best_effort",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(auto_benchmark_best_effort=True)
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=60.0,
        help="Seconds to wait for workers to become ready.",
    )


def _effective_benchmark_spec(workflow: Any, args: argparse.Namespace) -> BenchmarkSpec:
    spec = workflow.benchmark
    if spec is None:
        raise ValueError("The selected workflow does not declare BenchmarkSpec")

    updates = {}
    if args.auto_benchmark_batch_sizes:
        updates["batch_sizes"] = parse_batch_sizes(args.auto_benchmark_batch_sizes)
    if args.auto_benchmark_resolutions:
        updates["resolutions"] = parse_resolutions(args.auto_benchmark_resolutions)
    return replace(spec, **updates) if updates else spec


def parse_args(argv: Optional[Sequence[str]], prog: str) -> Tuple[
    argparse.ArgumentParser,
    argparse.Namespace,
    LoadedWorkflow,
    tuple[str, ...],
]:
    arguments = list(sys.argv[1:] if argv is None else argv)
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--workflow")
    known, _ = bootstrap.parse_known_args(arguments)

    parser = argparse.ArgumentParser(
        prog=prog,
        description="Start DiFlow workers and serve a workflow.",
    )
    _add_common_arguments(parser)
    loaded = None
    destinations: tuple[str, ...] = ()
    if known.workflow:
        try:
            loaded = load_workflow(known.workflow)
            destinations = add_workflow_arguments(parser, loaded)
        except WorkflowLoadError as exc:
            parser.error(str(exc))

    args = parser.parse_args(arguments)
    if loaded is None:
        parser.error("--workflow is required")
    return parser, args, loaded, destinations


def read_worker_hostnames(hostfile: str) -> List[str]:
    path = Path(hostfile).expanduser()
    if not path.is_file():
        raise ValueError(f"Hostfile does not exist: {path}")

    hostnames: List[str] = []
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        fields = line.split()
        hostname = fields[0]
        slots = 1
        for field in fields[1:]:
            match = re.fullmatch(r"slots=(\d+)", field)
            if match:
                slots = int(match.group(1))
                break
        if slots < 1:
            raise ValueError(f"Invalid slots value on {path}:{line_number}")
        hostnames.extend([hostname] * slots)

    if not hostnames:
        raise ValueError(f"Hostfile contains no worker hosts: {path}")
    return hostnames


def resolve_worker_layout(
    hostfile: Optional[str], num_workers: Optional[int]
) -> Tuple[List[str], int]:
    if num_workers is not None and num_workers < 1:
        raise ValueError("--num-workers must be at least 1")
    if hostfile:
        hostnames = read_worker_hostnames(hostfile)
        resolved_count = len(hostnames) if num_workers is None else num_workers
        if resolved_count != len(hostnames):
            raise ValueError(
                f"--num-workers ({resolved_count}) must match the expanded hostfile "
                f"entry count ({len(hostnames)})"
            )
        return hostnames, resolved_count

    resolved_count = 1 if num_workers is None else num_workers
    return ["localhost"] * resolved_count, resolved_count


def validate_transfer_layout(backend: str, hostfile: Optional[str]) -> None:
    if backend == "host" and hostfile:
        raise ValueError(
            "The host transfer backend supports only local single-node workers; "
            "remove --hostfile or use --transfer-backend nvshmem"
        )


def build_worker_command(args: argparse.Namespace, num_workers: int) -> List[str]:
    command = ["mpirun", "-n", str(num_workers)]
    if args.hostfile:
        command.extend(["--hostfile", str(Path(args.hostfile).expanduser())])
    command.extend(
        [
            sys.executable,
            "-m",
            "diflow.backend.worker",
            "--base-port",
            str(args.base_port),
            "--prefetch-models-config",
            args.prefetch_models_config,
            "--transfer-backend",
            args.transfer_backend,
            "--host-transfer-dir",
            args.host_transfer_dir,
            "--transfer-session-id",
            args.transfer_session_id,
        ]
    )
    return command


def _linux_descendant_pids(root_pid: int) -> Tuple[int, ...]:
    """Snapshot Linux descendants, including children in their own sessions."""
    if sys.platform != "linux":
        return ()

    children_by_parent: dict[int, list[int]] = {}
    try:
        process_entries = list(Path("/proc").iterdir())
    except OSError:
        return ()

    for entry in process_entries:
        if not entry.name.isdigit():
            continue
        try:
            status = (entry / "status").read_text()
            parent_line = next(
                line for line in status.splitlines() if line.startswith("PPid:")
            )
            parent_pid = int(parent_line.split()[1])
        except (OSError, StopIteration, ValueError):
            continue
        children_by_parent.setdefault(parent_pid, []).append(int(entry.name))

    descendants: list[int] = []
    frontier = [root_pid]
    while frontier:
        parent_pid = frontier.pop()
        children = children_by_parent.get(parent_pid, ())
        descendants.extend(children)
        frontier.extend(children)
    return tuple(descendants)


def _pid_is_running(pid: int) -> bool:
    if sys.platform == "linux":
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
        except OSError:
            return False
        closing_parenthesis = stat.rfind(")")
        if (
            closing_parenthesis >= 0
            and stat[closing_parenthesis + 2 : closing_parenthesis + 3] == "Z"
        ):
            return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _wait_for_pids(pids: Sequence[int], deadline: float) -> Tuple[int, ...]:
    remaining = {pid for pid in pids if _pid_is_running(pid)}
    while remaining and time.monotonic() < deadline:
        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        remaining = {pid for pid in remaining if _pid_is_running(pid)}
    return tuple(sorted(remaining))


class WorkerProcess:
    def __init__(self, command: Sequence[str]):
        self.command = list(command)
        self.process: Optional[subprocess.Popen] = None
        self.unexpected_returncode: Optional[int] = None
        self._stopping = False

    def start(self) -> None:
        environment = os.environ.copy()
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            environment.setdefault("OMPI_ALLOW_RUN_AS_ROOT", "1")
            environment.setdefault("OMPI_ALLOW_RUN_AS_ROOT_CONFIRM", "1")
        self.process = subprocess.Popen(
            self.command,
            env=environment,
            start_new_session=True,
        )

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def monitor(self, on_unexpected_exit: Callable[[int], None]) -> None:
        if self.process is None:
            raise RuntimeError("Worker process has not been started")

        def wait_for_exit() -> None:
            assert self.process is not None
            returncode = self.process.wait()
            if not self._stopping:
                self.unexpected_returncode = returncode
                on_unexpected_exit(returncode)

        threading.Thread(target=wait_for_exit, daemon=True).start()

    def stop(self, timeout: float = 10.0) -> None:
        self._stopping = True
        if self.process is None:
            return

        descendants = _linux_descendant_pids(self.process.pid)
        deadline = time.monotonic() + timeout
        if self.process.poll() is None:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                pass

        survivors = _wait_for_pids(descendants, deadline)
        if self.process.poll() is None:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        for pid in survivors:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        if self.process.poll() is None:
            try:
                self.process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                pass


def _validate_common_args(args: argparse.Namespace) -> None:
    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be between 1 and 65535")
    if not 1 <= args.base_port <= 65534:
        raise ValueError("--base-port must be between 1 and 65534")
    if args.startup_timeout <= 0:
        raise ValueError("--startup-timeout must be greater than 0")
    if args.no_auto_benchmark and not args.runtime_profile:
        raise ValueError("--no-auto-benchmark requires --runtime-profile")
    if not args.host_transfer_dir:
        raise ValueError("--host-transfer-dir cannot be empty")

    for option in (
        "preload_models_config",
        "prefetch_models_config",
        "model_batch_config",
    ):
        path = Path(getattr(args, option)).expanduser()
        if not path.is_file():
            raise ValueError(
                f"--{option.replace('_', '-')} file does not exist: {path}"
            )
        setattr(args, option, str(path))

    if args.runtime_profile:
        profile_path = Path(args.runtime_profile).expanduser()
        if not profile_path.is_file():
            raise ValueError(f"--runtime-profile file does not exist: {profile_path}")
        args.runtime_profile = str(profile_path)


def run(
    args: argparse.Namespace,
    loaded: LoadedWorkflow,
    destinations: tuple[str, ...],
) -> int:
    import uvicorn

    from diflow.backend.scheduler import SchedulingPolicy
    from diflow.backend.server import WorkflowService, create_app
    from diflow.interface.workflow import Workflow

    _validate_common_args(args)
    worker_hostnames, num_workers = resolve_worker_layout(
        args.hostfile, args.num_workers
    )
    args.transfer_backend = resolve_transfer_backend(args.transfer_backend)
    validate_transfer_layout(args.transfer_backend, args.hostfile)
    args.host_transfer_dir = str(Path(args.host_transfer_dir).expanduser().resolve())
    args.transfer_session_id = uuid.uuid4().hex
    try:
        factory_kwargs = workflow_kwargs(args, destinations)
        workflow = loaded.factory(**factory_kwargs)
    except Exception as exc:
        raise ValueError(
            f"Failed to build workflow from {loaded.source}: {exc}"
        ) from exc
    if not isinstance(workflow, Workflow):
        raise ValueError(
            f"{loaded.source} create_workflow() returned {type(workflow).__name__}; "
            "expected diflow.interface.Workflow"
        )

    if args.runtime_profile:
        print(f"Using runtime profile: {args.runtime_profile}")
        runtime_profile = RuntimeProfile.from_file(args.runtime_profile)
    else:
        benchmark_spec = _effective_benchmark_spec(workflow, args)
        benchmark_result = run_auto_benchmark(
            workflow_spec=args.workflow,
            loaded=loaded,
            workflow=workflow,
            workflow_kwargs=factory_kwargs,
            benchmark_spec=benchmark_spec,
            cache_dir=args.auto_benchmark_cache_dir,
            force=args.force_auto_benchmark,
            best_effort=args.auto_benchmark_best_effort,
        )
        runtime_profile = RuntimeProfile.from_file(benchmark_result.profile_path)

    worker = WorkerProcess(build_worker_command(args, num_workers))
    print(f"Starting {num_workers} DiFlow worker(s): {' '.join(worker.command)}")
    worker.start()

    service = WorkflowService(
        worker_hostnames=worker_hostnames,
        scheduling_policy=SchedulingPolicy(args.scheduling_policy),
        base_port=args.base_port,
        preload_models_config=args.preload_models_config,
        model_batch_config=args.model_batch_config,
        enable_early_abort=args.enable_early_abort,
        runtime_profile=runtime_profile,
    )

    def on_ready(service_id: Optional[str]) -> None:
        print(
            f"DiFlow is serving {loaded.source} at http://{args.host}:{args.port} "
            f"(service_id={service_id})"
        )

    app = create_app(
        service,
        initial_workflow=workflow,
        startup_timeout=args.startup_timeout,
        worker_health_check=worker.is_running,
        on_ready=on_ready,
        on_shutdown=worker.stop,
    )
    config = uvicorn.Config(
        app,
        host=args.host,
        port=args.port,
        loop="asyncio",
        timeout_keep_alive=30,
        timeout_graceful_shutdown=30,
    )
    server = uvicorn.Server(config)

    def stop_server(returncode: int) -> None:
        print(
            f"DiFlow worker process exited unexpectedly with status {returncode}",
            file=sys.stderr,
        )
        server.should_exit = True

    worker.monitor(stop_server)
    try:
        try:
            server.run()
        except KeyboardInterrupt:
            # Uvicorn re-raises a captured SIGINT after completing its graceful
            # shutdown. Cleanup still happens below; avoid a spurious traceback.
            pass
    finally:
        worker.stop()
        if args.transfer_backend == "host":
            session_dir = Path(args.host_transfer_dir) / args.transfer_session_id
            shutil.rmtree(session_dir, ignore_errors=True)
    if not server.started or worker.unexpected_returncode is not None:
        return 1
    return 0


def main(argv: Optional[Sequence[str]] = None, prog: str = "diflow serve") -> int:
    parser, args, loaded, destinations = parse_args(argv, prog)
    try:
        return run(args, loaded, destinations)
    except (OSError, RuntimeError, ValueError, WorkflowLoadError) as exc:
        parser.error(str(exc))
    return 2
