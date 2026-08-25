"""Startup-time automatic benchmark orchestration.

CUDA profiling runs in a short-lived child process. The serving process manages
only fingerprints and immutable cache entries, so profiler allocations cannot
leak into the server.
"""

from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from diflow.cli.workflow_loader import LoadedWorkflow
from diflow.interface.benchmark import BenchmarkSpec
from diflow.interface.workflow import Workflow

AUTO_BENCHMARK_SCHEMA_VERSION = 3
DEFAULT_AUTO_BENCHMARK_CACHE_DIR = "~/.cache/diflow/benchmarks"
MANIFEST_FILENAME = "manifest.json"
CACHEABLE_STATUSES = {"complete", "partial"}


@dataclass(frozen=True)
class AutoBenchmarkResult:
    fingerprint: str
    cache_dir: Path
    profile_path: Path
    cache_hit: bool
    status: str


def query_gpu_identity() -> Dict[str, str]:
    """Return stable GPU and driver identity without initializing CUDA."""
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version,uuid",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    devices = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not devices:
        raise RuntimeError("nvidia-smi did not report any GPUs")
    return {"nvidia_smi": "\n".join(devices)}


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "editable"


def _runtime_versions() -> Dict[str, str]:
    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "torch": _package_version("torch"),
        "diflow": _package_version("diflow"),
        "transformers": _package_version("transformers"),
    }


def _factory_source_digest(loaded: LoadedWorkflow) -> str:
    source_path = inspect.getsourcefile(loaded.factory)
    if source_path is None:
        return hashlib.sha256(repr(loaded.factory).encode()).hexdigest()
    try:
        source = Path(source_path).read_bytes()
    except OSError:
        source = repr(loaded.factory).encode()
    return hashlib.sha256(source).hexdigest()


def _runtime_source_digest() -> str:
    """Invalidate editable-install caches when profiler or operator code changes."""
    repository_root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for package_name in ("diflow", "benchmark_ops"):
        package_root = repository_root / package_name
        if not package_root.is_dir():
            continue
        for path in sorted(package_root.rglob("*.py")):
            digest.update(str(path.relative_to(repository_root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def build_fingerprint(
    loaded: LoadedWorkflow,
    workflow_kwargs: Mapping[str, Any],
    spec: BenchmarkSpec,
    gpu_identity: Mapping[str, str],
    best_effort: bool = True,
) -> str:
    payload = {
        "schema": AUTO_BENCHMARK_SCHEMA_VERSION,
        "workflow_name": loaded.name,
        "workflow_source": loaded.source,
        "factory_source_sha256": _factory_source_digest(loaded),
        "factory_kwargs": dict(workflow_kwargs),
        "runtime_source_sha256": _runtime_source_digest(),
        "benchmark": spec.to_dict(),
        "best_effort": best_effort,
        "gpu": dict(gpu_identity),
        "runtime": _runtime_versions(),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_child_command(
    workflow_spec: str,
    workflow_kwargs: Mapping[str, Any],
    benchmark_spec: BenchmarkSpec,
    results_dir: Path,
    case_name: str,
    best_effort: bool,
) -> Sequence[str]:
    command = [
        sys.executable,
        "-m",
        "benchmark_ops.auto_cli",
        "--workflow",
        workflow_spec,
        "--workflow-kwargs-json",
        json.dumps(dict(workflow_kwargs), sort_keys=True, default=str),
        "--benchmark-spec-json",
        json.dumps(benchmark_spec.to_dict(), sort_keys=True, default=str),
        "--results-dir",
        str(results_dir),
        "--case-name",
        case_name,
    ]
    if best_effort:
        command.append("--best-effort")
    return command


def _load_cached_result(entry: Path, fingerprint: str) -> Optional[AutoBenchmarkResult]:
    manifest_path = entry / MANIFEST_FILENAME
    try:
        manifest = json.loads(manifest_path.read_text())
        profile_path = entry / manifest["profile_path"]
    except (OSError, KeyError, json.JSONDecodeError):
        return None
    status = str(manifest.get("status", "invalid"))
    if manifest.get("schema") != AUTO_BENCHMARK_SCHEMA_VERSION:
        return None
    if manifest.get("fingerprint") != fingerprint:
        return None
    if status not in CACHEABLE_STATUSES or not profile_path.is_file():
        return None
    return AutoBenchmarkResult(
        fingerprint=fingerprint,
        cache_dir=entry,
        profile_path=profile_path,
        cache_hit=True,
        status=status,
    )


def _discover_output(results_dir: Path) -> Path:
    profiles = list(results_dir.glob("*/*.json"))
    if len(profiles) != 1:
        raise RuntimeError(
            "Automatic benchmark produced an unexpected result layout: "
            f"expected one profile, found {len(profiles)}"
        )
    return profiles[0]


def _profile_summary(profile: Mapping[str, Any]) -> Dict[str, Any]:
    ops = list(profile.get("ops", ()))
    op_errors = [str(op.get("error")) for op in ops if op.get("error")]
    capture_errors = [
        str(error.get("error")) for error in profile.get("profile_errors", ())
    ]
    successful_ops = sum(
        1 for op in ops if op.get("latency") is not None and not op.get("error")
    )
    status = "complete"
    if successful_ops == 0:
        status = "invalid"
    elif op_errors or capture_errors:
        status = "partial"
    return {
        "status": status,
        "successful_operator_profiles": successful_ops,
        "operator_errors": op_errors,
        "capture_errors": capture_errors,
    }


def run_auto_benchmark(
    *,
    workflow_spec: str,
    loaded: LoadedWorkflow,
    workflow: Workflow,
    workflow_kwargs: Mapping[str, Any],
    benchmark_spec: Optional[BenchmarkSpec] = None,
    cache_dir: str = DEFAULT_AUTO_BENCHMARK_CACHE_DIR,
    force: bool = False,
    best_effort: bool = True,
    gpu_identity: Optional[Mapping[str, str]] = None,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> AutoBenchmarkResult:
    """Benchmark a workflow if needed and return its runtime profile."""
    spec = benchmark_spec or workflow.benchmark
    if spec is None:
        raise ValueError(
            f"Workflow {loaded.source} must declare BenchmarkSpec before serving"
        )

    identity = dict(gpu_identity or query_gpu_identity())
    fingerprint = build_fingerprint(
        loaded, workflow_kwargs, spec, identity, best_effort=best_effort
    )
    root = Path(cache_dir).expanduser().resolve()
    entry = root / fingerprint
    root.mkdir(parents=True, exist_ok=True)

    lock_path = root / f"{fingerprint}.lock"
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        cached = _load_cached_result(entry, fingerprint)
        if cached is not None and not force:
            print(
                f"Automatic benchmark cache hit ({cached.status}): "
                f"{cached.cache_dir}"
            )
            return cached

        temporary = Path(tempfile.mkdtemp(prefix=f".{fingerprint}.", dir=root))
        try:
            results_dir = temporary / "results"
            command = build_child_command(
                workflow_spec=workflow_spec,
                workflow_kwargs=workflow_kwargs,
                benchmark_spec=spec,
                results_dir=results_dir,
                case_name=f"auto_{fingerprint[:16]}",
                best_effort=best_effort,
            )
            print(
                f"Running automatic workflow benchmark in child process: {loaded.source}"
            )
            try:
                runner(command, check=True)
            except subprocess.CalledProcessError as error:
                raise RuntimeError(
                    "Automatic benchmark child process exited with status "
                    f"{error.returncode}"
                ) from error

            profile_path = _discover_output(results_dir)
            profile = json.loads(profile_path.read_text())
            summary = _profile_summary(profile)
            if summary["status"] == "invalid":
                raise RuntimeError(
                    "Automatic benchmark produced no usable operator profiles. "
                    f"Capture errors: {summary['capture_errors']}; "
                    f"operator errors: {summary['operator_errors']}"
                )

            manifest = {
                "schema": AUTO_BENCHMARK_SCHEMA_VERSION,
                "fingerprint": fingerprint,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "status": summary["status"],
                "gpu": identity,
                "runtime": _runtime_versions(),
                "benchmark": spec.to_dict(),
                "summary": {
                    "successful_operator_profiles": summary[
                        "successful_operator_profiles"
                    ],
                    "operator_error_count": len(summary["operator_errors"]),
                    "capture_error_count": len(summary["capture_errors"]),
                },
                "profile_path": str(profile_path.relative_to(temporary)),
            }
            manifest_tmp = temporary / f".{MANIFEST_FILENAME}.tmp"
            manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
            os.replace(manifest_tmp, temporary / MANIFEST_FILENAME)

            if entry.exists():
                shutil.rmtree(entry)
            os.replace(temporary, entry)
            profile_path = entry / manifest["profile_path"]
            print(f"Automatic benchmark complete ({summary['status']}): {profile_path}")
            return AutoBenchmarkResult(
                fingerprint=fingerprint,
                cache_dir=entry,
                profile_path=profile_path,
                cache_hit=False,
                status=str(summary["status"]),
            )
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
