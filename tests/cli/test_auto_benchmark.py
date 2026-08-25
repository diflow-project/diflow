import json
import subprocess
from pathlib import Path

import pytest

from diflow.cli.auto_benchmark import build_fingerprint, run_auto_benchmark
from diflow.cli.workflow_loader import LoadedWorkflow
from diflow.interface import BenchmarkSpec, Workflow


def _factory() -> Workflow:
    return Workflow("test", benchmark=BenchmarkSpec(inputs={"prompt": "hello"}))


def _loaded() -> LoadedWorkflow:
    return LoadedWorkflow(
        name="test",
        source="test.py",
        factory=_factory,
    )


def _profile_payload(*, partial: bool = False) -> dict:
    op = {
        "op_id": "TestOp",
        "mode": "default",
        "shape": {"batch_size": 1, "height": 256, "width": 256},
        "latency": {"median": 0.1},
        "gpu_memory_used": 1,
    }
    if partial:
        op["error"] = "OOM"
        op.pop("latency")
        successful = {
            "op_id": "TestOp",
            "mode": "default",
            "shape": {"batch_size": 1, "height": 512, "width": 512},
            "latency": {"median": 0.2},
            "gpu_memory_used": 1,
        }
        ops = [op, successful]
    else:
        ops = [op]
    return {
        "schema_version": 2,
        "gpu_type": "fake",
        "gpu_memory_total": 1,
        "gpu_count": 1,
        "ops": ops,
        "model_load_profiles": [],
        "profile_errors": [],
    }


def _runner(calls, *, partial: bool = False):
    def fake_runner(command, check):
        assert check is True
        assert "--benchmark-spec-json" in command
        calls.append(command)
        results_dir = Path(command[command.index("--results-dir") + 1])
        case_name = command[command.index("--case-name") + 1]
        gpu_dir = results_dir / "fake_gpu"
        gpu_dir.mkdir(parents=True)
        (gpu_dir / f"{case_name}.json").write_text(
            json.dumps(_profile_payload(partial=partial))
        )
        return subprocess.CompletedProcess(command, 0)

    return fake_runner


def test_fingerprint_covers_gpu_factory_kwargs_and_sweep():
    loaded = _loaded()
    base = BenchmarkSpec(inputs={"prompt": "hello"})
    digest = build_fingerprint(loaded, {"model_path": "a"}, base, {"gpu": "A"})

    assert digest != build_fingerprint(loaded, {"model_path": "b"}, base, {"gpu": "A"})
    assert digest != build_fingerprint(loaded, {"model_path": "a"}, base, {"gpu": "B"})
    assert digest != build_fingerprint(
        loaded, {"model_path": "a"}, base, {"gpu": "A"}, best_effort=False
    )
    assert digest != build_fingerprint(
        loaded,
        {"model_path": "a"},
        BenchmarkSpec(inputs={"prompt": "hello"}, batch_sizes=(1, 2)),
        {"gpu": "A"},
    )


def test_run_uses_atomic_cache_entry(tmp_path):
    calls = []
    arguments = {
        "workflow_spec": "test.py",
        "loaded": _loaded(),
        "workflow": _factory(),
        "workflow_kwargs": {},
        "cache_dir": str(tmp_path),
        "gpu_identity": {"gpu": "fake"},
        "runner": _runner(calls),
    }

    first = run_auto_benchmark(**arguments)
    second = run_auto_benchmark(**arguments)

    assert first.cache_hit is False
    assert first.status == "complete"
    assert second.cache_hit is True
    assert second.status == "complete"
    assert first.profile_path.is_file()
    assert json.loads(first.profile_path.read_text())["schema_version"] == 2
    assert len(calls) == 1


def test_partial_result_is_labeled_and_cached(tmp_path):
    calls = []
    arguments = {
        "workflow_spec": "test.py",
        "loaded": _loaded(),
        "workflow": _factory(),
        "workflow_kwargs": {},
        "cache_dir": str(tmp_path),
        "gpu_identity": {"gpu": "fake"},
        "runner": _runner(calls, partial=True),
    }

    first = run_auto_benchmark(**arguments)
    second = run_auto_benchmark(**arguments)

    assert first.status == "partial"
    assert second.cache_hit is True
    manifest = json.loads((first.cache_dir / "manifest.json").read_text())
    assert manifest["status"] == "partial"
    assert manifest["summary"]["operator_error_count"] == 1
    assert len(calls) == 1


def test_old_manifest_is_not_a_cache_hit(tmp_path):
    calls = []
    arguments = {
        "workflow_spec": "test.py",
        "loaded": _loaded(),
        "workflow": _factory(),
        "workflow_kwargs": {},
        "cache_dir": str(tmp_path),
        "gpu_identity": {"gpu": "fake"},
        "runner": _runner(calls),
    }
    first = run_auto_benchmark(**arguments)
    manifest_path = first.cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema"] = 2
    manifest_path.write_text(json.dumps(manifest))

    second = run_auto_benchmark(**arguments)

    assert second.cache_hit is False
    assert len(calls) == 2


def test_workflow_without_spec_rejects_startup(tmp_path):
    def fail_runner(*args, **kwargs):
        raise AssertionError("runner must not be called")

    with pytest.raises(ValueError, match="must declare BenchmarkSpec"):
        run_auto_benchmark(
            workflow_spec="test.py",
            loaded=_loaded(),
            workflow=Workflow("plain"),
            workflow_kwargs={},
            cache_dir=str(tmp_path),
            gpu_identity={"gpu": "fake"},
            runner=fail_runner,
        )


def test_failed_child_does_not_publish_cache_entry(tmp_path):
    def fail_runner(command, check):
        assert check is True
        raise subprocess.CalledProcessError(7, command)

    with pytest.raises(RuntimeError, match="status 7"):
        run_auto_benchmark(
            workflow_spec="test.py",
            loaded=_loaded(),
            workflow=_factory(),
            workflow_kwargs={},
            cache_dir=str(tmp_path),
            gpu_identity={"gpu": "fake"},
            runner=fail_runner,
        )

    assert not [path for path in tmp_path.iterdir() if path.is_dir()]
