import math

import pytest

from diflow.backend.scheduler.scheduling_core import (
    CppSchedulingCore,
    ModelProfile,
    PythonSchedulingCore,
    TaskSpec,
    TransferProfile,
    create_scheduling_core,
)


def _profiles():
    return {
        "model": ModelProfile(
            loading_latency=0.5,
            execution_latencies={
                ("default", 1, 256, 256): 0.1,
                ("default", 2, 256, 256): 0.18,
            },
        )
    }


def _core(backend_cls=PythonSchedulingCore):
    return backend_cls(
        worker_host_ids={0: 0, 1: 0, 2: 1},
        active_models={0: ["model"], 1: [], 2: []},
        model_profiles=_profiles(),
        intra_profile=TransferProfile((1024, 4096), (1000.0, 4000.0)),
        inter_profile=TransferProfile((1024, 4096), (10000.0, 40000.0)),
        worker_latency_threshold=0.01,
    )


def _task(task_id, *, batch_size=1, source_rank=0, source_host=0):
    return TaskSpec(
        task_id=task_id,
        model_name="model",
        mode="default",
        batch_size=batch_size,
        uses_model_profile=True,
        tensor_offsets=(0, 1),
        source_worker_ranks=(source_rank,),
        source_host_ids=(source_host,),
        source_sizes_bytes=(512,),
    )


def test_select_reserve_complete_and_idempotency():
    core = _core()

    first = core.select_and_reserve(_task("first"))
    assert first.worker_rank == 0
    assert first.cost.queue == 0.0
    assert first.cost.transfer == 0.0
    assert first.cost.loading == 0.0
    assert first.cost.execution == 0.1

    duplicate = core.select_and_reserve(_task("first"))
    assert duplicate == first
    assert core.snapshot()[0]["queue_latency"] == pytest.approx(0.1)

    second = core.select_and_reserve(_task("second"))
    assert second.worker_rank == 1
    assert second.cost.transfer == pytest.approx(0.001)
    assert second.cost.loading == pytest.approx(0.5)
    assert second.cost.execution == pytest.approx(0.1)

    assert core.complete("first") is True
    assert core.complete("first") is False
    assert core.snapshot()[0]["queue_latency"] == 0.0


def test_reserve_on_worker_and_active_model_update():
    core = _core()
    core.update_active_models(2, ["model"])

    result = core.reserve_on_worker(_task("pinned"), 2)
    assert result.worker_rank == 2
    assert result.cost.transfer == pytest.approx(0.01)
    assert result.cost.loading == 0.0
    assert core.snapshot()[2]["reservations"] == {"pinned": pytest.approx(0.11)}

    with pytest.raises(ValueError, match="already reserved"):
        core.reserve_on_worker(_task("pinned"), 1)


def test_missing_profile_returns_no_selection_and_infinite_pinned_cost():
    core = _core()
    missing = TaskSpec(
        task_id="missing",
        model_name="unknown",
        mode="default",
        batch_size=1,
        uses_model_profile=True,
        tensor_offsets=(0,),
        source_worker_ranks=(),
        source_host_ids=(),
        source_sizes_bytes=(),
    )

    assert core.select_and_reserve(missing) is None
    result = core.reserve_on_worker(missing, 0)
    assert math.isinf(result.cost.total)
    assert core.complete("missing") is True
    assert core.snapshot()[0]["queue_latency"] == 0.0


def test_profile_free_task_only_pays_transfer_cost():
    core = _core()
    task = TaskSpec(
        task_id="scheduler-op",
        model_name="scheduler",
        mode="default",
        batch_size=1,
        uses_model_profile=False,
        tensor_offsets=(0, 1),
        source_worker_ranks=(2,),
        source_host_ids=(1,),
        source_sizes_bytes=(512,),
    )
    result = core.reserve_on_worker(task, 0)
    assert result.cost.transfer == pytest.approx(0.01)
    assert result.cost.loading == 0.0
    assert result.cost.execution == 0.0


def test_python_factory_accepts_typed_profiles():
    core = create_scheduling_core(
        worker_host_ids={0: 0},
        active_models={0: []},
        model_profiles=_profiles(),
        intra_profile=TransferProfile((1,), (1.0,)),
        inter_profile=TransferProfile((1,), (1.0,)),
        worker_latency_threshold=0.01,
        backend="python",
    )
    assert core.name == "python"


def test_cpp_backend_matches_python_state_transitions():
    try:
        cpp = _core(CppSchedulingCore)
    except (ImportError, OSError):
        pytest.skip("AOT C++ SchedulingCore is not built")
    python = _core(PythonSchedulingCore)

    operations = [
        _task("first"),
        _task("second", batch_size=2),
        _task("third", source_rank=2, source_host=1),
    ]
    for task in operations:
        python_result = python.select_and_reserve(task)
        cpp_result = cpp.select_and_reserve(task)
        assert cpp_result.worker_rank == python_result.worker_rank
        assert cpp_result.cost.queue == pytest.approx(python_result.cost.queue)
        assert cpp_result.cost.transfer == pytest.approx(python_result.cost.transfer)
        assert cpp_result.cost.loading == pytest.approx(python_result.cost.loading)
        assert cpp_result.cost.execution == pytest.approx(python_result.cost.execution)
        assert cpp_result.cost.total == pytest.approx(python_result.cost.total)

    python.update_active_models(1, ["model"])
    cpp.update_active_models(1, ["model"])
    assert python.complete("first") == cpp.complete("first")

    python_snapshot = python.snapshot()
    cpp_snapshot = cpp.snapshot()
    for worker_rank in python_snapshot:
        assert cpp_snapshot[worker_rank]["queue_latency"] == pytest.approx(
            python_snapshot[worker_rank]["queue_latency"]
        )
        assert cpp_snapshot[worker_rank]["reservations"] == pytest.approx(
            python_snapshot[worker_rank]["reservations"]
        )
