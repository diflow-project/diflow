import unittest

from diflow.backend.scheduler.scheduling_core import (
    CppSchedulingCore,
    ModelProfile,
    PythonSchedulingCore,
    TaskSpec,
    TransferProfile,
    lookup_execution_latency,
    max_profiled_batch_size,
)
from diflow.profiling.runtime_profile import RuntimeProfile


def _profile():
    return ModelProfile(
        loading_latency=0.5,
        execution_latencies={
            ("default", 1, 256, 256): 0.1,
            ("default", 1, 1024, 1024): 0.4,
            ("default", 2, 256, 256): 0.18,
        },
    )


def _core(backend):
    return backend(
        worker_host_ids={0: 0},
        active_models={0: ["model"]},
        model_profiles={"model": _profile()},
        intra_profile=TransferProfile((1,), (1.0,)),
        inter_profile=TransferProfile((1,), (1.0,)),
        worker_latency_threshold=1.0,
    )


def _task(task_id, height, width):
    return TaskSpec(
        task_id=task_id,
        model_name="model",
        mode="default",
        batch_size=1,
        uses_model_profile=True,
        tensor_offsets=(0,),
        source_worker_ranks=(),
        source_host_ids=(),
        source_sizes_bytes=(),
        height=height,
        width=width,
    )


class ShapeProfileTest(unittest.TestCase):
    def test_exact_and_nearest_resolution_lookup(self):
        profile = _profile()
        self.assertEqual(lookup_execution_latency(profile, "default", 1, 256, 256), 0.1)
        self.assertEqual(lookup_execution_latency(profile, "default", 1, 900, 900), 0.4)

        shaped_only = ModelProfile(
            loading_latency=0.0,
            execution_latencies={
                ("default", 1, 256, 256): 0.1,
                ("default", 1, 1024, 1024): 0.4,
            },
        )
        self.assertEqual(
            lookup_execution_latency(shaped_only, "default", 1, 900, 900), 0.4
        )

    def test_batch_limit_uses_successful_shapes(self):
        self.assertEqual(max_profiled_batch_size(_profile(), "default", 256, 256), 2)
        self.assertEqual(max_profiled_batch_size(_profile(), "default", 1024, 1024), 1)

    def test_runtime_profile_projects_only_successful_records(self):
        runtime_profile = RuntimeProfile.from_dict(
            {
                "schema_version": 2,
                "gpu_type": "NVIDIA H20",
                "gpu_memory_total": 100,
                "gpu_count": 1,
                "model_load_profiles": [
                    {
                        "op_id": "model",
                        "model_path": "/m",
                        "disk_to_host_latency_seconds": 1.25,
                        "host_to_gpu_latency_seconds": 0.5,
                        "model_memory_bytes": 10,
                    }
                ],
                "ops": [
                    {
                        "op_id": "model",
                        "mode": "default",
                        "shape": {"batch_size": 1, "height": 256, "width": 256},
                        "latency": {"median": 0.1},
                        "gpu_memory_used": 5,
                    },
                    {
                        "op_id": "model",
                        "mode": "default",
                        "shape": {"batch_size": 8, "height": 256, "width": 256},
                        "error": "OOM",
                    },
                ],
            }
        )
        profiles = runtime_profile.to_scheduling_profiles()

        self.assertEqual(profiles["model"].loading_latency, 0.5)
        self.assertEqual(
            profiles["model"].execution_latencies,
            {("default", 1, 256, 256): 0.1},
        )

    def test_cpp_and_python_use_same_shape(self):
        python = _core(PythonSchedulingCore)
        try:
            cpp = _core(CppSchedulingCore)
        except (ImportError, OSError):
            self.skipTest("AOT C++ SchedulingCore is not built")
        for index, resolution in enumerate(((256, 256), (1024, 1024))):
            task = _task(f"task-{index}", *resolution)
            python_result = python.reserve_on_worker(task, 0)
            cpp_result = cpp.reserve_on_worker(task, 0)
            self.assertAlmostEqual(
                cpp_result.cost.execution, python_result.cost.execution
            )


if __name__ == "__main__":
    unittest.main()
