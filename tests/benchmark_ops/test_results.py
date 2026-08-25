import os
import tempfile
import unittest
from unittest.mock import patch

from benchmark_ops.results import (
    OpLatencyRecord,
    build_result,
    get_result_path,
    load_existing_result,
    result_matches_current_gpu,
    save_result,
    summarize_latencies,
)
from benchmark_ops.shapes import Shape

GPU_INFO = ("NVIDIA H20", 102085492736, 8)
REFERENCE_SHAPE = Shape(batch_size=1, height=512, width=512)


def _record(op_id: str, median: float, shape: Shape, mode: str = "default"):
    return OpLatencyRecord(
        op_id=op_id,
        mode=mode,
        shape=shape,
        occurrences={"constant": 1, "per_step": 0},
        latency=summarize_latencies([median]),
    )


def _result(records, reference_shape: Shape = REFERENCE_SHAPE, case_name="case"):
    return build_result(
        case_name=case_name,
        workflow_name="a_workflow",
        suite="test",
        records=records,
        reference_shape=reference_shape,
        warmup=1,
        repeats=3,
        profiled_num_inference_steps=2,
        device="cuda",
        gpu_info=GPU_INFO,
    )


class TestSummarizeLatencies(unittest.TestCase):
    def test_odd_sample_count_takes_the_middle_value(self):
        stats = summarize_latencies([0.3, 0.1, 0.2])

        self.assertAlmostEqual(stats.median, 0.2)
        self.assertAlmostEqual(stats.min, 0.1)
        self.assertAlmostEqual(stats.max, 0.3)
        self.assertEqual(stats.samples, 3)

    def test_even_sample_count_averages_the_middle_two(self):
        stats = summarize_latencies([0.1, 0.2, 0.3, 0.4])

        self.assertAlmostEqual(stats.median, 0.25)

    def test_percentiles_use_nearest_rank(self):
        stats = summarize_latencies([0.1, 0.2, 0.3, 0.4])

        self.assertAlmostEqual(stats.p50, 0.2)
        self.assertAlmostEqual(stats.p99, 0.4)

    def test_empty_samples_are_rejected(self):
        with self.assertRaises(ValueError):
            summarize_latencies([])


class TestOpLatencyRecord(unittest.TestCase):
    def test_optional_fields_are_omitted_when_unset(self):
        record = OpLatencyRecord(op_id="Op", mode="default", shape=REFERENCE_SHAPE)

        serialized = record.to_dict()
        self.assertNotIn("latency", serialized)
        self.assertNotIn("gpu_memory_used", serialized)
        self.assertNotIn("error", serialized)

    def test_error_is_recorded_instead_of_latency(self):
        record = OpLatencyRecord(
            op_id="Op", mode="default", shape=REFERENCE_SHAPE, error="OOM"
        )

        self.assertEqual(record.to_dict()["error"], "OOM")

    def test_occurrences_scale_with_step_count(self):
        record = OpLatencyRecord(
            op_id="Op",
            mode="default",
            shape=REFERENCE_SHAPE,
            occurrences={"constant": 2, "per_step": 1},
        )

        self.assertEqual(record.occurrences_at(28), 30)


class TestProfileErrors(unittest.TestCase):
    def test_capture_errors_are_kept_in_the_result(self):
        profile_error = {
            "stage": "capture",
            "height": 1024,
            "width": 1024,
            "error": "OOM",
        }
        result = build_result(
            case_name="case",
            workflow_name="workflow",
            suite="test",
            records=[],
            reference_shape=REFERENCE_SHAPE,
            warmup=2,
            repeats=5,
            profiled_num_inference_steps=2,
            device="cuda",
            gpu_info=GPU_INFO,
            profile_errors=[profile_error],
        )

        self.assertEqual(result["profile_errors"], [profile_error])


class TestRuntimeSchema(unittest.TestCase):
    def test_result_is_schema_v2_with_model_load_profiles(self):
        load_profile = {
            "op_id": "Op",
            "model_path": "/model",
            "disk_to_host_latency_seconds": 1.0,
            "host_to_gpu_latency_seconds": 0.5,
            "model_memory_bytes": 1024,
        }
        result = build_result(
            case_name="case",
            workflow_name="workflow",
            suite="test",
            records=[_record("Op", 0.1, REFERENCE_SHAPE)],
            reference_shape=REFERENCE_SHAPE,
            warmup=2,
            repeats=5,
            profiled_num_inference_steps=2,
            device="cuda",
            gpu_info=GPU_INFO,
            model_load_profiles=[load_profile],
        )

        self.assertEqual(result["schema_version"], 2)
        self.assertEqual(result["model_load_profiles"], [load_profile])

    def test_gpu_match_rejects_schema_v1(self):
        result = _result([_record("Op", 0.1, REFERENCE_SHAPE)])
        with patch("benchmark_ops.results.get_gpu_info", return_value=GPU_INFO):
            self.assertTrue(result_matches_current_gpu(result))
            result["schema_version"] = 1
            self.assertFalse(result_matches_current_gpu(result))


class TestResultFiles(unittest.TestCase):
    def test_results_are_namespaced_by_normalized_gpu_name(self):
        with tempfile.TemporaryDirectory() as results_dir:
            path = get_result_path("my_case", results_dir, gpu_type="NVIDIA H20")

            self.assertEqual(
                path, os.path.join(results_dir, "nvidia_h20", "my_case.json")
            )

    def test_saved_schema_v2_result_round_trips(self):
        with tempfile.TemporaryDirectory() as results_dir:
            expected = _result(
                [_record("Shared", 0.1, REFERENCE_SHAPE)], case_name="case"
            )
            path = save_result(
                expected,
                "case",
                results_dir,
            )
            loaded = load_existing_result("case", results_dir, gpu_type="NVIDIA H20")

            self.assertTrue(os.path.isfile(path))
            self.assertEqual(loaded, expected)

    def test_invalid_result_file_is_ignored(self):
        with tempfile.TemporaryDirectory() as results_dir:
            gpu_dir = os.path.join(results_dir, "nvidia_h20")
            os.makedirs(gpu_dir)
            with open(os.path.join(gpu_dir, "case.json"), "w") as f:
                f.write("{not json")

            self.assertIsNone(
                load_existing_result("case", results_dir, gpu_type="NVIDIA H20")
            )


if __name__ == "__main__":
    unittest.main()
