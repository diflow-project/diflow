import pytest

from diflow.profiling.runtime_profile import (
    MissingProfileError,
    RuntimeProfile,
    RuntimeProfileError,
    UnsupportedProfileError,
)


def _payload():
    return {
        "schema_version": 2,
        "gpu_type": "NVIDIA H20",
        "gpu_memory_total": 100_000,
        "gpu_count": 1,
        "model_load_profiles": [
            {
                "op_id": "Model",
                "model_path": "/models/model",
                "disk_to_host_latency_seconds": 1.25,
                "host_to_gpu_latency_seconds": 0.5,
                "model_memory_bytes": 10_000,
            }
        ],
        "ops": [
            {
                "op_id": "Model",
                "mode": "default",
                "shape": {"batch_size": 1, "height": 256, "width": 256},
                "latency": {"median": 0.1},
                "gpu_memory_used": 1_000,
            },
            {
                "op_id": "Model",
                "mode": "default",
                "shape": {"batch_size": 1, "height": 512, "width": 512},
                "error": "OOM",
            },
            {
                "op_id": "Model",
                "mode": "default",
                "shape": {"batch_size": 1, "height": 1024, "width": 1024},
                "latency": {"median": 0.4},
                "gpu_memory_used": 4_000,
            },
            {
                "op_id": "Model",
                "mode": "default",
                "shape": {"batch_size": 2, "height": 256, "width": 256},
                "latency": {"median": 0.18},
                "gpu_memory_used": 2_000,
            },
        ],
    }


def test_exact_shape_returns_latency_and_separate_memory_components():
    profile = RuntimeProfile.from_dict(_payload())

    assert profile.execution_latency("Model", "default", 1, 256, 256) == 0.1
    assert profile.activation_memory("Model", "default", 1, 256, 256) == 1_000
    assert profile.model_memory("Model") == 10_000
    assert profile.loading_latency("Model") == 0.5
    assert profile.peak_total_memory("Model", "default", 1, 256, 256) == 11_000
    assert profile.max_peak_total_memory("Model") == 14_000


def test_missing_resolution_uses_nearest_success_at_the_same_batch():
    profile = RuntimeProfile.from_dict(_payload())

    assert profile.execution_latency("Model", "default", 1, 900, 900) == 0.4


def test_exact_error_rejects_instead_of_falling_back():
    profile = RuntimeProfile.from_dict(_payload())

    with pytest.raises(UnsupportedProfileError, match="OOM"):
        profile.execution("Model", "default", 1, 512, 512)


def test_duplicate_shape_error_is_not_masked_by_a_successful_record():
    payload = _payload()
    payload["ops"].append(
        {
            "op_id": "Model",
            "mode": "default",
            "shape": {"batch_size": 1, "height": 256, "width": 256},
            "error": "OOM",
        }
    )
    profile = RuntimeProfile.from_dict(payload)

    with pytest.raises(UnsupportedProfileError, match="OOM"):
        profile.execution("Model", "default", 1, 256, 256)


def test_capture_error_rejects_resolution_instead_of_using_fallback():
    payload = _payload()
    payload["profile_errors"] = [
        {"stage": "capture", "height": 768, "width": 768, "error": "OOM"}
    ]
    profile = RuntimeProfile.from_dict(payload)

    with pytest.raises(UnsupportedProfileError, match="capture failed: OOM"):
        profile.execution("Model", "default", 1, 768, 768)


def test_fallback_never_crosses_batch_or_mode():
    profile = RuntimeProfile.from_dict(_payload())

    with pytest.raises(MissingProfileError):
        profile.execution("Model", "other", 1, 300, 300)
    with pytest.raises(MissingProfileError):
        profile.execution("Model", "default", 4, 300, 300)


def test_non_power_of_two_batch_rounds_up_to_profiled_batch():
    payload = _payload()
    payload["ops"].append(
        {
            "op_id": "Model",
            "mode": "default",
            "shape": {"batch_size": 4, "height": 256, "width": 256},
            "latency": {"median": 0.3},
            "gpu_memory_used": 3_000,
        }
    )
    profile = RuntimeProfile.from_dict(payload)

    assert profile.execution_latency("Model", "default", 3, 256, 256) == 0.3


def test_schema_v1_cache_is_rejected():
    payload = _payload()
    payload["schema_version"] = 1

    with pytest.raises(RuntimeProfileError, match="expected 2"):
        RuntimeProfile.from_dict(payload)


def test_schema_v2_requires_model_load_data_for_profiled_ops():
    payload = _payload()
    payload["model_load_profiles"] = []

    with pytest.raises(RuntimeProfileError, match="missing model load data"):
        RuntimeProfile.from_dict(payload)


def test_transfer_profiles_are_loaded_from_packaged_data():
    profile = RuntimeProfile.from_dict(_payload())

    assert profile.intra_transfer.block_sizes
    assert profile.inter_transfer.block_sizes
    assert len(profile.intra_transfer.block_sizes) == len(
        profile.intra_transfer.fetch_overheads_us
    )
