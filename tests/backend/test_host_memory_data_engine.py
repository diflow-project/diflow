import time
import uuid

import pytest
import torch

import diflow.backend.data_engine.engine as engine_module
from diflow.backend.data_engine.engine import (
    FetchingTask,
    FreeingTask,
    resolve_transfer_backend,
)
from diflow.backend.data_engine.engine.host_memory_data_engine import (
    HostMemoryDataEngine,
)


def _engine(tmp_path, *, worker_id, world_size=2):
    return HostMemoryDataEngine(
        device_id=0,
        worker_id=worker_id,
        world_size=world_size,
        transfer_dir=str(tmp_path),
        session_id=_engine.session_id,
        device="cpu",
    )


_engine.session_id = uuid.uuid4().hex


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int64])
def test_host_backend_transfers_tensor_between_workers(tmp_path, dtype):
    _engine.session_id = uuid.uuid4().hex
    producer = _engine(tmp_path, worker_id=0)
    consumer = _engine(tmp_path, worker_id=1)
    source = torch.arange(24).reshape(4, 6).to(dtype).t()

    with producer, consumer:
        stored = producer.store_tensor(source)
        handle = producer.get_tensor_handle(stored)
        path = handle["path"]
        assert path is not None

        consumer.submit_fetch_task(
            FetchingTask(
                id="tensor",
                tensor_info=handle,
                size=list(stored.size()),
                dtype=stored.dtype,
                remote_worker_rank=0,
            )
        )
        fetched = consumer.get("tensor")

        assert fetched.is_contiguous()
        torch.testing.assert_close(fetched, source)
        assert consumer.get_tensor_handle(fetched) == handle

        consumer.submit_free_task(FreeingTask(fetched))
        producer.submit_free_task(FreeingTask(stored))
        deadline = time.monotonic() + 2
        while (
            time.monotonic() < deadline
            and producer.worker_dir.joinpath(path.rsplit("/", 1)[-1]).exists()
        ):
            time.sleep(0.01)
        assert not producer.worker_dir.joinpath(path.rsplit("/", 1)[-1]).exists()


def test_single_worker_keeps_tensor_on_device_without_shared_file(tmp_path):
    _engine.session_id = uuid.uuid4().hex
    engine = _engine(tmp_path, worker_id=0, world_size=1)
    source = torch.randn(2, 3)

    with engine:
        stored = engine.store_tensor(source)
        assert stored.data_ptr() == source.data_ptr()
        assert engine.get_tensor_handle(stored) == {
            "backend": "host",
            "path": None,
            "owner_rank": 0,
        }


def test_host_backend_rejects_path_outside_session(tmp_path):
    _engine.session_id = uuid.uuid4().hex
    consumer = _engine(tmp_path, worker_id=1)
    outside = tmp_path / "outside.tensor"
    outside.write_bytes(b"\0" * 16)

    with consumer:
        consumer.submit_fetch_task(
            FetchingTask(
                id="outside",
                tensor_info={
                    "backend": "host",
                    "path": str(outside),
                    "owner_rank": 0,
                },
                size=[4],
                dtype=torch.float32,
                remote_worker_rank=0,
            )
        )
        with pytest.raises(RuntimeError, match="Failed to fetch tensor outside"):
            consumer.get("outside")


def test_free_is_idempotent(tmp_path):
    _engine.session_id = uuid.uuid4().hex
    engine = _engine(tmp_path, worker_id=0)

    with engine:
        stored = engine.store_tensor(torch.ones(2))
        engine._free_tensor(stored)
        engine._free_tensor(stored)


def test_auto_backend_uses_host_when_nvshmem_is_unavailable(monkeypatch):
    monkeypatch.setattr(engine_module, "nvshmem_is_available", lambda: False)

    assert resolve_transfer_backend("auto") == "host"
    with pytest.raises(RuntimeError, match="NVSHMEM extension is unavailable"):
        resolve_transfer_backend("nvshmem")


def test_auto_backend_prefers_nvshmem_when_available(monkeypatch):
    monkeypatch.setattr(engine_module, "nvshmem_is_available", lambda: True)

    assert resolve_transfer_backend("auto") == "nvshmem"
