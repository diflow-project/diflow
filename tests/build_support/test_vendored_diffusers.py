import zipfile

import pytest

from build_support.vendored_diffusers import rewrite_private_imports
from tools.build_release_wheel import audit_wheel, write_release_index


def test_rewrite_private_imports_relocates_static_and_dynamic_imports():
    source = """\
from diffusers import schedulers
from diffusers.utils import load_image
import diffusers
import diffusers as library
module = importlib.import_module("diffusers")
child = importlib.import_module(f"diffusers.{name}")
"""

    rewritten = rewrite_private_imports(source)

    assert "from diflow._vendor.diffusers import schedulers" in rewritten
    assert "from diflow._vendor.diffusers.utils import load_image" in rewritten
    assert "from diflow._vendor import diffusers\n" in rewritten
    assert "from diflow._vendor import diffusers as library" in rewritten
    assert 'importlib.import_module("diflow._vendor.diffusers")' in rewritten
    assert 'importlib.import_module(f"diflow._vendor.diffusers.{name}")' in rewritten


def _write_fake_wheel(
    path,
    *,
    forbidden=False,
    external_dependency=False,
    absolute_import=False,
    include_nvshmem=True,
    hard_nvshmem_dependency=False,
):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("benchmark_ops/__init__.py", "")
        archive.writestr("diflow/_vendor/diffusers/__init__.py", "")
        archive.writestr("diflow/_vendor/diffusers/LICENSE", "Apache-2.0")
        archive.writestr("diflow/_vendor/diffusers/_diflow_vendor.py", "")
        archive.writestr(
            "diflow/backend/scheduler/_scheduling_core.cpython-310-x86_64-linux-gnu.so",
            b"extension",
        )
        if include_nvshmem:
            archive.writestr(
                "diflow/backend/data_engine/"
                "_data_engine.cpython-310-x86_64-linux-gnu.so",
                b"extension",
            )
        archive.writestr("workflow_hub/__init__.py", "")
        archive.writestr(
            "diflow-0.1.0.dist-info/METADATA",
            "Metadata-Version: 2.4\nName: diflow\nVersion: 0.1.0\n"
            "Provides-Extra: nvshmem\n"
            "Requires-Dist: nvidia-nvshmem-cu12==3.7.1; "
            'sys_platform == "linux" and extra == "nvshmem"\n'
            + ("Requires-Dist: diffusers>=0.40\n" if external_dependency else "")
            + (
                "Requires-Dist: nvidia-nvshmem-cu12==3.7.1\n"
                if hard_nvshmem_dependency
                else ""
            ),
        )
        if forbidden:
            archive.writestr("diffusers/__init__.py", "")
        if absolute_import:
            archive.writestr(
                "benchmark_ops/executor.py",
                "from diffusers.utils import load_image\n",
            )


def test_audit_wheel_accepts_private_vendor_and_native_core(tmp_path):
    wheel = tmp_path / "diflow-0.1.0-cp310-cp310-linux_x86_64.whl"
    _write_fake_wheel(wheel)

    audit_wheel(wheel)


def test_audit_wheel_rejects_top_level_diffusers(tmp_path):
    wheel = tmp_path / "diflow-0.1.0-cp310-cp310-linux_x86_64.whl"
    _write_fake_wheel(wheel, forbidden=True)

    with pytest.raises(RuntimeError, match="forbidden paths"):
        audit_wheel(wheel)


def test_audit_wheel_rejects_missing_nvshmem_extension(tmp_path):
    wheel = tmp_path / "diflow-0.1.0-cp310-cp310-linux_x86_64.whl"
    _write_fake_wheel(wheel, include_nvshmem=False)

    with pytest.raises(
        RuntimeError, match="missing the native NVSHMEM data-engine extension"
    ):
        audit_wheel(wheel)


def test_audit_wheel_rejects_external_diffusers_dependency(tmp_path):
    wheel = tmp_path / "diflow-0.1.0-cp310-cp310-linux_x86_64.whl"
    _write_fake_wheel(wheel, external_dependency=True)

    with pytest.raises(RuntimeError, match="must not depend"):
        audit_wheel(wheel)


def test_audit_wheel_rejects_hard_nvshmem_dependency(tmp_path):
    wheel = tmp_path / "diflow-0.1.0-cp310-cp310-linux_x86_64.whl"
    _write_fake_wheel(wheel, hard_nvshmem_dependency=True)

    with pytest.raises(RuntimeError, match="NVSHMEM runtime must remain an optional"):
        audit_wheel(wheel)


def test_audit_wheel_rejects_absolute_diffusers_import(tmp_path):
    wheel = tmp_path / "diflow-0.1.0-cp310-cp310-linux_x86_64.whl"
    _write_fake_wheel(wheel, absolute_import=True)

    with pytest.raises(RuntimeError, match="contains absolute Diffusers imports"):
        audit_wheel(wheel)


def test_release_index_contains_hash_pinned_github_links(tmp_path):
    wheel = tmp_path / "diflow-0.1.0-cp310-cp310-linux_x86_64.whl"
    wheel.write_bytes(b"wheel")

    index = write_release_index(tmp_path, "v0.1.0", [wheel])

    contents = index.read_text()
    assert "github.com/diflow-project/diflow/releases/download/v0.1.0" in contents
    assert "#sha256=" in contents
    assert (tmp_path / "SHA256SUMS").read_text().endswith(f"  {wheel.name}\n")
