import argparse

import pytest

from diflow.cli.workflow_loader import (
    LoadedWorkflow,
    WorkflowLoadError,
    add_workflow_arguments,
    load_workflow,
    workflow_kwargs,
)


def test_python_workflow_signature_becomes_cli_arguments(tmp_path):
    workflow_file = tmp_path / "example_workflow.py"
    workflow_file.write_text("""
from __future__ import annotations

def create_workflow(
    model_path: str,
    steps: int = 4,
    guidance: float | None = None,
    enabled: bool = False,
):
    return model_path, steps, guidance, enabled
""")

    loaded = load_workflow(str(workflow_file))
    parser = argparse.ArgumentParser()
    destinations = add_workflow_arguments(parser, loaded)
    args = parser.parse_args(
        ["--model-path", "/models/flux", "--steps", "8", "--enabled"]
    )

    assert destinations == ("model_path", "steps", "guidance", "enabled")
    assert workflow_kwargs(args, destinations) == {
        "model_path": "/models/flux",
        "steps": 8,
        "guidance": None,
        "enabled": True,
    }


def test_unknown_workflow_lists_builtins():
    with pytest.raises(
        WorkflowLoadError,
        match="flux-dev, flux-schnell, flux2-klein, zimage",
    ):
        load_workflow("does-not-exist")


@pytest.mark.parametrize("name", ["zimage", "flux2-klein"])
def test_new_builtin_workflows_load(name):
    loaded = load_workflow(name)
    assert loaded.name == name
    assert callable(loaded.factory)


def test_missing_factory_has_actionable_error(tmp_path):
    workflow_file = tmp_path / "invalid.py"
    workflow_file.write_text("value = 1\n")

    with pytest.raises(WorkflowLoadError, match="create_workflow"):
        load_workflow(str(workflow_file))


def test_python_workflow_can_define_dataclasses(tmp_path):
    workflow_file = tmp_path / "dataclass_workflow.py"
    workflow_file.write_text("""
from dataclasses import dataclass

@dataclass
class Options:
    model_path: str

def create_workflow(model_path: str):
    return Options(model_path)
""")

    loaded = load_workflow(str(workflow_file))
    assert loaded.factory("/models/flux").model_path == "/models/flux"


def test_unsupported_parameter_type_is_rejected():
    def factory(model_paths: list):
        return model_paths

    loaded = LoadedWorkflow("example", "test", factory)
    with pytest.raises(WorkflowLoadError, match="unsupported"):
        add_workflow_arguments(argparse.ArgumentParser(), loaded)


def test_workflow_option_conflict_is_rejected():
    def factory(port: int):
        return port

    parser = argparse.ArgumentParser()
    parser.add_argument("--port")
    loaded = LoadedWorkflow("example", "test", factory)

    with pytest.raises(WorkflowLoadError, match="conflicts"):
        add_workflow_arguments(parser, loaded)
