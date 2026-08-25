"""Registered workflows that can be profiled, and how to build them.

A case points at an existing `workflow_hub/register_*_workflow.py` factory
rather than redefining the workflow, so profiling measures the graph that gets
registered with the backend.
"""

import importlib
import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

from benchmark_ops.shapes import Shape, ShapeSweep
from diflow.interface.workflow import Workflow

# Inputs the profiler controls itself; a case that sets them is overridden per shape.
SHAPE_CONTROLLED_INPUTS = ("height", "width", "num_inference_steps")


@dataclass(frozen=True)
class WorkflowProfileCase:
    """One registered workflow plus the shape sweep to profile it over."""

    name: str
    suite: str
    # "<path to module>:<factory function>", e.g.
    # "examples/register_sd3_txt2img_workflow.py:create_workflow"
    factory: str
    factory_kwargs: Dict[str, Any] = field(default_factory=dict)
    # Non-shape workflow inputs: prompt, seed, guidance_scale, control_image, ...
    inputs: Dict[str, Any] = field(default_factory=dict)
    sweep: ShapeSweep = field(default_factory=ShapeSweep.default)
    _reference_shape: Optional[Shape] = None

    @property
    def reference_shape(self) -> Shape:
        """Canonical shape recorded in benchmark result metadata."""
        if self._reference_shape is not None:
            return self._reference_shape
        return self.sweep.reference_shape()

    def build_workflow(self) -> Workflow:
        factory = load_workflow_factory(self.factory)
        workflow = factory(**self.factory_kwargs)
        if not isinstance(workflow, Workflow):
            raise TypeError(
                f"Factory {self.factory} for case {self.name} returned "
                f"{type(workflow).__name__}, expected Workflow"
            )
        return workflow


def load_workflow_factory(factory_ref: str) -> Callable[..., Workflow]:
    """Load a `<module path>:<function>` reference.

    Packaged `workflow_hub/` references are imported by module name. Other
    references are loaded from a user-supplied file path.
    """
    if ":" not in factory_ref:
        raise ValueError(
            f"Factory reference {factory_ref!r} must be formatted as "
            f"<module path>:<function name>"
        )

    module_path, function_name = factory_ref.rsplit(":", 1)
    path = Path(module_path)
    packaged_module_name = None
    if path.suffix == ".py" and path.parts[:1] == ("workflow_hub",):
        packaged_module_name = ".".join(path.with_suffix("").parts)

    if packaged_module_name is not None:
        try:
            module = importlib.import_module(packaged_module_name)
        except ModuleNotFoundError as error:
            if error.name == packaged_module_name:
                raise FileNotFoundError(
                    f"Workflow factory module {module_path} does not exist "
                    f"(referenced by {factory_ref!r})"
                ) from error
            raise
    else:
        if not path.is_file():
            raise FileNotFoundError(
                f"Workflow factory module {module_path} does not exist "
                f"(referenced by {factory_ref!r})"
            )

        module_name = f"_benchmark_ops_workflow_{path.stem}"
        if module_name in sys.modules:
            module = sys.modules[module_name]
        else:
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load workflow factory module {module_path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

    if not hasattr(module, function_name):
        raise AttributeError(
            f"{module_path} has no attribute {function_name!r} "
            f"(referenced by {factory_ref!r})"
        )
    return getattr(module, function_name)


def load_cases_from_yaml(config_path: str) -> List[WorkflowProfileCase]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}

    suite = config["suite"]
    default_sweep = config.get("shapes")

    cases = []
    for workflow_config in config.get("workflows", []):
        sweep = ShapeSweep.from_dict(workflow_config.get("shapes") or default_sweep)
        raw_reference_shape = workflow_config.get("reference_shape")
        cases.append(
            WorkflowProfileCase(
                name=workflow_config["name"],
                suite=suite,
                factory=workflow_config["factory"],
                factory_kwargs=dict(workflow_config.get("factory_kwargs") or {}),
                inputs=dict(workflow_config.get("inputs") or {}),
                sweep=sweep,
                _reference_shape=(
                    Shape.from_dict(raw_reference_shape)
                    if raw_reference_shape
                    else None
                ),
            )
        )
    return cases
