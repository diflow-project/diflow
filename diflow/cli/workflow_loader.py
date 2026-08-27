from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import inspect
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Mapping,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from diflow.operators.utils import default_model_path


class WorkflowLoadError(ValueError):
    """Raised when a workflow specification does not satisfy the CLI contract."""


@dataclass(frozen=True)
class BuiltinWorkflow:
    target: str
    argument_defaults: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LoadedWorkflow:
    name: str
    source: str
    factory: Callable[..., Any]
    argument_defaults: Mapping[str, Any] = field(default_factory=dict)


BUILTIN_WORKFLOWS: Dict[str, BuiltinWorkflow] = {
    "flux-schnell": BuiltinWorkflow(
        target="workflow_hub.flux_schnell.register_txt2img_workflow:create_workflow",
        argument_defaults={"model_path": default_model_path("FLUX.1-schnell")},
    ),
    "flux-dev": BuiltinWorkflow(
        target="workflow_hub.flux_dev.register_txt2img_workflow:create_workflow",
        argument_defaults={"model_path": default_model_path("FLUX.1-dev")},
    ),
    "zimage": BuiltinWorkflow(
        target="workflow_hub.zimage.register_txt2img_workflow:create_workflow",
        argument_defaults={"model_path": default_model_path("Z-Image")},
    ),
    "flux2-klein": BuiltinWorkflow(
        target="workflow_hub.flux2_klein.register_txt2img_workflow:create_workflow",
        argument_defaults={"model_path": default_model_path("Flux.2-klein")},
    ),
}


def _load_module_from_file(path: Path):
    digest = hashlib.sha256(str(path).encode()).hexdigest()[:12]
    module_name = f"diflow_user_workflow_{digest}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise WorkflowLoadError(f"Unable to import workflow file: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise WorkflowLoadError(
            f"Failed to import workflow file {path}: {exc}"
        ) from exc
    return module


def _load_target(target: str):
    module_name, separator, attribute = target.partition(":")
    if not separator:
        raise WorkflowLoadError(f"Invalid built-in workflow target: {target}")
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise WorkflowLoadError(
            f"Failed to import built-in workflow {module_name}: {exc}"
        ) from exc
    return module, attribute


def _factory_from_module(
    module: Any, attribute: str, source: str
) -> Callable[..., Any]:
    factory = getattr(module, attribute, None)
    if not callable(factory):
        raise WorkflowLoadError(
            f"Workflow {source} must export a callable {attribute}(...)."
        )
    return factory


def load_workflow(specification: str) -> LoadedWorkflow:
    """Resolve a built-in workflow name or a Python workflow file."""
    builtin = BUILTIN_WORKFLOWS.get(specification)
    if builtin is not None:
        module, attribute = _load_target(builtin.target)
        return LoadedWorkflow(
            name=specification,
            source=f"built-in workflow '{specification}'",
            factory=_factory_from_module(module, attribute, builtin.target),
            argument_defaults={
                name: value
                for name, value in builtin.argument_defaults.items()
                if value is not None
            },
        )

    looks_like_path = (
        specification.endswith(".py") or "/" in specification or "\\" in specification
    )
    if looks_like_path:
        path = Path(specification).expanduser().resolve()
        if not path.is_file():
            raise WorkflowLoadError(f"Workflow file does not exist: {path}")
        module = _load_module_from_file(path)
        return LoadedWorkflow(
            name=path.stem,
            source=str(path),
            factory=_factory_from_module(module, "create_workflow", str(path)),
        )

    available = ", ".join(sorted(BUILTIN_WORKFLOWS))
    raise WorkflowLoadError(
        f"Unknown workflow '{specification}'. Available built-ins: {available}. "
        "Pass a .py file path for a custom workflow."
    )


def _argument_type(annotation: Any, parameter_name: str) -> type:
    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        members = [
            member for member in get_args(annotation) if member is not type(None)
        ]
        if len(members) != 1:
            raise WorkflowLoadError(
                f"Workflow parameter '{parameter_name}' has unsupported union type "
                f"{annotation!r}."
            )
        annotation = members[0]

    if annotation not in (str, int, float, bool):
        raise WorkflowLoadError(
            f"Workflow parameter '{parameter_name}' has unsupported type "
            f"{annotation!r}; expected str, int, float, bool, or an optional "
            "form of one of them."
        )
    return annotation


def _parse_bool(value: str) -> bool:
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected one of: true, false, yes, no, 1, 0")


def add_workflow_arguments(
    parser: argparse.ArgumentParser, loaded: LoadedWorkflow
) -> tuple[str, ...]:
    """Add CLI options derived from ``create_workflow`` and return their dests."""
    signature = inspect.signature(loaded.factory)
    try:
        type_hints = get_type_hints(loaded.factory)
    except Exception as exc:
        raise WorkflowLoadError(
            f"Unable to resolve type annotations for {loaded.source}: {exc}"
        ) from exc

    destinations = []
    existing_options = parser._option_string_actions
    for parameter in signature.parameters.values():
        if parameter.kind not in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            raise WorkflowLoadError(
                f"Workflow parameter '{parameter.name}' uses unsupported kind "
                f"{parameter.kind.description}."
            )

        annotation = type_hints.get(parameter.name, parameter.annotation)
        if annotation is inspect.Parameter.empty:
            raise WorkflowLoadError(
                f"Workflow parameter '{parameter.name}' must have a type annotation."
            )
        argument_type = _argument_type(annotation, parameter.name)
        option = f"--{parameter.name.replace('_', '-')}"
        if option in existing_options:
            raise WorkflowLoadError(
                f"Workflow parameter '{parameter.name}' conflicts with DiFlow "
                f"option {option}."
            )

        override = loaded.argument_defaults.get(parameter.name, inspect.Parameter.empty)
        default = (
            override if override is not inspect.Parameter.empty else parameter.default
        )
        required = default is inspect.Parameter.empty
        kwargs: Dict[str, Any] = {
            "dest": parameter.name,
            "required": required,
            "help": f"Argument for {loaded.name}.create_workflow().",
        }
        if argument_type is bool and not required:
            kwargs.update(action=argparse.BooleanOptionalAction, default=default)
        else:
            kwargs["type"] = _parse_bool if argument_type is bool else argument_type
            if not required:
                kwargs["default"] = default
        parser.add_argument(option, **kwargs)
        destinations.append(parameter.name)

    return tuple(destinations)


def workflow_kwargs(
    namespace: argparse.Namespace, destinations: tuple[str, ...]
) -> Dict[str, Any]:
    return {name: getattr(namespace, name) for name in destinations}
