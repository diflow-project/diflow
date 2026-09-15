from __future__ import annotations

import hashlib
import inspect
import json
import types
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Mapping,
    Optional,
    Union,
    get_args,
    get_origin,
)

from PIL import Image

from diflow.api.catalog import OPTIONAL_OPERATOR_INPUTS, PUBLIC_OPERATOR_IDS, type_name
from diflow.api.errors import WorkflowAPIError
from diflow.api.schemas import WorkflowInputSpec, WorkflowNodeSpec, WorkflowSpec
from diflow.interface.node_io import NodeIO
from diflow.interface.workflow import Workflow
from diflow.interface.workflow_node import WorkflowNode
from diflow.operators.utils import get_op

PUBLIC_INPUT_TYPES = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "image": Image.Image,
}


@dataclass(frozen=True)
class CompilationResult:
    workflow: Workflow
    digest: str
    source: str
    input_contract: Dict[str, Dict[str, Any]]
    output_contract: Dict[str, Dict[str, Any]]


class WorkflowCompiler:
    """Compile a safe public WorkflowSpec into DiFlow's internal graph."""

    def __init__(
        self,
        model_registry: Mapping[str, str],
        *,
        operator_factory: Callable[[str, Optional[str]], Any] = get_op,
        max_nodes: int = 64,
    ) -> None:
        self.model_registry = dict(model_registry)
        self.operator_factory = operator_factory
        self.max_nodes = max_nodes

    @staticmethod
    def digest(spec: WorkflowSpec) -> str:
        canonical = json.dumps(
            spec.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode()).hexdigest()

    def compile(self, spec: WorkflowSpec) -> CompilationResult:
        if len(spec.nodes) > self.max_nodes:
            raise WorkflowAPIError(
                "WORKFLOW_LIMIT",
                f"Workflow has {len(spec.nodes)} nodes; limit is {self.max_nodes}",
                path="nodes",
            )
        if spec.template is not None:
            workflow = self._compile_template(spec)
            source = f"template:{spec.template.id}"
        else:
            workflow = self._compile_dag(spec)
            source = "dag"
        input_contract = self._input_contract(spec, workflow)
        self._validate_preprocessors(spec, workflow)
        output_contract = self._output_contract(workflow)
        return CompilationResult(
            workflow=workflow,
            digest=self.digest(spec),
            source=source,
            input_contract=input_contract,
            output_contract=output_contract,
        )

    def _model_path(self, model_ref: str, path: str) -> str:
        try:
            return self.model_registry[model_ref]
        except KeyError as exc:
            raise WorkflowAPIError(
                "MODEL_REF_NOT_FOUND",
                f"Server model_ref {model_ref!r} was not found",
                path=path,
                status_code=404,
            ) from exc

    def _operator(self, node: WorkflowNodeSpec):
        if node.operator not in PUBLIC_OPERATOR_IDS:
            raise WorkflowAPIError(
                "OPERATOR_NOT_FOUND",
                f"Operator {node.operator!r} is not in the public catalog",
                path=f"nodes.{node.id}.operator",
                status_code=404,
            )
        model_path = None
        if node.model_ref is not None:
            model_path = self._model_path(node.model_ref, f"nodes.{node.id}.model_ref")
        try:
            operator = self.operator_factory(node.operator, model_path)
        except Exception as exc:
            raise WorkflowAPIError(
                "OPERATOR_UNAVAILABLE",
                f"Cannot instantiate operator {node.operator!r}: {exc}",
                path=f"nodes.{node.id}.operator",
            ) from exc
        if operator.config is not None and node.model_ref is None:
            raise WorkflowAPIError(
                "MODEL_REF_REQUIRED",
                f"Operator {node.operator!r} requires a server model_ref",
                path=f"nodes.{node.id}.model_ref",
            )
        return operator

    def _compile_template(self, spec: WorkflowSpec) -> Workflow:
        from diflow.cli.workflow_loader import BUILTIN_WORKFLOWS, load_workflow

        template = spec.template
        assert template is not None
        if template.id not in BUILTIN_WORKFLOWS:
            raise WorkflowAPIError(
                "TEMPLATE_NOT_FOUND",
                f"Built-in workflow template {template.id!r} was not found",
                path="template.id",
                status_code=404,
            )
        if "model_path" in template.parameters:
            raise WorkflowAPIError(
                "MODEL_PATH_FORBIDDEN",
                "model_path cannot be supplied through WorkflowSpec; use model_ref",
                path="template.parameters.model_path",
            )
        loaded = load_workflow(template.id)
        signature = inspect.signature(loaded.factory)
        unknown = sorted(set(template.parameters) - set(signature.parameters))
        if unknown:
            raise WorkflowAPIError(
                "UNKNOWN_TEMPLATE_PARAMETER",
                f"Unknown template parameters: {', '.join(unknown)}",
                path="template.parameters",
            )
        kwargs: Dict[str, Any] = dict(template.parameters)
        if "model_path" in signature.parameters:
            if template.model_ref is None:
                raise WorkflowAPIError(
                    "MODEL_REF_REQUIRED",
                    f"Template {template.id!r} requires a server model_ref",
                    path="template.model_ref",
                )
            kwargs["model_path"] = self._model_path(
                template.model_ref, "template.model_ref"
            )
        missing = [
            name
            for name, parameter in signature.parameters.items()
            if name not in kwargs and parameter.default is inspect.Parameter.empty
        ]
        if missing:
            raise WorkflowAPIError(
                "MISSING_TEMPLATE_PARAMETER",
                f"Missing template parameters: {', '.join(missing)}",
                path="template.parameters",
            )
        try:
            workflow = loaded.factory(**kwargs)
        except Exception as exc:
            raise WorkflowAPIError(
                "TEMPLATE_BUILD_FAILED",
                f"Failed to build template {template.id!r}: {exc}",
                path="template",
            ) from exc
        if not isinstance(workflow, Workflow):
            raise WorkflowAPIError(
                "TEMPLATE_BUILD_FAILED",
                f"Template returned {type(workflow).__name__}, expected Workflow",
                path="template",
            )
        workflow.name = spec.name
        return workflow

    @staticmethod
    def _topological_nodes(spec: WorkflowSpec) -> list[WorkflowNodeSpec]:
        nodes: Dict[str, WorkflowNodeSpec] = {}
        order: Dict[str, int] = {}
        for index, node in enumerate(spec.nodes):
            if node.id in nodes:
                raise WorkflowAPIError(
                    "DUPLICATE_NODE",
                    f"Node ID {node.id!r} is duplicated",
                    path=f"nodes.{index}.id",
                )
            nodes[node.id] = node
            order[node.id] = index
        dependencies: Dict[str, set[str]] = {node_id: set() for node_id in nodes}
        successors: Dict[str, set[str]] = {node_id: set() for node_id in nodes}
        for node in spec.nodes:
            for input_name, reference in node.inputs.items():
                if reference.node is None:
                    continue
                if reference.node not in nodes:
                    raise WorkflowAPIError(
                        "NODE_NOT_FOUND",
                        f"Referenced node {reference.node!r} was not found",
                        path=f"nodes.{node.id}.inputs.{input_name}.node",
                    )
                dependencies[node.id].add(reference.node)
                successors[reference.node].add(node.id)
        ready = sorted(
            (node_id for node_id, deps in dependencies.items() if not deps),
            key=order.get,
        )
        result = []
        while ready:
            node_id = ready.pop(0)
            result.append(nodes[node_id])
            for successor in sorted(successors[node_id], key=order.get):
                dependencies[successor].discard(node_id)
                if (
                    not dependencies[successor]
                    and successor not in {item.id for item in result}
                    and successor not in ready
                ):
                    ready.append(successor)
                    ready.sort(key=order.get)
        if len(result) != len(nodes):
            cyclic = sorted(node_id for node_id, deps in dependencies.items() if deps)
            raise WorkflowAPIError(
                "WORKFLOW_CYCLE",
                f"Workflow contains a cycle involving: {', '.join(cyclic)}",
                path="nodes",
            )
        return result

    def _compile_dag(self, spec: WorkflowSpec) -> Workflow:
        workflow = Workflow(spec.name)
        inputs = {
            name: workflow.add_input(name, PUBLIC_INPUT_TYPES[input_spec.type])
            for name, input_spec in spec.inputs.items()
        }
        produced: Dict[str, Dict[str, NodeIO]] = {}
        for node in self._topological_nodes(spec):
            operator = self._operator(node)
            modes = operator.get_execution_modes()
            if modes:
                if node.mode not in modes:
                    raise WorkflowAPIError(
                        "INVALID_OPERATOR_MODE",
                        f"Operator {node.operator!r} has no mode {node.mode!r}",
                        path=f"nodes.{node.id}.mode",
                    )
                expected = modes[node.mode]["inputs"]
                required = set(expected)
            else:
                if node.mode != "default":
                    raise WorkflowAPIError(
                        "INVALID_OPERATOR_MODE",
                        f"Operator {node.operator!r} only supports mode 'default'",
                        path=f"nodes.{node.id}.mode",
                    )
                expected = operator.get_inputs()
                required = set(expected) - OPTIONAL_OPERATOR_INPUTS.get(
                    node.operator, set()
                )
            unknown = sorted(set(node.inputs) - set(expected))
            if unknown:
                raise WorkflowAPIError(
                    "UNKNOWN_OPERATOR_INPUT",
                    f"Unknown inputs for {node.operator!r}: {', '.join(unknown)}",
                    path=f"nodes.{node.id}.inputs",
                )
            missing = sorted(required - set(node.inputs))
            if missing:
                raise WorkflowAPIError(
                    "MISSING_OPERATOR_INPUT",
                    f"Missing inputs for {node.operator!r}: {', '.join(missing)}",
                    path=f"nodes.{node.id}.inputs",
                )
            resolved_inputs: Dict[str, NodeIO] = {}
            for port, reference in node.inputs.items():
                if reference.input is not None:
                    try:
                        source = inputs[reference.input]
                    except KeyError as exc:
                        raise WorkflowAPIError(
                            "INPUT_NOT_FOUND",
                            f"Workflow input {reference.input!r} was not found",
                            path=f"nodes.{node.id}.inputs.{port}.input",
                        ) from exc
                else:
                    assert reference.node is not None and reference.output is not None
                    try:
                        source = produced[reference.node][reference.output]
                    except KeyError as exc:
                        raise WorkflowAPIError(
                            "OUTPUT_NOT_FOUND",
                            f"Output {reference.node!r}.{reference.output!r} was not found",
                            path=f"nodes.{node.id}.inputs.{port}",
                        ) from exc
                if not self._types_compatible(
                    source.data_type, expected[port].data_type
                ):
                    raise WorkflowAPIError(
                        "TYPE_MISMATCH",
                        f"Input {port!r} expects {type_name(expected[port].data_type)}, "
                        f"got {type_name(source.data_type)}",
                        path=f"nodes.{node.id}.inputs.{port}",
                    )
                resolved_inputs[port] = source
            workflow_node = WorkflowNode(
                op=operator,
                inputs=resolved_inputs,
                mode=node.mode,
                name=f"{operator.id}_{node.id}",
            )
            workflow.add_workflow_node(workflow_node)
            produced[node.id] = workflow_node.get_outputs()
        for public_name, reference in spec.outputs.items():
            if reference.input is not None:
                raise WorkflowAPIError(
                    "INVALID_WORKFLOW_OUTPUT",
                    "Workflow outputs must reference an operator output",
                    path=f"outputs.{public_name}",
                )
            assert reference.node is not None and reference.output is not None
            try:
                output = produced[reference.node][reference.output]
            except KeyError as exc:
                raise WorkflowAPIError(
                    "OUTPUT_NOT_FOUND",
                    f"Output {reference.node!r}.{reference.output!r} was not found",
                    path=f"outputs.{public_name}",
                ) from exc
            workflow.add_output(output, public_name)
        return workflow

    @staticmethod
    def _types_compatible(source: type, target: type) -> bool:
        if target is Any or source is Any or source == target:
            return True
        target_origin = get_origin(target)
        if target_origin in (Union, types.UnionType):
            return any(
                WorkflowCompiler._types_compatible(source, member)
                for member in get_args(target)
            )
        source_origin = get_origin(source)
        if source_origin in (Union, types.UnionType):
            return all(
                WorkflowCompiler._types_compatible(member, target)
                for member in get_args(source)
            )
        return False

    def _input_contract(
        self, spec: WorkflowSpec, workflow: Workflow
    ) -> Dict[str, Dict[str, Any]]:
        unknown = sorted(set(spec.inputs) - set(workflow.inputs))
        if unknown:
            raise WorkflowAPIError(
                "INPUT_NOT_FOUND",
                f"Declared inputs are not present in the workflow: {', '.join(unknown)}",
                path="inputs",
            )
        contract = {}
        for name, node_io in workflow.inputs.items():
            declared: Optional[WorkflowInputSpec] = spec.inputs.get(name)
            if declared is not None:
                declared_type = PUBLIC_INPUT_TYPES[declared.type]
                if not self._types_compatible(declared_type, node_io.data_type):
                    raise WorkflowAPIError(
                        "TYPE_MISMATCH",
                        f"Workflow input {name!r} is {declared.type}, but the graph "
                        f"expects {type_name(node_io.data_type)}",
                        path=f"inputs.{name}.type",
                    )
                required = declared.required
                default = declared.default
                if not required and not self._value_matches_public_type(
                    declared.type, default
                ):
                    raise WorkflowAPIError(
                        "INVALID_WORKFLOW_INPUT_DEFAULT",
                        f"Default for workflow input {name!r} must be {declared.type}",
                        path=f"inputs.{name}.default",
                    )
            else:
                required = True
                default = None
            contract[name] = {
                "type": type_name(node_io.data_type),
                "required": required,
                "default": default,
            }
        return contract

    @staticmethod
    def _value_matches_public_type(type_name: str, value: Any) -> bool:
        if type_name in ("string", "image"):
            return isinstance(value, str)
        if type_name == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if type_name == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if type_name == "boolean":
            return isinstance(value, bool)
        return False

    @staticmethod
    def _output_contract(workflow: Workflow) -> Dict[str, Dict[str, Any]]:
        all_outputs = {
            output.name: output
            for node in workflow.workflow_nodes
            for output in node.get_outputs().values()
        }
        for region in workflow.regions:
            for program in region.subprograms():
                for node in program.iter_nodes():
                    all_outputs.update(
                        (output.name, output) for output in node.get_outputs().values()
                    )
        return {
            public_name: {
                "type": (
                    type_name(all_outputs[internal_name].data_type)
                    if internal_name in all_outputs
                    else "unknown"
                )
            }
            for internal_name, public_name in workflow.outputs.items()
        }

    @staticmethod
    def _validate_preprocessors(spec: WorkflowSpec, workflow: Workflow) -> None:
        for index, preprocessor in enumerate(spec.preprocessors):
            input_io = workflow.inputs.get(preprocessor.input)
            if input_io is None:
                raise WorkflowAPIError(
                    "INPUT_NOT_FOUND",
                    f"Prompt preprocessor input {preprocessor.input!r} was not found",
                    path=f"preprocessors.{index}.input",
                )
            if not WorkflowCompiler._types_compatible(str, input_io.data_type):
                raise WorkflowAPIError(
                    "TYPE_MISMATCH",
                    "Prompt preprocessors can only target string inputs",
                    path=f"preprocessors.{index}.input",
                )
