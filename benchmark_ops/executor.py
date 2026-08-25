"""Single-process, single-GPU execution of an unrolled workflow.

The backend splits a workflow across workers and moves tensors over NVSHMEM. For
profiling we only want the operator compute time, so this module runs the same
unrolled graph locally and keeps hold of each node's real inputs. Input resolution
mirrors `Coordinator._prepare_task_inputs` and the node-input construction in
`DistributedWorker.process_task`, including the zero-arg callables used for lazy
inputs -- the forked diffusers `stream_forward` invokes those for controlnet block
samples, so passing bare tensors would not exercise the same code path.
"""

import copy
import inspect
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple, Union

import torch
from PIL import Image

from diflow.interface.workflow_node import WorkflowNode
from diflow.operators.base import Operator
from diflow.operators.operator_ids import (
    GUIDANCE_TENSOR_ID,
    INDEXED_TENSOR_ID,
)
from diflow.operators.schedulers.base_scheduler import BaseScheduler

logger = logging.getLogger(__name__)


# Mirrors the `is_batch_processing` exclusions in DistributedWorker.process_task:
# these ops are always executed one request at a time.
NON_BATCHABLE_OP_IDS = (INDEXED_TENSOR_ID, GUIDANCE_TENSOR_ID)


def is_batchable(op: Operator) -> bool:
    return not isinstance(op, BaseScheduler) and op.id not in NON_BATCHABLE_OP_IDS


def is_stateful(op: Operator) -> bool:
    """Schedulers mutate their components (timesteps, step index) when executed."""
    return isinstance(op, BaseScheduler)


@dataclass
class CapturedNode:
    """One node of the unrolled graph plus the real inputs it was executed with."""

    node: WorkflowNode
    # Raw values: tensors are unwrapped even when the input is lazy, so they can be
    # batched before being re-wrapped at call time.
    raw_inputs: Dict[str, Any]
    lazy_input_names: FrozenSet[str] = frozenset()
    # For stateful ops only: the components as they were *before* this node ran, so a
    # measurement can be replayed from the same state any number of times. A scheduler
    # `step` needs the timesteps that its `init` node installed.
    components_snapshot: Optional[Dict[str, Any]] = None

    @property
    def op(self) -> Operator:
        return self.node.op

    def input_shapes(self, batch_size: int = 1) -> Dict[str, List[int]]:
        """Tensor input shapes as they are at `batch_size`.

        Batching concatenates along dim 0, so that is the only dimension that scales.
        """
        shapes = {}
        for name, value in self.raw_inputs.items():
            if not isinstance(value, torch.Tensor):
                continue
            shape = list(value.shape)
            if shape:
                shape[0] *= batch_size
            shapes[name] = shape
        return shapes

    def static_signature(self) -> Tuple[Any, ...]:
        """Identity derivable without executing anything.

        Used to count how often an op runs per request, which is solved from unrolling
        alone. `static_signature_for` computes the same tuple from an un-executed node.
        """
        return _static_signature(self.op, self.node.mode, sorted(self.raw_inputs))

    def signature(self) -> Tuple[Any, ...]:
        """Identity for latency purposes.

        Tensor shapes and dtypes are included; scalar values are not, since an op does
        not get slower because a timestep index changed. This is what collapses a
        28-step denoise loop into a single measurement.
        """
        descriptors = []
        for name, value in sorted(self.raw_inputs.items()):
            if isinstance(value, torch.Tensor):
                descriptors.append((name, tuple(value.shape), str(value.dtype)))
            else:
                descriptors.append((name, type(value).__name__))

        return (self.static_signature(), tuple(descriptors))


def _static_signature(
    op: Operator, mode: str, input_names: Sequence[str]
) -> Tuple[Any, ...]:
    return (
        op.id,
        mode,
        tuple(sorted(patch.id for patch in op.get_patches())),
        tuple(input_names),
    )


def resolvable_input_names(
    workflow_nodes: Sequence[WorkflowNode], inputs: Dict[str, Any]
) -> Set[str]:
    """NodeIO names that will have a value at execution time.

    An input is resolvable when it is a workflow input or some node produces it.
    Anything else is a declared-but-unwired input the executor skips.
    """
    names = set(inputs)
    for node in workflow_nodes:
        for output_io in node.get_outputs().values():
            if output_io is not None:
                names.add(output_io.name)
    return names


def static_signature_for(node: WorkflowNode, resolvable: Set[str]) -> Tuple[Any, ...]:
    """Static signature of a node that has not been executed."""
    input_names = sorted(
        input_name
        for input_name, input_io in node.get_inputs().items()
        if input_io is not None and input_io.name in resolvable
    )
    return _static_signature(node.op, node.mode, input_names)


@dataclass
class CapturedGraph:
    """Deduplicated nodes of one unrolled workflow, with occurrence counts."""

    nodes: Dict[Tuple[Any, ...], CapturedNode] = field(default_factory=dict)
    occurrences: Dict[Tuple[Any, ...], int] = field(default_factory=dict)

    def add(self, captured_node: CapturedNode) -> None:
        signature = captured_node.signature()
        self.occurrences[signature] = self.occurrences.get(signature, 0) + 1
        # Keep the first occurrence only; the rest are structurally identical and
        # holding all of them would pin far more GPU memory than a real request does.
        self.nodes.setdefault(signature, captured_node)

    def __len__(self) -> int:
        return len(self.nodes)


def topological_sort(nodes: Sequence[WorkflowNode]) -> List[WorkflowNode]:
    """Order nodes so every producer runs before its consumers (Kahn's algorithm)."""
    # NodeIO name -> index of the node that produces it.
    producers: Dict[str, int] = {}
    for index, node in enumerate(nodes):
        for output_io in node.get_outputs().values():
            if output_io is not None:
                producers[output_io.name] = index

    dependencies: List[Set[int]] = [set() for _ in nodes]
    dependents: List[Set[int]] = [set() for _ in nodes]
    for index, node in enumerate(nodes):
        for input_io in node.get_inputs().values():
            if input_io is None:
                continue
            producer_index = producers.get(input_io.name)
            if producer_index is None or producer_index == index:
                continue
            dependencies[index].add(producer_index)
            dependents[producer_index].add(index)

    ready = deque(index for index in range(len(nodes)) if not dependencies[index])
    ordered: List[WorkflowNode] = []
    remaining = [len(dependency_set) for dependency_set in dependencies]
    while ready:
        index = ready.popleft()
        ordered.append(nodes[index])
        for dependent_index in sorted(dependents[index]):
            remaining[dependent_index] -= 1
            if remaining[dependent_index] == 0:
                ready.append(dependent_index)

    if len(ordered) != len(nodes):
        unresolved = [
            nodes[index].name for index in range(len(nodes)) if remaining[index] > 0
        ]
        raise ValueError(
            f"Workflow graph has a cycle or unreachable nodes: {unresolved}"
        )
    return ordered


def batch_input_value(value: Any, batch_size: int) -> Any:
    """Replicate one request's input into a batch of `batch_size`.

    Mirrors the batching in `DistributedWorker.process_task`: tensors are
    concatenated along dim 0, scalars are shared, strings and images become lists.
    """
    if batch_size == 1:
        return value
    if isinstance(value, torch.Tensor):
        return torch.cat([value] * batch_size, dim=0)
    if isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, (str, Image.Image)):
        return [value] * batch_size
    raise ValueError(f"Cannot batch input of type {type(value).__name__}")


def build_call_kwargs(
    raw_inputs: Dict[str, Any],
    lazy_input_names: FrozenSet[str],
    batch_size: int = 1,
) -> Dict[str, Any]:
    """Batch the raw inputs and re-wrap lazy ones as zero-arg callables."""
    call_kwargs = {}
    for name, value in raw_inputs.items():
        batched_value = batch_input_value(value, batch_size)
        if name in lazy_input_names:
            call_kwargs[name] = lambda captured=batched_value: captured
        else:
            call_kwargs[name] = batched_value
    return call_kwargs


class LocalWorkflowExecutor:
    """Loads operators on demand and executes an unrolled workflow locally."""

    def __init__(
        self,
        device: Union[str, torch.device] = "cuda",
        offload_idle_models: bool = False,
    ):
        self.device = device
        self.offload_idle_models = offload_idle_models
        # (op_id, model_path) -> model components
        self._components: Dict[Tuple[str, Optional[str]], Dict[str, Any]] = {}
        # Tracks where movable cached components currently reside. Cache membership
        # alone is insufficient because idle models may have been moved back to CPU.
        self._component_devices: Dict[Tuple[str, Optional[str]], str] = {}
        # Stateful components scoped to one graph execution, mirroring the
        # per-request scheduler copies in DistributedWorker._load_scheduler.
        self._run_components: Dict[Tuple[str, Optional[str]], Dict[str, Any]] = {}
        self._model_load_profiles: Dict[Tuple[str, Optional[str]], Dict[str, Any]] = {}

    def _component_key(self, op: Operator) -> Tuple[str, Optional[str]]:
        model_path = op.config.model_path if op.config is not None else None
        return op.id, model_path

    @staticmethod
    def _has_movable_components(components: Dict[str, Any]) -> bool:
        return any(hasattr(component, "to") for component in components.values())

    def _move_components(
        self, key: Tuple[str, Optional[str]], device: Union[str, torch.device]
    ) -> None:
        components = self._components[key]
        for name, component in components.items():
            if not hasattr(component, "to"):
                continue
            moved = component.to(device)
            if moved is not None:
                components[name] = moved
        self._component_devices[key] = str(device)

    def _activate_components(self, key: Tuple[str, Optional[str]]) -> float:
        on_cuda = str(self.device).startswith("cuda")
        if on_cuda:
            torch.cuda.synchronize()
        started = time.perf_counter()
        self._move_components(key, self.device)
        if on_cuda:
            torch.cuda.synchronize()
        return time.perf_counter() - started

    def base_components(self, op: Operator) -> Dict[str, Any]:
        """Load an op once and ensure cached model components are ready to execute."""
        key = self._component_key(op)
        if key not in self._components:
            model_path = key[1]
            logger.info("Initializing %s (model_path=%s) on cpu", op.id, model_path)
            started = time.perf_counter()
            components = op.initialize(model_path, "cpu")
            disk_to_host = time.perf_counter() - started

            self._components[key] = components
            self._component_devices[key] = "cpu"
            self._model_load_profiles[key] = {
                "op_id": op.id,
                "model_path": model_path,
                "disk_to_host_latency_seconds": disk_to_host,
                "host_to_gpu_latency_seconds": 0.0,
                "model_memory_bytes": 0,
            }

        components = self._components[key]
        uses_execution_device = not is_stateful(op) and self._has_movable_components(
            components
        )
        if not uses_execution_device:
            return components

        if self.offload_idle_models:
            self._offload_all_except(key)

        target_device = str(self.device)
        if self._component_devices[key] != target_device:
            on_cuda = target_device.startswith("cuda")
            memory_before = torch.cuda.memory_allocated() if on_cuda else 0
            host_to_gpu = self._activate_components(key)
            model_memory = (
                max(0, int(torch.cuda.memory_allocated() - memory_before))
                if on_cuda
                else 0
            )
            profile = self._model_load_profiles[key]
            # Reactivation happens before operator timing. Keep the worst observed
            # transfer cost as the conservative loading-latency estimate.
            profile["host_to_gpu_latency_seconds"] = max(
                profile["host_to_gpu_latency_seconds"], host_to_gpu
            )
            profile["model_memory_bytes"] = max(
                profile["model_memory_bytes"], model_memory
            )

        return self._components[key]

    def model_load_profiles(self) -> List[Dict[str, Any]]:
        return list(self._model_load_profiles.values())

    def _offload_all_except(self, key: Tuple[str, Optional[str]]) -> None:
        moved_any = False
        for component_key in self._components:
            if component_key == key or self._component_devices[component_key] == "cpu":
                continue
            self._move_components(component_key, "cpu")
            moved_any = True
        if moved_any and str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()

    def run_components(self, op: Operator) -> Dict[str, Any]:
        """Components to execute one graph run with.

        A stateful op keeps a single instance for the whole run: a scheduler's `step`
        nodes depend on the timesteps its `init` node installed, exactly as one request
        shares one scheduler copy across all of its nodes.
        """
        components = self.base_components(op)
        if not is_stateful(op):
            return components

        key = self._component_key(op)
        if key not in self._run_components:
            self._run_components[key] = copy.deepcopy(components)
        return self._run_components[key]

    def components_for_measurement(self, captured_node: CapturedNode) -> Dict[str, Any]:
        """Components to replay one captured node with.

        Stateful ops are restored from the snapshot taken during capture so repeated
        measurements do not advance past the end of the timestep schedule.
        """
        if captured_node.components_snapshot is not None:
            return copy.deepcopy(captured_node.components_snapshot)
        return self.base_components(captured_node.op)

    def run_node(
        self,
        node: WorkflowNode,
        call_kwargs: Dict[str, Any],
        components: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute one node. Generator ops are drained, as the worker drains them."""
        op = node.op
        if inspect.isgeneratorfunction(op.execute):
            outputs: Dict[str, Any] = {}
            for partial_outputs in op.execute(
                model_components=components,
                device=self.device,
                mode=node.mode,
                **call_kwargs,
            ):
                outputs.update(partial_outputs)
            return outputs

        return op.execute(
            model_components=components,
            device=self.device,
            mode=node.mode,
            **call_kwargs,
        )

    def capture(
        self, workflow_nodes: Sequence[WorkflowNode], inputs: Dict[str, Any]
    ) -> CapturedGraph:
        """Execute the unrolled graph at batch size 1, capturing each node's inputs."""
        captured_graph = CapturedGraph()
        # NodeIO name -> value produced by an upstream node.
        produced: Dict[str, Any] = {}
        self._run_components.clear()

        for node in topological_sort(workflow_nodes):
            raw_inputs, lazy_input_names = self._resolve_node_inputs(
                node, inputs, produced
            )
            components = self.run_components(node.op)
            captured_node = CapturedNode(
                node=node,
                raw_inputs=raw_inputs,
                lazy_input_names=lazy_input_names,
                # Snapshot before executing: this node's own execution mutates state.
                components_snapshot=(
                    copy.deepcopy(components) if is_stateful(node.op) else None
                ),
            )
            captured_graph.add(captured_node)

            outputs = self.run_node(
                node,
                build_call_kwargs(raw_inputs, lazy_input_names),
                components,
            )

            for output_name, output_io in node.get_outputs().items():
                if output_io is None:
                    continue
                if output_name not in outputs:
                    raise KeyError(
                        f"Operator {node.op.id} (mode={node.mode}) did not return "
                        f"declared output {output_name!r}"
                    )
                produced[output_io.name] = outputs[output_name]

        return captured_graph

    def _resolve_node_inputs(
        self,
        node: WorkflowNode,
        inputs: Dict[str, Any],
        produced: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], FrozenSet[str]]:
        raw_inputs: Dict[str, Any] = {}
        lazy_input_names: Set[str] = set()

        for input_name, input_io in node.get_inputs().items():
            if input_io is None:
                continue

            if input_io.name in produced:
                value = produced[input_io.name]
            elif input_io.name in inputs:
                value = _coerce_workflow_input(
                    inputs[input_io.name], input_io.data_type
                )
            else:
                # Optional inputs (declared but never wired) are simply not passed,
                # which is what the worker does when an input is missing.
                logger.debug(
                    "Skipping unwired input %s (%s) of node %s",
                    input_name,
                    input_io.name,
                    node.name,
                )
                continue

            raw_inputs[input_name] = value
            if input_io.lazy:
                lazy_input_names.add(input_name)

        return raw_inputs, frozenset(lazy_input_names)

    def cleanup(self) -> None:
        self._components.clear()
        self._component_devices.clear()
        self._run_components.clear()
        if str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()


def _coerce_workflow_input(value: Any, data_type: type) -> Any:
    """Turn a config-friendly workflow input into what the operator expects.

    The worker receives control images as base64 and decodes them in
    `_deserialize_inputs`; a profile config just names a file, so load it directly.
    """
    if data_type is Image.Image and isinstance(value, str):
        from diffusers.utils import load_image

        return load_image(value).convert("RGB")
    return value
