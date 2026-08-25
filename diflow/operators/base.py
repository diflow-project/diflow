from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import torch

from diflow.interface.node_io import NodeIO
from diflow.operators.utils import Config

if TYPE_CHECKING:
    from diflow.operators.models.patches.base_patch import BasePatch

logger = logging.getLogger(__name__)


def has_pretrained_weights(model_path: Optional[str], op_id: str) -> bool:
    """Whether ``initialize`` should load weights, or build an empty model.

    Returns False only when the caller deliberately asked for an uninitialised
    model by passing ``model_path=None``, which the memory-profiling entry points
    do -- with a warning, since such a model's outputs are meaningless.

    Raises when a path *was* supplied but does not exist. Operators used to fall
    back to a randomly initialised model there, silently: the pipeline ran end to
    end, every tensor had a plausible shape and magnitude, and the image came out
    as noise. That is indistinguishable from a genuine bug until you compare
    against a reference implementation, so it fails loudly instead.
    """
    if model_path is None:
        logger.warning(
            "%s: no model_path given, building an UNINITIALISED model. Any output "
            "will be meaningless; this is only useful for memory profiling.",
            op_id,
        )
        return False
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"{op_id}: model_path {model_path!r} does not exist. Refusing to fall "
            f"back to a randomly initialised model, which would run fine and "
            f"produce noise that looks like a bug somewhere else. Pass a real "
            f"checkpoint, or model_path=None to build an uninitialised model on "
            f"purpose."
        )
    return True


class Operator(ABC):
    def __init__(self, config: Config = None):
        self.config = config
        self._inputs: Dict[str, NodeIO] = {}
        self._outputs: Dict[str, NodeIO] = {}
        self._patches: List["BasePatch"] = []
        self.setup_io()

    @abstractmethod
    def id(self) -> str:
        pass

    @abstractmethod
    def setup_io(self):
        """Define input and output specifications"""
        pass

    def add_input(
        self,
        name: str,
        data_type: type,
        size: Optional[list[int]] = None,
        lazy: bool = False,
    ):
        self._inputs[name] = NodeIO(
            name=name, data_type=data_type, size=size, lazy=lazy
        )

    def add_output(
        self,
        name: str,
        data_type: type,
        size: Optional[list[int]] = None,
        lazy: bool = False,
    ):
        self._outputs[name] = NodeIO(
            name=name, data_type=data_type, size=size, lazy=lazy
        )

    def add_patch(self, patch: "BasePatch"):
        if patch in self._patches:
            raise ValueError(f"Patch {patch.id} already added to operator {self.id}.")
        self._patches.append(patch)

    def get_inputs(self) -> Dict[str, NodeIO]:
        return self._inputs

    def get_outputs(self) -> Dict[str, NodeIO]:
        return self._outputs

    def get_patches(self) -> List["BasePatch"]:
        return self._patches

    def add_execution_mode(
        self, mode: str, inputs: Dict[str, type], outputs: Dict[str, type]
    ):
        if not hasattr(self, "_execution_modes"):
            self._execution_modes = {}

        mode_spec = {
            "inputs": {name: NodeIO(name, dtype) for name, dtype in inputs.items()},
            "outputs": {name: NodeIO(name, dtype) for name, dtype in outputs.items()},
        }
        self._execution_modes[mode] = mode_spec

    def get_execution_modes(self) -> Dict[str, Dict[str, Dict[str, NodeIO]]]:
        """Get all available execution modes and their IO specifications"""
        return getattr(self, "_execution_modes", {})

    def initialize(
        self, model_path: str, device: Union[str, torch.device]
    ) -> Dict[str, Any]:
        """Default implementation for models that don't require initialization"""
        return {}

    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        pass

    def _check_inputs_are_graph_values(self, inputs: Dict[str, Any]) -> None:
        """Every input must be a ``NodeIO``; a bare value cannot be an edge.

        The executor resolves an input by the ``NodeIO``'s name, looking it up in
        the request inputs or in the tensor map. A literal has no name, and an
        :class:`Expr` is a recipe evaluated while the graph is being expanded
        rather than a value in it -- so neither can be wired to a node.

        Both used to be accepted here and blow up much later, during expansion,
        as ``AttributeError: 'int' object has no attribute 'name'``, with nothing
        pointing at the call that introduced it.
        """
        from diflow.interface.expr import Expr
        from diflow.interface.node_io import NodeIO

        for key, value in inputs.items():
            if isinstance(value, NodeIO):
                continue
            if isinstance(value, Expr):
                raise TypeError(
                    f"{self.id}: input {key!r} is an expression ({value}). "
                    f"Expressions are evaluated while the graph is being expanded, "
                    f"so they can decide a trip count or pick a branch, but they "
                    f"cannot be an input to a node. Pass a request input or an "
                    f"operator's output instead."
                )
            raise TypeError(
                f"{self.id}: input {key!r} is {value!r} ({type(value).__name__}), "
                f"not an operator output or a request input. A literal cannot be a "
                f"graph edge because the executor resolves inputs by name; declare "
                f"it with workflow.add_input(...) and supply it per request."
            )

    def __call__(self, **inputs):
        from diflow.interface.workflow_context import WorkflowContext
        from diflow.interface.workflow_node import WorkflowNode

        workflow = WorkflowContext.get_current_workflow()
        if workflow is None:
            raise RuntimeError("No active workflow context")

        # Remove None values from inputs
        inputs = {k: v for k, v in inputs.items() if v is not None}

        mode = inputs.pop("mode", "default")

        self._check_inputs_are_graph_values(inputs)

        workflow_node = WorkflowNode(op=self, inputs=inputs, mode=mode)

        workflow.add_workflow_node(workflow_node)

        output_list = list(workflow_node.get_outputs().values())
        if len(output_list) == 1:
            return output_list[0]
        return output_list

    def to_dict(self) -> Dict[str, Any]:
        """Serialize operator to dictionary with base_model and patches structure."""
        return {
            "base_model": {
                "model_id": self.id,
                "model_path": (
                    self.config.model_path if self.config is not None else None
                ),
            },
            "patches": [
                {
                    "model_id": patch.id,
                    "model_path": (
                        patch.config.model_path if patch.config is not None else None
                    ),
                }
                for patch in self._patches
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Operator":
        """Reconstruct operator from dictionary with base_model and patches structure."""
        from diflow.operators.utils import get_op

        base_model = data["base_model"]
        model_id = base_model["model_id"]
        model_path = base_model.get("model_path")

        # Create the base operator
        op = get_op(model_id, model_path)

        # Add patches to the operator if they exist
        patches = data.get("patches", [])
        for patch_data in patches:
            patch_id = patch_data["model_id"]
            patch_path = patch_data.get("model_path")
            patch_op = get_op(patch_id, patch_path)
            op.add_patch(patch_op)

        return op
