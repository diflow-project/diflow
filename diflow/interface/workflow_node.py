import uuid
from typing import Any, Dict

from diflow.interface.node_io import NodeIO, SourceType
from diflow.operators.base import Operator


class WorkflowNode:
    def __init__(
        self,
        op: Operator,
        inputs: Dict[str, NodeIO],
        outputs: Dict[str, NodeIO] = None,
        mode: str = "default",
        name: str = None,
    ):
        self.name = f"{op.id}_{uuid.uuid4()}" if name is None else name
        self.op = op
        self._inputs: Dict[str, NodeIO] = inputs
        self._outputs: Dict[str, NodeIO] = outputs
        if self._outputs is None:
            self._outputs = {}
            execution_modes = self.op.get_execution_modes()

            if not execution_modes:
                # Operator has no specific modes - use default outputs
                for output_name, output_io in op.get_outputs().items():
                    self._outputs[output_name] = NodeIO(
                        name=f"{self.name}:{output_name}",
                        data_type=output_io.data_type,
                        source_type=SourceType.NODE,
                        source_node=self.name,
                        size=output_io.size,
                        lazy=output_io.lazy,
                    )
            else:
                # Use outputs from specified mode
                if mode not in execution_modes:
                    raise ValueError(
                        f"Invalid execution mode '{mode}' for operator {op.id}"
                    )

                mode_outputs = execution_modes[mode]["outputs"]
                for output_name, output_io in mode_outputs.items():
                    self._outputs[output_name] = NodeIO(
                        name=f"{self.name}:{output_name}",
                        data_type=output_io.data_type,
                        source_type=SourceType.NODE,
                        source_node=self.name,
                        size=output_io.size,
                        lazy=output_io.lazy,
                    )

        self.mode = mode

    def __repr__(self) -> str:
        return f"""
        WorkflowNode(
            name={self.name},
            op={self.op},
            mode={self.mode},
            inputs={self.get_inputs()},
            outputs={self.get_outputs()}
        )
        """

    def set_input(self, input_name: str, input_io: NodeIO):
        self._inputs[input_name] = input_io

    def get_inputs(self) -> Dict[str, NodeIO]:
        return self._inputs

    def get_outputs(self) -> Dict[str, NodeIO]:
        return self._outputs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "op": self.op.to_dict(),
            "mode": self.mode,
            "inputs": {
                input_name: input_io.to_dict() if input_io is not None else None
                for input_name, input_io in self.get_inputs().items()
            },
            "outputs": {
                output_name: output_io.to_dict() if output_io is not None else None
                for output_name, output_io in self.get_outputs().items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowNode":
        inputs = {
            input_name: NodeIO.from_dict(input_data)
            for input_name, input_data in data["inputs"].items()
        }
        outputs = {
            output_name: NodeIO.from_dict(output_data)
            for output_name, output_data in data["outputs"].items()
        }

        return cls(
            name=data["name"],
            op=Operator.from_dict(data["op"]),
            mode=data.get("mode", "default"),
            inputs=inputs,
            outputs=outputs,
        )
