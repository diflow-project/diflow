import unittest
from typing import Any, Dict

import torch
from PIL import Image

from benchmark_ops.executor import (
    LocalWorkflowExecutor,
    batch_input_value,
    build_call_kwargs,
    is_batchable,
    resolvable_input_names,
    static_signature_for,
    topological_sort,
)
from diflow.interface.node_io import NodeIO, SourceType
from diflow.interface.workflow_node import WorkflowNode
from diflow.operators.base import Operator
from diflow.operators.custom.indexed_tensor import IndexedTensor
from diflow.operators.schedulers.flux_flow_match_euler_discrete_scheduler import (
    FluxFlowMatchEulerDiscreteScheduler,
)


class DoubleOp(Operator):
    """Doubles `tensor`, and records the inputs it was called with."""

    def __init__(self, op_id: str = "DoubleOp"):
        self._op_id = op_id
        self.calls: list = []
        super().__init__(config=None)

    @property
    def id(self) -> str:
        return self._op_id

    def setup_io(self):
        self.add_input("tensor", torch.Tensor)
        self.add_input("scale", int)
        self.add_output("doubled", torch.Tensor)

    def execute(
        self, model_components: Dict[str, Any], device: Any, **kwargs
    ) -> Dict[str, Any]:
        self.calls.append(kwargs)
        tensor = kwargs["tensor"]
        if callable(tensor):
            tensor = tensor()
        return {"doubled": tensor * 2}


class SourceOp(Operator):
    """Produces a tensor from a scalar workflow input.

    `lazy_output` mirrors the controlnet adapters, which declare their
    `control_block_sample_*` *outputs* lazy -- that producer-side flag is what makes
    the consuming node receive a callable.
    """

    def __init__(self, lazy_output: bool = False):
        self._lazy_output = lazy_output
        super().__init__(config=None)

    @property
    def id(self) -> str:
        return "SourceOp"

    def setup_io(self):
        self.add_input("size", int)
        self.add_output("tensor", torch.Tensor, lazy=self._lazy_output)

    def execute(
        self, model_components: Dict[str, Any], device: Any, **kwargs
    ) -> Dict[str, Any]:
        return {"tensor": torch.ones(1, int(kwargs["size"]))}


class DeviceComponent:
    def __init__(self):
        self.device = "cpu"
        self.moves = []

    def to(self, device):
        self.device = str(device)
        self.moves.append(self.device)
        return self


class DeviceOp(Operator):
    def __init__(self, op_id: str):
        self._op_id = op_id
        self.component = DeviceComponent()
        self.initialize_calls = 0
        super().__init__(config=None)

    @property
    def id(self) -> str:
        return self._op_id

    def setup_io(self):
        pass

    def initialize(self, model_path, device):
        self.initialize_calls += 1
        self.component.to(device)
        return {"model": self.component}

    def execute(self, model_components, device, **kwargs):
        return {}


def _input_io(name: str) -> NodeIO:
    return NodeIO(name=name, data_type=int, source_type=SourceType.INPUT)


def _build_two_node_graph(lazy_tensor: bool = False):
    """SourceOp(size) -> DoubleOp(tensor, scale)."""
    source_op = SourceOp(lazy_output=lazy_tensor)
    source_node = WorkflowNode(op=source_op, inputs={"size": _input_io("size")})

    double_op = DoubleOp()
    double_node = WorkflowNode(
        op=double_op,
        inputs={
            "tensor": source_node.get_outputs()["tensor"],
            "scale": _input_io("scale"),
        },
    )
    return source_node, double_node


class TestTopologicalSort(unittest.TestCase):
    def test_producer_runs_before_consumer_regardless_of_list_order(self):
        source_node, double_node = _build_two_node_graph()

        ordered = topological_sort([double_node, source_node])

        self.assertEqual(
            [node.name for node in ordered],
            [source_node.name, double_node.name],
        )

    def test_raises_on_cycle(self):
        first_op = DoubleOp(op_id="First")
        second_op = DoubleOp(op_id="Second")
        first_node = WorkflowNode(op=first_op, inputs={"scale": _input_io("scale")})
        second_node = WorkflowNode(
            op=second_op,
            inputs={
                "tensor": first_node.get_outputs()["doubled"],
                "scale": _input_io("scale"),
            },
        )
        first_node.set_input("tensor", second_node.get_outputs()["doubled"])

        with self.assertRaises(ValueError):
            topological_sort([first_node, second_node])


class TestBatching(unittest.TestCase):
    def test_tensor_is_concatenated_along_dim_zero(self):
        batched = batch_input_value(torch.ones(1, 4), batch_size=3)

        self.assertEqual(list(batched.shape), [3, 4])

    def test_scalars_are_shared_and_strings_become_lists(self):
        self.assertEqual(batch_input_value(7, 3), 7)
        self.assertEqual(batch_input_value(1.5, 3), 1.5)
        self.assertEqual(batch_input_value("a prompt", 2), ["a prompt", "a prompt"])

    def test_images_become_lists(self):
        image = Image.new("RGB", (8, 8))

        self.assertEqual(batch_input_value(image, 2), [image, image])

    def test_batch_size_one_is_passed_through_untouched(self):
        tensor = torch.ones(1, 4)

        self.assertIs(batch_input_value(tensor, 1), tensor)

    def test_unsupported_type_is_rejected(self):
        with self.assertRaises(ValueError):
            batch_input_value({"unsupported": True}, 2)

    def test_lazy_inputs_are_rewrapped_as_callables_after_batching(self):
        call_kwargs = build_call_kwargs(
            raw_inputs={"tensor": torch.ones(1, 4), "scale": 2},
            lazy_input_names=frozenset({"tensor"}),
            batch_size=3,
        )

        self.assertTrue(callable(call_kwargs["tensor"]))
        self.assertEqual(list(call_kwargs["tensor"]().shape), [3, 4])
        self.assertEqual(call_kwargs["scale"], 2)


class TestIsBatchable(unittest.TestCase):
    def test_schedulers_and_indexed_tensor_are_not_batchable(self):
        self.assertFalse(is_batchable(FluxFlowMatchEulerDiscreteScheduler()))
        self.assertFalse(is_batchable(IndexedTensor()))

    def test_ordinary_ops_are_batchable(self):
        self.assertTrue(is_batchable(DoubleOp()))


class TestComponentResidency(unittest.TestCase):
    def setUp(self):
        self.executor = LocalWorkflowExecutor(
            device="test-device", offload_idle_models=True
        )

    def test_cached_model_is_reactivated_after_another_model_is_loaded(self):
        first = DeviceOp("FirstModel")
        second = DeviceOp("SecondModel")

        self.executor.base_components(first)
        self.executor.base_components(second)
        self.assertEqual(first.component.device, "cpu")
        self.executor.base_components(first)

        self.assertEqual(first.component.device, "test-device")
        self.assertEqual(second.component.device, "cpu")
        self.assertEqual(first.initialize_calls, 1)
        self.assertEqual(second.initialize_calls, 1)
        self.assertEqual(
            first.component.moves,
            ["cpu", "test-device", "cpu", "test-device"],
        )

    def test_scheduler_access_does_not_offload_resident_model(self):
        model = DeviceOp("Model")
        scheduler = FluxFlowMatchEulerDiscreteScheduler()

        self.executor.base_components(model)
        self.executor.base_components(scheduler)

        self.assertEqual(model.component.device, "test-device")
        self.assertEqual(model.component.moves, ["cpu", "test-device"])

    def test_component_free_op_does_not_offload_resident_model(self):
        model = DeviceOp("Model")

        self.executor.base_components(model)
        self.executor.base_components(DoubleOp())

        self.assertEqual(model.component.device, "test-device")
        self.assertEqual(model.component.moves, ["cpu", "test-device"])

    def test_offloading_disabled_keeps_models_resident(self):
        executor = LocalWorkflowExecutor(
            device="test-device", offload_idle_models=False
        )
        first = DeviceOp("FirstModel")
        second = DeviceOp("SecondModel")

        executor.base_components(first)
        executor.base_components(second)

        self.assertEqual(first.component.device, "test-device")
        self.assertEqual(second.component.device, "test-device")


class TestCapture(unittest.TestCase):
    def setUp(self):
        self.executor = LocalWorkflowExecutor(device="cpu")

    def test_captures_inputs_from_workflow_inputs_and_upstream_nodes(self):
        source_node, double_node = _build_two_node_graph()
        inputs = {"size": 4, "scale": 2}

        captured = self.executor.capture([source_node, double_node], inputs)

        self.assertEqual(len(captured), 2)
        double_captured = next(
            node for node in captured.nodes.values() if node.op.id == "DoubleOp"
        )
        self.assertEqual(double_captured.raw_inputs["scale"], 2)
        self.assertEqual(list(double_captured.raw_inputs["tensor"].shape), [1, 4])

    def test_lazy_input_is_passed_to_the_op_as_a_callable(self):
        source_node, double_node = _build_two_node_graph(lazy_tensor=True)

        self.executor.capture([source_node, double_node], {"size": 4, "scale": 2})

        received = double_node.op.calls[0]
        self.assertTrue(callable(received["tensor"]))
        # The raw tensor is kept unwrapped so it can still be batched.
        captured_double = next(
            node
            for node in self.executor.capture(
                [source_node, double_node], {"size": 4, "scale": 2}
            ).nodes.values()
            if node.op.id == "DoubleOp"
        )
        self.assertIsInstance(captured_double.raw_inputs["tensor"], torch.Tensor)
        self.assertEqual(captured_double.lazy_input_names, frozenset({"tensor"}))

    def test_unwired_inputs_are_skipped_rather_than_failing(self):
        double_op = DoubleOp()
        # `scale` is declared by the op but never wired, and nothing produces it.
        node = WorkflowNode(op=double_op, inputs={"tensor": _input_io("tensor")})

        captured = self.executor.capture([node], {"tensor": torch.ones(1, 2)})

        self.assertEqual(list(captured.nodes.values())[0].raw_inputs.keys(), {"tensor"})

    def test_missing_declared_output_is_reported(self):
        class SilentOp(SourceOp):
            def execute(self, model_components, device, **kwargs):
                return {}

        node = WorkflowNode(op=SilentOp(), inputs={"size": _input_io("size")})

        with self.assertRaises(KeyError):
            self.executor.capture([node], {"size": 2})

    def test_identical_nodes_collapse_to_one_measurement(self):
        source_node, first_double = _build_two_node_graph()
        second_double = WorkflowNode(
            op=DoubleOp(),
            inputs={
                "tensor": source_node.get_outputs()["tensor"],
                "scale": _input_io("scale"),
            },
        )

        captured = self.executor.capture(
            [source_node, first_double, second_double], {"size": 4, "scale": 2}
        )

        self.assertEqual(len(captured), 2)
        double_signature = next(
            signature
            for signature, node in captured.nodes.items()
            if node.op.id == "DoubleOp"
        )
        self.assertEqual(captured.occurrences[double_signature], 2)

    def test_different_tensor_shapes_do_not_collapse(self):
        small_source = WorkflowNode(op=SourceOp(), inputs={"size": _input_io("small")})
        large_source = WorkflowNode(op=SourceOp(), inputs={"size": _input_io("large")})
        small_double = WorkflowNode(
            op=DoubleOp(),
            inputs={
                "tensor": small_source.get_outputs()["tensor"],
                "scale": _input_io("scale"),
            },
        )
        large_double = WorkflowNode(
            op=DoubleOp(),
            inputs={
                "tensor": large_source.get_outputs()["tensor"],
                "scale": _input_io("scale"),
            },
        )

        captured = self.executor.capture(
            [small_source, large_source, small_double, large_double],
            {"small": 4, "large": 8, "scale": 2},
        )

        double_nodes = [
            node for node in captured.nodes.values() if node.op.id == "DoubleOp"
        ]
        self.assertEqual(len(double_nodes), 2)

    def test_input_shapes_scale_only_dim_zero(self):
        source_node, double_node = _build_two_node_graph()
        captured = self.executor.capture(
            [source_node, double_node], {"size": 4, "scale": 2}
        )
        double_captured = next(
            node for node in captured.nodes.values() if node.op.id == "DoubleOp"
        )

        self.assertEqual(double_captured.input_shapes(1), {"tensor": [1, 4]})
        self.assertEqual(double_captured.input_shapes(3), {"tensor": [3, 4]})


class TestStaticSignature(unittest.TestCase):
    def test_static_signature_matches_the_captured_one(self):
        source_node, double_node = _build_two_node_graph()
        nodes = [source_node, double_node]
        inputs = {"size": 4, "scale": 2}

        captured = LocalWorkflowExecutor(device="cpu").capture(nodes, inputs)
        resolvable = resolvable_input_names(nodes, inputs)

        static_signatures = {static_signature_for(node, resolvable) for node in nodes}
        captured_signatures = {
            node.static_signature() for node in captured.nodes.values()
        }
        self.assertEqual(static_signatures, captured_signatures)

    def test_unresolvable_inputs_are_excluded_from_the_static_signature(self):
        node = WorkflowNode(op=DoubleOp(), inputs={"tensor": _input_io("tensor")})
        inputs = {"tensor": torch.ones(1, 2)}

        captured = LocalWorkflowExecutor(device="cpu").capture([node], inputs)
        resolvable = resolvable_input_names([node], inputs)

        self.assertEqual(
            static_signature_for(node, resolvable),
            list(captured.nodes.values())[0].static_signature(),
        )


if __name__ == "__main__":
    unittest.main()
