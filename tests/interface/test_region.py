"""Serialization and structure tests for control-flow regions.

The JSON round-trip is the important one: a workflow is registered over HTTP, so
a field omitted from ``to_dict`` would leave the server expanding an incomplete
region.
"""

import json
import unittest

import torch

from diflow.interface.expr import BinOp, Const, InputRef
from diflow.interface.node_io import NodeIO, SourceType
from diflow.interface.region import (
    CondRegion,
    LoopRegion,
    Region,
    RegionBuilder,
    RegionProgram,
    make_carry_placeholder,
    make_induction_var,
    make_region_result,
)
from diflow.interface.workflow import Workflow
from diflow.interface.workflow_context import WorkflowContext, workflow_context
from diflow.interface.workflow_node import WorkflowNode
from diflow.operators.custom.indexed_tensor import IndexedTensor


def tensor_io(name: str, source_node: str = "producer") -> NodeIO:
    return NodeIO(
        name=name,
        data_type=torch.Tensor,
        source_type=SourceType.NODE,
        source_node=source_node,
    )


def make_node(tensor: NodeIO, index: NodeIO) -> WorkflowNode:
    """A real WorkflowNode; constructing one needs no workflow context."""
    return WorkflowNode(op=IndexedTensor(), inputs={"tensor": tensor, "index": index})


def simple_loop(region_id: str = "for_test") -> LoopRegion:
    latents = tensor_io("LatentsGenerator_x:latents", "LatentsGenerator_x")
    timesteps = tensor_io("Scheduler_x:timesteps", "Scheduler_x")

    induction_var = make_induction_var(region_id)
    carry_placeholder = make_carry_placeholder(region_id, "latents", latents)
    node = make_node(timesteps, induction_var)
    carry_out = node.get_outputs()["indexed_tensor"]

    return LoopRegion(
        id=region_id,
        trip_count=InputRef("num_inference_steps"),
        induction_var=induction_var,
        carry_placeholders={"latents": carry_placeholder},
        carry_init={"latents": latents},
        carry_out={"latents": carry_out},
        body=RegionProgram(ops=[node]),
        results={"latents": make_region_result(region_id, "latents", latents)},
    )


def simple_cond(region_id: str = "cond_test", with_else: bool = True) -> CondRegion:
    latents = tensor_io("LatentsGenerator_x:latents", "LatentsGenerator_x")
    index = NodeIO(name="idx", data_type=int, source_type=SourceType.INPUT)

    then_node = make_node(latents, index)
    then_results = {"out": then_node.get_outputs()["indexed_tensor"]}

    else_body = None
    else_results = None
    if with_else:
        else_node = make_node(latents, index)
        else_body = RegionProgram(ops=[else_node])
        else_results = {"out": else_node.get_outputs()["indexed_tensor"]}

    return CondRegion(
        id=region_id,
        predicate=BinOp(">", InputRef("guidance_scale"), Const(1.0)),
        then_body=RegionProgram(ops=[then_node]),
        then_results=then_results,
        results={"out": make_region_result(region_id, "out", latents)},
        else_body=else_body,
        else_results=else_results,
    )


class TestPlaceholders(unittest.TestCase):
    def test_induction_var_is_an_input(self):
        iv = make_induction_var("for_1")
        self.assertEqual(iv.source_type, SourceType.INPUT)
        self.assertEqual(iv.data_type, int)

    def test_carry_placeholder_is_node_sourced_so_leaks_are_detectable(self):
        like = tensor_io("a:b")
        ph = make_carry_placeholder("for_1", "latents", like)
        self.assertEqual(ph.source_type, SourceType.NODE)
        self.assertEqual(ph.source_node, "for_1_carry")
        self.assertEqual(ph.data_type, torch.Tensor)

    def test_region_result_matches_the_legacy_forward_reference_shape(self):
        """Same shape as DenoiseNode.denoised_latents, so the downstream patch
        is the same name-equality sweep."""
        like = tensor_io("a:b")
        result = make_region_result("Model_Sched_uuid", "denoised_latents", like)
        self.assertEqual(result.name, "Model_Sched_uuid:denoised_latents")
        self.assertEqual(result.source_node, "Model_Sched_uuid")
        self.assertEqual(result.source_type, SourceType.NODE)


class TestLoopRegionSerialization(unittest.TestCase):
    def test_round_trip_through_json(self):
        region = simple_loop()
        revived = Region.from_dict(json.loads(json.dumps(region.to_dict())))

        self.assertIsInstance(revived, LoopRegion)
        self.assertEqual(revived.id, region.id)
        self.assertEqual(revived.trip_count, region.trip_count)
        self.assertEqual(revived.induction_var.name, region.induction_var.name)
        self.assertEqual(revived.iv_name_template, region.iv_name_template)
        self.assertEqual(
            {k: v.name for k, v in revived.carry_init.items()},
            {k: v.name for k, v in region.carry_init.items()},
        )
        self.assertEqual(
            {k: v.name for k, v in revived.carry_out.items()},
            {k: v.name for k, v in region.carry_out.items()},
        )
        self.assertEqual(
            {k: v.name for k, v in revived.results.items()},
            {k: v.name for k, v in region.results.items()},
        )

    def test_body_nodes_survive_with_names_and_ops(self):
        region = simple_loop()
        revived = Region.from_dict(json.loads(json.dumps(region.to_dict())))

        original = list(region.body.iter_nodes())
        restored = list(revived.body.iter_nodes())
        self.assertEqual(len(restored), len(original))
        self.assertEqual(restored[0].name, original[0].name)
        self.assertEqual(restored[0].op.id, original[0].op.id)
        self.assertEqual(set(restored[0].get_inputs()), set(original[0].get_inputs()))

    def test_iv_name_template_is_applied(self):
        region = simple_loop("for_abc")
        self.assertEqual(region.iv_name("3"), "for_abc_iv_3")
        region.iv_name_template = "{region_id}_timestep_{i}"
        self.assertEqual(region.iv_name(2), "for_abc_timestep_2")

    def test_placeholder_names(self):
        region = simple_loop("for_abc")
        self.assertEqual(
            region.placeholder_names(),
            {"for_abc_iv", "for_abc_carry:latents", "for_abc:latents"},
        )

    def test_mismatched_carry_keys_are_rejected_on_parse(self):
        payload = simple_loop().to_dict()
        payload["carry_out"] = {}
        with self.assertRaises(ValueError) as ctx:
            Region.from_dict(payload)
        self.assertIn("carry_out", str(ctx.exception))

    def test_missing_field_is_rejected(self):
        for field in ("id", "trip_count", "induction_var", "body", "results"):
            payload = simple_loop().to_dict()
            del payload[field]
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    Region.from_dict(payload)


class TestCondRegionSerialization(unittest.TestCase):
    def test_round_trip_with_else(self):
        region = simple_cond()
        revived = Region.from_dict(json.loads(json.dumps(region.to_dict())))

        self.assertIsInstance(revived, CondRegion)
        self.assertEqual(revived.predicate, region.predicate)
        self.assertIsNotNone(revived.else_body)
        self.assertEqual(set(revived.then_results), {"out"})
        self.assertEqual(set(revived.else_results), {"out"})

    def test_round_trip_without_else(self):
        region = CondRegion(
            id="cond_bare",
            predicate=Const(True),
            then_body=RegionProgram(ops=[]),
            then_results={},
            results={},
        )
        revived = Region.from_dict(json.loads(json.dumps(region.to_dict())))
        self.assertIsNone(revived.else_body)
        self.assertEqual(revived.results, {})

    def test_results_without_an_else_branch_are_rejected(self):
        """A result the false path cannot produce would dangle."""
        payload = simple_cond(with_else=False).to_dict()
        payload["else"] = None
        with self.assertRaises(ValueError) as ctx:
            Region.from_dict(payload)
        self.assertIn("no", str(ctx.exception).lower())

    def test_branch_result_key_mismatch_is_rejected(self):
        payload = simple_cond().to_dict()
        payload["else"]["results"] = {}
        with self.assertRaises(ValueError) as ctx:
            Region.from_dict(payload)
        self.assertIn("else", str(ctx.exception))


class TestNestedRegions(unittest.TestCase):
    def _nested(self):
        inner = simple_cond("cond_inner")
        outer = simple_loop("for_outer")
        outer.body.ops.append(inner)
        return outer, inner

    def test_round_trip(self):
        outer, inner = self._nested()
        revived = Region.from_dict(json.loads(json.dumps(outer.to_dict())))

        nested = list(revived.body.iter_regions())
        self.assertEqual(len(nested), 1)
        self.assertIsInstance(nested[0], CondRegion)
        self.assertEqual(nested[0].id, inner.id)

    def test_iter_nodes_descends_into_nested_bodies(self):
        outer, _ = self._nested()
        # 1 loop-body node + then-branch node + else-branch node
        self.assertEqual(len(list(outer.body.iter_nodes())), 3)

    def test_placeholder_names_include_nested(self):
        outer, _ = self._nested()
        names = outer.placeholder_names()
        self.assertIn("for_outer_iv", names)
        self.assertIn("cond_inner:out", names)


class TestRegionProgram(unittest.TestCase):
    def test_unknown_body_item_kind_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            RegionProgram.from_dict([{"op_kind": "subprocess", "node": {}}])
        self.assertIn("unknown body item kind", str(ctx.exception))

    def test_unknown_region_kind_is_rejected(self):
        with self.assertRaises(ValueError):
            Region.from_dict({"region_kind": "while", "id": "x"})

    def test_non_dict_region_is_rejected(self):
        with self.assertRaises(ValueError):
            Region.from_dict("loop")


class TestRegionBuilder(unittest.TestCase):
    def test_collects_nodes_and_regions_in_authoring_order(self):
        builder = RegionBuilder()
        first = make_node(tensor_io("a:b"), make_induction_var("r"))
        region = simple_cond()
        second = make_node(tensor_io("c:d"), make_induction_var("r"))

        builder.add_workflow_node(first)
        builder.add_region(region)
        builder.add_workflow_node(second)

        self.assertEqual(builder.program.ops, [first, region, second])

    def test_satisfies_the_sink_protocol_used_by_operator_call(self):
        builder = RegionBuilder()
        self.assertTrue(hasattr(builder, "add_workflow_node"))
        self.assertTrue(hasattr(builder, "add_region"))


class TestWorkflowContextScoping(unittest.TestCase):
    def setUp(self):
        self._saved = WorkflowContext.get_current_workflow()

    def tearDown(self):
        WorkflowContext.set_current_workflow(self._saved)

    def test_restores_the_previous_sink(self):
        outer = RegionBuilder()
        WorkflowContext.set_current_workflow(outer)

        inner = RegionBuilder()
        with workflow_context(inner):
            self.assertIs(WorkflowContext.get_current_workflow(), inner)
        self.assertIs(WorkflowContext.get_current_workflow(), outer)

    def test_restores_on_exception(self):
        outer = RegionBuilder()
        WorkflowContext.set_current_workflow(outer)

        with self.assertRaises(RuntimeError):
            with workflow_context(RegionBuilder()):
                raise RuntimeError("boom")
        self.assertIs(WorkflowContext.get_current_workflow(), outer)

    def test_nests(self):
        a, b, c = RegionBuilder(), RegionBuilder(), RegionBuilder()
        WorkflowContext.set_current_workflow(a)
        with workflow_context(b):
            with workflow_context(c):
                self.assertIs(WorkflowContext.get_current_workflow(), c)
            self.assertIs(WorkflowContext.get_current_workflow(), b)
        self.assertIs(WorkflowContext.get_current_workflow(), a)


class TestWorkflowWithRegions(unittest.TestCase):
    """Regions must survive the registration round-trip carried by Workflow."""

    def _workflow_with_region(self):
        workflow = Workflow("region_round_trip")
        seed = workflow.add_input("seed", int)

        # A plain top-level node, so we cover both containers.
        latents = tensor_io("LatentsGenerator_x:latents", "LatentsGenerator_x")
        top_node = make_node(latents, seed)
        workflow.add_workflow_node(top_node)

        region = simple_loop("for_wf")
        # Give the body a genuine request input alongside the induction variable,
        # so we can tell the two apart after parsing.
        real_input = NodeIO(
            name="strength", data_type=float, source_type=SourceType.INPUT
        )
        region.body.ops.append(make_node(latents, real_input))
        workflow.add_region(region)

        workflow.add_output(region.results["latents"], "image")
        return workflow

    def test_regions_survive_json_round_trip(self):
        workflow = self._workflow_with_region()
        revived = Workflow.from_dict(json.loads(workflow.to_json()))

        self.assertEqual(len(revived.regions), 1)
        self.assertEqual(len(revived.workflow_nodes), 1)
        loop = revived.regions[0]
        self.assertIsInstance(loop, LoopRegion)
        self.assertEqual(loop.id, "for_wf")
        self.assertEqual(len(list(loop.body.iter_nodes())), 2)

    def test_body_request_inputs_are_declared_but_placeholders_are_not(self):
        workflow = self._workflow_with_region()
        revived = Workflow.from_dict(json.loads(workflow.to_json()))

        self.assertIn("strength", revived.inputs)
        self.assertIn("seed", revived.inputs)
        self.assertNotIn("for_wf_iv", revived.inputs)

    def test_output_produced_inside_a_region_is_preserved(self):
        """The old from_dict rebuilt outputs by scanning node outputs, which
        cannot see inside a region body."""
        workflow = self._workflow_with_region()
        revived = Workflow.from_dict(json.loads(workflow.to_json()))

        self.assertEqual(revived.outputs, {"for_wf:latents": "image"})

    def test_workflow_without_regions_still_round_trips(self):
        workflow = Workflow("plain")
        seed = workflow.add_input("seed", int)
        node = make_node(tensor_io("a:b"), seed)
        workflow.add_workflow_node(node)
        workflow.add_output(node.get_outputs()["indexed_tensor"], "out")

        revived = Workflow.from_dict(json.loads(workflow.to_json()))
        self.assertEqual(revived.regions, [])
        self.assertEqual(len(revived.workflow_nodes), 1)
        self.assertEqual(list(revived.outputs.values()), ["out"])


if __name__ == "__main__":
    unittest.main()
