"""Tests for the for_range / cond authoring API.

These cover tracing and validation only -- what gets recorded in the region.
Expansion behaviour is covered in ``test_workflow_expand.py``.
"""

import unittest

import torch

from diflow.interface.control_flow import cond, for_range
from diflow.interface.expr import BinOp, Const, InputRef
from diflow.interface.node_io import NodeIO, SourceType
from diflow.interface.region import CondRegion, LoopRegion
from diflow.interface.workflow import Workflow
from diflow.interface.workflow_context import WorkflowContext
from diflow.operators.custom.indexed_tensor import IndexedTensor


def tensor_io(name: str, source_node: str = "producer") -> NodeIO:
    return NodeIO(
        name=name,
        data_type=torch.Tensor,
        source_type=SourceType.NODE,
        source_node=source_node,
    )


class ControlFlowTestCase(unittest.TestCase):
    """Each test authors into its own workflow and restores the global sink."""

    def setUp(self):
        self._saved = WorkflowContext.get_current_workflow()
        self.workflow = Workflow("test")  # installs itself as the current sink
        self.latents = tensor_io("LatentsGenerator_x:latents", "LatentsGenerator_x")
        self.timesteps = tensor_io("Scheduler_x:timesteps", "Scheduler_x")
        self.steps = self.workflow.add_input("num_inference_steps", int)
        self.guidance = self.workflow.add_input("guidance_scale", float)
        self.index_op = IndexedTensor()

    def tearDown(self):
        WorkflowContext.set_current_workflow(self._saved)

    def step(self, tensor, index):
        """One node, standing in for a denoising step."""
        return self.index_op(tensor=tensor, index=index)


class TestForRange(ControlFlowTestCase):
    def test_records_a_loop_region_and_emits_no_top_level_nodes(self):
        def body(i, carry):
            return {"latents": self.step(carry["latents"], i)}

        results = for_range(self.steps, body, carry={"latents": self.latents})

        self.assertEqual(len(self.workflow.regions), 1)
        self.assertEqual(
            self.workflow.workflow_nodes,
            [],
            "loop body nodes must land in the region, not the top-level graph",
        )
        region = self.workflow.regions[0]
        self.assertIsInstance(region, LoopRegion)
        self.assertEqual(set(results), {"latents"})

    def test_body_is_traced_exactly_once(self):
        calls = []

        def body(i, carry):
            calls.append(i)
            return {"latents": self.step(carry["latents"], i)}

        for_range(self.steps, body, carry={"latents": self.latents})
        self.assertEqual(len(calls), 1)

    def test_body_sees_carry_placeholders_not_the_initial_values(self):
        seen = {}

        def body(i, carry):
            seen.update(carry)
            return {"latents": self.step(carry["latents"], i)}

        for_range(self.steps, body, carry={"latents": self.latents})
        self.assertNotEqual(seen["latents"].name, self.latents.name)
        self.assertEqual(
            seen["latents"].source_node, self.workflow.regions[0].id + "_carry"
        )

    def test_trip_count_from_request_input(self):
        def body(i, carry):
            return {"latents": self.step(carry["latents"], i)}

        for_range(self.steps, body, carry={"latents": self.latents})
        self.assertEqual(
            self.workflow.regions[0].trip_count, InputRef("num_inference_steps")
        )

    def test_trip_count_accepts_a_literal_and_an_expression(self):
        def body(i, carry):
            return {"latents": self.step(carry["latents"], i)}

        for_range(4, body, carry={"latents": self.latents})
        self.assertEqual(self.workflow.regions[0].trip_count, Const(4))

        expr = BinOp("-", InputRef("num_inference_steps"), Const(1))
        for_range(expr, body, carry={"latents": self.latents})
        self.assertEqual(self.workflow.regions[1].trip_count, expr)

    def test_carry_init_and_out_are_recorded(self):
        def body(i, carry):
            return {"latents": self.step(carry["latents"], i)}

        for_range(self.steps, body, carry={"latents": self.latents})
        region = self.workflow.regions[0]
        self.assertEqual(region.carry_init["latents"].name, self.latents.name)
        # carry_out is whatever the body produced last
        self.assertIn(":indexed_tensor", region.carry_out["latents"].name)

    def test_results_are_forward_references_to_the_region(self):
        def body(i, carry):
            return {"latents": self.step(carry["latents"], i)}

        results = for_range(self.steps, body, carry={"latents": self.latents})
        region = self.workflow.regions[0]
        self.assertEqual(results["latents"].name, f"{region.id}:latents")
        self.assertEqual(results["latents"].source_node, region.id)

    def test_nested_loops(self):
        def inner_body(j, carry):
            return {"latents": self.step(carry["latents"], j)}

        def outer_body(i, carry):
            inner = for_range(2, inner_body, carry={"latents": carry["latents"]})
            return {"latents": inner["latents"]}

        for_range(self.steps, outer_body, carry={"latents": self.latents})

        self.assertEqual(len(self.workflow.regions), 1)
        outer = self.workflow.regions[0]
        nested = list(outer.body.iter_regions())
        self.assertEqual(len(nested), 1)
        self.assertIsInstance(nested[0], LoopRegion)

    def test_multiple_carry_keys(self):
        other = tensor_io("Other_x:t", "Other_x")

        def body(i, carry):
            return {
                "latents": self.step(carry["latents"], i),
                "aux": self.step(carry["aux"], i),
            }

        results = for_range(
            self.steps, body, carry={"latents": self.latents, "aux": other}
        )
        self.assertEqual(set(results), {"latents", "aux"})


class TestForRangeValidation(ControlFlowTestCase):
    def test_empty_carry_is_rejected_with_the_reason(self):
        with self.assertRaises(ValueError) as ctx:
            for_range(self.steps, lambda i, c: {}, carry={})
        message = str(ctx.exception)
        self.assertIn("non-empty carry", message)
        self.assertIn("concurrently", message)

    def test_carry_key_mismatch_is_rejected(self):
        def body(i, carry):
            return {"wrong": self.step(carry["latents"], i)}

        with self.assertRaises(ValueError) as ctx:
            for_range(self.steps, body, carry={"latents": self.latents})
        message = str(ctx.exception)
        self.assertIn("latents", message)
        self.assertIn("wrong", message)

    def test_body_returning_none_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            for_range(self.steps, lambda i, c: None, carry={"latents": self.latents})
        self.assertIn("must return the updated carry", str(ctx.exception))

    def test_non_node_io_carry_is_rejected(self):
        with self.assertRaises(ValueError):
            for_range(self.steps, lambda i, c: c, carry={"latents": 5})

    def test_body_returning_non_node_io_is_rejected(self):
        with self.assertRaises(ValueError):
            for_range(
                self.steps,
                lambda i, c: {"latents": "not-an-io"},
                carry={"latents": self.latents},
            )

    def test_node_sourced_trip_count_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            for_range(self.latents, lambda i, c: c, carry={"latents": self.latents})
        self.assertIn("request input", str(ctx.exception))

    def test_iv_template_without_the_index_is_rejected(self):
        """Otherwise every iteration would share one induction variable."""
        with self.assertRaises(ValueError) as ctx:
            for_range(
                self.steps,
                lambda i, c: c,
                carry={"latents": self.latents},
                iv_name_template="{region_id}_iv",
            )
        self.assertIn("{i}", str(ctx.exception))

    def test_escaped_body_output_warns(self):
        def body(i, carry):
            self.step(self.timesteps, i)  # result never used or surfaced
            return {"latents": self.step(carry["latents"], i)}

        with self.assertLogs("diflow.interface.control_flow", "WARNING") as logs:
            for_range(self.steps, body, carry={"latents": self.latents})
        self.assertIn("unreachable", "\n".join(logs.output))

    def test_no_warning_when_everything_is_used(self):
        def body(i, carry):
            timestep = self.step(self.timesteps, i)
            return {"latents": self.step(timestep, i)}

        with self.assertNoLogs("diflow.interface.control_flow", "WARNING"):
            for_range(self.steps, body, carry={"latents": self.latents})


class TestCond(ControlFlowTestCase):
    def test_records_a_cond_region_with_both_branches(self):
        index = self.workflow.add_input("idx", int)

        def then_fn():
            return {"out": self.step(self.latents, index)}

        def else_fn():
            return {"out": self.step(self.timesteps, index)}

        results = cond(BinOp(">", self.guidance, Const(1.0)), then_fn, else_fn)

        self.assertEqual(len(self.workflow.regions), 1)
        region = self.workflow.regions[0]
        self.assertIsInstance(region, CondRegion)
        self.assertIsNotNone(region.else_body)
        self.assertEqual(set(results), {"out"})
        self.assertEqual(self.workflow.workflow_nodes, [])

    def test_both_branches_are_traced(self):
        index = self.workflow.add_input("idx", int)
        traced = []

        def then_fn():
            traced.append("then")
            return {"out": self.step(self.latents, index)}

        def else_fn():
            traced.append("else")
            return {"out": self.step(self.timesteps, index)}

        cond(Const(True), then_fn, else_fn)
        self.assertEqual(sorted(traced), ["else", "then"])

    def test_node_io_predicate_from_request_input(self):
        index = self.workflow.add_input("idx", int)
        flag = self.workflow.add_input("use_cfg", bool)

        cond(
            flag,
            lambda: {"out": self.step(self.latents, index)},
            lambda: {"out": self.step(self.timesteps, index)},
        )
        self.assertEqual(self.workflow.regions[0].predicate, InputRef("use_cfg"))

    def test_side_effect_only_branch_needs_no_else(self):
        index = self.workflow.add_input("idx", int)

        def then_fn():
            self.step(self.latents, index)
            return None

        results = cond(Const(True), then_fn)
        self.assertEqual(results, {})
        self.assertIsNone(self.workflow.regions[0].else_body)

    def test_results_without_an_else_are_rejected(self):
        index = self.workflow.add_input("idx", int)
        with self.assertRaises(ValueError) as ctx:
            cond(Const(True), lambda: {"out": self.step(self.latents, index)})
        self.assertIn("no else branch", str(ctx.exception))

    def test_branch_result_key_mismatch_is_rejected(self):
        index = self.workflow.add_input("idx", int)
        with self.assertRaises(ValueError) as ctx:
            cond(
                Const(True),
                lambda: {"a": self.step(self.latents, index)},
                lambda: {"b": self.step(self.timesteps, index)},
            )
        self.assertIn("same result keys", str(ctx.exception))

    def test_node_sourced_predicate_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            cond(self.latents, lambda: None)
        self.assertIn("request input", str(ctx.exception))


class TestCondInsideForRange(ControlFlowTestCase):
    """The shape the denoise loop actually uses: a CFG branch per iteration."""

    def _build(self):
        def body(i, carry):
            timestep = self.step(self.timesteps, i)

            def with_cfg():
                return {"latents": self.step(timestep, i)}

            def without_cfg():
                return {"latents": self.step(carry["latents"], i)}

            return cond(BinOp(">", self.guidance, Const(1.0)), with_cfg, without_cfg)

        return for_range(self.steps, body, carry={"latents": self.latents})

    def test_cond_is_nested_in_the_loop_body(self):
        self._build()
        loop = self.workflow.regions[0]
        self.assertIsInstance(loop, LoopRegion)

        nested = list(loop.body.iter_regions())
        self.assertEqual(len(nested), 1)
        self.assertIsInstance(nested[0], CondRegion)

    def test_loop_carry_out_is_the_cond_result(self):
        self._build()
        loop = self.workflow.regions[0]
        inner = list(loop.body.iter_regions())[0]
        self.assertEqual(loop.carry_out["latents"].name, inner.results["latents"].name)

    def test_body_ops_are_in_authoring_order(self):
        self._build()
        loop = self.workflow.regions[0]
        kinds = [
            "node" if not isinstance(op, (LoopRegion, CondRegion)) else "region"
            for op in loop.body.ops
        ]
        self.assertEqual(kinds, ["node", "region"])

    def test_branch_capturing_an_enclosing_body_value_is_recorded(self):
        """The then-branch consumes the per-iteration timestep node."""
        self._build()
        loop = self.workflow.regions[0]
        timestep_node = loop.body.ops[0]
        timestep_name = timestep_node.get_outputs()["indexed_tensor"].name

        inner = list(loop.body.iter_regions())[0]
        then_inputs = {
            io.name
            for node in inner.then_body.iter_nodes()
            for io in node.get_inputs().values()
            if io is not None
        }
        self.assertIn(timestep_name, then_inputs)


class TestNoActiveContext(unittest.TestCase):
    def setUp(self):
        self._saved = WorkflowContext.get_current_workflow()
        WorkflowContext.set_current_workflow(None)

    def tearDown(self):
        WorkflowContext.set_current_workflow(self._saved)

    def test_for_range_requires_a_workflow(self):
        io = tensor_io("a:b")
        with self.assertRaises(RuntimeError):
            for_range(1, lambda i, c: c, carry={"latents": io})

    def test_cond_requires_a_workflow(self):
        with self.assertRaises(RuntimeError):
            cond(Const(True), lambda: None)


if __name__ == "__main__":
    unittest.main()


class TestTensorSubscript(ControlFlowTestCase):
    """``timesteps[i]`` emits the indexing node, so callers never name the operator."""

    def test_subscript_emits_a_node_and_returns_its_output(self):
        picked = self.timesteps[self.steps]

        self.assertEqual(len(self.workflow.workflow_nodes), 1)
        node = self.workflow.workflow_nodes[0]
        self.assertEqual(node.op.id, "IndexedTensor")
        self.assertEqual(node.get_inputs()["tensor"].name, self.timesteps.name)
        self.assertEqual(node.get_inputs()["index"].name, self.steps.name)
        self.assertEqual(picked.name, node.get_outputs()["indexed_tensor"].name)

    def test_subscript_with_a_literal_is_refused(self):
        """A literal cannot be a graph edge: the executor resolves inputs by name.

        This used to be accepted and then failed during expansion as
        ``AttributeError: 'int' object has no attribute 'name'``, with nothing
        pointing back at the subscript that caused it.
        """
        with self.assertRaises(TypeError) as ctx:
            self.timesteps[2]
        message = str(ctx.exception)
        self.assertIn("index", message)
        self.assertIn("add_input", message)

    def test_subscript_with_an_expression_is_refused(self):
        """``timesteps[i + offset]`` reads naturally but cannot work.

        The index becomes an Expr, which is resolved while expanding rather than
        being a value in the graph. It used to slip through and crash later inside
        the escaped-output check.
        """
        offset = self.workflow.add_input("offset", int)
        with self.assertRaises(TypeError) as ctx:
            self.timesteps[self.steps + offset]
        self.assertIn("expression", str(ctx.exception))

    def test_subscript_inside_a_loop_body_lands_in_the_region(self):
        def body(i, carry):
            return {"latents": self.step(self.timesteps[i], i)}

        for_range(self.steps, body, carry={"latents": self.latents})

        self.assertEqual(
            self.workflow.workflow_nodes, [], "must land in the region, not the graph"
        )
        op_ids = [n.op.id for n in self.workflow.regions[0].body.iter_nodes()]
        self.assertEqual(op_ids.count("IndexedTensor"), 2)

    def test_subscript_without_a_workflow_is_refused(self):
        WorkflowContext.set_current_workflow(None)
        with self.assertRaises(RuntimeError):
            self.timesteps[0]

    def test_all_iterations_share_the_same_source_tensor(self):
        """The scheduler fuses per-step lookups by comparing the tensor they read,
        so the capture must not be rewritten per iteration."""

        def body(i, carry):
            return {"latents": self.step(self.timesteps[i], i)}

        for_range(self.steps, body, carry={"latents": self.latents})
        index_nodes = [
            n
            for n in self.workflow.regions[0].body.iter_nodes()
            if n.get_inputs().get("tensor") is not None
            and n.get_inputs()["tensor"].name == self.timesteps.name
        ]
        self.assertEqual(len(index_nodes), 1)
