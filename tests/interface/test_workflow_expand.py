"""Tests for expanding control-flow regions into a flat graph."""

import json
import unittest

import torch

from diflow.interface.control_flow import cond, for_range
from diflow.interface.expr import BinOp, Const, InputRef
from diflow.interface.node_io import NodeIO, SourceType
from diflow.interface.workflow import Workflow
from diflow.interface.workflow_context import WorkflowContext
from diflow.interface.workflow_expand import (
    Scope,
    expand_workflow,
    expand_workflow_pure,
)
from diflow.interface.workflow_node import WorkflowNode
from diflow.operators.custom.indexed_tensor import IndexedTensor


def tensor_io(name: str, source_node: str = "producer") -> NodeIO:
    return NodeIO(
        name=name,
        data_type=torch.Tensor,
        source_type=SourceType.NODE,
        source_node=source_node,
    )


class ExpandTestCase(unittest.TestCase):
    def setUp(self):
        self._saved = WorkflowContext.get_current_workflow()
        self.workflow = Workflow("expand_test")
        self.steps = self.workflow.add_input("num_inference_steps", int)
        self.guidance = self.workflow.add_input("guidance_scale", float)
        self.seed = self.workflow.add_input("seed", int)
        self.base = self.workflow.add_input("base_tensor", torch.Tensor)
        self.index_op = IndexedTensor()
        # Real nodes outside any region, so loops have something to capture and
        # every edge has a producer. `source` stands in for the initial latents,
        # `captured` for a loop-invariant tensor like the timestep schedule.
        self.source = self.index_op(tensor=self.base, index=self.seed)
        self.captured = self.index_op(tensor=self.base, index=self.seed)
        # Nodes present before any region is added, so counts below stay correct
        # if the fixture grows.
        self.base_nodes = len(self.workflow.workflow_nodes)

    def tearDown(self):
        WorkflowContext.set_current_workflow(self._saved)

    def step(self, tensor, index):
        return self.index_op(tensor=tensor, index=index)

    def expand(self, **request):
        inputs = {
            "num_inference_steps": 3,
            "guidance_scale": 7.5,
            "seed": 0,
            "base_tensor": None,
        }
        inputs.update(request)
        graph, injected = expand_workflow_pure(self.workflow, inputs)
        return graph, injected, inputs

    def producers(self, graph):
        return {
            io.name: node
            for node in graph.workflow_nodes
            for io in node.get_outputs().values()
        }

    def consumers_of(self, graph):
        consumers = {}
        for node in graph.workflow_nodes:
            for io in node.get_inputs().values():
                if io is not None:
                    consumers.setdefault(io.name, []).append(node)
        return consumers

    def chain_from(self, graph, start_name):
        """Follow the single-consumer chain forward from a tensor name."""
        consumers = self.consumers_of(graph)
        chain = []
        current = start_name
        while current in consumers:
            node = consumers[current][0]
            chain.append(node)
            outs = list(node.get_outputs().values())
            if len(outs) != 1:
                break
            current = outs[0].name
        return chain

    def longest_chain_from(self, graph, start_name):
        """Length of the longest forward path from a tensor name.

        Needed where a value has several consumers (e.g. a loop body reads the
        carry from more than one node), so following the first consumer is not
        necessarily following the carry.
        """
        consumers = self.consumers_of(graph)
        memo = {}

        def depth(tensor_name):
            if tensor_name in memo:
                return memo[tensor_name]
            memo[tensor_name] = 0  # guards against revisiting
            best = 0
            for node in consumers.get(tensor_name, []):
                for io in node.get_outputs().values():
                    best = max(best, 1 + depth(io.name))
            memo[tensor_name] = best
            return best

        return depth(start_name)


class TestLoopExpansion(ExpandTestCase):
    def test_emits_one_body_copy_per_iteration(self):
        def body(i, carry):
            return {"latents": self.step(carry["latents"], i)}

        for_range(self.steps, body, carry={"latents": self.source})
        graph, _, _ = self.expand(num_inference_steps=3)

        self.assertEqual(len(graph.workflow_nodes), self.base_nodes + 3)

    def test_trip_count_scales_linearly(self):
        def body(i, carry):
            return {"latents": self.step(carry["latents"], i)}

        for_range(self.steps, body, carry={"latents": self.source})
        for steps in (1, 2, 5, 9):
            graph, _, _ = self.expand(num_inference_steps=steps)
            with self.subTest(steps=steps):
                self.assertEqual(len(graph.workflow_nodes), self.base_nodes + steps)

    def test_clones_get_distinct_names_prefixed_by_the_op_id(self):
        def body(i, carry):
            return {"latents": self.step(carry["latents"], i)}

        for_range(self.steps, body, carry={"latents": self.source})
        graph, _, _ = self.expand(num_inference_steps=4)

        names = [n.name for n in graph.workflow_nodes]
        self.assertEqual(len(names), len(set(names)))
        for node in graph.workflow_nodes:
            self.assertTrue(node.name.startswith(node.op.id))

    def test_carry_forms_a_chain_across_iterations(self):
        def body(i, carry):
            return {"latents": self.step(carry["latents"], i)}

        results = for_range(self.steps, body, carry={"latents": self.source})
        graph, _, _ = self.expand(num_inference_steps=4)

        chain = self.chain_from(graph, self.source.name)
        self.assertEqual(len(chain), 4, "each iteration should consume the previous")
        # And the last link is what the region result resolves to.
        last_output = list(chain[-1].get_outputs().values())[0]
        producers = self.producers(graph)
        self.assertIn(last_output.name, producers)

    def test_induction_variable_is_injected_once_per_iteration(self):
        def body(i, carry):
            return {"latents": self.step(carry["latents"], i)}

        for_range(self.steps, body, carry={"latents": self.source})
        _, injected, _ = self.expand(num_inference_steps=5)

        self.assertEqual(sorted(injected.values()), [0, 1, 2, 3, 4])
        self.assertEqual(len(set(injected)), 5, "names must be distinct per iteration")

    def test_body_nodes_read_the_injected_induction_variable(self):
        def body(i, carry):
            return {"latents": self.step(carry["latents"], i)}

        for_range(self.steps, body, carry={"latents": self.source})
        graph, injected, _ = self.expand(num_inference_steps=3)

        index_inputs = {
            node.get_inputs()["index"].name
            for node in graph.workflow_nodes
            if node.get_inputs().get("index") is not None
            and node.get_inputs()["index"].source_type == SourceType.INPUT
        }
        self.assertTrue(set(injected).issubset(index_inputs))

    def test_expand_workflow_injects_into_the_caller_dict(self):
        def body(i, carry):
            return {"latents": self.step(carry["latents"], i)}

        for_range(self.steps, body, carry={"latents": self.source})
        inputs = {
            "num_inference_steps": 3,
            "guidance_scale": 1.0,
            "seed": 0,
            "base_tensor": None,
        }
        expand_workflow(self.workflow, inputs)
        self.assertEqual(len(inputs), 7, "three induction values were added")

    def test_zero_iterations_aliases_the_initial_carry(self):
        """The old unroller left a dangling reference here."""
        results = for_range(
            self.steps,
            lambda i, c: {"latents": self.step(c["latents"], i)},
            carry={"latents": self.source},
        )
        consumer = self.step(results["latents"], self.seed)
        graph, _, _ = self.expand(num_inference_steps=0)

        # Only the pre-existing nodes and the downstream consumer survive.
        self.assertEqual(len(graph.workflow_nodes), self.base_nodes + 1)
        downstream = next(
            n for n in graph.workflow_nodes if n.name == consumer.source_node
        )
        self.assertEqual(downstream.get_inputs()["tensor"].name, self.source.name)

    def test_negative_trip_count_is_rejected(self):
        for_range(
            BinOp("-", self.steps, Const(10)),
            lambda i, c: {"latents": self.step(c["latents"], i)},
            carry={"latents": self.source},
        )
        with self.assertRaises(ValueError) as ctx:
            self.expand(num_inference_steps=1)
        self.assertIn("negative", str(ctx.exception))

    def test_non_integer_trip_count_is_rejected(self):
        for_range(
            InputRef("guidance_scale"),
            lambda i, c: {"latents": self.step(c["latents"], i)},
            carry={"latents": self.source},
        )
        with self.assertRaises(ValueError) as ctx:
            self.expand(guidance_scale=7.5)
        self.assertIn("must evaluate to an int", str(ctx.exception))

    def test_missing_trip_count_input_names_it(self):
        for_range(
            InputRef("absent_input"),
            lambda i, c: {"latents": self.step(c["latents"], i)},
            carry={"latents": self.source},
        )
        with self.assertRaises(KeyError) as ctx:
            self.expand()
        self.assertIn("absent_input", str(ctx.exception))


class TestExternalCaptures(ExpandTestCase):
    def test_captured_value_is_not_remapped_per_iteration(self):
        """All iterations must keep referring to the same external tensor.

        The scheduler groups the per-step index lookups by comparing the tensor
        name they read, so rewriting the capture would break that batching.
        """
        external = self.step(self.base, self.seed)

        def body(i, carry):
            picked = self.step(external, i)
            return {"latents": self.step(picked, i)}

        for_range(self.steps, body, carry={"latents": self.source})
        graph, _, _ = self.expand(num_inference_steps=4)

        readers = [
            node
            for node in graph.workflow_nodes
            if node.get_inputs().get("tensor") is not None
            and node.get_inputs()["tensor"].name == external.name
        ]
        self.assertEqual(len(readers), 4)
        tensor_names = {n.get_inputs()["tensor"].name for n in readers}
        self.assertEqual(len(tensor_names), 1)


class TestCondExpansion(ExpandTestCase):
    def _build(self):
        def then_fn():
            return {"out": self.step(self.source, self.seed)}

        def else_fn():
            return {"out": self.step(self.step(self.source, self.seed), self.seed)}

        results = cond(BinOp(">", self.guidance, Const(1.0)), then_fn, else_fn)
        self.consumer = self.step(results["out"], self.seed)
        return results

    def test_only_the_taken_branch_contributes_nodes(self):
        self._build()

        taken, _, _ = self.expand(guidance_scale=7.5)
        # sources + then-branch node + consumer
        self.assertEqual(len(taken.workflow_nodes), self.base_nodes + 2)

        not_taken, _, _ = self.expand(guidance_scale=1.0)
        # sources + two else-branch nodes + consumer
        self.assertEqual(len(not_taken.workflow_nodes), self.base_nodes + 3)

    def test_cond_emits_no_node_of_its_own(self):
        self._build()
        graph, _, _ = self.expand(guidance_scale=7.5)
        for node in graph.workflow_nodes:
            self.assertEqual(node.op.id, IndexedTensor().id)

    def test_downstream_consumer_is_wired_to_the_taken_branch(self):
        self._build()
        graph, _, _ = self.expand(guidance_scale=7.5)

        producers = self.producers(graph)
        consumer = next(
            n for n in graph.workflow_nodes if n.name == self.consumer.source_node
        )
        upstream_name = consumer.get_inputs()["tensor"].name
        self.assertIn(upstream_name, producers, "must point at a real node")

    def test_side_effect_only_cond_without_else(self):
        def then_fn():
            self.step(self.source, self.seed)  # emitted for its side effect only
            return None

        cond(BinOp(">", self.guidance, Const(1.0)), then_fn)

        taken, _, _ = self.expand(guidance_scale=7.5)
        self.assertEqual(len(taken.workflow_nodes), self.base_nodes + 1)

        skipped, _, _ = self.expand(guidance_scale=1.0)
        self.assertEqual(len(skipped.workflow_nodes), self.base_nodes)


class TestCondInsideLoop(ExpandTestCase):
    """The denoise shape: a per-iteration branch capturing per-iteration values."""

    def _build(self):
        """Mirrors the real denoise body: a per-iteration timestep, then a
        branch that consumes both that timestep and the loop-carried latents."""

        def body(i, carry):
            timestep = self.step(self.captured, i)

            def with_cfg():
                noise = self.step(timestep, i)  # an extra "model" pass
                return {"latents": self.step(carry["latents"], noise)}

            def without_cfg():
                return {"latents": self.step(carry["latents"], timestep)}

            return cond(BinOp(">", self.guidance, Const(1.0)), with_cfg, without_cfg)

        return for_range(self.steps, body, carry={"latents": self.source})

    def test_node_counts_per_branch(self):
        self._build()
        cfg, _, _ = self.expand(num_inference_steps=3, guidance_scale=7.5)
        # per iteration: timestep + noise + step
        self.assertEqual(len(cfg.workflow_nodes), self.base_nodes + 3 * 3)

        plain, _, _ = self.expand(num_inference_steps=3, guidance_scale=1.0)
        # per iteration: timestep + step
        self.assertEqual(len(plain.workflow_nodes), self.base_nodes + 3 * 2)

    def test_branch_captures_the_current_iterations_timestep(self):
        """The scope chain must resolve per-iteration captures freshly.

        A flat substitution map would wire every branch copy to one iteration's
        timestep node.
        """
        self._build()
        graph, _, _ = self.expand(num_inference_steps=3, guidance_scale=7.5)

        # Timestep nodes are the ones reading the captured loop-invariant tensor.
        timestep_nodes = [
            n
            for n in graph.workflow_nodes
            if n.get_inputs().get("tensor") is not None
            and n.get_inputs()["tensor"].name == self.captured.name
        ]
        self.assertEqual(len(timestep_nodes), 3)

        timestep_outputs = {
            list(n.get_outputs().values())[0].name for n in timestep_nodes
        }
        # Each timestep output must be consumed by exactly one branch copy.
        consumers_of_timesteps = [
            n
            for n in graph.workflow_nodes
            if n.get_inputs().get("tensor") is not None
            and n.get_inputs()["tensor"].name in timestep_outputs
        ]
        self.assertEqual(len(consumers_of_timesteps), 3)
        consumed = {n.get_inputs()["tensor"].name for n in consumers_of_timesteps}
        self.assertEqual(
            consumed,
            timestep_outputs,
            "each iteration's branch must read its own timestep, not a shared one",
        )

    def test_carry_serializes_iterations_through_either_branch(self):
        """Graph depth must grow with the step count.

        This is the property that keeps a stateful scheduler correct: iterations
        are ordered only because each consumes the previous one's output. If the
        carry were mis-wired, depth would stay flat and the executor would be free
        to run all iterations at once.
        """
        self._build()
        for guidance in (1.0, 7.5):
            depths = [
                self.longest_chain_from(
                    self.expand(num_inference_steps=steps, guidance_scale=guidance)[0],
                    self.source.name,
                )
                for steps in (1, 2, 3)
            ]
            with self.subTest(guidance=guidance):
                delta = depths[1] - depths[0]
                self.assertGreater(delta, 0, "depth must grow with the step count")
                self.assertEqual(depths[2] - depths[1], delta, "and grow linearly")


class TestSequentialLoops(ExpandTestCase):
    def test_second_loop_consumes_the_first_loops_result(self):
        first = for_range(
            self.steps,
            lambda i, c: {"latents": self.step(c["latents"], i)},
            carry={"latents": self.source},
        )
        second = for_range(
            Const(2),
            lambda i, c: {"latents": self.step(c["latents"], i)},
            carry={"latents": first["latents"]},
        )
        self.workflow.add_output(second["latents"], "image")

        graph, _, _ = self.expand(num_inference_steps=3)
        self.assertEqual(len(graph.workflow_nodes), self.base_nodes + 3 + 2)

        # The whole thing is one chain: source -> 3 -> 2
        self.assertEqual(self.longest_chain_from(graph, self.source.name), 5)

    def test_region_output_mapping_is_remapped(self):
        results = for_range(
            self.steps,
            lambda i, c: {"latents": self.step(c["latents"], i)},
            carry={"latents": self.source},
        )
        self.workflow.add_output(results["latents"], "image")

        graph, _, _ = self.expand(num_inference_steps=3)
        self.assertEqual(list(graph.outputs.values()), ["image"])
        output_key = list(graph.outputs)[0]
        self.assertNotIn(":latents", output_key.replace(":indexed_tensor", ""))
        self.assertIn(output_key, self.producers(graph))


class TestJsonRoundTrip(ExpandTestCase):
    """What the server actually does: parse, then expand."""

    def test_expansion_after_a_round_trip_matches_in_process_expansion(self):
        def body(i, carry):
            timestep = self.step(self.source, i)

            def with_cfg():
                return {"latents": self.step(timestep, i)}

            def without_cfg():
                return {"latents": self.step(carry["latents"], i)}

            return cond(BinOp(">", self.guidance, Const(1.0)), with_cfg, without_cfg)

        results = for_range(self.steps, body, carry={"latents": self.source})
        self.workflow.add_output(results["latents"], "image")

        revived = Workflow.from_dict(json.loads(self.workflow.to_json()))

        for steps in (0, 1, 3):
            for guidance in (1.0, 7.5):
                inputs = {
                    "num_inference_steps": steps,
                    "guidance_scale": guidance,
                    "seed": 0,
                    "base_tensor": None,
                }
                direct, direct_injected = expand_workflow_pure(
                    self.workflow, dict(inputs)
                )
                parsed, parsed_injected = expand_workflow_pure(revived, dict(inputs))
                with self.subTest(steps=steps, guidance=guidance):
                    self.assertEqual(
                        len(direct.workflow_nodes), len(parsed.workflow_nodes)
                    )
                    self.assertEqual(
                        sorted(direct_injected.values()),
                        sorted(parsed_injected.values()),
                    )
                    self.assertEqual(
                        [n.op.id for n in direct.workflow_nodes],
                        [n.op.id for n in parsed.workflow_nodes],
                    )
                    self.assertEqual(
                        list(direct.outputs.values()), list(parsed.outputs.values())
                    )


class TestInvariantsRaise(ExpandTestCase):
    """Deliberately corrupt a region; expansion must refuse it."""

    def _loop_workflow(self):
        results = for_range(
            self.steps,
            lambda i, c: {"latents": self.step(c["latents"], i)},
            carry={"latents": self.source},
        )
        self.workflow.add_output(results["latents"], "image")
        return self.workflow.regions[0]

    def test_carry_passthrough_is_legitimate_not_a_leak(self):
        """``return carry`` unchanged must expand, not trip the leak check."""
        region = self._loop_workflow()
        region.carry_out = {"latents": region.carry_placeholders["latents"]}

        graph, _, _ = self.expand(num_inference_steps=2)
        # The body still emits its nodes, but the carry never advances, so the
        # downstream output resolves back to the loop's input.
        self.assertEqual(list(graph.outputs), [self.source.name])

    def test_unsubstituted_region_result_is_caught(self):
        """A cond declaring results with no else branch cannot bind them.

        The authoring API and the parser both reject this, so it is only
        reachable by constructing the region directly -- which is exactly what the
        defensive check is for.
        """
        index = self.seed

        def then_fn():
            return {"out": self.step(self.source, index)}

        # Build a well-formed cond, then strip the else branch behind its back.
        results = cond(
            BinOp(">", self.guidance, Const(1.0)),
            then_fn,
            lambda: {"out": self.step(self.source, index)},
        )
        self.step(results["out"], index)  # a downstream consumer
        region = self.workflow.regions[-1]
        region.else_body = None
        region.else_results = None

        with self.assertRaises(ValueError) as ctx:
            self.expand(guidance_scale=1.0)  # take the now-missing else branch
        self.assertIn("placeholder", str(ctx.exception))

    def test_body_input_with_no_producer_is_caught(self):
        region = self._loop_workflow()
        orphan = tensor_io("Ghost_x:out", "Ghost_x")
        list(region.body.iter_nodes())[0].set_input("tensor", orphan)
        with self.assertRaises(ValueError) as ctx:
            self.expand(num_inference_steps=2)
        self.assertIn("no node", str(ctx.exception))

    def test_unsupplied_request_input_is_caught(self):
        region = self._loop_workflow()
        missing = NodeIO(
            name="never_supplied", data_type=int, source_type=SourceType.INPUT
        )
        list(region.body.iter_nodes())[0].set_input("index", missing)
        with self.assertRaises(ValueError) as ctx:
            self.expand(num_inference_steps=2)
        self.assertIn("never_supplied", str(ctx.exception))

    def test_output_without_a_producer_is_caught(self):
        self._loop_workflow()
        self.workflow.outputs["Nonexistent_x:out"] = "extra"
        with self.assertRaises(ValueError) as ctx:
            self.expand(num_inference_steps=2)
        self.assertIn("no producer", str(ctx.exception))

    def test_duplicate_node_names_are_caught(self):
        """Two top-level nodes sharing a name collide in the coordinator's maps."""
        clone = WorkflowNode(
            op=IndexedTensor(),
            inputs={"tensor": self.base, "index": self.seed},
            name=self.workflow.workflow_nodes[0].name,
        )
        self.workflow.add_workflow_node(clone)
        with self.assertRaises(ValueError) as ctx:
            self.expand()
        self.assertIn("duplicate node names", str(ctx.exception))


class TestScope(unittest.TestCase):
    def test_lookup_walks_outward(self):
        outer = Scope(mapping={"a": tensor_io("A")})
        inner = outer.child()
        inner.mapping["b"] = tensor_io("B")

        self.assertEqual(inner.lookup("a").name, "A")
        self.assertEqual(inner.lookup("b").name, "B")
        self.assertIsNone(inner.lookup("c"))

    def test_inner_shadows_outer(self):
        outer = Scope(mapping={"a": tensor_io("OUTER")})
        inner = outer.child()
        inner.mapping["a"] = tensor_io("INNER")

        self.assertEqual(inner.lookup("a").name, "INNER")
        self.assertEqual(outer.lookup("a").name, "OUTER")

    def test_child_does_not_leak_into_parent(self):
        outer = Scope()
        inner = outer.child()
        inner.mapping["a"] = tensor_io("A")
        self.assertIsNone(outer.lookup("a"))


if __name__ == "__main__":
    unittest.main()
