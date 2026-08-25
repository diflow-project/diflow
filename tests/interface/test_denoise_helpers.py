"""The three ways to write a denoise loop must agree.

The default loop and a hand-written ``step_fn`` share the same scaffolding, so
they cannot drift; this pins that, and pins the hand-written control-flow example
too -- an example that quietly builds a different graph teaches the API wrongly.
All three are compared against the same golden signatures the library path is held
to:

1. ``denoise_loop(...)`` with its default step
2. ``denoise_loop(step_fn=...)``, the step written out
3. ``for_range``/``cond`` written out by hand

A ``step_fn`` that does something the default cannot is exercised separately,
since that one is *supposed* to produce a different graph.
"""

import json
import unittest

from diflow.interface.workflow import Workflow
from diflow.interface.workflow_expand import expand_workflow
from tests.interface.graph_canon import (
    canonical_form,
    canonical_json,
    injected_values,
)
from tests.interface.hub_workflows import build_workflow, make_inputs

STEP_COUNTS = (1, 2, 4)
GUIDANCE_SCALES = (1.0, 7.5)

VIA_DEFAULT_STEP = "flux_schnell.register_txt2img_workflow"
VIA_STEP_FN = "flux_schnell.register_txt2img_helper_workflow"
VIA_RAW_CONTROL_FLOW = "flux_schnell.register_txt2img_control_flow_workflow"


def form(example, steps, guidance, via_json=False):
    workflow = build_workflow(example)
    base = make_inputs(workflow, steps, guidance)
    if via_json:
        workflow = Workflow.from_dict(json.loads(workflow.to_json()))
    request_inputs = dict(base)
    graph = expand_workflow(workflow, request_inputs)
    injected = injected_values(base, request_inputs)
    return canonical_json(canonical_form(graph, injected))


class TestAllThreeTiersAgree(unittest.TestCase):
    def test_a_hand_written_step_matches_the_default_one(self):
        for steps in STEP_COUNTS:
            for guidance in GUIDANCE_SCALES:
                with self.subTest(steps=steps, guidance=guidance):
                    self.assertEqual(
                        form(VIA_DEFAULT_STEP, steps, guidance),
                        form(VIA_STEP_FN, steps, guidance),
                    )

    def test_raw_control_flow_matches_the_default_step(self):
        for steps in STEP_COUNTS:
            for guidance in GUIDANCE_SCALES:
                with self.subTest(steps=steps, guidance=guidance):
                    self.assertEqual(
                        form(VIA_DEFAULT_STEP, steps, guidance),
                        form(VIA_RAW_CONTROL_FLOW, steps, guidance),
                    )

    def test_a_step_fn_loop_survives_the_registration_json_boundary(self):
        """A step_fn's loop has to serialize like any other region."""
        self.assertEqual(
            form(VIA_DEFAULT_STEP, 3, 7.5),
            form(VIA_STEP_FN, 3, 7.5, via_json=True),
        )

    def test_the_cfg_choice_wraps_the_loop_rather_than_the_step(self):
        """One cond at the top, a fixed step in each branch.

        The branch has to enclose the whole loop. Deciding per step instead would
        leave the negative prompt's encoders above the branch, running on every
        request; the else branch below is what proves they are inside it.
        """
        workflow = build_workflow(VIA_STEP_FN)
        self.assertEqual(len(workflow.regions), 1)
        top = workflow.regions[0]
        self.assertEqual(top.region_kind, "cond")

        for body in (top.then_body, top.else_body):
            self.assertEqual(
                [region.region_kind for region in body.iter_regions()], ["loop"]
            )

        def encoder_count(body):
            return sum(
                1
                for region in body.iter_regions()
                for node in region.body.iter_nodes()
                if node.op.id in ("T5_Flux", "CLIP_Flux")
            ) + sum(
                1
                for node in body.iter_nodes()
                if node.op.id in ("T5_Flux", "CLIP_Flux")
            )

        self.assertEqual(encoder_count(top.then_body), 2, "negative prompt encoders")
        self.assertEqual(encoder_count(top.else_body), 0, "nothing to encode")


class TestStepFnEscapeHatch(unittest.TestCase):
    """``step_fn`` lets the per-step work be anything, not just a model pass."""

    def _build(self, step_fn):
        import torch

        from diflow.interface import denoise_loop
        from diflow.operators.custom.indexed_tensor import IndexedTensor

        workflow = Workflow("custom_step")
        steps = workflow.add_input("num_inference_steps", int)
        latents = workflow.add_input("latents", torch.Tensor)
        embeds = workflow.add_input("prompt_embeds", torch.Tensor)

        # A stand-in model/scheduler pair: IndexedTensor takes a tensor and an
        # index, which is enough to build a loop body without loading weights.
        class ToyScheduler(IndexedTensor):
            @property
            def id(self):
                return "ToyScheduler"

            def setup_io(self):
                self.add_execution_mode(
                    "init",
                    inputs={"num_inference_steps": int, "latents": torch.Tensor},
                    outputs={"timesteps": torch.Tensor},
                )

        denoised = denoise_loop(
            model=None,
            scheduler=ToyScheduler(),
            latents=latents,
            num_inference_steps=steps,
            prompt_embeds=embeds,
            step_fn=step_fn,
        )
        workflow.add_output(denoised, "out")
        return workflow

    def test_a_custom_step_replaces_only_the_per_step_work(self):
        """The scaffolding still runs: scheduler init, indexing, carry threading."""
        from diflow.operators.custom.indexed_tensor import IndexedTensor

        double = IndexedTensor()

        def step_fn(context):
            # Two nodes per step instead of the usual one model pass.
            once = double(tensor=context.latents, index=context.timestep)
            return double(tensor=once, index=context.timestep)

        workflow = self._build(step_fn)
        self.assertEqual(len(workflow.regions), 1)

        body_ops = [n.op.id for n in workflow.regions[0].body.iter_nodes()]
        # one indexing node for the timestep, plus the two the step_fn emitted
        self.assertEqual(body_ops.count("IndexedTensor"), 3)

        graph = expand_workflow(
            workflow,
            {"num_inference_steps": 3, "latents": None, "prompt_embeds": None},
        )
        # 3 iterations x 3 nodes, plus the scheduler init outside the loop
        self.assertEqual(len(graph.workflow_nodes), 3 * 3 + 1)

    def test_the_carry_still_serializes_the_iterations(self):
        """A custom step must not break the ordering the carry provides."""
        from diflow.operators.custom.indexed_tensor import IndexedTensor

        op = IndexedTensor()

        def step_fn(context):
            return op(tensor=context.latents, index=context.timestep)

        workflow = self._build(step_fn)
        graph = expand_workflow(
            workflow,
            {"num_inference_steps": 4, "latents": None, "prompt_embeds": None},
        )

        consumers = {}
        for node in graph.workflow_nodes:
            for node_io in node.get_inputs().values():
                if node_io is not None:
                    consumers.setdefault(node_io.name, []).append(node)

        # Walk the carry chain forward from the initial latents.
        seen = 0
        current = "latents"
        while current in consumers:
            following = [
                n
                for n in consumers[current]
                if n.get_inputs().get("tensor") is not None
                and n.get_inputs()["tensor"].name == current
            ]
            if not following:
                break
            seen += 1
            current = list(following[0].get_outputs().values())[0].name
        self.assertEqual(seen, 4, "each iteration should consume the previous one")


class TestPublicSurface(unittest.TestCase):
    """The loop builder has to be importable after a plain pip install.

    It lived alongside the examples first, which meant reading the repo was the
    only way to get at it.
    """

    def test_the_control_flow_api_is_exported_from_the_package(self):
        import diflow.interface as api

        for name in ("denoise_loop", "for_range", "cond", "DenoiseContext"):
            with self.subTest(name=name):
                self.assertIn(name, api.__all__)
                self.assertTrue(hasattr(api, name))

    def test_a_model_or_a_step_fn_is_required(self):
        from diflow.interface import denoise_loop

        workflow = Workflow("needs_one")
        with self.assertRaises(ValueError) as ctx:
            denoise_loop(
                scheduler=object(),
                latents=workflow.add_input("latents", int),
                num_inference_steps=workflow.add_input("n", int),
            )
        self.assertIn("step_fn", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
