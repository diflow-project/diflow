"""A new model family should need no edits to shared code.

``interface/denoise_ops.py`` used to dispatch on ``diffusion_model_id``: which
extra kwargs a model wanted, how its adapter was called, and what its residuals
were named all lived in one ``if/elif`` chain there. Adding a family meant editing
it, and forgetting to meant either a ``ValueError`` at registration or a model
invoked without the kwargs its ``execute`` reads.

Those three decisions now sit on the operators. The proof is a family defined
entirely here -- no production file mentions it -- that wants a *different* set of
step kwargs and names its residuals the way the UNet families did, and still
builds and expands.
"""

import unittest

import torch

from diflow.interface import denoise_loop
from diflow.interface.denoise_ops import (
    DenoiseContext,
    prepare_model_kwargs,
    run_adapters,
)
from diflow.interface.node_io import AdapterInputs, NodeIO, SourceType
from diflow.interface.workflow import Workflow
from diflow.interface.workflow_context import WorkflowContext
from diflow.interface.workflow_expand import expand_workflow
from diflow.operators.models.adapters.base_adapter import BaseAdapter
from diflow.operators.models.diffusion_models.base_diffusion_model import (
    BaseDiffusionModel,
)
from diflow.operators.schedulers.base_scheduler import BaseScheduler


class ToyModel(BaseDiffusionModel):
    """Wants only ``height`` per step -- not Flux's guidance/height/width."""

    @property
    def id(self) -> str:
        return "ToyModel"

    def setup_io(self):
        super().setup_io()
        for i in range(2):
            self.add_input(f"down_block_res_sample_{i}", torch.Tensor, lazy=True)
        self.add_input("mid_block_res_sample", torch.Tensor, lazy=True)

    def denoise_step_kwargs(self, context):
        return {"height": context.height}

    def initialize(self, model_path, device):
        return {}

    def execute(self, model_components, device, **kwargs):
        return {"noise_pred": kwargs["latents"]}


class ToyAdapter(BaseAdapter):
    """Names its residuals the way the UNet families did.

    ``down_block_res_sample_{i}`` plus a single ``mid_block_res_sample`` -- a shape
    the old central table had a dedicated branch for.
    """

    @property
    def id(self) -> str:
        return "ToyAdapter"

    def setup_io(self):
        super().setup_io()
        self.add_input("controlnet_cond", torch.Tensor)
        self.add_input("conditioning_scale", float)
        for i in range(2):
            self.add_output(f"down_block_res_sample_{i}", torch.Tensor, lazy=True)
        self.add_output("mid_block_res_sample", torch.Tensor, lazy=True)

    def pack_block_samples(self, outputs):
        *down, mid = outputs
        packed = {f"down_block_res_sample_{i}": s for i, s in enumerate(down)}
        packed["mid_block_res_sample"] = mid
        return packed

    def initialize(self, model_path, device):
        return {}

    def execute(self, model_components, device, **kwargs):
        raise NotImplementedError


class ToyScheduler(BaseScheduler):
    @property
    def id(self) -> str:
        return "ToyScheduler"

    def setup_io(self):
        self.add_execution_mode(
            "init",
            inputs={"num_inference_steps": int, "latents": torch.Tensor},
            outputs={"timesteps": torch.Tensor},
        )
        self.add_execution_mode(
            "step",
            inputs={
                "latents": torch.Tensor,
                "timestep": torch.Tensor,
                "noise_pred": torch.Tensor,
            },
            outputs={"output_latents": torch.Tensor},
        )

    def initialize(self, model_path, device):
        return {}

    def execute(self, model_components, device, mode, **kwargs):
        raise NotImplementedError


def request_input(name, data_type):
    return NodeIO(name=name, data_type=data_type, source_type=SourceType.INPUT)


def node_output(name, source_node="Producer_x"):
    return NodeIO(
        name=name,
        data_type=torch.Tensor,
        source_type=SourceType.NODE,
        source_node=source_node,
    )


class ExtensibilityTestCase(unittest.TestCase):
    def setUp(self):
        self._saved = WorkflowContext.get_current_workflow()

    def tearDown(self):
        WorkflowContext.set_current_workflow(self._saved)

    def context(self, **overrides):
        base = dict(
            latents=node_output("Gen_x:latents", "Gen_x"),
            timestep=node_output("Index_x:t", "Index_x"),
            prompt_embeds=node_output("Text_x:embeds", "Text_x"),
            pooled_prompt_embeds=None,
            guidance=node_output("Guidance_x:g", "Guidance_x"),
            height=request_input("height", int),
            width=request_input("width", int),
        )
        base.update(overrides)
        return DenoiseContext(**base)


class TestModelHookDecidesItsOwnKwargs(ExtensibilityTestCase):
    def test_only_what_the_model_asked_for_is_passed(self):
        Workflow("toy")  # installs a sink so operator calls can emit
        kwargs = prepare_model_kwargs(ToyModel(), self.context(), None)

        self.assertEqual(kwargs["height"].name, "height")
        # The context also carries guidance and width; this model wants neither.
        self.assertNotIn("guidance", kwargs)
        self.assertNotIn("width", kwargs)
        # The four shared inputs are always present.
        for key in ("latents", "timestep", "prompt_embeds", "pooled_prompt_embeds"):
            self.assertIn(key, kwargs)

    def test_a_model_wanting_nothing_extra_is_a_valid_answer(self):
        """The old code raised for any id it did not enumerate; ``{}`` is normal."""

        class Bare(ToyModel):
            @property
            def id(self):
                return "Bare"

            def denoise_step_kwargs(self, context):
                return {}

        Workflow("toy")
        kwargs = prepare_model_kwargs(Bare(), self.context(), None)
        self.assertEqual(
            set(kwargs),
            {"latents", "timestep", "prompt_embeds", "pooled_prompt_embeds"},
        )


class TestAdapterHookNamesItsOwnResiduals(ExtensibilityTestCase):
    def test_unet_style_residual_naming_round_trips_into_model_kwargs(self):
        workflow = Workflow("toy")
        adapter_input = AdapterInputs(
            controlnet_cond=node_output("Cond_x:img", "Cond_x"),
            conditioning_scale=request_input("conditioning_scale", float),
        )
        packed = run_adapters([ToyAdapter()], [adapter_input], self.context())

        self.assertEqual(len(packed), 1)
        self.assertEqual(
            set(packed[0]),
            {
                "down_block_res_sample_0",
                "down_block_res_sample_1",
                "mid_block_res_sample",
            },
        )

        kwargs = prepare_model_kwargs(ToyModel(), self.context(), packed)
        self.assertIn("mid_block_res_sample", kwargs)
        self.assertIn("down_block_res_sample_1", kwargs)
        # The adapter emitted a real node.
        self.assertEqual(len(workflow.workflow_nodes), 1)
        self.assertEqual(workflow.workflow_nodes[0].op.id, "ToyAdapter")

    def test_no_adapters_yields_none_not_an_empty_list(self):
        """Callers distinguish the two; ``prepare_model_kwargs`` skips on falsy."""
        self.assertIsNone(run_adapters(None, None, self.context()))
        self.assertIsNone(run_adapters([], [], self.context()))

    def test_the_default_pack_explains_what_to_implement(self):
        class Unpacked(ToyAdapter):
            pack_block_samples = BaseAdapter.pack_block_samples

        with self.assertRaises(NotImplementedError) as ctx:
            Unpacked().pack_block_samples([])
        self.assertIn("pack_block_samples", str(ctx.exception))


class TestMultipleAdaptersAreRefused(ExtensibilityTestCase):
    def test_a_second_adapter_raises_rather_than_being_dropped(self):
        """It used to take ``[0]`` and discard the rest, leaving their nodes in the
        graph with nothing consuming them."""
        Workflow("toy")
        packed = [
            {"mid_block_res_sample": node_output("A:x", "A")},
            {"mid_block_res_sample": node_output("B:x", "B")},
        ]
        with self.assertRaises(ValueError) as ctx:
            prepare_model_kwargs(ToyModel(), self.context(), packed)
        message = str(ctx.exception)
        self.assertIn("ToyModel", message)
        self.assertIn("only be wired to one", message)


class TestDenoiseLoopWorksForANewFamily(ExtensibilityTestCase):
    """The end that matters: the shared builder handles a family it never heard of."""

    def _build(self, with_adapter):
        workflow = Workflow("toy_denoise")
        steps = workflow.add_input("num_inference_steps", int)
        height = workflow.add_input("height", int)
        width = workflow.add_input("width", int)
        # Request inputs rather than fake node outputs: the expander refuses an
        # edge with no producer, which is the point of its invariant checks.
        latents = workflow.add_input("latents", torch.Tensor)
        embeds = workflow.add_input("prompt_embeds", torch.Tensor)

        adapters = None
        adapter_inputs = None
        if with_adapter:
            adapters = [ToyAdapter()]
            adapter_inputs = [
                AdapterInputs(
                    controlnet_cond=workflow.add_input("controlnet_cond", torch.Tensor),
                    conditioning_scale=workflow.add_input("conditioning_scale", float),
                )
            ]

        denoised = denoise_loop(
            model=ToyModel(),
            scheduler=ToyScheduler(),
            latents=latents,
            num_inference_steps=steps,
            prompt_embeds=embeds,
            height=height,
            width=width,
            adapters=adapters,
            adapter_inputs=adapter_inputs,
        )
        return workflow, denoised

    def test_the_loop_is_built_and_the_model_gets_its_own_kwargs(self):
        workflow, _ = self._build(with_adapter=False)
        self.assertEqual(len(workflow.regions), 1)

        model_node = next(
            n for n in workflow.regions[0].body.iter_nodes() if n.op.id == "ToyModel"
        )
        self.assertIn("height", model_node.get_inputs())
        self.assertNotIn("guidance", model_node.get_inputs())
        self.assertNotIn("width", model_node.get_inputs())

    def test_the_adapter_residuals_reach_the_model(self):
        workflow, _ = self._build(with_adapter=True)
        model_node = next(
            n for n in workflow.regions[0].body.iter_nodes() if n.op.id == "ToyModel"
        )
        for key in (
            "down_block_res_sample_0",
            "down_block_res_sample_1",
            "mid_block_res_sample",
        ):
            self.assertIn(key, model_node.get_inputs())

    def test_it_expands(self):
        """Expansion runs the invariant checks, so a clean pass is the assertion."""
        for with_adapter in (False, True):
            with self.subTest(with_adapter=with_adapter):
                workflow, denoised = self._build(with_adapter)
                workflow.add_output(denoised, "out")
                inputs = {
                    "num_inference_steps": 3,
                    "height": 64,
                    "width": 64,
                    "latents": None,
                    "prompt_embeds": None,
                    "controlnet_cond": None,
                    "conditioning_scale": 1.0,
                }
                graph = expand_workflow(workflow, inputs)
                model_nodes = [n for n in graph.workflow_nodes if n.op.id == "ToyModel"]
                self.assertEqual(len(model_nodes), 3)


if __name__ == "__main__":
    unittest.main()
