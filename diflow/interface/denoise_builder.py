"""The iterative denoising loop, expressed with control flow.

:func:`denoise_loop` is the standard way to write a denoising pipeline. Pass
``step_fn`` to replace just the per-step computation while keeping the
scaffolding: scheduler init, timestep indexing, carry threading. That scaffolding
is the fiddly part, and it has nothing to do with what a given pipeline wants to
change. See ``workflow_hub/flux_schnell/register_txt2img_helper_workflow.py``.

Below it there is only :func:`~diflow.interface.control_flow.for_range` and
:func:`~diflow.interface.control_flow.cond`, for a shape neither of the
above covers, such as nested loops or several carried values. See
``workflow_hub/flux_schnell/register_txt2img_control_flow_workflow.py``.

Nothing here is privileged: the loop is ``for_range`` over a body containing a
``cond``, which any caller could write.

Nothing here is model specific either. Which extra inputs a model wants per step,
and how an adapter's residuals are named, come from the operators themselves via
:func:`~diflow.interface.denoise_ops.prepare_model_kwargs` and
:func:`~diflow.interface.denoise_ops.run_adapters`.
"""

import logging
import uuid
from typing import Callable, List, Optional

from diflow.interface.control_flow import cond, for_range
from diflow.interface.denoise_ops import (
    DenoiseContext,
    prepare_model_kwargs,
    run_adapters,
)
from diflow.interface.node_io import AdapterInputs, NodeIO
from diflow.operators.custom.guidance_tensor import GuidanceTensor

logger = logging.getLogger(__name__)

CARRY_KEY = "latents"

StepFn = Callable[[DenoiseContext], NodeIO]


def denoise_loop(
    *,
    scheduler,
    latents: NodeIO,
    num_inference_steps: NodeIO,
    model=None,
    prompt_embeds: Optional[NodeIO] = None,
    encoder_attention_mask: Optional[NodeIO] = None,
    pooled_prompt_embeds: Optional[NodeIO] = None,
    negative_prompt_embeds: Optional[NodeIO] = None,
    negative_encoder_attention_mask: Optional[NodeIO] = None,
    negative_pooled_prompt_embeds: Optional[NodeIO] = None,
    cfg_guidance_scale: Optional[NodeIO] = None,
    cfg_threshold: float = 1.0,
    guidance_scale: Optional[NodeIO] = None,
    height: Optional[NodeIO] = None,
    width: Optional[NodeIO] = None,
    adapters: Optional[List] = None,
    adapter_inputs: Optional[List[AdapterInputs]] = None,
    step_fn: Optional[StepFn] = None,
    region_id: Optional[str] = None,
    iv_name_template: str = "{region_id}_timestep_{i}",
) -> NodeIO:
    """Denoise ``latents`` for ``num_inference_steps`` and return the result.

    Args:
        scheduler: The scheduler operator. Called once with ``mode="init"``, then
            per step with ``mode="step"`` or ``mode="step_classifier_free_guidance"``.
        latents: The initial latents.
        model: The diffusion model operator. Optional only when ``step_fn`` is
            given, since the step then makes its own model call.
        num_inference_steps: Normally a request input. The loop is expanded to
            this many iterations once a request arrives, so the count does not have
            to be known while the workflow is being built.
        cfg_guidance_scale: Enables classifier-free guidance when given. The choice
            is made per request through ``cond``, so one registered workflow serves
            both; ``negative_prompt_embeds`` is then required.
        cfg_threshold: Selects the CFG branch when ``cfg_guidance_scale`` is above
            this value. It defaults to the conventional ``1.0``; model families
            whose pipeline defines CFG for every positive scale can pass ``0.0``.
        guidance_scale: Flux's distilled guidance embedding, which is unrelated to
            classifier-free guidance. Hoisted out of the loop when present.
        step_fn: Replaces the per-step computation. Receives a
            :class:`~diflow.interface.denoise_ops.DenoiseContext` whose
            ``latents`` are the current ones, and returns the updated latents.
            Everything else -- scheduler init, timestep indexing, carry threading --
            still happens. Defaults to one model pass, or two under CFG.
        region_id: Override the generated region id, and with it the names of the
            induction variable and the loop's result placeholder. Cosmetic.
        iv_name_template: Per-iteration name for the induction variable. Must
            contain ``{i}``.

    Returns:
        A handle on the latents after the final step. With
        ``num_inference_steps=0`` that is the input latents, unchanged.
    """
    adapters = adapters or []
    adapter_inputs = adapter_inputs or []
    if len(adapters) != len(adapter_inputs):
        raise ValueError(
            f"got {len(adapters)} adapters but {len(adapter_inputs)} adapter inputs"
        )

    if model is None and step_fn is None:
        raise ValueError(
            "denoise_loop needs either a model, or a step_fn that performs the "
            "model call itself"
        )

    if cfg_guidance_scale is not None and negative_prompt_embeds is None:
        # Both branches are traced, so the CFG branch would otherwise build a model
        # node with prompt_embeds silently dropped (Operator.__call__ discards None
        # inputs). Fail while building instead.
        raise ValueError(
            "cfg_guidance_scale enables the classifier-free guidance branch, so "
            "negative_prompt_embeds is required: the unconditional pass has "
            "nothing to run on without it."
        )

    if region_id is None:
        # Cosmetic: it names the induction variable and the result placeholder.
        # A custom step_fn may not have a model to name it after.
        named = "_".join(op.id for op in (model, scheduler) if op is not None)
        region_id = f"{named}_{uuid.uuid4()}"

    # ---- once, outside the loop ------------------------------------------------
    # Neither the timestep schedule nor Flux's guidance embedding varies per step,
    # so computing them inside would emit a redundant node per iteration.
    timesteps = scheduler(
        num_inference_steps=num_inference_steps,
        # Flux's scheduler init derives its sigmas from the latent shape.
        latents=latents,
        mode="init",
    )
    guidance = (
        GuidanceTensor()(guidance_scale=guidance_scale)
        if guidance_scale is not None
        else None
    )

    if step_fn is None:
        step_fn = _default_step(
            model=model,
            scheduler=scheduler,
            adapters=adapters,
            adapter_inputs=adapter_inputs,
            cfg_guidance_scale=cfg_guidance_scale,
            cfg_threshold=cfg_threshold,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_encoder_attention_mask=negative_encoder_attention_mask,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        )

    def body(index, carry):
        # `index` is a handle on the loop counter, not a number. Subscripting with
        # it emits the indexing node; arithmetic on it does not work, because the
        # value only exists per iteration, once the loop is expanded.
        context = DenoiseContext(
            latents=carry[CARRY_KEY],
            timestep=timesteps[index],
            prompt_embeds=prompt_embeds,
            encoder_attention_mask=encoder_attention_mask,
            pooled_prompt_embeds=pooled_prompt_embeds,
            guidance=guidance,
            height=height,
            width=width,
        )
        return {CARRY_KEY: step_fn(context)}

    # The carry threads latents from one iteration to the next, and it is also what
    # makes them run in order: the executor schedules on data dependencies alone,
    # and the scheduler operator is stateful.
    return for_range(
        num_inference_steps,
        body,
        carry={CARRY_KEY: latents},
        region_id=region_id,
        iv_name_template=iv_name_template,
    )[CARRY_KEY]


def _default_step(
    *,
    model,
    scheduler,
    adapters,
    adapter_inputs,
    cfg_guidance_scale,
    cfg_threshold,
    negative_prompt_embeds,
    negative_encoder_attention_mask,
    negative_pooled_prompt_embeds,
) -> StepFn:
    """The standard per-step computation: one model pass, or two under CFG."""

    def predict(context: DenoiseContext) -> NodeIO:
        """One model pass, with the adapters that feed it."""
        block_samples = run_adapters(adapters, adapter_inputs, context)
        return model(**prepare_model_kwargs(model, context, block_samples))

    def step(context: DenoiseContext) -> NodeIO:
        def with_cfg():
            uncond = context.with_conditioning(
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
                negative_encoder_attention_mask,
            )
            return {
                CARRY_KEY: scheduler(
                    latents=context.latents,
                    timestep=context.timestep,
                    noise_pred_uncond=predict(uncond),
                    noise_pred_text=predict(context),
                    guidance_scale=cfg_guidance_scale,
                    mode="step_classifier_free_guidance",
                )
            }

        def without_cfg():
            return {
                CARRY_KEY: scheduler(
                    latents=context.latents,
                    timestep=context.timestep,
                    noise_pred=predict(context),
                    mode="step",
                )
            }

        if cfg_guidance_scale is None:
            # CFG can never be selected, so emit the single pass directly. A cond
            # here would still trace the CFG branch, building a model node out of
            # embeddings that were never supplied.
            return without_cfg()[CARRY_KEY]

        # Decided per request. Both branches are traced now, so both must be
        # buildable even though only one will end up in the graph.
        return cond(cfg_guidance_scale > cfg_threshold, with_cfg, without_cfg)[
            CARRY_KEY
        ]

    return step
