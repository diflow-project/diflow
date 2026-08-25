"""Assembling one denoising step without knowing which model family it is.

These are pure graph builders: they call operators, which emit nodes, and they
return ``NodeIO`` handles. Nothing here executes.

Both functions used to dispatch on ``diffusion_model_id`` with a branch per model
family, so adding a family meant editing this module -- and forgetting to meant a
``ValueError`` at registration at best, or a model invoked without the kwargs its
``execute`` reads at worst. The per-family knowledge now lives on the operators:

* what extra inputs a model wants per step -> :meth:`BaseDiffusionModel.denoise_step_kwargs`
* how an adapter is called, and how its outputs are named for the model ->
  :meth:`BaseAdapter.adapter_step_kwargs` / :meth:`BaseAdapter.pack_block_samples`

Adding a model family is now adding an operator. Note that the hooks are the
authority, not ``setup_io``: several operators declare inputs they never read and
read inputs they never declared, so the declarations cannot be trusted to drive
this.
"""

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional

from diflow.interface.node_io import NodeIO


@dataclass(frozen=True)
class DenoiseContext:
    """Everything one denoising step can draw on.

    Fields are graph handles, not tensors. ``pooled_prompt_embeds`` and below are
    optional because not every family has them; an operator asking for one that is
    ``None`` gets ``None``, and ``Operator.__call__`` drops it, which is how the
    graph stays free of edges a model does not use.
    """

    latents: NodeIO
    timestep: NodeIO
    prompt_embeds: Optional[NodeIO] = None
    pooled_prompt_embeds: Optional[NodeIO] = None
    # Flux's distilled guidance embedding. Unrelated to classifier-free guidance.
    guidance: Optional[NodeIO] = None
    height: Optional[NodeIO] = None
    width: Optional[NodeIO] = None

    def with_conditioning(
        self,
        prompt_embeds: Optional[NodeIO],
        pooled_prompt_embeds: Optional[NodeIO],
    ) -> "DenoiseContext":
        """The same step under different text conditioning.

        Classifier-free guidance runs the step twice, once on the negative
        embeddings and once on the positive ones; everything else is shared.
        """
        return replace(
            self,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
        )


def run_adapters(
    adapters: Optional[List],
    adapter_inputs: Optional[List],
    context: DenoiseContext,
) -> Optional[List[Dict[str, NodeIO]]]:
    """Emit each adapter's nodes and name its outputs the way the model expects.

    Returns one dict of model kwargs per adapter, or ``None`` when there are no
    adapters -- distinct from an empty list, which the callers rely on.
    """
    if not adapters:
        return None

    packed = []
    for adapter, adapter_input in zip(adapters, adapter_inputs):
        outputs = adapter(**adapter.adapter_step_kwargs(context, adapter_input))
        packed.append(adapter.pack_block_samples(outputs))
    return packed


def prepare_model_kwargs(
    model,
    context: DenoiseContext,
    adapter_block_samples: Optional[List[Dict[str, NodeIO]]],
) -> Dict[str, Any]:
    """The kwargs for one call of the diffusion model."""
    model_kwargs: Dict[str, Any] = {
        "latents": context.latents,
        "timestep": context.timestep,
        "prompt_embeds": context.prompt_embeds,
        "pooled_prompt_embeds": context.pooled_prompt_embeds,
    }
    model_kwargs.update(model.denoise_step_kwargs(context))

    if adapter_block_samples:
        if len(adapter_block_samples) > 1:
            # Previously the extras were dropped here without a word, leaving
            # their nodes in the graph with nothing consuming them.
            raise ValueError(
                f"{model.id}: {len(adapter_block_samples)} adapters were given, but "
                f"a diffusion model can only be wired to one. Supporting several "
                f"means deciding how their residuals combine, which is model "
                f"specific."
            )
        model_kwargs.update(adapter_block_samples[0])

    return model_kwargs
