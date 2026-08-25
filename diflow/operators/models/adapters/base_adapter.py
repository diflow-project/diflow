from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List

import torch

from diflow.operators.base import Operator

if TYPE_CHECKING:  # pragma: no cover
    from diflow.interface.denoise_ops import DenoiseContext
    from diflow.interface.node_io import AdapterInputs, NodeIO


class BaseAdapter(Operator):
    @abstractmethod
    def id(self) -> str:
        pass

    def setup_io(self):
        self.add_input("latents", torch.Tensor)
        self.add_input("timestep", torch.Tensor)
        self.add_input("prompt_embeds", torch.Tensor)
        self.add_input("pooled_prompt_embeds", torch.Tensor)
        # self.add_output("block_samples", List[torch.Tensor])

    def adapter_step_kwargs(
        self, context: "DenoiseContext", adapter_input: "AdapterInputs"
    ) -> Dict[str, Any]:
        """The kwargs for one call of this adapter.

        The default is what every family shares. Override to add whatever else the
        adapter reads -- Flux's controlnet also wants the guidance embedding and
        the image dimensions.
        """
        return {
            "latents": context.latents,
            "timestep": context.timestep,
            "prompt_embeds": context.prompt_embeds,
            "pooled_prompt_embeds": context.pooled_prompt_embeds,
            "controlnet_cond": adapter_input.controlnet_cond,
            "conditioning_scale": adapter_input.conditioning_scale,
        }

    def pack_block_samples(self, outputs) -> Dict[str, "NodeIO"]:
        """Name this adapter's residuals the way its diffusion model reads them.

        ``outputs`` is what calling the adapter returned: a list of ``NodeIO`` in
        declaration order, or a single one if it declares a single output.

        The naming is family specific -- Flux takes ``control_block_sample_{i}``,
        while the UNet families took ``down_block_res_sample_{i}`` plus a separate
        ``mid_block_res_sample`` -- which is why it belongs next to the adapter
        that produces them rather than in a central table.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement pack_block_samples to say how "
            f"its outputs map onto the diffusion model's residual inputs"
        )
