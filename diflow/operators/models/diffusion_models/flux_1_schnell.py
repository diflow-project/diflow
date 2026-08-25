import logging
import time
from typing import Any, Dict, List, Union

import torch
from diffusers.models.transformers import FluxTransformer2DModel

from diflow.operators.base import has_pretrained_weights
from diflow.operators.flux_utils import (
    prepare_latent_image_ids,
    prepare_text_ids,
)
from diflow.operators.models.diffusion_models.base_diffusion_model import (
    BaseDiffusionModel,
)
from diflow.operators.operator_ids import FLUX_1_SCHNELL_ID
from diflow.operators.utils import test_model_memory_allocation

logger = logging.getLogger(__name__)


class Flux1Schnell(BaseDiffusionModel):
    def setup_io(self):
        super().setup_io()
        self.add_input("guidance_scale", float)
        self.add_input("text_ids", torch.Tensor)
        self.add_input("latent_image_ids", torch.Tensor)
        # ! Suyi: hard coding the length of the list of controlnet input
        for i in range(2):
            self.add_input("control_block_sample_{}".format(i), torch.Tensor, lazy=True)

    @property
    def id(self) -> str:
        return FLUX_1_SCHNELL_ID

    def denoise_step_kwargs(self, context) -> Dict[str, Any]:
        """Flux also reads its distilled guidance embedding and the image size.

        ``guidance`` is passed even by schnell, whose transformer has
        ``guidance_embeds: false`` and ignores the value. The edge is kept because
        it is what orders the guidance node before the model, and dropping it would
        change the served graph.
        """
        return {
            "guidance": context.guidance,
            "height": context.height,
            "width": context.width,
        }

    def initialize(
        self, model_path: Union[str, None], device: Union[str, torch.device]
    ) -> Dict[str, Any]:
        if has_pretrained_weights(model_path, self.id):
            transformer = FluxTransformer2DModel.from_pretrained(
                model_path,
                subfolder="transformer",
                torch_dtype=torch.bfloat16,
            ).to(device)
        else:
            config = self._default_dummy_config()
            transformer = FluxTransformer2DModel(**config).to(
                device=device, dtype=torch.bfloat16
            )

        return {"transformer": transformer}

    @staticmethod
    def _default_dummy_config() -> Dict[str, Any]:
        return {
            "attention_head_dim": 128,
            "guidance_embeds": False,
            "in_channels": 64,
            "joint_attention_dim": 4096,
            "num_attention_heads": 24,
            "num_layers": 19,
            "num_single_layers": 38,
            "patch_size": 1,
            "pooled_projection_dim": 768,
        }

    @torch.no_grad()
    def execute(
        self,
        model_components: Dict[str, Any],
        device: Union[str, torch.device],
        **kwargs,
    ) -> Dict[str, Any]:
        # Images with 1024x1024 resolution:
        # latents: (batch_size, 4096, 64)
        # prompt embeddings: (batch_size, 512, 4096)
        # pooled prompt embeddings: (batch_size, 768)
        transformer = model_components["transformer"]

        latents = kwargs["latents"]
        timestep = kwargs["timestep"]

        prompt_embeds = kwargs["prompt_embeds"]
        pooled_prompt_embeds = kwargs["pooled_prompt_embeds"]

        height = kwargs["height"]
        width = kwargs["width"]
        dtype = latents.dtype
        vae_scale_factor = kwargs.get("vae_scale_factor", 8)

        text_ids = prepare_text_ids(prompt_embeds.shape[1], device, dtype)
        latent_image_ids = prepare_latent_image_ids(
            height // (vae_scale_factor * 2),
            width // (vae_scale_factor * 2),
            device,
            dtype,
        )

        control_block_samples = []
        for i in range(2):
            control_block_samples.append(
                kwargs.get("control_block_sample_{}".format(i), None)
            )
        if None in control_block_samples:
            control_block_samples = None

        ### Flux 1.0 Schnell ###
        timestep = timestep.expand(latents.shape[0]).to(latents.dtype)
        noise_pred = transformer.stream_forward(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=None,  # Suyi: according to the transformer config, the guidance is None, where the "guidance_embeds": false, so we set it to None.
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            controlnet_block_samples=control_block_samples,
            controlnet_single_block_samples=None,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs={},
        )

        return {"noise_pred": noise_pred}


if __name__ == "__main__":
    test_model_memory_allocation(model=Flux1Schnell(), model_path=None)
