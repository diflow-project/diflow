from typing import Any, Dict, Union

import torch
from diffusers.models.transformers import Flux2Transformer2DModel

from diflow.operators.base import require_pretrained_weights
from diflow.operators.flux2_utils import (
    prepare_latent_ids_4d,
    prepare_text_ids_4d,
)
from diflow.operators.models.diffusion_models.base_diffusion_model import (
    BaseDiffusionModel,
)
from diflow.operators.operator_ids import FLUX_2_KLEIN_ID


class Flux2Klein(BaseDiffusionModel):
    """Transformer operator for the distilled FLUX.2 Klein checkpoint."""

    def setup_io(self):
        super().setup_io()
        self.add_input("height", int)
        self.add_input("width", int)

    @property
    def id(self) -> str:
        return FLUX_2_KLEIN_ID

    def denoise_step_kwargs(self, context) -> Dict[str, Any]:
        return {"height": context.height, "width": context.width}

    def initialize(
        self, model_path: str, device: Union[str, torch.device]
    ) -> Dict[str, Any]:
        require_pretrained_weights(model_path, self.id)
        transformer = Flux2Transformer2DModel.from_pretrained(
            model_path,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        ).to(device)
        return {"transformer": transformer}

    @torch.no_grad()
    def execute(
        self,
        model_components: Dict[str, Any],
        device: Union[str, torch.device],
        **kwargs,
    ) -> Dict[str, Any]:
        transformer = model_components["transformer"]
        latents = kwargs["latents"]
        prompt_embeds = kwargs["prompt_embeds"]
        batch_size = latents.shape[0]
        patch_height = int(kwargs["height"]) // 16
        patch_width = int(kwargs["width"]) // 16

        # FLUX.2 uses four-axis integer coordinates. Keeping these as int64 is
        # essential because bf16 rounds coordinates beyond 256.
        text_ids = prepare_text_ids_4d(batch_size, prompt_embeds.shape[1], device)
        latent_ids = prepare_latent_ids_4d(
            batch_size, patch_height, patch_width, device
        )
        timestep = kwargs["timestep"].expand(batch_size).to(latents.dtype)
        noise_pred = transformer(
            hidden_states=latents,
            encoder_hidden_states=prompt_embeds,
            timestep=timestep / 1000,
            guidance=None,
            img_ids=latent_ids,
            txt_ids=text_ids,
            joint_attention_kwargs={},
            return_dict=False,
        )[0]
        return {"noise_pred": noise_pred[:, : latents.shape[1]]}
