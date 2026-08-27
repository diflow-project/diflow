from typing import Any, Dict, Union

import torch
from diffusers.models.transformers import ZImageTransformer2DModel

from diflow.operators.base import require_pretrained_weights
from diflow.operators.models.diffusion_models.base_diffusion_model import (
    BaseDiffusionModel,
)
from diflow.operators.operator_ids import ZIMAGE_ID


class ZImage(BaseDiffusionModel):
    """Z-Image transformer with list-valued latent and text inputs."""

    def setup_io(self):
        super().setup_io()
        self.add_input("encoder_attention_mask", torch.Tensor)

    @property
    def id(self) -> str:
        return ZIMAGE_ID

    def denoise_step_kwargs(self, context) -> Dict[str, Any]:
        return {"encoder_attention_mask": context.encoder_attention_mask}

    def initialize(
        self, model_path: str, device: Union[str, torch.device]
    ) -> Dict[str, Any]:
        require_pretrained_weights(model_path, self.id)
        transformer = ZImageTransformer2DModel.from_pretrained(
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
        attention_mask = kwargs.get("encoder_attention_mask")
        batch_size = latents.shape[0]

        # The reference transformer consumes one (C, 1, H, W) tensor and one
        # variable-length text tensor per sample rather than dense batched tensors.
        latent_list = list(latents.to(transformer.dtype).unsqueeze(2).unbind(dim=0))
        if attention_mask is None:
            cap_feats = list(prompt_embeds.unbind(dim=0))
        else:
            cap_feats = [
                prompt_embeds[index][attention_mask[index].bool()]
                for index in range(batch_size)
            ]

        timestep = kwargs["timestep"].expand(batch_size)
        timestep = (1000 - timestep) / 1000
        outputs = transformer(
            latent_list,
            timestep,
            cap_feats,
            return_dict=False,
        )[0]
        # Scheduler state remains float32 in the diffusers pipeline.
        noise_pred = torch.stack([output.float() for output in outputs]).squeeze(2)
        return {"noise_pred": noise_pred}
