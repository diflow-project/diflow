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
        self.add_execution_mode(
            "default",
            inputs={
                "latents": torch.Tensor,
                "timestep": torch.Tensor,
                "prompt_embeds": torch.Tensor,
                "encoder_attention_mask": torch.Tensor,
            },
            outputs={"noise_pred": torch.Tensor},
        )
        self.add_execution_mode(
            "batch_cfg",
            inputs={
                "latents": torch.Tensor,
                "timestep": torch.Tensor,
                "prompt_embeds": torch.Tensor,
                "negative_prompt_embeds": torch.Tensor,
                "encoder_attention_mask": torch.Tensor,
                "negative_encoder_attention_mask": torch.Tensor,
            },
            outputs={
                "noise_pred_text": torch.Tensor,
                "noise_pred_uncond": torch.Tensor,
            },
        )

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
        mode: str = "default",
        **kwargs,
    ) -> Dict[str, Any]:
        transformer = model_components["transformer"]
        latents = kwargs["latents"]
        prompt_embeds = kwargs["prompt_embeds"]
        attention_mask = kwargs.get("encoder_attention_mask")
        batch_size = latents.shape[0]

        if mode == "batch_cfg":
            negative_embeds = kwargs["negative_prompt_embeds"]
            negative_mask = kwargs.get("negative_encoder_attention_mask")

            # The reference pipeline concatenates positive then negative
            # conditioning into one batch=2 transformer invocation. Keeping that
            # batching is required for an apple-to-apple numerical comparison.
            latent_input = latents.to(transformer.dtype).repeat(2, 1, 1, 1)
            latent_list = list(latent_input.unsqueeze(2).unbind(dim=0))

            def variable_length(embeds, mask):
                if mask is None:
                    return list(embeds.unbind(dim=0))
                return [
                    embeds[index][mask[index].bool()] for index in range(batch_size)
                ]

            cap_feats = variable_length(prompt_embeds, attention_mask)
            cap_feats += variable_length(negative_embeds, negative_mask)
            timestep = kwargs["timestep"].expand(batch_size)
            timestep = ((1000 - timestep) / 1000).repeat(2)
            outputs = transformer(
                latent_list,
                timestep,
                cap_feats,
                return_dict=False,
            )[0]
            positive = torch.stack(
                [output.float() for output in outputs[:batch_size]]
            ).squeeze(2)
            negative = torch.stack(
                [output.float() for output in outputs[batch_size:]]
            ).squeeze(2)
            return {
                "noise_pred_text": positive,
                "noise_pred_uncond": negative,
            }

        if mode != "default":
            raise ValueError(f"Invalid execution mode: {mode}")

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
