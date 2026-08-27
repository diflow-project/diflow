from typing import Any, Dict, Union

import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders import AutoencoderKL
from PIL.Image import Image

from diflow.operators.base import Operator, require_pretrained_weights
from diflow.operators.operator_ids import ZIMAGE_VAE_ID


class ZImageVAE(Operator):
    """Decode Z-Image's unpacked 16-channel latent tensor."""

    def setup_io(self):
        self.add_execution_mode(
            "decode_latents",
            inputs={"latents": torch.Tensor},
            outputs={"image": Image},
        )

    @property
    def id(self) -> str:
        return ZIMAGE_VAE_ID

    def initialize(
        self, model_path: str, device: Union[str, torch.device]
    ) -> Dict[str, Any]:
        require_pretrained_weights(model_path, self.id)
        vae = AutoencoderKL.from_pretrained(
            model_path,
            subfolder="vae",
            torch_dtype=torch.bfloat16,
        ).to(device)
        scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        processor = VaeImageProcessor(vae_scale_factor=scale_factor * 2)
        return {"vae": vae, "image_processor": processor}

    @torch.no_grad()
    def execute(
        self,
        model_components: Dict[str, Any],
        device: Union[str, torch.device],
        mode: str,
        **kwargs,
    ) -> Dict[str, Any]:
        if mode != "decode_latents":
            raise ValueError(f"Invalid execution mode: {mode}")
        vae = model_components["vae"]
        latents = kwargs["latents"].to(dtype=vae.dtype)
        latents = latents / vae.config.scaling_factor + vae.config.shift_factor
        image = vae.decode(latents, return_dict=False)[0]
        image = model_components["image_processor"].postprocess(
            image, output_type="pil"
        )
        return {"image": image}
