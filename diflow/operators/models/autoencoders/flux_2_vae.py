from typing import Any, Dict, Union

import torch
from diffusers.models.autoencoders import AutoencoderKLFlux2
from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
from PIL.Image import Image

from diflow.operators.base import Operator, require_pretrained_weights
from diflow.operators.flux2_utils import prepare_latent_ids_4d
from diflow.operators.operator_ids import FLUX_2_VAE_ID


def unpack_latents_with_ids(
    tokens: torch.Tensor, ids: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    """Scatter raster tokens back into their patchified spatial positions."""
    images = []
    for data, positions in zip(tokens, ids):
        channels = data.shape[1]
        flat_ids = positions[:, 1].long() * width + positions[:, 2].long()
        spatial = torch.zeros(
            (height * width, channels), device=data.device, dtype=data.dtype
        )
        spatial.scatter_(0, flat_ids[:, None].expand(-1, channels), data)
        images.append(spatial.view(height, width, channels).permute(2, 0, 1))
    return torch.stack(images)


def unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
    """Reverse FLUX.2's 2x2 channel patchification."""
    batch, channels, height, width = latents.shape
    latents = latents.reshape(batch, channels // 4, 2, 2, height, width)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    return latents.reshape(batch, channels // 4, height * 2, width * 2)


class Flux2VAE(Operator):
    """Decode FLUX.2 tokens with its BatchNorm latent statistics."""

    def setup_io(self):
        self.add_execution_mode(
            "decode_latents",
            inputs={"latents": torch.Tensor, "height": int, "width": int},
            outputs={"image": Image},
        )

    @property
    def id(self) -> str:
        return FLUX_2_VAE_ID

    def initialize(
        self, model_path: str, device: Union[str, torch.device]
    ) -> Dict[str, Any]:
        require_pretrained_weights(model_path, self.id)
        vae = AutoencoderKLFlux2.from_pretrained(
            model_path,
            subfolder="vae",
            torch_dtype=torch.bfloat16,
        ).to(device)
        scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        processor = Flux2ImageProcessor(vae_scale_factor=scale_factor * 2)
        return {
            "vae": vae,
            "image_processor": processor,
            "vae_scale_factor": scale_factor,
        }

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
        latents = kwargs["latents"]
        scale_factor = model_components["vae_scale_factor"]
        latent_height = 2 * (int(kwargs["height"]) // (scale_factor * 2))
        latent_width = 2 * (int(kwargs["width"]) // (scale_factor * 2))
        patch_height, patch_width = latent_height // 2, latent_width // 2
        ids = prepare_latent_ids_4d(
            latents.shape[0], patch_height, patch_width, latents.device
        )
        latents = unpack_latents_with_ids(latents, ids, patch_height, patch_width)

        # AutoencoderKLFlux2 replaces a scalar scaling factor with stored BN
        # statistics; apply the inverse normalization before unpatchifying.
        mean = vae.bn.running_mean.view(1, -1, 1, 1).to(
            device=latents.device, dtype=latents.dtype
        )
        std = torch.sqrt(
            vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps
        ).to(device=latents.device, dtype=latents.dtype)
        latents = unpatchify_latents(latents * std + mean)
        image = vae.decode(latents, return_dict=False)[0]
        image = model_components["image_processor"].postprocess(
            image, output_type="pil"
        )
        return {"image": image}
