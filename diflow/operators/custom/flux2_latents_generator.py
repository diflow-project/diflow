from typing import Any, Dict, Union

import torch
from diffusers.utils.torch_utils import randn_tensor

from diflow.operators.base import Operator
from diflow.operators.operator_ids import FLUX2_LATENTS_GENERATOR_ID


def pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """Pack ``(B, C, H, W)`` into raster-ordered ``(B, H*W, C)`` tokens."""
    batch_size, channels, height, width = latents.shape
    return latents.reshape(batch_size, channels, height * width).permute(0, 2, 1)


class Flux2LatentsGenerator(Operator):
    """Generate the patchified and packed latent tokens used by FLUX.2."""

    def setup_io(self):
        self.add_input("height", int)
        self.add_input("width", int)
        self.add_input("seed", int)
        self.add_output("latents", torch.Tensor)

    @property
    def id(self) -> str:
        return FLUX2_LATENTS_GENERATOR_ID

    @torch.no_grad()
    def execute(
        self,
        model_components: Dict[str, Any],
        device: Union[str, torch.device],
        **kwargs,
    ) -> Dict[str, Any]:
        latent_height = 2 * (int(kwargs["height"]) // 16)
        latent_width = 2 * (int(kwargs["width"]) // 16)
        # A CPU generator matches the reference pipeline's normal seeded usage
        # and remains deterministic when randn_tensor transfers to CUDA.
        generator = torch.manual_seed(kwargs["seed"])
        latents = randn_tensor(
            (1, 128, latent_height // 2, latent_width // 2),
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        return {"latents": pack_latents(latents)}
