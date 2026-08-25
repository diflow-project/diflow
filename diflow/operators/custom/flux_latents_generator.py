import logging
from typing import Any, Dict, Union

import torch
from diffusers.utils.torch_utils import randn_tensor

from diflow.operators.base import Operator
from diflow.operators.operator_ids import FLUX_LATENTS_GENERATOR_ID


def _pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(
        batch_size, num_channels_latents, height // 2, 2, width // 2, 2
    )
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(
        batch_size, (height // 2) * (width // 2), num_channels_latents * 4
    )

    return latents


class FluxLatentsGenerator(Operator):
    def setup_io(self):
        self.add_input("batch_size", int)
        self.add_input("num_channels_latents", int)
        self.add_input("height", int)
        self.add_input("width", int)
        self.add_input("dtype", torch.dtype)
        self.add_input("seed", int)
        self.add_input("latents", torch.Tensor)
        self.add_output("latents", torch.Tensor)

    @property
    def id(self) -> str:
        return FLUX_LATENTS_GENERATOR_ID

    @torch.no_grad()
    def execute(
        self,
        model_components: Dict[str, Any],
        device: Union[str, torch.device],
        **kwargs,
    ) -> Dict[str, Any]:
        batch_size = kwargs.get("batch_size", 1)
        num_channels_latents = kwargs.get("num_channels_latents", 16)
        height = kwargs.get("height", 512)
        width = kwargs.get("width", 512)
        dtype = kwargs.get("dtype", torch.bfloat16)
        seed = kwargs.get("seed", 0)
        latents = kwargs.get("latents", None)
        vae_scale_factor = kwargs.get("vae_scale_factor", 8)

        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))

        shape = (batch_size, num_channels_latents, height, width)

        if latents is not None:
            return {
                "latents": latents.to(device=device, dtype=dtype),
            }

        generator = torch.manual_seed(seed)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if isinstance(device, str):
            device = torch.device(device)
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = _pack_latents(
            latents, batch_size, num_channels_latents, height, width
        )

        return {
            "latents": latents,
        }


if __name__ == "__main__":
    flux_latents_generator = FluxLatentsGenerator()
    # Flux
    result = flux_latents_generator.execute(
        model_components={},
        device="cuda",
        batch_size=1,
        num_channels_latents=16,
        height=1024,
        width=1024,
        dtype=torch.bfloat16,
        seed=0,
        latents=None,
        vae_scale_factor=8,
    )
    print(result["latents"].shape)
    print(result["latents"][0][0][:10])
