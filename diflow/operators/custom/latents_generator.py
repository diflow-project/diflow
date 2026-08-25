from typing import Any, Dict, Union

import torch
from diffusers.utils.torch_utils import randn_tensor

from diflow.operators.base import Operator
from diflow.operators.operator_ids import LATENTS_GENERATOR_ID


class LatentsGenerator(Operator):
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
        return LATENTS_GENERATOR_ID

    # Adapted from https://github.com/huggingface/diffusers/blob/3579cd2bb7d2e8f8fa97b198a513f4e02ecccfc1/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py#L632
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
        dtype = kwargs.get("dtype", torch.float16)
        seed = kwargs.get("seed", 0)
        latents = kwargs.get("latents", None)
        vae_scale_factor = kwargs.get("vae_scale_factor", 8)

        if latents is not None:
            return {"latents": latents.to(device=device, dtype=dtype)}

        shape = (
            batch_size,
            num_channels_latents,
            int(height) // vae_scale_factor,
            int(width) // vae_scale_factor,
        )

        generator = torch.manual_seed(seed)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if isinstance(device, str):
            device = torch.device(device)
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        return {"latents": latents}


if __name__ == "__main__":
    latents_generator = LatentsGenerator()
    result = latents_generator.execute(
        model_components={},
        device="cuda",
        batch_size=1,
        num_channels_latents=4,
    )
    # SD1.5
    print(result["latents"].shape)
    print(result["latents"][0][0][0][:10])
