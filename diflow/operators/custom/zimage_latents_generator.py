from typing import Any, Dict, Union

import torch
from diffusers.utils.torch_utils import randn_tensor

from diflow.operators.base import Operator
from diflow.operators.operator_ids import ZIMAGE_LATENTS_GENERATOR_ID


class ZImageLatentsGenerator(Operator):
    """Generate the unpacked float32 latents consumed by Z-Image."""

    def setup_io(self):
        self.add_input("height", int)
        self.add_input("width", int)
        self.add_input("seed", int)
        self.add_output("latents", torch.Tensor)

    @property
    def id(self) -> str:
        return ZIMAGE_LATENTS_GENERATOR_ID

    @torch.no_grad()
    def execute(
        self,
        model_components: Dict[str, Any],
        device: Union[str, torch.device],
        **kwargs,
    ) -> Dict[str, Any]:
        if int(kwargs["height"]) % 16 or int(kwargs["width"]) % 16:
            raise ValueError("Z-Image height and width must be divisible by 16")
        height = 2 * (int(kwargs["height"]) // 16)
        width = 2 * (int(kwargs["width"]) // 16)
        generator = torch.Generator(device=device).manual_seed(kwargs["seed"])
        # Z-Image deliberately keeps scheduler latents in float32 even though
        # the transformer itself runs in bf16.
        latents = randn_tensor(
            (1, 16, height, width),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        return {"latents": latents}
