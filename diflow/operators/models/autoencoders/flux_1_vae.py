import logging
from typing import Any, Dict, List, Union

import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders import AutoencoderKL
from PIL.Image import Image

from diflow.operators.base import Operator, has_pretrained_weights
from diflow.operators.operator_ids import FLUX_1_VAE_ID
from diflow.operators.utils import test_model_memory_allocation

logger = logging.getLogger(__name__)


class Flux1VAE(Operator):
    def setup_io(self):
        # Add execution modes with their IO specifications
        self.add_execution_mode(
            "prepare_image",
            inputs={
                "image": Image,
                "width": int,
                "height": int,
            },
            outputs={"image_embedding": torch.Tensor},
        )
        self.add_execution_mode(
            "decode_latents", inputs={"latents": torch.Tensor}, outputs={"image": Image}
        )

    @property
    def id(self) -> str:
        return FLUX_1_VAE_ID

    def initialize(
        self, model_path: Union[str, None], device: Union[str, torch.device]
    ) -> Dict[str, Any]:
        if has_pretrained_weights(model_path, self.id):
            vae = AutoencoderKL.from_pretrained(
                model_path,
                subfolder="vae",
                torch_dtype=torch.bfloat16,
            ).to(device)
        else:
            config = self._default_dummy_config()
            vae = AutoencoderKL(**config).to(device=device, dtype=torch.bfloat16)

        self.vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

        image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)

        return {
            "vae": vae,
            "image_processor": image_processor,
        }

    @staticmethod
    def _default_dummy_config() -> Dict[str, Any]:
        return {
            "act_fn": "silu",
            "block_out_channels": [128, 256, 512, 512],
            "down_block_types": [
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
            ],
            "force_upcast": True,
            "in_channels": 3,
            "latent_channels": 16,
            "latents_mean": None,
            "latents_std": None,
            "layers_per_block": 2,
            "mid_block_add_attention": True,
            "norm_num_groups": 32,
            "out_channels": 3,
            "sample_size": 1024,
            "scaling_factor": 0.3611,
            "shift_factor": 0.1159,
            "up_block_types": [
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
            ],
            "use_post_quant_conv": False,
            "use_quant_conv": False,
        }

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

        return latents

    @torch.no_grad()
    def execute(
        self,
        model_components: Dict[str, Any],
        device: Union[str, torch.device],
        mode: str,
        **kwargs,
    ) -> Dict[str, Any]:
        vae = model_components["vae"]
        image_processor = model_components["image_processor"]

        if mode == "prepare_image":
            image = kwargs["image"]
            width = kwargs["width"]
            height = kwargs["height"]

            # Preprocess image
            if isinstance(image, torch.Tensor):
                pass
            else:
                image = image_processor.preprocess(image, height=height, width=width)

            image = image.to(device=device, dtype=torch.bfloat16)

            return {"image_embedding": image}

        elif mode == "decode_latents":
            # Images with 1024x1024 resolution:
            # latents: (batch_size, 4096, 64)
            # height: 1024
            # width: 1024
            latents = kwargs["latents"]
            height = kwargs["height"]
            width = kwargs["width"]

            latents = self._unpack_latents(
                latents, height, width, self.vae_scale_factor
            )
            latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
            image = vae.decode(latents, return_dict=False)[0]
            image = image_processor.postprocess(image, output_type="pil")

            return {"image": image}
        else:
            raise ValueError(f"Invalid execution mode: {mode}")


if __name__ == "__main__":
    test_model_memory_allocation(model=Flux1VAE(), model_path=None)
