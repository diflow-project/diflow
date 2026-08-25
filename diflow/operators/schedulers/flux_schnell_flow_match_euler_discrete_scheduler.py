import logging
from typing import Any, Dict, Union

import numpy as np
import torch
from diffusers import schedulers
from overrides import override

from diflow.operators.schedulers.base_scheduler import BaseScheduler
from diflow.operators.utils import require_model_path

logger = logging.getLogger(__name__)


def _calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


class FluxSchnellFlowMatchEulerDiscreteScheduler(BaseScheduler):
    @property
    def id(self) -> str:
        return "FluxSchnellFlowMatchEulerDiscreteScheduler"

    def initialize(
        self, model_path: str, device: Union[str, torch.device]
    ) -> Dict[str, Any]:
        scheduler = schedulers.FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )
        return {"scheduler": scheduler}

    @torch.no_grad()
    @override
    def execute(
        self,
        model_components: Dict[str, Any],
        device: Union[str, torch.device],
        mode: str,
        **kwargs,
    ) -> Dict[str, Any]:
        scheduler = model_components["scheduler"]

        if mode == "init":
            num_inference_steps = kwargs["num_inference_steps"]
            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            image_seq_len = kwargs["latents"].shape[1]
            mu = _calculate_shift(
                image_seq_len,
                scheduler.config.base_image_seq_len,
                scheduler.config.max_image_seq_len,
                scheduler.config.base_shift,
                scheduler.config.max_shift,
            )

            scheduler.set_timesteps(sigmas=sigmas, device=device, mu=mu)

            return {"timesteps": scheduler.timesteps}
        elif mode == "step":
            latents = kwargs["latents"]
            noise_pred = kwargs["noise_pred"]
            timestep = kwargs["timestep"].item()

            # compute the previous noisy sample x_t -> x_t-1
            output_latents = scheduler.step(
                noise_pred, timestep, latents, return_dict=False
            )[0]

            return {"output_latents": output_latents}
        elif mode == "step_classifier_free_guidance":
            latents = kwargs["latents"]
            timestep = kwargs["timestep"].item()
            noise_pred_uncond = kwargs["noise_pred_uncond"]
            noise_pred_text = kwargs["noise_pred_text"]
            guidance_scale = kwargs["guidance_scale"]

            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            # compute the previous noisy sample x_t -> x_t-1
            output_latents = scheduler.step(
                noise_pred, timestep, latents, return_dict=False
            )[0]

            return {"output_latents": output_latents}
        else:
            raise ValueError(f"Invalid execution mode: {self.execution_mode}")


if __name__ == "__main__":
    scheduler = FluxSchnellFlowMatchEulerDiscreteScheduler()
    model_components = scheduler.initialize(
        require_model_path("FLUX.1-schnell"), "cuda"
    )
    result = scheduler.execute(
        model_components=model_components,
        device="cuda",
        mode="init",
        num_inference_steps=4,
        latents=torch.randn(1, 4096, 1024).cuda(),
    )
    print(result["timesteps"])
