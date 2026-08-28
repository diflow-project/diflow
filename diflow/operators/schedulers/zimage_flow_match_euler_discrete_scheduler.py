from typing import Any, Dict, Union

import torch
from diffusers import schedulers
from overrides import override

from diflow.operators.base import require_pretrained_weights
from diflow.operators.operator_ids import (
    ZIMAGE_FLOW_MATCH_EULER_DISCRETE_SCHEDULER_ID,
    ZIMAGE_TURBO_FLOW_MATCH_EULER_DISCRETE_SCHEDULER_ID,
)
from diflow.operators.schedulers.base_scheduler import BaseScheduler


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    return image_seq_len * slope + base_shift - slope * base_seq_len


class ZImageFlowMatchEulerDiscreteScheduler(BaseScheduler):
    """FlowMatch Euler scheduler with Z-Image's sign and CFG conventions."""

    def setup_io(self):
        super().setup_io()
        self.add_execution_mode(
            "init",
            inputs={"num_inference_steps": int, "latents": torch.Tensor},
            outputs={"timesteps": torch.Tensor},
        )

    @property
    def id(self) -> str:
        return ZIMAGE_FLOW_MATCH_EULER_DISCRETE_SCHEDULER_ID

    def initialize(
        self, model_path: str, device: Union[str, torch.device]
    ) -> Dict[str, Any]:
        require_pretrained_weights(model_path, self.id)
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
            latents = kwargs["latents"]
            image_seq_len = (latents.shape[2] // 2) * (latents.shape[3] // 2)
            mu = calculate_shift(
                image_seq_len,
                scheduler.config.get("base_image_seq_len", 256),
                scheduler.config.get("max_image_seq_len", 4096),
                scheduler.config.get("base_shift", 0.5),
                scheduler.config.get("max_shift", 1.15),
            )
            num_inference_steps = kwargs["num_inference_steps"]
            # Match ZImagePipeline's explicit schedule. Letting the scheduler
            # interpolate to sigma_min=0 would make the final denoising step a
            # no-op because FlowMatch also appends a terminal zero sigma.
            sigmas = torch.linspace(
                1.0, 1.0 / num_inference_steps, num_inference_steps
            ).tolist()
            scheduler.set_timesteps(sigmas=sigmas, device=device, mu=mu)
            return {"timesteps": scheduler.timesteps}

        latents = kwargs["latents"]
        timestep = kwargs["timestep"]
        if mode == "step":
            noise_pred = -kwargs["noise_pred"]
        elif mode == "step_classifier_free_guidance":
            positive = kwargs["noise_pred_text"]
            negative = kwargs["noise_pred_uncond"]
            # Z-Image defines CFG relative to the positive prediction rather
            # than the conventional negative-prediction anchor.
            noise_pred = positive + kwargs["guidance_scale"] * (positive - negative)
            noise_pred = -noise_pred
        else:
            raise ValueError(f"Invalid execution mode: {mode}")

        output = scheduler.step(
            noise_pred.to(torch.float32),
            timestep,
            latents,
            return_dict=False,
        )[0]
        return {"output_latents": output}


class ZImageTurboFlowMatchEulerDiscreteScheduler(ZImageFlowMatchEulerDiscreteScheduler):
    """The same scheduler behavior under the Turbo checkpoint identity."""

    @property
    def id(self) -> str:
        return ZIMAGE_TURBO_FLOW_MATCH_EULER_DISCRETE_SCHEDULER_ID
