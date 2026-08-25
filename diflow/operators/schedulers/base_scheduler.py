import logging
import time
from abc import abstractmethod
from typing import Any, Dict, Union

import torch

from diflow.operators.base import Operator

logger = logging.getLogger(__name__)


class BaseScheduler(Operator):
    def setup_io(self):
        self.add_execution_mode(
            "init",
            inputs={
                "num_inference_steps": int,
            },
            outputs={
                "timesteps": torch.Tensor,
            },
        )

        self.add_execution_mode(
            "step",
            inputs={
                "latents": torch.Tensor,
                "timestep": torch.Tensor,
                "noise_pred": torch.Tensor,
            },
            outputs={
                "output_latents": torch.Tensor,
            },
        )

        self.add_execution_mode(
            "step_classifier_free_guidance",
            inputs={
                "latents": torch.Tensor,
                "timestep": torch.Tensor,
                "noise_pred_uncond": torch.Tensor,
                "noise_pred_text": torch.Tensor,
                "guidance_scale": float,
            },
            outputs={
                "output_latents": torch.Tensor,
            },
        )

    @abstractmethod
    def id(self) -> str:
        pass

    @abstractmethod
    def initialize(
        self, model_path: str, device: Union[str, torch.device]
    ) -> Dict[str, Any]:
        pass

    @torch.no_grad()
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
            scheduler.set_timesteps(num_inference_steps, device=device)
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
