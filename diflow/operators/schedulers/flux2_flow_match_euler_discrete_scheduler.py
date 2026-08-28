from typing import Any, Dict, Union

import numpy as np
import torch
from diffusers import schedulers
from overrides import override

from diflow.operators.base import require_pretrained_weights
from diflow.operators.flux2_utils import compute_empirical_mu
from diflow.operators.operator_ids import (
    FLUX2_FLOW_MATCH_EULER_DISCRETE_SCHEDULER_ID,
)
from diflow.operators.schedulers.base_scheduler import BaseScheduler


class Flux2FlowMatchEulerDiscreteScheduler(BaseScheduler):
    """FlowMatch Euler scheduler using FLUX.2's empirical timestep shift."""

    def setup_io(self):
        self.add_execution_mode(
            "init",
            inputs={"num_inference_steps": int, "latents": torch.Tensor},
            outputs={"timesteps": torch.Tensor},
        )
        self.add_execution_mode(
            "step",
            inputs={
                "latents": torch.Tensor,
                "timestep": torch.Tensor,
                "noise_pred": torch.Tensor,
            },
            outputs={"output_latents": torch.Tensor},
        )

    @property
    def id(self) -> str:
        return FLUX2_FLOW_MATCH_EULER_DISCRETE_SCHEDULER_ID

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
            steps = kwargs["num_inference_steps"]
            mu = compute_empirical_mu(kwargs["latents"].shape[1], steps)
            if scheduler.config.get("use_flow_sigmas", False):
                scheduler.set_timesteps(steps, device=device, mu=mu)
            else:
                sigmas = np.linspace(1.0, 1 / steps, steps)
                scheduler.set_timesteps(sigmas=sigmas, device=device, mu=mu)
            # Starting from a known index avoids a timestep-search DtoH sync on
            # every request, matching Flux2KleinPipeline.
            scheduler.set_begin_index(0)
            return {"timesteps": scheduler.timesteps}

        if mode != "step":
            raise ValueError(
                "Distilled FLUX.2 Klein only supports the single-pass scheduler step"
            )
        output = scheduler.step(
            kwargs["noise_pred"],
            kwargs["timestep"],
            kwargs["latents"],
            return_dict=False,
        )[0]
        return {"output_latents": output}
