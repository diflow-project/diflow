import logging
from typing import Any, Dict, Union

import torch
from diffusers import schedulers

from diflow.operators.operator_ids import PNDM_SCHEDULER_ID
from diflow.operators.schedulers.base_scheduler import BaseScheduler
from diflow.operators.utils import require_model_path

logger = logging.getLogger(__name__)


class PNDMScheduler(BaseScheduler):
    @property
    def id(self) -> str:
        return PNDM_SCHEDULER_ID

    def initialize(
        self, model_path: str, device: Union[str, torch.device]
    ) -> Dict[str, Any]:
        scheduler = schedulers.PNDMScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )
        return {"scheduler": scheduler}


if __name__ == "__main__":
    scheduler = PNDMScheduler()
    model_components = scheduler.initialize(
        require_model_path("stable-diffusion-v1-5"), "cuda"
    )
    result = scheduler.execute(
        model_components=model_components,
        device="cuda",
        mode="init",
        num_inference_steps=50,
    )
    print(result)
