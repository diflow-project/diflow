import logging
from typing import Any, Dict, Union

import torch

from diflow.operators.base import Operator
from diflow.operators.operator_ids import FLUX_TEXT_ENCODER_ID

logger = logging.getLogger(__name__)


# Deprecated: FluxTextEncoder is deprecated and will be removed in the future
class FluxTextEncoder(Operator):
    def setup_io(self):
        self.add_input("clip_prompt_embeds", torch.Tensor)
        self.add_input("t5_prompt_embeds", torch.Tensor)
        self.add_output("prompt_embeds", torch.Tensor)
        self.add_output("pooled_prompt_embeds", torch.Tensor)
        self.add_output("text_ids", torch.Tensor)

    @property
    def id(self) -> str:
        return FLUX_TEXT_ENCODER_ID

    @torch.no_grad()
    def execute(
        self,
        model_components: Dict[str, Any],
        device: Union[str, torch.device],
        **kwargs,
    ) -> Dict[str, Any]:
        # Images with 1024x1024 resolution:
        # clip_prompt_embeds: (batch_size, 768)
        # t5_prompt_embeds: (batch_size, 512, 4096)
        # text_ids: (512, 3)
        clip_prompt_embeds = kwargs["clip_prompt_embeds"]
        t5_prompt_embeds = kwargs["t5_prompt_embeds"]

        text_ids = torch.zeros(t5_prompt_embeds.shape[1], 3).to(
            device=device, dtype=torch.bfloat16
        )

        return {
            "prompt_embeds": t5_prompt_embeds,
            "pooled_prompt_embeds": clip_prompt_embeds,
            "text_ids": text_ids,
        }
