from typing import Any, Dict, Union

import torch

from diflow.operators.base import Operator
from diflow.operators.operator_ids import GUIDANCE_TENSOR_ID


class GuidanceTensor(Operator):
    def setup_io(self):
        self.add_input("guidance_scale", float)
        self.add_output("guidance_tensor", torch.Tensor)

    @property
    def id(self) -> str:
        return GUIDANCE_TENSOR_ID

    @torch.no_grad()
    def execute(
        self,
        model_components: Dict[str, Any],
        device: Union[str, torch.device],
        **kwargs,
    ) -> Dict[str, Any]:
        guidance_scale = kwargs["guidance_scale"]
        return {
            "guidance_tensor": torch.full(
                [1], guidance_scale, device=device, dtype=torch.float32
            )
        }
