from typing import Any, Dict, Union

import torch

from diflow.operators.base import Operator
from diflow.operators.operator_ids import INDEXED_TENSOR_ID


class IndexedTensor(Operator):
    def setup_io(self):
        self.add_input("tensor", torch.Tensor)
        self.add_input("index", int)
        self.add_output("indexed_tensor", torch.Tensor)

    @property
    def id(self) -> str:
        return INDEXED_TENSOR_ID

    @torch.no_grad()
    def execute(
        self,
        model_components: Dict[str, Any],
        device: Union[str, torch.device],
        **kwargs,
    ) -> Dict[str, Any]:
        tensor = kwargs["tensor"]
        index = kwargs["index"]
        return {"indexed_tensor": tensor[index : index + 1]}
