from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Dict

import torch

from diflow.operators.base import Operator

if TYPE_CHECKING:  # pragma: no cover
    from diflow.interface.denoise_ops import DenoiseContext


class BaseDiffusionModel(Operator):
    @abstractmethod
    def id(self) -> str:
        pass

    def setup_io(self):
        self.add_input("latents", torch.Tensor)
        self.add_input("timestep", torch.Tensor)
        self.add_input("prompt_embeds", torch.Tensor)
        self.add_input("pooled_prompt_embeds", torch.Tensor)
        self.add_output("noise_pred", torch.Tensor)

    def denoise_step_kwargs(self, context: "DenoiseContext") -> Dict[str, Any]:
        """Inputs this model needs per denoising step beyond the shared four.

        ``latents``, ``timestep``, ``prompt_embeds`` and ``pooled_prompt_embeds``
        are passed for every family and are not repeated here. Override to draw
        anything else out of the context -- Flux needs its guidance embedding and
        the image dimensions, for instance.

        Returning ``{}`` is a normal answer, not a missing implementation: several
        families need nothing else. This is what replaced a dispatch on model id in
        ``interface/denoise_ops.py``, so a new family is now an operator rather
        than another branch there.
        """
        return {}
