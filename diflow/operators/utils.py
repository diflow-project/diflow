from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

import torch

if TYPE_CHECKING:
    from diflow.operators.base import Operator


@dataclass
class Config:
    model_path: str | None = None


def test_model_memory_allocation(
    model: "Operator",
    model_path: Union[str, None] = None,
):
    """
    Helper function to test memory allocation before and after model initialization.

    Args:
        model: Model to test
        model_path: Optional path to model weights (None for dummy weights)
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for memory allocation testing")

    device = torch.device("cuda")
    print(f"Device: {device}")

    memory_before = torch.cuda.memory_allocated() / (1024**3)
    print(f"{model.id} - Before initialization: {memory_before:.2f} GiB")

    model_components = model.initialize(model_path=model_path, device=device)

    memory_after = torch.cuda.memory_allocated() / (1024**3)
    print(f"{model.id} - After initialization: {memory_after:.2f} GiB")


MODEL_ROOT_ENV_VAR = "DIFLOW_MODEL_ROOT"


def default_model_path(checkpoint: str) -> "str | None":
    """Where ``checkpoint`` lives, per ``$DIFLOW_MODEL_ROOT``.

    Returns ``None`` when the variable is unset, so a caller can make its
    ``--model-path`` flag required rather than defaulting to a path that only
    existed on one machine. Checkpoint locations are a property of the machine,
    not of the code.
    """
    root = os.environ.get(MODEL_ROOT_ENV_VAR)
    return os.path.join(root, checkpoint) if root else None


def require_model_path(checkpoint: str) -> str:
    """:func:`default_model_path`, for scripts with no flag to fall back on."""
    path = default_model_path(checkpoint)
    if path is None:
        raise RuntimeError(
            f"set {MODEL_ROOT_ENV_VAR} to the directory holding {checkpoint} "
            f"(for example: export {MODEL_ROOT_ENV_VAR}=/path/to/checkpoints)"
        )
    return path


def get_op(op: str, model_path: str | None = None):
    if op == "LatentsGenerator":
        from diflow.operators.custom.latents_generator import LatentsGenerator

        return LatentsGenerator()

    elif op == "FlowMatchEulerDiscreteScheduler":
        from diflow.operators.schedulers.flow_match_euler_discrete_scheduler import (
            FlowMatchEulerDiscreteScheduler,
        )

        return FlowMatchEulerDiscreteScheduler(Config(model_path=model_path))

    elif op == "IndexedTensor":
        from diflow.operators.custom.indexed_tensor import IndexedTensor

        return IndexedTensor()

    elif op == "GuidanceTensor":
        from diflow.operators.custom.guidance_tensor import GuidanceTensor

        return GuidanceTensor()

    elif op == "PNDMScheduler":
        from diflow.operators.schedulers.pndm_scheduler import PNDMScheduler

        return PNDMScheduler(Config(model_path=model_path))

    ### Flux 1.0 Dev
    elif op == "FluxLatentsGenerator":
        from diflow.operators.custom.flux_latents_generator import (
            FluxLatentsGenerator,
        )

        return FluxLatentsGenerator()
    elif op == "CLIP_Flux":
        from diflow.operators.models.text_encoders.clip_flux import CLIP_Flux

        return CLIP_Flux(Config(model_path=model_path))
    elif op == "T5_Flux":
        from diflow.operators.models.text_encoders.t5_flux import T5_Flux

        return T5_Flux(Config(model_path=model_path))
    elif op == "FluxTextEncoder":
        from diflow.operators.custom.flux_text_encoder import FluxTextEncoder

        return FluxTextEncoder()
    elif op == "Flux1VAE":
        from diflow.operators.models.autoencoders.flux_1_vae import Flux1VAE

        return Flux1VAE(Config(model_path=model_path))
    elif op == "Flux1Dev":
        from diflow.operators.models.diffusion_models.flux_1_dev import Flux1Dev

        return Flux1Dev(Config(model_path=model_path))
    elif op == "Flux1Schnell":
        from diflow.operators.models.diffusion_models.flux_1_schnell import (
            Flux1Schnell,
        )

        return Flux1Schnell(Config(model_path=model_path))
    elif op == "FluxFlowMatchEulerDiscreteScheduler":
        from diflow.operators.schedulers.flux_flow_match_euler_discrete_scheduler import (
            FluxFlowMatchEulerDiscreteScheduler,
        )

        return FluxFlowMatchEulerDiscreteScheduler(Config(model_path=model_path))
    elif op == "FluxSchnellFlowMatchEulerDiscreteScheduler":
        from diflow.operators.schedulers.flux_flow_match_euler_discrete_scheduler import (
            FluxSchnellFlowMatchEulerDiscreteScheduler,
        )

        return FluxSchnellFlowMatchEulerDiscreteScheduler(Config(model_path=model_path))
    elif op == "Flux1DevControlNet":
        from diflow.operators.models.adapters.flux_1_dev_controlnet import (
            Flux1DevControlNet,
        )

        return Flux1DevControlNet(Config(model_path=model_path))

    elif op == "Flux1DevControlNetDepth":
        from diflow.operators.models.adapters.flux_1_dev_controlnet import (
            Flux1DevControlNetDepth,
        )

        return Flux1DevControlNetDepth(Config(model_path=model_path))
    elif op == "Flux1DevControlNetCanny":
        from diflow.operators.models.adapters.flux_1_dev_controlnet import (
            Flux1DevControlNetCanny,
        )

        return Flux1DevControlNetCanny(Config(model_path=model_path))

    ### Patches
    raise ValueError(f"Operator with ID {op} not found")
