from __future__ import annotations

import types
from functools import lru_cache
from typing import Any, Dict, Union, get_args, get_origin

from diflow.operators.utils import get_op

PUBLIC_OPERATOR_IDS = (
    "LatentsGenerator",
    "IndexedTensor",
    "GuidanceTensor",
    "PNDMScheduler",
    "FluxLatentsGenerator",
    "CLIP_Flux",
    "T5_Flux",
    "FluxTextEncoder",
    "Flux1VAE",
    "Flux1Dev",
    "Flux1Schnell",
    "FluxFlowMatchEulerDiscreteScheduler",
    "FluxSchnellFlowMatchEulerDiscreteScheduler",
    "ZImageLatentsGenerator",
    "Qwen3_ZImage",
    "ZImage",
    "ZImageTurbo",
    "ZImageVAE",
    "ZImageFlowMatchEulerDiscreteScheduler",
    "ZImageTurboFlowMatchEulerDiscreteScheduler",
    "Flux2LatentsGenerator",
    "Qwen3_Flux2Klein",
    "Flux2Klein",
    "Flux2VAE",
    "Flux2FlowMatchEulerDiscreteScheduler",
    "Flux1DevControlNet",
    "Flux1DevControlNetDepth",
    "Flux1DevControlNetCanny",
)

# These operators implement defaults inside execute(**kwargs). The base Operator
# interface predates explicit required/default metadata, so publish the known
# optional ports here instead of falsely telling agents that every port is needed.
OPTIONAL_OPERATOR_INPUTS = {
    "LatentsGenerator": {
        "batch_size",
        "num_channels_latents",
        "height",
        "width",
        "dtype",
        "seed",
        "latents",
    },
    "FluxLatentsGenerator": {
        "batch_size",
        "num_channels_latents",
        "height",
        "width",
        "dtype",
        "seed",
        "latents",
    },
    "Flux2LatentsGenerator": {
        "batch_size",
        "num_channels_latents",
        "height",
        "width",
        "dtype",
        "seed",
        "latents",
    },
    "ZImageLatentsGenerator": {
        "batch_size",
        "num_channels_latents",
        "height",
        "width",
        "dtype",
        "seed",
        "latents",
    },
}


def type_name(value: type) -> str:
    origin = get_origin(value)
    if origin in (Union, types.UnionType):
        return " | ".join(type_name(member) for member in get_args(value))
    if origin is list:
        args = get_args(value)
        return f"list[{type_name(args[0])}]" if args else "list"
    if origin is not None:
        args = get_args(value)
        rendered = ", ".join(type_name(item) for item in args)
        return f"{getattr(origin, '__name__', str(origin))}[{rendered}]"
    module = getattr(value, "__module__", "")
    name = getattr(value, "__qualname__", str(value))
    if module in ("", "builtins"):
        return name
    if module == "PIL.Image":
        return "image"
    if module == "torch":
        return f"torch.{name}"
    return f"{module}.{name}"


def _ports(ports: Dict[str, Any], optional: set[str]) -> Dict[str, Any]:
    return {
        name: {
            "type": type_name(io.data_type),
            "required": name not in optional,
            "lazy": io.lazy,
            "size": io.size,
        }
        for name, io in ports.items()
    }


@lru_cache(maxsize=1)
def operator_catalog() -> tuple[Dict[str, Any], ...]:
    catalog = []
    for operator_id in PUBLIC_OPERATOR_IDS:
        try:
            operator = get_op(operator_id, None)
            optional = OPTIONAL_OPERATOR_INPUTS.get(operator_id, set())
            modes = operator.get_execution_modes()
            if modes:
                mode_specs = {
                    mode: {
                        "inputs": _ports(spec["inputs"], set()),
                        "outputs": _ports(spec["outputs"], set()),
                    }
                    for mode, spec in modes.items()
                }
            else:
                mode_specs = {
                    "default": {
                        "inputs": _ports(operator.get_inputs(), optional),
                        "outputs": _ports(operator.get_outputs(), set()),
                    }
                }
            doc = (operator.__class__.__doc__ or "").strip().splitlines()
            catalog.append(
                {
                    "id": operator_id,
                    "description": doc[0].strip() if doc else "",
                    "requires_model_ref": operator.config is not None,
                    "modes": mode_specs,
                    "available": True,
                }
            )
        except Exception as exc:
            catalog.append(
                {
                    "id": operator_id,
                    "available": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return tuple(catalog)


def template_catalog(model_refs: set[str]) -> list[Dict[str, Any]]:
    from diflow.cli.workflow_loader import BUILTIN_WORKFLOWS

    return [
        {
            "id": template_id,
            "description": f"Built-in DiFlow workflow {template_id}",
            "requires_model_ref": True,
            "suggested_model_ref": template_id if template_id in model_refs else None,
            "available": template_id in model_refs,
        }
        for template_id in sorted(BUILTIN_WORKFLOWS)
    ]
