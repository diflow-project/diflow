"""Helpers for driving the ``workflow_hub/**/register_*.py`` workflows from tests.

Authoring a workflow is fully offline: ``Config`` only carries a ``model_path``
string and operators touch disk in ``initialize()``, which registration never
calls.  So a dummy path is enough to build any hub workflow's graph.

Workflows are identified by their dotted path below ``workflow_hub/``, e.g.
``flux_dev.register_txt2img_workflow``. The family directory is part of the
identifier because the filenames repeat across families.
"""

import ast
import importlib
import inspect
from pathlib import Path
from typing import Any, Dict, List

import workflow_hub

HUB_DIR = Path(workflow_hub.__file__).resolve().parent
DUMMY_MODEL_PATH = "/dummy/model/path"


def standard_loop_workflow_names() -> List[str]:
    """Every hub workflow whose denoise loop is the standard one.

    Discovered by source inspection rather than hard-coded, so adding a workflow
    automatically extends coverage.
    """
    names = []
    for path in sorted(HUB_DIR.rglob("register_*.py")):
        if _calls_denoise_loop_with_the_default_step(path.read_text()):
            names.append(".".join(path.relative_to(HUB_DIR).with_suffix("").parts))
    return names


def _calls_denoise_loop_with_the_default_step(source: str) -> bool:
    """Whether the source calls ``denoise_loop(...)`` and takes the default step.

    Parsed rather than grepped: a comment or docstring mentioning the name would
    otherwise pull a workflow into the oracle comparison, where it mismatches for
    the wrong reason.

    A ``step_fn`` replaces the per-step computation, so such a workflow has no
    business matching the golden graphs. Recognising that from the call rather
    than from a name list is what keeps this free of an exclusion list to
    maintain.
    """
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "denoise_loop"
            and not any(keyword.arg == "step_fn" for keyword in node.keywords)
        ):
            return True
    return False


def build_workflow(name: str, model_path: str = DUMMY_MODEL_PATH):
    """Import a hub workflow and call its ``create_workflow``.

    They are ``__main__``-guarded, so importing has no side effects.

    ``model_path`` defaults to a path that does not exist, which is fine for
    graph-shape tests -- registration never reads the weights. It is NOT fine for
    anything that executes: the operators fall back to randomly-initialised
    models when the path is missing, silently, so a run with the default produces
    noise. Pass a real checkpoint for end-to-end work.

    """
    module = importlib.import_module(f"workflow_hub.{name}")
    signature = inspect.signature(module.create_workflow)
    required = sum(
        1 for p in signature.parameters.values() if p.default is inspect.Parameter.empty
    )
    return module.create_workflow(*([model_path] * required))


def make_inputs(
    workflow, num_inference_steps: int, guidance_scale: float
) -> Dict[str, Any]:
    """Plausible request inputs covering every declared workflow input.

    Only ``num_inference_steps``, ``guidance_scale`` and ``strength`` affect the
    expanded graph's shape; the rest just need to be present.
    """
    values: Dict[str, Any] = {}
    for name, node_io in workflow.inputs.items():
        if name == "num_inference_steps":
            values[name] = num_inference_steps
        elif name in ("guidance_scale", "cfg_guidance_scale", "conditioning_scale"):
            # cfg_guidance_scale is the one that gates the CFG branch on Flux;
            # plain guidance_scale there is the distilled guidance embedding.
            values[name] = guidance_scale
        elif name == "strength":
            values[name] = 0.6
        elif node_io.data_type is int:
            values[name] = 64
        elif node_io.data_type is float:
            values[name] = 1.0
        elif node_io.data_type is str:
            values[name] = "a photo of a cat"
        else:
            values[name] = None
    return values
