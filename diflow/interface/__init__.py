from diflow.interface.benchmark import BenchmarkSpec
from diflow.interface.control_flow import cond, for_range
from diflow.interface.denoise_builder import denoise_loop
from diflow.interface.denoise_ops import DenoiseContext
from diflow.interface.workflow import Workflow, register_workflow, run_inference

__all__ = [
    "BenchmarkSpec",
    "Workflow",
    "register_workflow",
    "run_inference",
    # Control flow: build a loop or a branch whose shape depends on the request.
    "for_range",
    "cond",
    # The standard denoising loop, built from the two above. Pass step_fn to
    # replace the per-step computation and keep the scaffolding.
    "denoise_loop",
    "DenoiseContext",
]
