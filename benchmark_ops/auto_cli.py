"""Child-process entry point for startup-time workflow profiling."""

from __future__ import annotations

import argparse
import json
from typing import Optional, Sequence

from benchmark_ops.profiler import ProfileSettings, profile_case
from benchmark_ops.shapes import ShapeSweep
from benchmark_ops.workflow_cases import WorkflowProfileCase
from diflow.cli.workflow_loader import load_workflow
from diflow.interface.benchmark import BenchmarkSpec
from diflow.interface.workflow import Workflow


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile one served DiFlow workflow.")
    parser.add_argument("--workflow", required=True)
    parser.add_argument("--workflow-kwargs-json", required=True)
    parser.add_argument("--benchmark-spec-json", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--case-name", required=True)
    parser.add_argument("--best-effort", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    loaded = load_workflow(args.workflow)
    workflow_kwargs = json.loads(args.workflow_kwargs_json)
    workflow = loaded.factory(**workflow_kwargs)
    if not isinstance(workflow, Workflow):
        raise TypeError(
            f"{loaded.source} returned {type(workflow).__name__}; expected Workflow"
        )

    spec = BenchmarkSpec(**json.loads(args.benchmark_spec_json))
    workflow.benchmark = spec
    case = WorkflowProfileCase(
        name=args.case_name,
        suite="auto",
        factory="",
        factory_kwargs=dict(workflow_kwargs),
        inputs=dict(spec.inputs),
        sweep=ShapeSweep(
            batch_sizes=spec.batch_sizes,
            resolutions=spec.resolutions,
        ),
    )
    settings = ProfileSettings(
        warmup=spec.warmup,
        repeats=spec.repeats,
        profile_steps=spec.profile_steps,
        device="cuda",
        offload_idle_models=spec.offload_idle_models,
        best_effort=args.best_effort,
    )
    profile_case(
        case=case,
        settings=settings,
        results_dir=args.results_dir,
        force_benchmark=True,
        workflow=workflow,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
