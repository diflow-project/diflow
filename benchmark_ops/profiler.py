"""Automated warmup and per-shape op latency measurement for a registered workflow.

For each resolution the profiler runs the unrolled workflow once at batch size 1 to
capture every op's real inputs, deduplicates structurally identical nodes, then warms
up and times each unique op at every requested batch size.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from benchmark_ops.executor import (
    CapturedNode,
    LocalWorkflowExecutor,
    build_call_kwargs,
    is_batchable,
    resolvable_input_names,
    static_signature_for,
)
from benchmark_ops.results import (
    DEFAULT_RESULTS_DIR,
    OpLatencyRecord,
    build_result,
    load_existing_result,
    result_matches_current_gpu,
    save_result,
    summarize_latencies,
)
from benchmark_ops.shapes import Shape, ShapeSweep, group_by_resolution
from benchmark_ops.workflow_cases import (
    SHAPE_CONTROLLED_INPUTS,
    WorkflowProfileCase,
)
from diflow.interface.workflow import Workflow
from diflow.interface.workflow_expand import expand_workflow

logger = logging.getLogger(__name__)


DEFAULT_PROFILE_STEPS = 2


@dataclass
class ProfileSettings:
    warmup: int = 2
    repeats: int = 5
    # A single denoise step's latency does not depend on the total step count, so the
    # graph is unrolled with very few steps just to keep capture cheap.
    profile_steps: int = DEFAULT_PROFILE_STEPS
    device: str = "cuda"
    offload_idle_models: bool = False
    best_effort: bool = False


def build_inputs(
    case: WorkflowProfileCase, shape: Shape, num_inference_steps: int
) -> Dict[str, Any]:
    """Case inputs with the shape-controlled ones filled in by the profiler."""
    inputs = dict(case.inputs)
    overridden = [name for name in SHAPE_CONTROLLED_INPUTS if name in inputs]
    if overridden:
        logger.debug(
            "Overriding case inputs %s for %s with the profiled shape",
            overridden,
            case.name,
        )

    inputs["height"] = shape.height
    inputs["width"] = shape.width
    inputs["num_inference_steps"] = num_inference_steps
    return inputs


def solve_occurrences(
    workflow: Workflow,
    case: WorkflowProfileCase,
    shape: Shape,
    profile_steps: int,
) -> Dict[Tuple[Any, ...], Dict[str, int]]:
    """How many times each op runs per request, as `constant + per_step * steps`.

    Unrolling is graph construction only -- no operator executes -- so this is cheap
    and needs no GPU. Two step counts are enough to solve the linear relation.
    """
    steps_a = max(1, profile_steps)
    steps_b = steps_a * 2

    counts = []
    for steps in (steps_a, steps_b):
        inputs = build_inputs(case, shape, steps)
        expanded = expand_workflow(workflow, inputs)
        resolvable = resolvable_input_names(expanded.workflow_nodes, inputs)

        step_counts: Dict[Tuple[Any, ...], int] = {}
        for node in expanded.workflow_nodes:
            signature = static_signature_for(node, resolvable)
            step_counts[signature] = step_counts.get(signature, 0) + 1
        counts.append(step_counts)

    counts_a, counts_b = counts
    occurrences: Dict[Tuple[Any, ...], Dict[str, int]] = {}
    for signature in set(counts_a) | set(counts_b):
        count_a = counts_a.get(signature, 0)
        count_b = counts_b.get(signature, 0)
        per_step = (count_b - count_a) // (steps_b - steps_a)
        occurrences[signature] = {
            "constant": count_a - per_step * steps_a,
            "per_step": per_step,
        }
    return occurrences


def _split_occurrences(
    occurrences: Dict[str, int], runtime_count: int, total_runtime_count: int
) -> Dict[str, int]:
    """Attribute a static signature's occurrences across its runtime signatures.

    Almost always one-to-one; the split only matters if one op runs at two different
    tensor shapes within a single request.
    """
    if total_runtime_count == runtime_count:
        return dict(occurrences)

    share = runtime_count / total_runtime_count
    return {
        "constant": round(occurrences.get("constant", 0) * share),
        "per_step": round(occurrences.get("per_step", 0) * share),
    }


def _is_oom_error(error: Exception) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        "out of memory" in str(error).lower()
    )


def measure_captured_node(
    executor: LocalWorkflowExecutor,
    captured_node: CapturedNode,
    shape: Shape,
    warmup: int,
    repeats: int,
    best_effort: bool = False,
) -> Tuple[Optional[List[float]], Optional[int], Optional[str]]:
    """Warm up then time one op at one shape.

    Returns `(samples, gpu_memory_used, error)`. OOM is reported, not raised, so one
    shape falling over does not abandon the rest of the sweep -- the same contract as
    `benchmark.benchmark_utils.run_benchmark_with_oom_handling`.
    """
    node = captured_node.node
    on_cuda = str(executor.device).startswith("cuda")

    try:
        call_kwargs = build_call_kwargs(
            captured_node.raw_inputs,
            captured_node.lazy_input_names,
            shape.batch_size,
        )

        for _ in range(warmup):
            components = executor.components_for_measurement(captured_node)
            executor.run_node(node, call_kwargs, components)

        if on_cuda:
            torch.cuda.synchronize()
            memory_before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()

        samples = []
        for _ in range(repeats):
            # Restoring stateful components must not be timed.
            components = executor.components_for_measurement(captured_node)

            if on_cuda:
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            executor.run_node(node, call_kwargs, components)
            if on_cuda:
                torch.cuda.synchronize()
            samples.append(time.perf_counter() - start_time)

        gpu_memory_used = None
        if on_cuda:
            gpu_memory_used = int(torch.cuda.max_memory_allocated() - memory_before)

        return samples, gpu_memory_used, None

    except Exception as error:  # noqa: BLE001 - reported per shape, see docstring
        is_oom = _is_oom_error(error)
        if not is_oom and not best_effort:
            raise
        error_label = "OOM" if is_oom else f"{type(error).__name__}: {error}"
        logger.warning(
            "Failed profiling %s (mode=%s) at %s: %s",
            node.op.id,
            node.mode,
            shape,
            error_label,
        )
        if on_cuda:
            torch.cuda.empty_cache()
        return None, None, error_label


def _build_record(
    captured_node: CapturedNode,
    shape: Shape,
    occurrences: Dict[str, int],
    samples: Optional[Sequence[float]],
    gpu_memory_used: Optional[int],
    error: Optional[str],
) -> OpLatencyRecord:
    op = captured_node.op
    return OpLatencyRecord(
        op_id=op.id,
        mode=captured_node.node.mode,
        shape=shape,
        model_path=op.config.model_path if op.config is not None else None,
        patches=sorted(patch.id for patch in op.get_patches()),
        occurrences=occurrences,
        input_shapes=captured_node.input_shapes(shape.batch_size),
        latency=summarize_latencies(samples) if samples else None,
        gpu_memory_used=gpu_memory_used,
        error=error,
    )


def profile_resolution(
    executor: LocalWorkflowExecutor,
    workflow: Workflow,
    case: WorkflowProfileCase,
    resolution: Tuple[int, int],
    batch_sizes: Sequence[int],
    settings: ProfileSettings,
    occurrences: Dict[Tuple[Any, ...], Dict[str, int]],
) -> List[OpLatencyRecord]:
    """Capture the graph once at this resolution, then measure every batch size."""
    height, width = resolution
    capture_shape = Shape(batch_size=1, height=height, width=width)
    inputs = build_inputs(case, capture_shape, settings.profile_steps)

    expanded = expand_workflow(workflow, inputs)
    print(
        f"--- {case.name} @ {height}x{width}: capturing "
        f"{len(expanded.workflow_nodes)} nodes "
        f"(num_inference_steps={settings.profile_steps}) ---"
    )
    captured_graph = executor.capture(expanded.workflow_nodes, inputs)
    print(
        f"Captured {len(captured_graph)} unique ops from "
        f"{sum(captured_graph.occurrences.values())} nodes"
    )

    runtime_counts_per_static: Dict[Tuple[Any, ...], int] = {}
    for signature, count in captured_graph.occurrences.items():
        static_signature = signature[0]
        runtime_counts_per_static[static_signature] = (
            runtime_counts_per_static.get(static_signature, 0) + count
        )

    records = []
    for signature, captured_node in captured_graph.nodes.items():
        static_signature = signature[0]
        node_occurrences = _split_occurrences(
            occurrences.get(static_signature, {}),
            captured_graph.occurrences[signature],
            runtime_counts_per_static[static_signature],
        )

        for batch_size in batch_sizes:
            if batch_size > 1 and not is_batchable(captured_node.op):
                # The worker never batches these, so a batched number would be
                # fiction. Recorded at batch size 1 only.
                continue

            shape = Shape(batch_size=batch_size, height=height, width=width)
            samples, gpu_memory_used, error = measure_captured_node(
                executor=executor,
                captured_node=captured_node,
                shape=shape,
                warmup=settings.warmup,
                repeats=settings.repeats,
                best_effort=settings.best_effort,
            )
            record = _build_record(
                captured_node=captured_node,
                shape=shape,
                occurrences=node_occurrences,
                samples=samples,
                gpu_memory_used=gpu_memory_used,
                error=error,
            )
            records.append(record)

            if record.latency is not None:
                print(
                    f"{record.op_id} (mode={record.mode}, {shape}): "
                    f"median {record.latency.median * 1000:.2f} ms "
                    f"over {record.latency.samples} runs"
                )
            else:
                print(f"{record.op_id} (mode={record.mode}, {shape}): {error}")

    # Intermediate activations from capture are no longer needed.
    del captured_graph
    if str(settings.device).startswith("cuda"):
        torch.cuda.empty_cache()

    return records


def profile_case(
    case: WorkflowProfileCase,
    settings: Optional[ProfileSettings] = None,
    sweep: Optional[ShapeSweep] = None,
    results_dir: str = DEFAULT_RESULTS_DIR,
    force_benchmark: bool = False,
    workflow: Optional[Workflow] = None,
) -> Dict[str, Any]:
    """Profile every op of one registered workflow across a shape sweep."""
    settings = settings or ProfileSettings()
    sweep = sweep or case.sweep

    workflow = workflow or case.build_workflow()

    existing_result = load_existing_result(case.name, results_dir)
    if (
        existing_result is not None
        and result_matches_current_gpu(existing_result)
        and not force_benchmark
    ):
        print(
            f"Skipping {case.name}: op latency result already exists for the "
            f"current GPU. Use --force-benchmark to re-run."
        )
        return existing_result

    shapes = sweep.expand()
    grouped_batch_sizes = group_by_resolution(shapes)
    reference_shape = (
        case.reference_shape if sweep is case.sweep else sweep.reference_shape()
    )

    occurrences = solve_occurrences(
        workflow=workflow,
        case=case,
        shape=shapes[0],
        profile_steps=settings.profile_steps,
    )

    executor = LocalWorkflowExecutor(
        device=settings.device,
        offload_idle_models=settings.offload_idle_models,
    )
    records: List[OpLatencyRecord] = []
    profile_errors: List[Dict[str, Any]] = []
    try:
        for resolution, batch_sizes in grouped_batch_sizes.items():
            try:
                records.extend(
                    profile_resolution(
                        executor=executor,
                        workflow=workflow,
                        case=case,
                        resolution=resolution,
                        batch_sizes=batch_sizes,
                        settings=settings,
                        occurrences=occurrences,
                    )
                )
            except Exception as error:
                is_oom = _is_oom_error(error)
                if not is_oom and not settings.best_effort:
                    raise
                error_label = "OOM" if is_oom else f"{type(error).__name__}: {error}"
                profile_errors.append(
                    {
                        "stage": "capture",
                        "height": resolution[0],
                        "width": resolution[1],
                        "error": error_label,
                    }
                )
                logger.warning(
                    "Skipping failed automatic benchmark resolution %sx%s: %s",
                    *resolution,
                    error_label,
                )
                if str(settings.device).startswith("cuda"):
                    torch.cuda.empty_cache()
    finally:
        executor.cleanup()

    result = build_result(
        case_name=case.name,
        workflow_name=workflow.name,
        suite=case.suite,
        records=records,
        reference_shape=reference_shape,
        warmup=settings.warmup,
        repeats=settings.repeats,
        profiled_num_inference_steps=settings.profile_steps,
        device=settings.device,
        model_load_profiles=executor.model_load_profiles(),
        profile_errors=profile_errors,
    )
    save_result(result, case.name, results_dir)
    return result


def plan_case(
    case: WorkflowProfileCase,
    settings: Optional[ProfileSettings] = None,
    sweep: Optional[ShapeSweep] = None,
) -> List[Dict[str, Any]]:
    """Enumerate what a run would measure, without touching the GPU.

    Shapes are not known before execution, so this reports ops per (mode, resolution,
    batch size) from the unrolled graph alone.
    """
    settings = settings or ProfileSettings()
    workflow = case.build_workflow()
    sweep = sweep or case.sweep
    shapes = sweep.expand()

    occurrences = solve_occurrences(
        workflow=workflow,
        case=case,
        shape=shapes[0],
        profile_steps=settings.profile_steps,
    )

    planned = []
    for resolution, batch_sizes in group_by_resolution(shapes).items():
        height, width = resolution
        inputs = build_inputs(
            case,
            Shape(batch_size=1, height=height, width=width),
            settings.profile_steps,
        )
        expanded = expand_workflow(workflow, inputs)
        resolvable = resolvable_input_names(expanded.workflow_nodes, inputs)

        seen = set()
        for node in expanded.workflow_nodes:
            signature = static_signature_for(node, resolvable)
            if signature in seen:
                continue
            seen.add(signature)

            for batch_size in batch_sizes:
                if batch_size > 1 and not is_batchable(node.op):
                    continue
                planned.append(
                    {
                        "op_id": node.op.id,
                        "mode": node.mode,
                        "patches": sorted(patch.id for patch in node.op.get_patches()),
                        "occurrences": occurrences.get(signature, {}),
                        "shape": Shape(
                            batch_size=batch_size, height=height, width=width
                        ),
                    }
                )
    return planned
