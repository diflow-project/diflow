"""Building workflows with control flow: :func:`for_range` and :func:`cond`.

Both trace their body **once**, while the workflow is being authored, and record
it as a region. The region is expanded into plain nodes per request, once the
trip count and predicates can be evaluated against the concrete inputs.

Loops thread state explicitly through a ``carry``, in the style of
``jax.lax.fori_loop``::

    def body(i, carry):
        latents = carry["latents"]
        timestep = index_op(tensor=timesteps, index=i)
        noise = unet(latents=latents, timestep=timestep, **kwargs)
        return {"latents": scheduler(
            latents=latents, timestep=timestep, noise_pred=noise, mode="step")}

    out = for_range(num_inference_steps, body, carry={"latents": latents})
    image = vae(latents=out["latents"], mode="decode_latents")

Two consequences of tracing once, both of which mirror ``jax``:

* A Python ``if`` inside a body is evaluated at authoring time, so it may only
  test values known then -- ``model.id``, whether an adapter list is empty.
  Anything that depends on the request must use :func:`cond`.
* :func:`cond` traces *both* branches, so neither may raise even when it will not
  be taken.
"""

import logging
import uuid
from typing import Any, Callable, Dict, Optional, Union

from diflow.interface.expr import Expr, to_expr
from diflow.interface.node_io import NodeIO
from diflow.interface.region import (
    CondRegion,
    LoopRegion,
    NodeSink,
    RegionBuilder,
    RegionProgram,
    make_carry_placeholder,
    make_induction_var,
    make_region_result,
)
from diflow.interface.workflow_context import WorkflowContext, workflow_context

logger = logging.getLogger(__name__)

BodyFn = Callable[[NodeIO, Dict[str, NodeIO]], Dict[str, NodeIO]]
BranchFn = Callable[[], Optional[Dict[str, NodeIO]]]


def _current_sink() -> NodeSink:
    sink = WorkflowContext.get_current_workflow()
    if sink is None:
        raise RuntimeError(
            "No active workflow context; construct a Workflow before adding "
            "control flow"
        )
    return sink


def _trace(fn: Callable, *args) -> tuple:
    """Run ``fn`` with a fresh collector installed as the current sink."""
    builder = RegionBuilder()
    with workflow_context(builder):
        returned = fn(*args)
    return builder.program, returned


def _check_io_mapping(mapping: Any, what: str) -> Dict[str, NodeIO]:
    if not isinstance(mapping, dict):
        raise ValueError(f"{what} must be a dict, got {type(mapping).__name__}")
    for key, value in mapping.items():
        if not isinstance(key, str):
            raise ValueError(f"{what} keys must be strings, got {key!r}")
        if not isinstance(value, NodeIO):
            raise ValueError(
                f"{what}[{key!r}] must be a NodeIO (an operator's output), got "
                f"{type(value).__name__}"
            )
    return mapping


def _warn_on_escaped_outputs(
    program: RegionProgram, surfaced: Dict[str, NodeIO], region_id: str
) -> None:
    """Warn about body outputs that nothing can ever read.

    Forgetting to return a value in the carry is the easy mistake here, and it
    manifests as a silently shorter expanded graph rather than an error.
    """
    produced = {
        io.name for node in program.iter_nodes() for io in node.get_outputs().values()
    }
    consumed = {
        io.name
        for node in program.iter_nodes()
        for io in node.get_inputs().values()
        if io is not None
    }
    consumed |= {io.name for io in surfaced.values()}

    # A nested region surfaces some of its body's outputs as its own results;
    # those count as used even though no node in this body reads them directly.
    for region in program.iter_regions():
        if isinstance(region, CondRegion):
            consumed |= {io.name for io in region.then_results.values()}
            consumed |= {io.name for io in (region.else_results or {}).values()}
        elif isinstance(region, LoopRegion):
            consumed |= {io.name for io in region.carry_out.values()}
            consumed |= {io.name for io in region.carry_init.values()}

    escaped = produced - consumed
    if escaped:
        logger.warning(
            "region %s: %d body output(s) are neither consumed inside the region "
            "nor surfaced as results, so they will be unreachable after "
            "expansion: %s. Did you forget to return one in the carry?",
            region_id,
            len(escaped),
            ", ".join(sorted(escaped)),
        )


def for_range(
    trip_count: Union[int, NodeIO, Expr],
    body: BodyFn,
    carry: Dict[str, NodeIO],
    *,
    region_id: Optional[str] = None,
    id_prefix: str = "for",
    iv_name_template: str = "{region_id}_iv_{i}",
) -> Dict[str, NodeIO]:
    """Repeat ``body`` ``trip_count`` times, threading ``carry`` between rounds.

    Args:
        trip_count: How many iterations. A literal, a request-input ``NodeIO``,
            or an :class:`Expr` over request inputs. Node outputs are rejected --
            the graph's shape has to be fixed before it runs.
        body: ``(index, carry) -> carry``. Called once, to trace the body.
            ``index`` is a ``NodeIO`` standing for the loop counter; pass it to
            operators, don't do Python arithmetic on it.
        carry: Initial loop-carried values, keyed by name. Must be non-empty and
            the body must return exactly the same keys.
        region_id: Override the generated region id. Intended for
            compatibility shims that need to reproduce historical names.
        id_prefix: Prefix for the generated region id.
        iv_name_template: Per-iteration name for the induction variable. Must
            contain ``{i}``.

    Returns:
        The carry as it stands after the final iteration, keyed the same way. If
        ``trip_count`` evaluates to 0 these are the initial values.
    """
    sink = _current_sink()
    trip_count_expr = to_expr(trip_count)

    _check_io_mapping(carry, "carry")
    if not carry:
        raise ValueError(
            "for_range requires a non-empty carry. The loop-carried dependency is "
            "the only thing that serializes iterations: the executor schedules "
            "purely on data dependencies, so a carry-free body would expand into "
            "independent iterations it is free to run concurrently -- which would "
            "corrupt any stateful operator in the loop (the diffusers schedulers "
            "track a step index internally)."
        )

    if "{i}" not in iv_name_template:
        raise ValueError(
            f"iv_name_template must contain '{{i}}' so each iteration gets a "
            f"distinct induction variable; got {iv_name_template!r}"
        )

    rid = region_id or f"{id_prefix}_{uuid.uuid4()}"

    induction_var = make_induction_var(rid)
    carry_placeholders = {
        key: make_carry_placeholder(rid, key, io) for key, io in carry.items()
    }

    program, returned = _trace(body, induction_var, dict(carry_placeholders))

    if returned is None:
        raise ValueError(
            f"for_range body must return the updated carry as a dict; got None. "
            f"Expected keys: {sorted(carry)}"
        )
    _check_io_mapping(returned, "the carry returned by the for_range body")
    if set(returned) != set(carry):
        missing = sorted(set(carry) - set(returned))
        extra = sorted(set(returned) - set(carry))
        raise ValueError(
            f"for_range body must return the same carry keys it was given. "
            f"missing={missing} unexpected={extra}"
        )

    _warn_on_escaped_outputs(program, returned, rid)

    results = {key: make_region_result(rid, key, returned[key]) for key in returned}

    sink.add_region(
        LoopRegion(
            id=rid,
            trip_count=trip_count_expr,
            induction_var=induction_var,
            carry_placeholders=carry_placeholders,
            carry_init=dict(carry),
            carry_out=returned,
            body=program,
            results=results,
            iv_name_template=iv_name_template,
        )
    )
    return results


def cond(
    predicate: Union[bool, NodeIO, Expr],
    then_fn: BranchFn,
    else_fn: Optional[BranchFn] = None,
    *,
    region_id: Optional[str] = None,
    id_prefix: str = "cond",
) -> Dict[str, NodeIO]:
    """Emit one of two subgraphs, chosen when the workflow is expanded.

    Only the taken branch contributes nodes to the expanded graph; the region
    itself emits nothing. Both branches are traced while authoring, though, so
    neither may raise.

    Args:
        predicate: A literal, a request-input ``NodeIO``, or an :class:`Expr`.
            Node outputs are rejected -- branching on a value only known while
            running would require the executor to skip nodes, which its one-shot
            tensor reference counting and node-count completion check cannot do.
        then_fn: ``() -> results | None``. Takes no arguments; close over the
            surrounding scope instead.
        else_fn: Optional. If omitted, ``then_fn`` must not return any results,
            since nothing would produce them when the predicate is false.

    Returns:
        The chosen branch's results, keyed as the branches returned them. Empty
        if there are none.
    """
    sink = _current_sink()
    predicate_expr = to_expr(predicate)

    rid = region_id or f"{id_prefix}_{uuid.uuid4()}"

    then_body, then_returned = _trace(then_fn)
    then_results = _check_io_mapping(
        then_returned or {}, "the results returned by the cond then-branch"
    )

    else_body = None
    else_results = None
    if else_fn is not None:
        else_body, else_returned = _trace(else_fn)
        else_results = _check_io_mapping(
            else_returned or {}, "the results returned by the cond else-branch"
        )
        if set(else_results) != set(then_results):
            raise ValueError(
                f"cond branches must return the same result keys; "
                f"then={sorted(then_results)} else={sorted(else_results)}"
            )
    elif then_results:
        raise ValueError(
            f"cond was given results {sorted(then_results)} but no else branch. "
            f"Nothing would produce them when the predicate is false. Either add "
            f"an else branch or return nothing."
        )

    results = {
        key: make_region_result(rid, key, then_results[key]) for key in then_results
    }

    sink.add_region(
        CondRegion(
            id=rid,
            predicate=predicate_expr,
            then_body=then_body,
            then_results=then_results,
            results=results,
            else_body=else_body,
            else_results=else_results,
        )
    )
    return results
