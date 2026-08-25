"""Expand control-flow regions into a flat, fully-unrolled graph.

Replaces the hard-coded ``unroll_workflow``. Runs once per request, on the
server, after the workflow has been deserialized -- which is the first moment the
trip counts and predicates can be evaluated.

Unrolling completely is not a shortcut, it is what the executor requires:

* tensor reference counts are static and one-shot (each consumer decrements
  exactly once when it completes), so a node that ran twice would double-free;
* a request is finished when the completed-node count equals the total, so the
  node set has to be known up front;
* readiness is a monotone subset test, so nodes cannot be skipped.

Unlike its predecessor this does not re-run authoring code -- it only clones
nodes that were already traced. So it never touches the global workflow context,
and is re-entrant.
"""

import logging
from collections import ChainMap
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple

from diflow.interface.node_io import NodeIO, SourceType
from diflow.interface.region import CondRegion, LoopRegion, Region, RegionProgram
from diflow.interface.workflow import Workflow
from diflow.interface.workflow_node import WorkflowNode

logger = logging.getLogger(__name__)

Emit = Callable[[WorkflowNode], None]


@dataclass
class Scope:
    """A chain of name -> NodeIO bindings, innermost first.

    A flat dict will not do. Inside a loop body, a nested region may consume
    values produced by *this iteration* (which change every round) while also
    capturing values produced outside the loop (which must not be rewritten).
    The chain resolves the first from an iteration scope and lets the second fall
    through to identity.

    That identity fallthrough is also what keeps per-step index nodes batchable:
    they all keep referring to the one ``timesteps`` tensor produced outside the
    loop, which is what the scheduler's grouping check compares.
    """

    parent: Optional["Scope"] = None
    mapping: Dict[str, NodeIO] = field(default_factory=dict)

    def lookup(self, name: str) -> Optional[NodeIO]:
        scope: Optional["Scope"] = self
        while scope is not None:
            if name in scope.mapping:
                return scope.mapping[name]
            scope = scope.parent
        return None

    def child(self) -> "Scope":
        return Scope(parent=self)


def expand_workflow(workflow: Workflow, inputs: Dict[str, Any]) -> Workflow:
    """Expand ``workflow`` for one request.

    Induction-variable values are added to ``inputs`` in place, matching the
    previous behaviour: the coordinator hands the same dict to every task, and
    resolves those synthetic names there.
    """
    expanded, injected = expand_workflow_pure(workflow, inputs)
    inputs.update(injected)
    return expanded


def expand_workflow_pure(
    workflow: Workflow, inputs: Mapping[str, Any]
) -> Tuple[Workflow, Dict[str, Any]]:
    """Side-effect-free core: returns the expanded graph and the values to inject."""
    subst: Dict[str, NodeIO] = {}
    injected: Dict[str, Any] = {}
    emitted: List[WorkflowNode] = []

    # Pass 1: plain nodes, names unchanged. Their inputs get a fresh dict because
    # pass 3 rewrites them, and the registered workflow must not be mutated.
    for node in workflow.workflow_nodes:
        emitted.append(_copy_preserving_name(node))

    # Pass 2: regions, in authoring order. Each consults `subst` so a later
    # region can consume an earlier one's results.
    root = Scope()
    for region in workflow.regions:
        _expand_region(region, inputs, {}, root, subst, emitted.append, injected)

    # Pass 3: point pass-1 consumers at the real nodes that replaced the region
    # result placeholders.
    for node in emitted:
        for key, node_io in list(node.get_inputs().items()):
            if node_io is not None and node_io.name in subst:
                node.set_input(key, subst[node_io.name])

    expanded = _new_workflow_like(workflow)
    expanded.workflow_nodes = emitted
    # Pass 4: an output produced inside a region body is registered under the
    # placeholder name, so remap it too.
    expanded.outputs = {
        (subst[key].name if key in subst else key): value
        for key, value in workflow.outputs.items()
    }

    _check_invariants(expanded, workflow, inputs, injected)

    logger.debug(
        "expanded workflow %s: %d node(s) from %d plain node(s) and %d region(s); "
        "injected %d induction value(s)",
        workflow.name,
        len(emitted),
        len(workflow.workflow_nodes),
        len(workflow.regions),
        len(injected),
    )
    return expanded, injected


# --------------------------------------------------------------------------- #
# Node copying
# --------------------------------------------------------------------------- #


def _new_workflow_like(workflow: Workflow) -> Workflow:
    """A bare Workflow carrying over identity but no nodes.

    Bypasses ``__init__`` on purpose: it would install the new object as the
    global authoring sink, which has nothing to do with expanding a request.
    """
    expanded = Workflow.__new__(Workflow)
    expanded.name = workflow.name
    expanded.benchmark = getattr(workflow, "benchmark", None)
    expanded.workflow_nodes = []
    expanded.inputs = dict(workflow.inputs)
    expanded.outputs = {}
    expanded.regions = []
    return expanded


def _copy_preserving_name(node: WorkflowNode) -> WorkflowNode:
    """Shallow-copy a top-level node, keeping its name and output identities."""
    return WorkflowNode(
        op=node.op,
        inputs=dict(node.get_inputs()),
        outputs=dict(node.get_outputs()),
        mode=node.mode,
        name=node.name,
    )


def _resolve(
    node_io: Optional[NodeIO], scope: Scope, subst: Dict[str, NodeIO]
) -> Optional[NodeIO]:
    """Innermost scope, then outer scopes, then region results, then identity."""
    if node_io is None:
        return None
    found = scope.lookup(node_io.name)
    if found is not None:
        return found
    if node_io.name in subst:
        return subst[node_io.name]
    return node_io


def _clone(src: WorkflowNode, scope: Scope, subst: Dict[str, NodeIO]) -> WorkflowNode:
    """Copy a body node with its inputs rewired for the current iteration.

    ``outputs=None``/``name=None`` make ``WorkflowNode.__init__`` re-derive
    everything: a fresh ``f"{op.id}_{uuid4()}"`` name and matching output IOs.
    That is deliberate rather than incidental -- it is what keeps names unique per
    request and keeps the op id recoverable from the name (the coordinator's
    overload estimator reads it back with ``name.split("_")[0]``). There is no
    code path here that mints a name any other way.

    The operator object is shared, not copied, exactly as the old unroller shared
    one model across all steps. Operator identity is never used for anything: the
    worker rebuilds the op from its serialized form for every task.
    """
    resolved = {
        key: _resolve(node_io, scope, subst)
        for key, node_io in src.get_inputs().items()
    }
    clone = WorkflowNode(
        op=src.op, inputs=resolved, outputs=None, mode=src.mode, name=None
    )
    if set(clone.get_outputs()) != set(src.get_outputs()):
        raise ValueError(
            f"cloning {src.name} produced outputs "
            f"{sorted(clone.get_outputs())} but the original had "
            f"{sorted(src.get_outputs())}"
        )
    return clone


# --------------------------------------------------------------------------- #
# Region expansion
# --------------------------------------------------------------------------- #


def _expand_program(
    program: RegionProgram,
    inputs: Mapping[str, Any],
    env: Mapping[str, Any],
    scope: Scope,
    subst: Dict[str, NodeIO],
    emit: Emit,
    injected: Dict[str, Any],
) -> None:
    for op in program.ops:
        if isinstance(op, WorkflowNode):
            clone = _clone(op, scope, subst)
            for key, source_io in op.get_outputs().items():
                scope.mapping[source_io.name] = clone.get_outputs()[key]
            emit(clone)
        else:
            _expand_region(op, inputs, env, scope, subst, emit, injected)


def _expand_region(
    region: Region,
    inputs: Mapping[str, Any],
    env: Mapping[str, Any],
    scope: Scope,
    subst: Dict[str, NodeIO],
    emit: Emit,
    injected: Dict[str, Any],
) -> None:
    if isinstance(region, LoopRegion):
        _expand_loop(region, inputs, env, scope, subst, emit, injected)
    elif isinstance(region, CondRegion):
        _expand_cond(region, inputs, env, scope, subst, emit, injected)
    else:
        raise ValueError(f"cannot expand region of type {type(region).__name__}")


def _expand_loop(
    region: LoopRegion,
    inputs: Mapping[str, Any],
    env: Mapping[str, Any],
    scope: Scope,
    subst: Dict[str, NodeIO],
    emit: Emit,
    injected: Dict[str, Any],
) -> None:
    trip_count = region.trip_count.eval(ChainMap(dict(env), dict(inputs)))
    if isinstance(trip_count, bool) or not isinstance(trip_count, int):
        raise ValueError(
            f"loop {region.id}: trip count must evaluate to an int, got "
            f"{trip_count!r} ({type(trip_count).__name__})"
        )
    if trip_count < 0:
        raise ValueError(
            f"loop {region.id}: trip count evaluated to {trip_count}, which is negative"
        )

    current = {
        key: _resolve(node_io, scope, subst)
        for key, node_io in region.carry_init.items()
    }

    for iteration in range(trip_count):
        iteration_scope = scope.child()

        # The induction variable becomes a per-iteration request input, the same
        # channel the old unroller used for the timestep index.
        iv_name = region.iv_name(iteration)
        injected[iv_name] = iteration
        iteration_scope.mapping[region.induction_var.name] = NodeIO(
            name=iv_name,
            data_type=region.induction_var.data_type,
            source_type=SourceType.INPUT,
        )

        for key, placeholder in region.carry_placeholders.items():
            iteration_scope.mapping[placeholder.name] = current[key]

        _expand_program(
            region.body,
            inputs,
            ChainMap({region.id: iteration}, dict(env)),
            iteration_scope,
            subst,
            emit,
            injected,
        )

        current = {
            key: _resolve(node_io, iteration_scope, subst)
            for key, node_io in region.carry_out.items()
        }

    # With a trip count of 0 this aliases the initial values, so downstream
    # consumers see the loop's input rather than a dangling reference.
    for key, placeholder in region.results.items():
        subst[placeholder.name] = current[key]


def _expand_cond(
    region: CondRegion,
    inputs: Mapping[str, Any],
    env: Mapping[str, Any],
    scope: Scope,
    subst: Dict[str, NodeIO],
    emit: Emit,
    injected: Dict[str, Any],
) -> None:
    taken = bool(region.predicate.eval(ChainMap(dict(env), dict(inputs))))
    if taken:
        body, branch_results = region.then_body, region.then_results
    else:
        body, branch_results = region.else_body, region.else_results

    if body is None:
        # No else branch; a cond without one has no results to bind.
        return

    branch_scope = scope.child()
    _expand_program(body, inputs, env, branch_scope, subst, emit, injected)

    for key, placeholder in region.results.items():
        subst[placeholder.name] = _resolve(
            (branch_results or {})[key], branch_scope, subst
        )


# --------------------------------------------------------------------------- #
# Invariants
# --------------------------------------------------------------------------- #


def _all_placeholder_names(workflow: Workflow) -> Set[str]:
    names: Set[str] = set()
    for region in workflow.regions:
        names |= region.placeholder_names()
    return names


def _check_invariants(
    expanded: Workflow,
    source: Workflow,
    inputs: Mapping[str, Any],
    injected: Mapping[str, Any],
) -> None:
    """Fail loudly on a malformed expansion.

    Worth the cost because the executor will not: when an input has no producer
    and is missing from the request, ``Coordinator._prepare_task_inputs`` logs an
    error and carries on, so the symptom is a confusing failure inside an
    operator on a worker -- or a silently wrong result.
    """
    nodes = expanded.workflow_nodes

    names = [node.name for node in nodes]
    duplicates = {name for name in names if names.count(name) > 1}
    if duplicates:
        raise ValueError(
            f"expanded workflow {expanded.name} has duplicate node names: "
            f"{sorted(duplicates)}. Task ids are request id + node name, so "
            f"duplicates collide across the coordinator's bookkeeping."
        )

    for node in nodes:
        if not node.name.startswith(node.op.id):
            raise ValueError(
                f"node {node.name!r} does not start with its operator id "
                f"{node.op.id!r}; the overload estimator recovers the op with "
                f"name.split('_')[0]"
            )

    produced: Dict[str, WorkflowNode] = {}
    for node in nodes:
        for node_io in node.get_outputs().values():
            produced[node_io.name] = node

    placeholders = _all_placeholder_names(source)
    request_values = set(inputs) | set(injected)
    # An INPUT-sourced edge may also be satisfied by a registered workflow output,
    # which is how non-tensor intermediates reach downstream nodes
    # (Coordinator._prepare_task_inputs falls back to request_required_outputs).
    resolvable_inputs = request_values | set(expanded.outputs)

    for node in nodes:
        for key, node_io in node.get_inputs().items():
            if node_io is None:
                continue
            if node_io.name in placeholders:
                raise ValueError(
                    f"{node.name}.{key} still refers to the unexpanded "
                    f"placeholder {node_io.name!r}; a carry or region result was "
                    f"not substituted"
                )
            if node_io.source_type == SourceType.INPUT:
                if node_io.name not in resolvable_inputs:
                    raise ValueError(
                        f"{node.name}.{key} reads request input "
                        f"{node_io.name!r}, which was not supplied"
                    )
            elif node_io.name not in produced:
                raise ValueError(
                    f"{node.name}.{key} consumes {node_io.name!r}, which no node "
                    f"produces"
                )

    for name in produced:
        if name in placeholders:
            raise ValueError(
                f"expanded graph produces {name!r}, which is a region placeholder"
            )

    for output_name in expanded.outputs:
        # Deliberately checked against request_values, not resolvable_inputs:
        # the latter contains the output names themselves, which would make this
        # vacuous.
        if output_name not in produced and output_name not in request_values:
            raise ValueError(
                f"workflow output {output_name!r} has no producer in the expanded "
                f"graph"
            )

    _check_acyclic(expanded, produced)


def _check_acyclic(expanded: Workflow, produced: Dict[str, WorkflowNode]) -> None:
    """Verify a topological order exists.

    The executor advances only on completion events, so a cycle would not raise
    anything, it would simply never finish.

    Iterative (Kahn) rather than recursive: a denoising chain is as long as the
    step count times the body size, which would put a depth-first walk within
    reach of the recursion limit.
    """
    prerequisites: Dict[str, Set[str]] = {}
    for node in expanded.workflow_nodes:
        deps = set()
        for node_io in node.get_inputs().values():
            if node_io is None or node_io.source_type == SourceType.INPUT:
                continue
            producer = produced.get(node_io.name)
            if producer is not None and producer.name != node.name:
                deps.add(producer.name)
        prerequisites[node.name] = deps

    remaining = {name: set(deps) for name, deps in prerequisites.items()}
    dependents: Dict[str, Set[str]] = {name: set() for name in remaining}
    for name, deps in prerequisites.items():
        for dep in deps:
            dependents[dep].add(name)

    ready = [name for name, deps in remaining.items() if not deps]
    settled = 0
    while ready:
        name = ready.pop()
        settled += 1
        for dependent in dependents[name]:
            remaining[dependent].discard(name)
            if not remaining[dependent]:
                ready.append(dependent)

    if settled != len(remaining):
        stuck = sorted(name for name, deps in remaining.items() if deps)
        raise ValueError(
            f"expanded workflow has a dependency cycle involving "
            f"{stuck[:10]}{' ...' if len(stuck) > 10 else ''}"
        )
