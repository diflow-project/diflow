"""Canonical form for an expanded workflow graph, ignoring generated names.

Both the old unroller and the new expander mint node names as
``f"{op.id}_{uuid4()}"``, so two runs never agree on names even when they build
the identical graph. To compare them we need a name-free fingerprint.

Each node's signature hashes what the executor actually acts on -- operator id,
model path, patches, execution mode -- plus, recursively, the signatures of the
nodes producing its inputs. Two graphs are isomorphic when their multisets of
signatures agree.

The one subtlety is the induction variable. It reaches a node as a request input
whose *name* contains the region's uuid, so hashing the name would always differ.
Its identity is really the integer value bound to it, so that is what gets
hashed. The set of such names is recovered by diffing the request inputs before
and after expansion, which avoids assuming any naming convention.
"""

import hashlib
from collections import Counter
from typing import Any, Dict, List, Mapping, Set, Tuple

from diflow.interface.node_io import SourceType
from diflow.interface.workflow import Workflow
from diflow.interface.workflow_node import WorkflowNode


def injected_values(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> Dict[str, Any]:
    """Request inputs that expansion added (the induction variables)."""
    return {key: value for key, value in after.items() if key not in before}


def build_producer_index(graph: Workflow) -> Dict[str, Tuple[WorkflowNode, str]]:
    """tensor name -> (producing node, its output key)."""
    index: Dict[str, Tuple[WorkflowNode, str]] = {}
    for node in graph.workflow_nodes:
        for output_key, node_io in node.get_outputs().items():
            index[node_io.name] = (node, output_key)
    return index


def _op_identity(node: WorkflowNode) -> Tuple:
    op = node.op
    return (
        op.id,
        op.config.model_path if getattr(op, "config", None) is not None else None,
        tuple(sorted(patch.id for patch in op.get_patches())),
        node.mode,
    )


def _order_nodes(
    graph: Workflow, producers: Dict[str, Tuple[WorkflowNode, str]]
) -> List[WorkflowNode]:
    """Dependency order, computed iteratively.

    Signatures are built bottom-up rather than by recursion because a denoising
    chain is as deep as the step count times the body size, which would put a
    recursive walk near the interpreter's limit.
    """
    by_name = {node.name: node for node in graph.workflow_nodes}
    prerequisites: Dict[str, Set[str]] = {}
    for node in graph.workflow_nodes:
        deps = set()
        for node_io in node.get_inputs().values():
            if node_io is None or node_io.source_type == SourceType.INPUT:
                continue
            producer = producers.get(node_io.name)
            if producer is not None and producer[0].name != node.name:
                deps.add(producer[0].name)
        prerequisites[node.name] = deps

    remaining = {name: set(deps) for name, deps in prerequisites.items()}
    dependents: Dict[str, Set[str]] = {name: set() for name in remaining}
    for name, deps in prerequisites.items():
        for dep in deps:
            dependents[dep].add(name)

    ready = sorted(name for name, deps in remaining.items() if not deps)
    order: List[WorkflowNode] = []
    while ready:
        name = ready.pop()
        order.append(by_name[name])
        for dependent in sorted(dependents[name]):
            remaining[dependent].discard(name)
            if not remaining[dependent]:
                ready.append(dependent)

    if len(order) != len(graph.workflow_nodes):
        raise ValueError("graph has a dependency cycle; cannot canonicalize")
    return order


def node_signatures(graph: Workflow, injected: Mapping[str, Any]) -> Dict[str, str]:
    """node name -> signature hash."""
    producers = build_producer_index(graph)
    signatures: Dict[str, str] = {}

    for node in _order_nodes(graph, producers):
        edges = []
        for key in sorted(node.get_inputs()):
            node_io = node.get_inputs()[key]
            if node_io is None:
                edges.append((key, "none"))
            elif node_io.source_type == SourceType.INPUT:
                if node_io.name in injected:
                    # Identity is the bound value, not the uuid-bearing name.
                    edges.append((key, "iv", injected[node_io.name]))
                else:
                    edges.append((key, "input", node_io.name))
            else:
                producer = producers.get(node_io.name)
                if producer is None:
                    edges.append((key, "dangling", node_io.name))
                else:
                    producer_node, output_key = producer
                    edges.append(
                        (key, "node", signatures[producer_node.name], output_key)
                    )
        payload = repr((_op_identity(node), tuple(edges)))
        signatures[node.name] = hashlib.sha256(payload.encode()).hexdigest()[:20]

    return signatures


def canonical_form(graph: Workflow, injected: Mapping[str, Any]) -> Dict[str, Any]:
    """A comparable, name-free description of the graph."""
    signatures = node_signatures(graph, injected)
    producers = build_producer_index(graph)

    outputs = set()
    for tensor_name, output_name in graph.outputs.items():
        producer = producers.get(tensor_name)
        if producer is None:
            outputs.add(("unproduced", tensor_name, output_name))
        else:
            producer_node, output_key = producer
            outputs.add((signatures[producer_node.name], output_key, output_name))

    return {
        "node_count": len(graph.workflow_nodes),
        "signatures": Counter(signatures.values()),
        "outputs": outputs,
        "injected": sorted(injected.values()),
    }


def canonical_json(form: Dict[str, Any]) -> Dict[str, Any]:
    """A JSON-serializable rendering of :func:`canonical_form`.

    Used to freeze the graphs the original unroller produced, so the guarantee it
    provided as a test oracle outlives it.
    """
    return {
        "node_count": form["node_count"],
        "signatures": sorted([sig, count] for sig, count in form["signatures"].items()),
        "outputs": sorted(list(entry) for entry in form["outputs"]),
        "injected": list(form["injected"]),
    }


def executor_view(graph: Workflow, injected: Mapping[str, Any]) -> Dict[str, Any]:
    """What the coordinator derives from the graph, keyed by signature.

    Stronger than comparing graph shape: it pins the dependency edges, the lazy
    subset, the tensor reference counts and the scheduling depths. Reference
    counts in particular catch a missing edge that a shape comparison would
    tolerate, because a one-shot refcount that is too low frees a tensor while a
    consumer still needs it.

    Everything is a ``Counter`` of tuples rather than a dict keyed by signature,
    because two nodes with identical signatures are legitimate (and by definition
    interchangeable), and a dict would silently collapse them.
    """
    from diflow.backend import dependency

    signatures = node_signatures(graph, injected)
    producers = build_producer_index(graph)

    prerequisites = dependency.get_node_dependencies(graph)
    lazy_prerequisites = dependency.get_node_dependencies(graph, lazy_only=True)
    successors = dependency.build_successors(graph, prerequisites)
    depth = dependency.compute_node_depth(graph, prerequisites, successors)
    reference_count = dependency.build_tensor_reference_count(graph)

    def edge_counter(mapping):
        return Counter(
            (signatures[name], frozenset(signatures[d] for d in deps))
            for name, deps in mapping.items()
        )

    refcounts = Counter()
    for tensor_name, count in reference_count.items():
        producer = producers.get(tensor_name)
        key = (
            (signatures[producer[0].name], producer[1])
            if producer is not None
            else ("unproduced", tensor_name)
        )
        refcounts[(key, count)] += 1

    return {
        "prerequisites": edge_counter(prerequisites),
        "lazy_prerequisites": edge_counter(lazy_prerequisites),
        "successors": edge_counter(successors),
        "depth": Counter((signatures[n], d) for n, d in depth.items()),
        "reference_counts": refcounts,
        "sinks": Counter(
            signatures[name] for name, succ in successors.items() if not succ
        ),
    }


def describe_node(node: WorkflowNode) -> str:
    inputs = ", ".join(sorted(node.get_inputs()))
    return f"{node.op.id}(mode={node.mode}, inputs=[{inputs}])"


def explain_difference(
    left: Dict[str, Any],
    right: Dict[str, Any],
    left_graph: Workflow,
    right_graph: Workflow,
    left_injected: Mapping[str, Any],
    right_injected: Mapping[str, Any],
    left_label: str = "left",
    right_label: str = "right",
) -> str:
    """Human-readable diff; a bare hash mismatch is unactionable."""
    lines = []
    if left["node_count"] != right["node_count"]:
        lines.append(
            f"node count: {left_label}={left['node_count']} "
            f"{right_label}={right['node_count']}"
        )

    left_sigs = node_signatures(left_graph, left_injected)
    right_sigs = node_signatures(right_graph, right_injected)
    left_by_sig = {sig: name for name, sig in left_sigs.items()}
    right_by_sig = {sig: name for name, sig in right_sigs.items()}
    left_nodes = {n.name: n for n in left_graph.workflow_nodes}
    right_nodes = {n.name: n for n in right_graph.workflow_nodes}

    only_left = left["signatures"] - right["signatures"]
    only_right = right["signatures"] - left["signatures"]

    if only_left:
        lines.append(f"only in {left_label}:")
        for sig, count in sorted(only_left.items()):
            node = left_nodes[left_by_sig[sig]]
            lines.append(f"  {count}x {describe_node(node)}")
    if only_right:
        lines.append(f"only in {right_label}:")
        for sig, count in sorted(only_right.items()):
            node = right_nodes[right_by_sig[sig]]
            lines.append(f"  {count}x {describe_node(node)}")

    if left["outputs"] != right["outputs"]:
        lines.append(
            f"outputs differ: {left_label}={sorted(left['outputs'])} "
            f"{right_label}={sorted(right['outputs'])}"
        )
    if left["injected"] != right["injected"]:
        lines.append(
            f"injected values differ: {left_label}={left['injected']} "
            f"{right_label}={right['injected']}"
        )

    return "\n".join(lines) or "(canonical forms are equal)"
