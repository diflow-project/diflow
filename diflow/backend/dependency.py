"""Pure graph analyses derived from a :class:`Workflow`.

These were methods on :class:`~diflow.backend.coordinator.Coordinator`.
They are extracted here so they can be exercised against a bare ``Workflow``
without standing up a coordinator (ZMQ sockets, workers, event loop) -- the
control-flow expansion tests compare the executor's *view* of two graphs, not
just their shape.

The algorithms are moved verbatim, including the O(N^2) producer search in
:func:`get_node_dependencies`.  Replacing that with a producer index is a
separate, purely-performance change; keeping it identical here means any
regression is unambiguously attributable.
"""

from collections import deque
from typing import Dict, List, Set

from diflow.interface.node_io import SourceType
from diflow.interface.workflow import Workflow
from diflow.interface.workflow_node import WorkflowNode


def get_node_dependencies(
    workflow: Workflow, lazy_only: bool = False
) -> Dict[str, Set[str]]:
    """Map each node name to the names of the nodes producing its inputs.

    With ``lazy_only`` set, only edges whose consumed ``NodeIO`` is marked
    ``lazy`` are reported.
    """
    dependencies = {node.name: set() for node in workflow.workflow_nodes}

    for node in workflow.workflow_nodes:
        for _, input_info in node.get_inputs().items():
            if input_info.source_type == SourceType.INPUT:
                continue
            if lazy_only and not input_info.lazy:
                continue
            # Find which node produces this input
            for other_node in workflow.workflow_nodes:
                if other_node == node:
                    continue
                for _, output_info in other_node.get_outputs().items():
                    if output_info.name == input_info.name:
                        dependencies[node.name].add(other_node.name)

    return dependencies


def build_successors(
    workflow: Workflow, prerequisites: Dict[str, Set[str]]
) -> Dict[str, Set[str]]:
    """Reverse ``prerequisites`` into a successor map."""
    successors = {node.name: set() for node in workflow.workflow_nodes}
    for node_name, deps in prerequisites.items():
        for dep in deps:
            successors[dep].add(node_name)
    return successors


def compute_node_depth(
    workflow: Workflow,
    prerequisites: Dict[str, Set[str]],
    successors: Dict[str, Set[str]],
) -> Dict[str, int]:
    """Depth of each node, counting backwards from the sinks (sinks are 1).

    Multi-source BFS, first visit wins. Used as the ``-node_depth`` tiebreak in
    ready-task ordering, so deeper (closer to a sink) runs first.
    """
    node_depth: Dict[str, int] = {}
    temp_visited = set()

    queue = deque(
        [
            (node.name, 1)
            for node in workflow.workflow_nodes
            if len(successors[node.name]) == 0
        ]
    )
    while queue:
        node_name, depth = queue.popleft()
        if node_name in temp_visited:
            continue
        temp_visited.add(node_name)
        node_depth[node_name] = depth

        for prerequisite in prerequisites[node_name]:
            if prerequisite not in temp_visited:
                queue.append((prerequisite, depth + 1))

    return node_depth


def build_tensor_reference_count(workflow: Workflow) -> Dict[str, int]:
    """Count static consumers per produced tensor.

    One-shot: each consumer decrements exactly once when it completes, so this
    is only correct for a fully-expanded graph in which every node runs once.
    """
    reference_count: Dict[str, int] = {}
    for node in workflow.workflow_nodes:
        for _, input_info in node.get_inputs().items():
            if input_info.source_type == SourceType.INPUT:
                continue
            if input_info.name not in reference_count:
                reference_count[input_info.name] = 1
            else:
                reference_count[input_info.name] += 1
    return reference_count


def topological_sort(workflow: Workflow) -> List[WorkflowNode]:
    """Sort workflow nodes in topological order.

    Raises ``ValueError`` if the graph has a cycle.
    """
    dependencies = get_node_dependencies(workflow)
    sorted_nodes = []
    visited = set()
    temp_visited = set()

    def visit(node_name):
        if node_name in temp_visited:
            raise ValueError("Workflow has cyclic dependencies")
        if node_name in visited:
            return

        temp_visited.add(node_name)
        for dep in dependencies[node_name]:
            visit(dep)
        temp_visited.remove(node_name)
        visited.add(node_name)
        node = next(n for n in workflow.workflow_nodes if n.name == node_name)
        sorted_nodes.append(node)

    for node in workflow.workflow_nodes:
        if node.name not in visited:
            visit(node.name)

    return sorted_nodes
