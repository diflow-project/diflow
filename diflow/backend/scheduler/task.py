"""
Task representation for workflow execution and scheduling.

This module defines the Task dataclass that represents a single task
in a workflow execution, containing all necessary information for
both coordination and scheduling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from diflow.interface.workflow_node import WorkflowNode


def get_task_id(task_group: List[Task] | List[Dict[str, Any]]) -> str:
    """Get the task ID for a task group."""
    if isinstance(task_group[0], Task):
        return task_group[0].request_id + "_" + task_group[0].workflow_node.name
    elif isinstance(task_group[0], Dict):
        return task_group[0]["request_id"] + "_" + task_group[0]["node_name"]
    else:
        raise ValueError(f"Invalid task group type: {type(task_group[0])}")


@dataclass
class Task:
    """
    Task representation for workflow execution and scheduling.

    A Task represents a single workflow node that needs to be executed,
    containing all the necessary information for coordination and scheduling.
    """

    request_id: str
    workflow_node: "WorkflowNode"
    inputs: Dict[str, Any] = None
    # map input_name -> input_info.name
    input_map: Dict[str, str] = None
    # map output_name -> output_info.name
    output_map: Dict[str, str] = None
    # map input_name -> input_value
    node_inputs: Dict[str, Any] = None
    # map input_name -> {worker_rank: tensor_info}
    node_input_locations: Dict[str, str] = None
    required_outputs: List[str] = None
    # Lazy input tracking
    # map input_name -> input_info.name for lazy inputs that need to be provided at execution time
    lazy_inputs: Dict[str, str] = None
