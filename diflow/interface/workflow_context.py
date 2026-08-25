from contextlib import contextmanager
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from diflow.interface.region import NodeSink


class WorkflowContext:
    """The sink that ``Operator.__call__`` appends newly built nodes to.

    Usually the :class:`~diflow.interface.workflow.Workflow` being
    authored, but while a control-flow body is traced it is a
    :class:`~diflow.interface.region.RegionBuilder` instead, so the body's
    nodes are collected into the region rather than the top-level graph. The
    method names are kept for compatibility; anything satisfying ``NodeSink``
    works.
    """

    _current_workflow: Optional["NodeSink"] = None

    @classmethod
    def get_current_workflow(cls) -> Optional["NodeSink"]:
        return cls._current_workflow

    @classmethod
    def set_current_workflow(cls, workflow: Optional["NodeSink"]) -> None:
        cls._current_workflow = workflow


@contextmanager
def workflow_context(workflow: "NodeSink"):
    """Install ``workflow`` as the current sink for the duration of the block."""
    previous_workflow = WorkflowContext.get_current_workflow()
    WorkflowContext.set_current_workflow(workflow)
    try:
        yield workflow
    finally:
        WorkflowContext.set_current_workflow(previous_workflow)
