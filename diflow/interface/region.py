"""Control-flow regions: a serializable subgraph plus a loop or branch around it.

A region is authored by tracing its body **once** (see
:mod:`diflow.interface.control_flow`) and expanded into ordinary nodes at
request time (see :mod:`diflow.interface.workflow_expand`). Tracing once
is forced by the wire format: a workflow is registered as JSON, so the body has
to survive serialization as *structure* -- a Python callable cannot be shipped to
the server. That is also why a Python ``if`` inside a body only works on values
known while authoring (``model.id``, whether an adapter list is empty); anything
that depends on the request must go through :func:`~control_flow.cond`.

Bodies hold real :class:`WorkflowNode` objects, which already round-trip through
``to_dict``/``from_dict``. So region serialization is plain recursion with no new
payload format, and expansion reduces to cloning nodes and resolving names.

Three placeholder flavours stand in for values that only exist during expansion.
All reuse the existing :class:`SourceType` members so nothing downstream has to
learn a new edge kind:

``induction variable``
    An ``INPUT``-sourced ``NodeIO``. The expander renames it per iteration and
    injects the integer into the request inputs, which is how the old hard-coded
    unroller passed the timestep index too.
``carry placeholder``
    A ``NODE``-sourced ``NodeIO`` naming a node that never exists. It must be
    substituted away entirely; ``NODE`` makes a leak *detectable*, since the
    expander's invariant check can see it has no producer.
``region result``
    Also ``NODE``-sourced, and shaped exactly like the old
    ``DenoiseNode.denoised_latents`` forward reference, so patching downstream
    consumers is the same name-equality sweep as before.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    Iterator,
    List,
    Optional,
    Protocol,
    Set,
    Union,
)

from diflow.interface.expr import Expr
from diflow.interface.node_io import NodeIO, SourceType
from diflow.interface.workflow_node import WorkflowNode

if TYPE_CHECKING:  # pragma: no cover
    pass

# region_kind -> class, populated at the bottom of the module.
_REGION_REGISTRY: Dict[str, type] = {}


class NodeSink(Protocol):
    """Anything operator calls can append to.

    ``Operator.__call__`` looks up the current sink and calls
    ``add_workflow_node``; that is the whole hook. A :class:`Workflow` is the
    top-level sink, and a :class:`RegionBuilder` is the sink while a region body
    is being traced.
    """

    def add_workflow_node(self, workflow_node: WorkflowNode) -> None: ...

    def add_region(self, region: "Region") -> None: ...


# --------------------------------------------------------------------------- #
# Placeholder construction
# --------------------------------------------------------------------------- #


def make_induction_var(region_id: str) -> NodeIO:
    """The loop counter, as a request input the expander materializes per step."""
    return NodeIO(
        name=f"{region_id}_iv",
        data_type=int,
        source_type=SourceType.INPUT,
    )


def make_carry_placeholder(region_id: str, key: str, like: NodeIO) -> NodeIO:
    """The value a body reads at the top of each iteration."""
    return NodeIO(
        name=f"{region_id}_carry:{key}",
        data_type=like.data_type,
        source_type=SourceType.NODE,
        source_node=f"{region_id}_carry",
        size=like.size,
        lazy=like.lazy,
    )


def make_region_result(region_id: str, key: str, like: NodeIO) -> NodeIO:
    """The forward reference downstream nodes consume before expansion."""
    return NodeIO(
        name=f"{region_id}:{key}",
        data_type=like.data_type,
        source_type=SourceType.NODE,
        source_node=region_id,
        size=like.size,
        lazy=like.lazy,
    )


def _ios_to_dict(ios: Dict[str, NodeIO]) -> Dict[str, Any]:
    return {key: io.to_dict() for key, io in ios.items()}


def _ios_from_dict(data: Dict[str, Any]) -> Dict[str, NodeIO]:
    return {key: NodeIO.from_dict(value) for key, value in data.items()}


# --------------------------------------------------------------------------- #
# Program (an ordered body)
# --------------------------------------------------------------------------- #


@dataclass
class RegionProgram:
    """An ordered list of nodes and nested regions, in authoring order."""

    ops: List[Union[WorkflowNode, "Region"]] = field(default_factory=list)

    def to_dict(self) -> List[Dict[str, Any]]:
        items = []
        for op in self.ops:
            if isinstance(op, WorkflowNode):
                items.append({"op_kind": "node", "node": op.to_dict()})
            else:
                items.append({"op_kind": "region", "region": op.to_dict()})
        return items

    @classmethod
    def from_dict(cls, items: List[Dict[str, Any]]) -> "RegionProgram":
        ops: List[Union[WorkflowNode, "Region"]] = []
        for item in items:
            op_kind = item.get("op_kind")
            if op_kind == "node":
                ops.append(WorkflowNode.from_dict(item["node"]))
            elif op_kind == "region":
                ops.append(Region.from_dict(item["region"]))
            else:
                raise ValueError(
                    f"unknown body item kind {op_kind!r}; expected 'node' or 'region'"
                )
        return cls(ops=ops)

    def iter_nodes(self) -> Iterator[WorkflowNode]:
        """Every node in this body, including those inside nested regions."""
        for op in self.ops:
            if isinstance(op, WorkflowNode):
                yield op
            else:
                for program in op.subprograms():
                    yield from program.iter_nodes()

    def iter_regions(self) -> Iterator["Region"]:
        """Every nested region, depth-first."""
        for op in self.ops:
            if not isinstance(op, WorkflowNode):
                yield op
                for program in op.subprograms():
                    yield from program.iter_regions()

    def nested_placeholder_names(self) -> Set[str]:
        names: Set[str] = set()
        for region in self.iter_regions():
            names |= region.own_placeholder_names()
        return names


# --------------------------------------------------------------------------- #
# Regions
# --------------------------------------------------------------------------- #


class Region(ABC):
    """Base class for a control-flow construct wrapping a traced body."""

    region_kind: ClassVar[str]

    id: str
    results: Dict[str, NodeIO]

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]: ...

    @abstractmethod
    def subprograms(self) -> List[RegionProgram]:
        """Bodies belonging to this region, for recursive traversal."""

    @abstractmethod
    def own_placeholder_names(self) -> Set[str]:
        """Placeholder names introduced by this region itself."""

    def placeholder_names(self) -> Set[str]:
        """Placeholder names introduced here or by any nested region."""
        names = self.own_placeholder_names()
        for program in self.subprograms():
            names |= program.nested_placeholder_names()
        return names

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Region":
        if not isinstance(data, dict):
            raise ValueError(f"region must be an object, got {type(data).__name__}")
        region_kind = data.get("region_kind")
        if region_kind not in _REGION_REGISTRY:
            raise ValueError(
                f"unknown region kind {region_kind!r}; "
                f"expected one of {sorted(_REGION_REGISTRY)}"
            )
        return _REGION_REGISTRY[region_kind]._from_dict(data)

    @classmethod
    @abstractmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "Region": ...


@dataclass
class LoopRegion(Region):
    """A counted loop whose trip count is resolved from the request.

    ``carry_init`` -> ``carry_placeholders`` -> body -> ``carry_out`` is the
    loop-carried chain. That chain is the *only* thing serializing iterations:
    readiness in the executor is purely dependency-driven, so a body with no
    carry would produce N mutually independent iterations that the scheduler is
    free to run concurrently.
    """

    id: str
    trip_count: Expr
    induction_var: NodeIO
    carry_placeholders: Dict[str, NodeIO]
    carry_init: Dict[str, NodeIO]
    carry_out: Dict[str, NodeIO]
    body: RegionProgram
    results: Dict[str, NodeIO]
    # Per-iteration name for the induction variable. Configurable only so the
    # denoise compatibility path can reproduce the historical name exactly.
    iv_name_template: str = "{region_id}_iv_{i}"

    region_kind: ClassVar[str] = "loop"

    def iv_name(self, iteration: int) -> str:
        return self.iv_name_template.format(region_id=self.id, i=iteration)

    def subprograms(self) -> List[RegionProgram]:
        return [self.body]

    def own_placeholder_names(self) -> Set[str]:
        return (
            {self.induction_var.name}
            | {io.name for io in self.carry_placeholders.values()}
            | {io.name for io in self.results.values()}
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "region_kind": self.region_kind,
            "id": self.id,
            "trip_count": self.trip_count.to_dict(),
            "induction_var": self.induction_var.to_dict(),
            "iv_name_template": self.iv_name_template,
            "carry_placeholders": _ios_to_dict(self.carry_placeholders),
            "carry_init": _ios_to_dict(self.carry_init),
            "carry_out": _ios_to_dict(self.carry_out),
            "body": self.body.to_dict(),
            "results": _ios_to_dict(self.results),
        }

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "LoopRegion":
        required = (
            "id",
            "trip_count",
            "induction_var",
            "carry_placeholders",
            "carry_init",
            "carry_out",
            "body",
            "results",
        )
        for key in required:
            if key not in data:
                raise ValueError(f"loop region is missing {key!r}")

        region = cls(
            id=data["id"],
            trip_count=Expr.from_dict(data["trip_count"]),
            induction_var=NodeIO.from_dict(data["induction_var"]),
            carry_placeholders=_ios_from_dict(data["carry_placeholders"]),
            carry_init=_ios_from_dict(data["carry_init"]),
            carry_out=_ios_from_dict(data["carry_out"]),
            body=RegionProgram.from_dict(data["body"]),
            results=_ios_from_dict(data["results"]),
            iv_name_template=data.get("iv_name_template", "{region_id}_iv_{i}"),
        )
        keys = set(region.carry_init)
        for label, mapping in (
            ("carry_placeholders", region.carry_placeholders),
            ("carry_out", region.carry_out),
            ("results", region.results),
        ):
            if set(mapping) != keys:
                raise ValueError(
                    f"loop region {region.id!r}: {label} keys {sorted(mapping)} "
                    f"do not match carry keys {sorted(keys)}"
                )
        return region


@dataclass
class CondRegion(Region):
    """A branch resolved at expansion time.

    Emits no node of its own: the expander inlines the taken branch and points
    ``results`` at that branch's real outputs. Both branches are traced while
    authoring, so tracing either one must not raise.
    """

    id: str
    predicate: Expr
    then_body: RegionProgram
    then_results: Dict[str, NodeIO]
    results: Dict[str, NodeIO]
    else_body: Optional[RegionProgram] = None
    else_results: Optional[Dict[str, NodeIO]] = None

    region_kind: ClassVar[str] = "cond"

    def subprograms(self) -> List[RegionProgram]:
        programs = [self.then_body]
        if self.else_body is not None:
            programs.append(self.else_body)
        return programs

    def own_placeholder_names(self) -> Set[str]:
        return {io.name for io in self.results.values()}

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "region_kind": self.region_kind,
            "id": self.id,
            "predicate": self.predicate.to_dict(),
            "then": {
                "body": self.then_body.to_dict(),
                "results": _ios_to_dict(self.then_results),
            },
            "results": _ios_to_dict(self.results),
            "else": None,
        }
        if self.else_body is not None:
            payload["else"] = {
                "body": self.else_body.to_dict(),
                "results": _ios_to_dict(self.else_results or {}),
            }
        return payload

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "CondRegion":
        for key in ("id", "predicate", "then", "results"):
            if key not in data:
                raise ValueError(f"cond region is missing {key!r}")

        then_payload = data["then"]
        else_payload = data.get("else")

        region = cls(
            id=data["id"],
            predicate=Expr.from_dict(data["predicate"]),
            then_body=RegionProgram.from_dict(then_payload["body"]),
            then_results=_ios_from_dict(then_payload["results"]),
            results=_ios_from_dict(data["results"]),
            else_body=(
                RegionProgram.from_dict(else_payload["body"])
                if else_payload is not None
                else None
            ),
            else_results=(
                _ios_from_dict(else_payload["results"])
                if else_payload is not None
                else None
            ),
        )
        keys = set(region.results)
        if set(region.then_results) != keys:
            raise ValueError(
                f"cond region {region.id!r}: then-branch results "
                f"{sorted(region.then_results)} do not match {sorted(keys)}"
            )
        if region.else_results is not None and set(region.else_results) != keys:
            raise ValueError(
                f"cond region {region.id!r}: else-branch results "
                f"{sorted(region.else_results)} do not match {sorted(keys)}"
            )
        if region.else_body is None and keys:
            raise ValueError(
                f"cond region {region.id!r}: has results {sorted(keys)} but no "
                f"else branch to produce them when the predicate is false"
            )
        return region


# --------------------------------------------------------------------------- #
# Body collector
# --------------------------------------------------------------------------- #


class RegionBuilder:
    """Collects a region body while it is being traced.

    Deliberately *not* a :class:`~diflow.interface.workflow.Workflow`:
    ``Workflow.__init__`` installs itself as the current sink, so using a
    throwaway workflow as the collector would clobber the real one before the
    scoping context manager could save it.

    Nodes and nested regions share one ordered list, so a nested region lands at
    its authoring position relative to surrounding nodes.
    """

    def __init__(self):
        self.program = RegionProgram()

    def add_workflow_node(self, workflow_node: WorkflowNode) -> None:
        self.program.ops.append(workflow_node)

    def add_region(self, region: Region) -> None:
        self.program.ops.append(region)

    def __repr__(self) -> str:
        return f"RegionBuilder(ops={len(self.program.ops)})"


_REGION_REGISTRY.update({cls.region_kind: cls for cls in (LoopRegion, CondRegion)})
