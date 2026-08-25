from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Dict,
    NamedTuple,
    Optional,
    _GenericAlias,
    get_args,
    get_origin,
)

# expr imports node_io only lazily (inside to_expr), so this direction is safe.
from diflow.interface.expr import SymbolicOperand


def type_to_string(t):
    # Handle typing generics (List[...], Dict[...], etc)
    origin = get_origin(t)
    if origin is not None:
        args = get_args(t)
        args_str = ",".join(type_to_string(arg) for arg in args)
        # Preserve the original format (typing.List or list)
        if isinstance(t, _GenericAlias) and t.__module__ == "typing":
            return f"typing.{origin.__name__.capitalize()}[{args_str}]"
        return f"{origin.__module__}.{origin.__name__}[{args_str}]"

    # Handle regular types
    if isinstance(t, type):
        return f"{t.__module__}.{t.__qualname__}"

    # Handle special cases like typing.Any, typing.Union, etc
    return str(t)


def string_to_type(type_string):
    # Handle generic types
    if "[" in type_string:
        base_type, args = type_string.split("[", 1)
        args = args.rstrip("]")

        # Convert base type
        module_name, type_name = base_type.rsplit(".", 1)
        import importlib

        module = importlib.import_module(module_name)
        base = getattr(module, type_name)

        # Convert argument types
        arg_types = [string_to_type(arg.strip()) for arg in args.split(",")]

        return base[tuple(arg_types) if len(arg_types) > 1 else arg_types[0]]

    # Handle regular types
    module_name, type_name = type_string.rsplit(".", 1)
    import importlib

    module = importlib.import_module(module_name)
    return getattr(module, type_name)


class SourceType(Enum):
    INPUT = "input"
    NODE = "node"


@dataclass
class NodeIO(SymbolicOperand):
    """A value in the graph: a request input, or one operator's output.

    Inherits comparison and arithmetic overloads so a request input can be used
    directly in a control-flow predicate or trip count -- ``guidance_scale > 1.0``
    builds an expression rather than comparing objects. Converting one to
    ``bool`` or ``int`` raises, since Python would have to decide something only
    the request can decide; the error names the API to use instead.
    """

    name: str
    data_type: type
    source_type: Optional[SourceType] = None
    source_node: Optional[str] = None  # TODO (Lingyun): Is it optional?
    size: Optional[list[int]] = None  # Used for pre-allocating output tensors
    lazy: bool = False

    def __getitem__(self, index) -> "NodeIO":
        """Take one slice of this value, as a node in the graph.

        ``timesteps[i]`` inside a loop body is how you pick the current step's
        timestep. ``index`` is normally the loop's induction variable, but a plain
        int works too.

        The slice is a real operator call, so it emits a node into whatever is
        currently being authored -- which means it needs an active workflow, and
        the returned handle is that node's output rather than a value. Callers used
        to have to know the operator by name and wire it up themselves; they no
        longer do.
        """
        # Imported here rather than at module scope: operators import node_io, so
        # the reverse direction can only be resolved at call time.
        from diflow.operators.custom.indexed_tensor import IndexedTensor

        return IndexedTensor()(tensor=self, index=index)

    def __iter__(self):
        """Refuse iteration, which defining ``__getitem__`` would otherwise allow.

        This is not decoration. With ``__getitem__`` present and no ``__iter__``,
        Python falls back to the legacy sequence protocol: ``iter(io)`` starts
        calling ``io[0]``, ``io[1]``, ... and stops at ``IndexError``. Every one of
        those calls emits an indexing node and returns a handle, so nothing ever
        raises and the loop runs forever, growing the graph as it goes. A stray
        ``list(io)`` or an over-wide tuple unpack would hang the process.
        """
        raise TypeError(
            f"cannot iterate {self._symbolic_description()}. To loop a "
            f"request-dependent number of times use for_range(<count>, body, "
            f"carry={{...}}). To read one element, subscript it: tensor[index]. "
            f"If you meant to unpack an operator's outputs, this one returns a "
            f"single value rather than several."
        )

    def _symbolic_description(self) -> str:
        if self.source_type == SourceType.INPUT:
            return f"request input {self.name!r}"
        return f"the operator output {self.name!r}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "data_type": type_to_string(self.data_type),
            "source_type": self.source_type.value if self.source_type else None,
            "source_node": self.source_node,
            "size": self.size,
            "lazy": self.lazy,
        }

    @classmethod
    def from_dict(cls, io_dict: Dict[str, Any]) -> "NodeIO":
        return cls(
            name=io_dict["name"],
            data_type=string_to_type(io_dict["data_type"]),
            source_type=(
                SourceType(io_dict["source_type"]) if io_dict["source_type"] else None
            ),
            source_node=io_dict["source_node"],
            size=io_dict["size"],
            lazy=io_dict["lazy"],
        )


class AdapterInputs(NamedTuple):
    controlnet_cond: NodeIO
    conditioning_scale: NodeIO

    def to_dict(self) -> Dict[str, Any]:
        return {
            "controlnet_cond": self.controlnet_cond.to_dict(),
            "conditioning_scale": self.conditioning_scale.to_dict(),
        }

    @classmethod
    def from_dict(cls, io_dict: Dict[str, Any]) -> "AdapterInputs":
        return cls(
            controlnet_cond=NodeIO.from_dict(io_dict["controlnet_cond"]),
            conditioning_scale=NodeIO.from_dict(io_dict["conditioning_scale"]),
        )
