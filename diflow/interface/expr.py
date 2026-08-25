"""Static expressions over request inputs, evaluated when a workflow is expanded.

Control-flow regions need two things the graph cannot supply: how many times a
loop runs, and which branch of a conditional is taken. Both are fixed for a
given request but unknown when the workflow is authored -- ``num_inference_steps``
and ``guidance_scale`` arrive in the request body. An :class:`Expr` is the
serializable stand-in: authored symbolically, evaluated against the concrete
request inputs at expansion time.

Expressions may only read *request inputs*, never node outputs. The executor
requires a fully-expanded acyclic graph (its tensor reference counting is
one-shot and completion is a node-count comparison), so the graph's shape has to
be settled before any node runs.

Security note: ``from_dict`` is reachable from ``POST /api/workflow/register``,
so operators are looked up in closed dictionaries -- never ``eval``, never
``getattr(operator, name)`` on wire data. Unknown ``kind`` or ``op`` values are
rejected at parse time.
"""

import operator
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Dict, Mapping, Set, Union

# Closed operator tables. Adding an entry here is the only way to widen the
# expression language; that is deliberate, since these names arrive over HTTP.
_BINOPS: Dict[str, Callable[[Any, Any], Any]] = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "//": operator.floordiv,
    "%": operator.mod,
    "min": min,
    "max": max,
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
    # Eager, unlike Python's short-circuiting forms. Operands are pure values,
    # so the only difference would be a spurious lookup of a missing input.
    "and": lambda a, b: bool(a and b),
    "or": lambda a, b: bool(a or b),
}

_UNOPS: Dict[str, Callable[[Any], Any]] = {
    "int": int,
    "float": float,
    "bool": bool,
    "not": operator.not_,
    "neg": operator.neg,
}

_CONST_TYPES = (int, float, bool, str)

# kind -> class, populated at the bottom of the module.
_EXPR_REGISTRY: Dict[str, type] = {}


class SymbolicOperand:
    """Operator overloads shared by :class:`Expr` and ``NodeIO``.

    Comparisons and arithmetic build an :class:`Expr` rather than computing a
    value, so a predicate reads like ordinary Python::

        cond(guidance_scale > 1.0, with_cfg, without_cfg)

    ``__index__`` raises, so ``range(num_inference_steps)`` reports something
    actionable instead of "cannot be interpreted as an integer".

    ``__bool__`` is defined on :class:`Expr` but deliberately NOT here, so it does
    not apply to ``NodeIO``. Truth-testing a ``NodeIO`` is the ordinary
    presence-check idiom -- ``if input_info and input_info.data_type is ...`` --
    and raising there breaks unrelated code that has nothing to do with control
    flow. Comparisons return an ``Expr``, so the mistake that actually matters,
    ``if guidance_scale > 1.0:``, is still caught.

    ``__iter__`` is likewise omitted: defining it would make
    ``isinstance(io, Iterable)`` true and could reroute duck-typed code. Python's
    own "not iterable" error is clear enough.

    ``__eq__`` is deliberately left alone. ``NodeIO`` is a dataclass whose
    equality is used as ordinary equality; returning an expression from it would
    break comparisons throughout the codebase. Use ``BinOp("==", a, b)`` for a
    symbolic equality test.
    """

    def _symbolic_description(self) -> str:
        return f"{type(self).__name__} {self!s}"

    # -- comparisons. Python supplies the reflected forms, so `1.0 < x` works. --
    def __gt__(self, other) -> "BinOp":
        return BinOp(">", self, other)

    def __ge__(self, other) -> "BinOp":
        return BinOp(">=", self, other)

    def __lt__(self, other) -> "BinOp":
        return BinOp("<", self, other)

    def __le__(self, other) -> "BinOp":
        return BinOp("<=", self, other)

    # -- arithmetic, for derived trip counts like `num_inference_steps - 1` -----
    def __add__(self, other) -> "BinOp":
        return BinOp("+", self, other)

    def __radd__(self, other) -> "BinOp":
        return BinOp("+", other, self)

    def __sub__(self, other) -> "BinOp":
        return BinOp("-", self, other)

    def __rsub__(self, other) -> "BinOp":
        return BinOp("-", other, self)

    def __mul__(self, other) -> "BinOp":
        return BinOp("*", self, other)

    def __rmul__(self, other) -> "BinOp":
        return BinOp("*", other, self)

    def __truediv__(self, other) -> "BinOp":
        return BinOp("/", self, other)

    def __rtruediv__(self, other) -> "BinOp":
        return BinOp("/", other, self)

    def __floordiv__(self, other) -> "BinOp":
        return BinOp("//", self, other)

    def __rfloordiv__(self, other) -> "BinOp":
        return BinOp("//", other, self)

    def __mod__(self, other) -> "BinOp":
        return BinOp("%", self, other)

    def __neg__(self) -> "UnaryOp":
        return UnaryOp("neg", self)

    # -- refuse the conversion Python would otherwise guess at -----------------
    def __index__(self):
        raise TypeError(
            f"cannot use {self._symbolic_description()} as an integer: its value "
            f"is only known once a request arrives, so range() cannot be given it "
            f"while the workflow is being built. Use for_range() instead:\n"
            f"    for_range(<count>, body, carry={{...}})"
        )


class Expr(SymbolicOperand, ABC):
    """A pure expression over request inputs."""

    kind: ClassVar[str]

    def __bool__(self):
        """Refuse to collapse a predicate into a build-time decision.

        ``if guidance_scale > 1.0:`` lands here, because the comparison produced
        an expression. Left off ``NodeIO`` on purpose -- see
        :class:`SymbolicOperand`.
        """
        raise TypeError(
            f"cannot take the truth value of {self._symbolic_description()}: it "
            f"depends on values that only exist once a request arrives, but a "
            f"Python `if` is evaluated while the workflow is being built. Use "
            f"cond() instead:\n"
            f"    cond(<predicate>, then_fn, else_fn)\n"
            f"A Python `if` is still fine for anything known at build time, such "
            f"as a model id or whether an adapter list is empty."
        )

    @abstractmethod
    def eval(self, env: Mapping[str, Any]) -> Any:
        """Evaluate against a mapping of input name -> concrete value."""

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable form."""

    @abstractmethod
    def free_inputs(self) -> Set[str]:
        """Names of the request inputs this expression reads."""

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Expr":
        if not isinstance(data, dict):
            raise ValueError(f"expression must be an object, got {type(data).__name__}")
        kind = data.get("kind")
        if kind not in _EXPR_REGISTRY:
            raise ValueError(
                f"unknown expression kind {kind!r}; "
                f"expected one of {sorted(_EXPR_REGISTRY)}"
            )
        return _EXPR_REGISTRY[kind]._from_dict(data)

    @classmethod
    @abstractmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "Expr":
        """Parse the payload for this specific kind."""


@dataclass(frozen=True)
class Const(Expr):
    """A literal."""

    value: Union[int, float, bool, str]

    kind: ClassVar[str] = "const"

    def __post_init__(self):
        if not isinstance(self.value, _CONST_TYPES):
            raise ValueError(
                f"constant must be one of "
                f"{', '.join(t.__name__ for t in _CONST_TYPES)}; "
                f"got {type(self.value).__name__}"
            )

    def eval(self, env: Mapping[str, Any]) -> Any:
        return self.value

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "value": self.value}

    def free_inputs(self) -> Set[str]:
        return set()

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "Const":
        if "value" not in data:
            raise ValueError("const expression is missing 'value'")
        return cls(value=data["value"])

    def __str__(self) -> str:
        return repr(self.value)

    def _symbolic_description(self) -> str:
        return f"the constant {self.value!r}"


@dataclass(frozen=True)
class InputRef(Expr):
    """The value of a named request input."""

    input_name: str

    kind: ClassVar[str] = "input"

    def __post_init__(self):
        if not isinstance(self.input_name, str) or not self.input_name:
            raise ValueError(
                f"input name must be a non-empty string, got {self.input_name!r}"
            )

    def eval(self, env: Mapping[str, Any]) -> Any:
        try:
            return env[self.input_name]
        except KeyError:
            raise KeyError(
                f"request input {self.input_name!r} is required to expand this "
                f"workflow's control flow but was not supplied"
            ) from None

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "input_name": self.input_name}

    def free_inputs(self) -> Set[str]:
        return {self.input_name}

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "InputRef":
        if "input_name" not in data:
            raise ValueError("input expression is missing 'input_name'")
        return cls(input_name=data["input_name"])

    def __str__(self) -> str:
        return f"${self.input_name}"

    def _symbolic_description(self) -> str:
        return f"request input {self.input_name!r}"


@dataclass(frozen=True)
class BinOp(Expr):
    """A binary operation drawn from :data:`_BINOPS`."""

    op: str
    lhs: Expr
    rhs: Expr

    kind: ClassVar[str] = "binop"

    def __post_init__(self):
        if self.op not in _BINOPS:
            raise ValueError(
                f"unknown binary operator {self.op!r}; "
                f"expected one of {sorted(_BINOPS)}"
            )
        # Coerce operands so callers can write the natural
        # BinOp(">", guidance_scale_io, 1.0) instead of wrapping by hand.
        # to_expr still rejects node outputs and non-values.
        for side in ("lhs", "rhs"):
            value = getattr(self, side)
            if not isinstance(value, Expr):
                object.__setattr__(self, side, to_expr(value))

    def eval(self, env: Mapping[str, Any]) -> Any:
        return _BINOPS[self.op](self.lhs.eval(env), self.rhs.eval(env))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "op": self.op,
            "lhs": self.lhs.to_dict(),
            "rhs": self.rhs.to_dict(),
        }

    def free_inputs(self) -> Set[str]:
        return self.lhs.free_inputs() | self.rhs.free_inputs()

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "BinOp":
        for key in ("op", "lhs", "rhs"):
            if key not in data:
                raise ValueError(f"binop expression is missing {key!r}")
        return cls(
            op=data["op"],
            lhs=Expr.from_dict(data["lhs"]),
            rhs=Expr.from_dict(data["rhs"]),
        )

    def __str__(self) -> str:
        if self.op in ("min", "max"):
            return f"{self.op}({self.lhs}, {self.rhs})"
        return f"({self.lhs} {self.op} {self.rhs})"


@dataclass(frozen=True)
class UnaryOp(Expr):
    """A unary operation drawn from :data:`_UNOPS`."""

    op: str
    operand: Expr

    kind: ClassVar[str] = "unaryop"

    def __post_init__(self):
        if self.op not in _UNOPS:
            raise ValueError(
                f"unknown unary operator {self.op!r}; "
                f"expected one of {sorted(_UNOPS)}"
            )
        if not isinstance(self.operand, Expr):
            object.__setattr__(self, "operand", to_expr(self.operand))

    def eval(self, env: Mapping[str, Any]) -> Any:
        return _UNOPS[self.op](self.operand.eval(env))

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "op": self.op, "operand": self.operand.to_dict()}

    def free_inputs(self) -> Set[str]:
        return self.operand.free_inputs()

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "UnaryOp":
        for key in ("op", "operand"):
            if key not in data:
                raise ValueError(f"unaryop expression is missing {key!r}")
        return cls(op=data["op"], operand=Expr.from_dict(data["operand"]))

    def __str__(self) -> str:
        return f"{self.op}({self.operand})"


def to_expr(value: Any) -> Expr:
    """Coerce a user-supplied trip count or predicate into an :class:`Expr`.

    Accepts a literal, a request-input ``NodeIO``, or an already-built
    expression. A ``NodeIO`` fed by another node is rejected: the graph's shape
    must be decidable before execution starts.
    """
    # Imported here rather than at module scope to keep the dependency one-way
    # for readers -- node_io knows nothing about expressions.
    from diflow.interface.node_io import NodeIO, SourceType

    if isinstance(value, Expr):
        return value
    if isinstance(value, _CONST_TYPES):
        return Const(value)
    if isinstance(value, NodeIO):
        if value.source_type == SourceType.INPUT:
            return InputRef(value.name)
        raise ValueError(
            f"cannot use node output {value.name!r} as a trip count or predicate: "
            f"it is only known while the graph is running, but control flow is "
            f"resolved before it starts (the executor needs a fully-expanded "
            f"graph -- its tensor reference counts are one-shot and completion "
            f"is a node-count comparison). Use a request input instead."
        )
    raise ValueError(
        f"expected a number, a request-input NodeIO, or an Expr; "
        f"got {type(value).__name__}"
    )


# A predicate is just an expression the expander coerces with bool().
Predicate = Expr

_EXPR_REGISTRY.update({cls.kind: cls for cls in (Const, InputRef, BinOp, UnaryOp)})
