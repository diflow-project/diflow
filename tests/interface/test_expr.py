"""Tests for the expand-time expression tree."""

import json
import unittest

from diflow.interface.expr import (
    BinOp,
    Const,
    Expr,
    InputRef,
    UnaryOp,
    to_expr,
)
from diflow.interface.node_io import NodeIO, SourceType


def derived_img2img_trip_count() -> Expr:
    """A step count derived from another input, as img2img needs::

        init_timestep = min(int(n * strength), n)
        n = n - max(n - init_timestep, 0)

    This is the most demanding trip count the old hard-coded unroller ever
    computed (in an SDXL img2img path since removed). Kept as evidence that the
    expression language can express a derived trip count, not just read one
    straight out of the request.
    """
    n = InputRef("num_inference_steps")
    strength = InputRef("strength")
    init_timestep = BinOp("min", UnaryOp("int", BinOp("*", n, strength)), n)
    return BinOp("-", n, BinOp("max", BinOp("-", n, init_timestep), Const(0)))


class TestEval(unittest.TestCase):
    def test_const(self):
        self.assertEqual(Const(5).eval({}), 5)
        self.assertEqual(Const(1.5).eval({}), 1.5)
        self.assertEqual(Const(True).eval({}), True)
        self.assertEqual(Const("x").eval({}), "x")

    def test_input_ref(self):
        self.assertEqual(InputRef("n").eval({"n": 7}), 7)

    def test_input_ref_missing_names_the_input(self):
        with self.assertRaises(KeyError) as ctx:
            InputRef("num_inference_steps").eval({})
        self.assertIn("num_inference_steps", str(ctx.exception))

    def test_arithmetic(self):
        env = {"a": 10, "b": 4}
        a, b = InputRef("a"), InputRef("b")
        self.assertEqual(BinOp("+", a, b).eval(env), 14)
        self.assertEqual(BinOp("-", a, b).eval(env), 6)
        self.assertEqual(BinOp("*", a, b).eval(env), 40)
        self.assertEqual(BinOp("//", a, b).eval(env), 2)
        self.assertEqual(BinOp("%", a, b).eval(env), 2)
        self.assertEqual(BinOp("min", a, b).eval(env), 4)
        self.assertEqual(BinOp("max", a, b).eval(env), 10)

    def test_comparison_and_logic(self):
        env = {"gs": 7.5}
        gs = InputRef("gs")
        self.assertTrue(BinOp(">", gs, Const(1.0)).eval(env))
        self.assertFalse(BinOp("<=", gs, Const(1.0)).eval(env))
        self.assertTrue(BinOp("and", BinOp(">", gs, Const(1.0)), Const(True)).eval(env))
        self.assertFalse(UnaryOp("not", BinOp(">", gs, Const(1.0))).eval(env))

    def test_unary_casts(self):
        self.assertEqual(UnaryOp("int", Const(3.7)).eval({}), 3)
        self.assertEqual(UnaryOp("neg", Const(3)).eval({}), -3)
        self.assertEqual(UnaryOp("bool", Const(0)).eval({}), False)

    def test_cfg_predicate(self):
        """The one predicate the denoise loop actually needs."""
        pred = BinOp(">", InputRef("guidance_scale"), Const(1.0))
        self.assertTrue(pred.eval({"guidance_scale": 7.5}))
        self.assertFalse(pred.eval({"guidance_scale": 1.0}))
        self.assertFalse(pred.eval({"guidance_scale": 0.5}))

    def test_derived_trip_count_matches_the_equivalent_python(self):
        expr = derived_img2img_trip_count()
        for n in (1, 2, 4, 10, 28, 50):
            for strength in (0.0, 0.1, 0.3, 0.5, 0.6, 0.9, 1.0):
                init_timestep = min(int(n * strength), n)
                expected = n - max(n - init_timestep, 0)
                with self.subTest(n=n, strength=strength):
                    self.assertEqual(
                        expr.eval({"num_inference_steps": n, "strength": strength}),
                        expected,
                    )


class TestFreeInputs(unittest.TestCase):
    def test_collects_transitively(self):
        self.assertEqual(
            derived_img2img_trip_count().free_inputs(),
            {"num_inference_steps", "strength"},
        )

    def test_const_has_none(self):
        self.assertEqual(Const(1).free_inputs(), set())


class TestSerialization(unittest.TestCase):
    def _round_trip(self, expr):
        payload = json.loads(json.dumps(expr.to_dict()))
        return Expr.from_dict(payload)

    def test_round_trip_preserves_structure_and_value(self):
        env = {"num_inference_steps": 28, "strength": 0.6, "guidance_scale": 7.5}
        for expr in (
            Const(3),
            Const("hello"),
            InputRef("num_inference_steps"),
            BinOp(">", InputRef("guidance_scale"), Const(1.0)),
            UnaryOp("not", BinOp("<", InputRef("num_inference_steps"), Const(4))),
            derived_img2img_trip_count(),
        ):
            with self.subTest(expr=str(expr)):
                revived = self._round_trip(expr)
                self.assertEqual(revived, expr)
                self.assertEqual(revived.eval(env), expr.eval(env))


class TestRejectsUntrustedPayloads(unittest.TestCase):
    """``from_dict`` is reachable from the registration endpoint."""

    def test_unknown_kind(self):
        with self.assertRaises(ValueError) as ctx:
            Expr.from_dict({"kind": "exec", "value": 1})
        self.assertIn("unknown expression kind", str(ctx.exception))

    def test_unknown_binop(self):
        payload = {
            "kind": "binop",
            "op": "__import__",
            "lhs": {"kind": "const", "value": 1},
            "rhs": {"kind": "const", "value": 2},
        }
        with self.assertRaises(ValueError) as ctx:
            Expr.from_dict(payload)
        self.assertIn("unknown binary operator", str(ctx.exception))

    def test_unknown_unaryop(self):
        payload = {
            "kind": "unaryop",
            "op": "eval",
            "operand": {"kind": "const", "value": 1},
        }
        with self.assertRaises(ValueError):
            Expr.from_dict(payload)

    def test_non_dict_payload(self):
        for bad in ("const", 3, None, []):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    Expr.from_dict(bad)

    def test_missing_fields(self):
        for payload in (
            {"kind": "const"},
            {"kind": "input"},
            {"kind": "binop", "op": "+"},
            {"kind": "unaryop", "op": "int"},
        ):
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    Expr.from_dict(payload)

    def test_constructor_rejects_bad_operators(self):
        with self.assertRaises(ValueError):
            BinOp("**", Const(2), Const(3))
        with self.assertRaises(ValueError):
            UnaryOp("abs", Const(-1))

    def test_constructor_rejects_operands_that_cannot_be_coerced(self):
        """Operands go through to_expr, which still refuses node outputs."""
        node_output = NodeIO(
            name="Node_x:out",
            data_type=int,
            source_type=SourceType.NODE,
            source_node="Node_x",
        )
        with self.assertRaises(ValueError):
            BinOp("+", node_output, Const(2))
        with self.assertRaises(ValueError):
            BinOp("+", Const(1), None)
        with self.assertRaises(ValueError):
            UnaryOp("int", [1, 2])

    def test_const_rejects_unsupported_types(self):
        for bad in ([1, 2], {"a": 1}, object()):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaises(ValueError):
                    Const(bad)


class TestToExpr(unittest.TestCase):
    def test_passes_through_expressions(self):
        expr = Const(1)
        self.assertIs(to_expr(expr), expr)

    def test_wraps_literals(self):
        self.assertEqual(to_expr(4), Const(4))
        self.assertEqual(to_expr(1.5), Const(1.5))

    def test_operands_are_coerced_so_callers_need_not_wrap(self):
        io = NodeIO(
            name="guidance_scale", data_type=float, source_type=SourceType.INPUT
        )
        self.assertEqual(
            BinOp(">", io, 1.0),
            BinOp(">", InputRef("guidance_scale"), Const(1.0)),
        )
        self.assertEqual(UnaryOp("int", 3.7), UnaryOp("int", Const(3.7)))

    def test_request_input_node_io_becomes_input_ref(self):
        io = NodeIO(
            name="num_inference_steps", data_type=int, source_type=SourceType.INPUT
        )
        self.assertEqual(to_expr(io), InputRef("num_inference_steps"))

    def test_node_sourced_node_io_is_rejected_with_an_explanation(self):
        io = NodeIO(
            name="Scheduler_abc:latents",
            data_type=int,
            source_type=SourceType.NODE,
            source_node="Scheduler_abc",
        )
        with self.assertRaises(ValueError) as ctx:
            to_expr(io)
        message = str(ctx.exception)
        self.assertIn("Scheduler_abc:latents", message)
        self.assertIn("request input", message)

    def test_rejects_other_types(self):
        for bad in (None, [1], {"a": 1}):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    to_expr(bad)


if __name__ == "__main__":
    unittest.main()


class TestSymbolicOperators(unittest.TestCase):
    """NodeIO and Expr build expressions from Python operators.

    This is what lets a predicate read as ``cond(guidance_scale > 1.0, ...)``
    rather than ``cond(BinOp(">", guidance_scale, Const(1.0)), ...)``.
    """

    def setUp(self):
        self.steps = NodeIO(
            name="num_inference_steps", data_type=int, source_type=SourceType.INPUT
        )
        self.guidance = NodeIO(
            name="guidance_scale", data_type=float, source_type=SourceType.INPUT
        )

    def test_comparisons_build_expressions(self):
        self.assertEqual(
            self.guidance > 1.0, BinOp(">", InputRef("guidance_scale"), Const(1.0))
        )
        self.assertEqual(
            self.guidance >= 1.0, BinOp(">=", InputRef("guidance_scale"), Const(1.0))
        )
        self.assertEqual(
            self.steps < 4, BinOp("<", InputRef("num_inference_steps"), Const(4))
        )
        self.assertEqual(
            self.steps <= 4, BinOp("<=", InputRef("num_inference_steps"), Const(4))
        )

    def test_reflected_comparison(self):
        """``1.0 < x`` is handed to x as ``x > 1.0`` by Python."""
        self.assertEqual(1.0 < self.guidance, self.guidance > 1.0)

    def test_arithmetic_builds_expressions(self):
        self.assertEqual(
            (self.steps - 1).eval({"num_inference_steps": 4}),
            3,
        )
        self.assertEqual((2 * self.steps).eval({"num_inference_steps": 4}), 8)
        self.assertEqual((-self.steps).eval({"num_inference_steps": 4}), -4)

    def test_expressions_compose(self):
        predicate = (self.steps - 1) > 0
        self.assertTrue(predicate.eval({"num_inference_steps": 4}))
        self.assertFalse(predicate.eval({"num_inference_steps": 1}))

    def test_equality_is_left_alone(self):
        """Overloading __eq__ would break ordinary equality on a dataclass."""
        same = NodeIO(
            name="guidance_scale", data_type=float, source_type=SourceType.INPUT
        )
        self.assertTrue(self.guidance == same)
        self.assertIsInstance(self.guidance == same, bool)
        self.assertTrue(Const(1) == Const(1))


class TestSymbolicGuards(unittest.TestCase):
    """Python constructs that would silently guess get an actionable error."""

    def setUp(self):
        self.steps = NodeIO(
            name="num_inference_steps", data_type=int, source_type=SourceType.INPUT
        )
        self.guidance = NodeIO(
            name="guidance_scale", data_type=float, source_type=SourceType.INPUT
        )

    def test_python_if_on_a_predicate_raises_and_names_cond(self):
        """``if guidance_scale > 1.0:`` used to be a TypeError about '>'; now the
        comparison succeeds and the truth test is what refuses."""
        with self.assertRaises(TypeError) as ctx:
            if self.guidance > 1.0:
                pass
        message = str(ctx.exception)
        self.assertIn("guidance_scale", message)
        self.assertIn("cond(", message)

    def test_truth_testing_a_node_io_still_works(self):
        """Regression: __bool__ must NOT be defined on NodeIO.

        ``if input_info and input_info.data_type == ...`` is the ordinary
        presence-check idiom and appears in the worker's input deserialization.
        Raising there broke every served request, and no interface-level test
        covered it because none of them go through the worker.
        """
        self.assertTrue(bool(self.guidance))
        self.assertTrue(bool(self.guidance and self.guidance.data_type is float))
        self.assertFalse(bool(None and self.guidance))

    def test_iterating_a_node_io_raises_rather_than_hanging(self):
        """Regression: ``__getitem__`` exists for ``timesteps[i]``, and without an
        ``__iter__`` Python falls back to the legacy sequence protocol -- ``iter``
        calls ``io[0]``, ``io[1]``, ... until IndexError. Each of those emits a
        node and returns a handle, so nothing raises and the loop never ends. A
        stray ``list(io)`` hung the test suite.

        The cost is that ``isinstance(io, Iterable)`` is now true, which is a
        lesser evil than an unbounded loop.
        """
        import collections.abc

        with self.assertRaises(TypeError):
            list(self.guidance)
        self.assertTrue(isinstance(self.guidance, collections.abc.Iterable))

    def test_range_raises_and_names_for_range(self):
        with self.assertRaises(TypeError) as ctx:
            range(self.steps)
        message = str(ctx.exception)
        self.assertIn("num_inference_steps", message)
        self.assertIn("for_range(", message)

    def test_iteration_raises_and_points_at_for_range(self):
        with self.assertRaises(TypeError) as ctx:
            list(self.steps)
        self.assertIn("for_range(", str(ctx.exception))

    def test_node_output_is_described_as_such(self):
        output = NodeIO(
            name="Flux_x:noise_pred",
            data_type=int,
            source_type=SourceType.NODE,
            source_node="Flux_x",
        )
        with self.assertRaises(TypeError) as ctx:
            range(output)
        self.assertIn("Flux_x:noise_pred", str(ctx.exception))
