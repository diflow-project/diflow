"""The regression net for the control-flow refactor.

The denoise loop used to be a macro node expanded by a hard-coded unroller. That
unroller was the oracle for this suite: the control-flow graph had to come out
isomorphic to what it produced. It is gone now, so its output is frozen in
``golden_denoise_graphs.json``, generated from it while it still existed. That is
what lets the guarantee outlive the code it came from.

A diff in the golden file means the served graph changed shape. Regenerate it only
for a deliberate change, and say why.

The canonical form is checked for teeth first: a fingerprint that called every
graph equal would pass everything.
"""

import json
import pathlib
import unittest

from diflow.backend import dependency
from diflow.interface.node_io import SourceType
from diflow.interface.workflow import Workflow
from diflow.interface.workflow_expand import expand_workflow
from tests.interface.graph_canon import (
    canonical_form,
    canonical_json,
    executor_view,
    injected_values,
    node_signatures,
)
from tests.interface.hub_workflows import (
    build_workflow,
    make_inputs,
    standard_loop_workflow_names,
)

GOLDEN_PATH = pathlib.Path(__file__).with_name("golden_denoise_graphs.json")
GOLDEN = json.loads(GOLDEN_PATH.read_text())["cases"]

STEP_COUNTS = (1, 2, 4)
# 1.0 takes the non-CFG path, 7.5 the CFG path. On the Flux examples the input
# that gates CFG is cfg_guidance_scale; make_inputs maps both names.
GUIDANCE_SCALES = (1.0, 7.5)

# A spread of the remaining variants, for tests that do not need all of them:
# schnell and dev, plus a controlnet of each for lazy edges. CFG is not a variant
# here -- every workflow serves both, selected by GUIDANCE_SCALES.
REPRESENTATIVE = [
    "flux_schnell.register_txt2img_workflow",
    "flux_dev.register_txt2img_workflow",
    "flux_schnell.register_txt2img_controlnet_canny_workflow",
    "flux_dev.register_txt2img_controlnet_depth_workflow",
]


def golden_key(example, steps, guidance):
    return f"{example}|{steps}|{guidance}"


def golden_form(example, steps, guidance):
    """The frozen canonical form the original unroller produced."""
    key = golden_key(example, steps, guidance)
    if key not in GOLDEN:
        raise AssertionError(
            f"no golden entry for {key}; regenerate {GOLDEN_PATH.name} if this "
            f"case is new"
        )
    return GOLDEN[key]


def expand(example, steps, guidance, via_json=False):
    """Expand an example, returning (graph, injected values).

    ``via_json`` round-trips through registration first, which is what the server
    actually does.
    """
    workflow = build_workflow(example)
    base = make_inputs(workflow, steps, guidance)
    if via_json:
        workflow = Workflow.from_dict(json.loads(workflow.to_json()))
    request_inputs = dict(base)
    graph = expand_workflow(workflow, request_inputs)
    return graph, injected_values(base, request_inputs)


def expanded_form(example, steps, guidance, via_json=False):
    graph, injected = expand(example, steps, guidance, via_json=via_json)
    return canonical_json(canonical_form(graph, injected)), graph, injected


def describe_mismatch(expected, got, graph, injected):
    """A bare hash diff is unactionable; expand it into op ids and modes."""
    lines = []
    if expected["node_count"] != got["node_count"]:
        lines.append(
            f"node count: golden={expected['node_count']} got={got['node_count']}"
        )
    expected_sigs = dict(expected["signatures"])
    got_sigs = dict(got["signatures"])
    by_sig = {sig: name for name, sig in node_signatures(graph, injected).items()}
    nodes = {n.name: n for n in graph.workflow_nodes}

    for sig, count in sorted(got_sigs.items()):
        if expected_sigs.get(sig) != count:
            node = nodes.get(by_sig.get(sig, ""))
            what = (
                f"{node.op.id}(mode={node.mode})" if node is not None else "<unknown>"
            )
            lines.append(
                f"  got {count}x {what}, golden has {expected_sigs.get(sig, 0)}x"
            )
    for sig, count in sorted(expected_sigs.items()):
        if sig not in got_sigs:
            lines.append(f"  golden has {count}x a node absent from the expansion")
    if expected["outputs"] != got["outputs"]:
        lines.append(f"outputs: golden={expected['outputs']} got={got['outputs']}")
    if expected["injected"] != got["injected"]:
        lines.append(f"injected: golden={expected['injected']} got={got['injected']}")
    return "\n".join(lines) or "(no structural difference found)"


class TestCanonicalFormDiscriminates(unittest.TestCase):
    """Negative controls. Without these the golden comparison proves nothing."""

    EXAMPLE = "flux_schnell.register_txt2img_workflow"

    def test_identical_configurations_match(self):
        left, _, _ = expanded_form(self.EXAMPLE, 2, 7.5)
        right, _, _ = expanded_form(self.EXAMPLE, 2, 7.5)
        self.assertEqual(left, right)

    def test_different_step_counts_differ(self):
        left, _, _ = expanded_form(self.EXAMPLE, 2, 7.5)
        right, _, _ = expanded_form(self.EXAMPLE, 3, 7.5)
        self.assertNotEqual(left, right)

    def test_cfg_and_non_cfg_differ(self):
        left, _, _ = expanded_form(self.EXAMPLE, 2, 7.5)
        right, _, _ = expanded_form(self.EXAMPLE, 2, 1.0)
        self.assertNotEqual(left, right)

    def test_different_models_differ(self):
        left, _, _ = expanded_form(self.EXAMPLE, 2, 7.5)
        right, _, _ = expanded_form("flux_dev.register_txt2img_workflow", 2, 7.5)
        self.assertNotEqual(left, right)

    def test_controlnet_differs_from_plain(self):
        left, _, _ = expanded_form(self.EXAMPLE, 2, 7.5)
        right, _, _ = expanded_form(
            "flux_schnell.register_txt2img_controlnet_canny_workflow", 2, 7.5
        )
        self.assertNotEqual(left, right)

    def test_a_dropped_node_is_detected(self):
        graph, injected = expand(self.EXAMPLE, 3, 7.5)
        before = canonical_json(canonical_form(graph, injected))
        graph.workflow_nodes = graph.workflow_nodes[:-1]
        self.assertNotEqual(before, canonical_json(canonical_form(graph, injected)))

    def test_a_rewired_edge_is_detected(self):
        graph, injected = expand(self.EXAMPLE, 3, 7.5)
        before = canonical_json(canonical_form(graph, injected))
        target = next(n for n in graph.workflow_nodes if "latents" in n.get_inputs())
        other = next(
            io
            for n in graph.workflow_nodes
            for io in n.get_outputs().values()
            if io.name != target.get_inputs()["latents"].name
        )
        target.set_input("latents", other)
        self.assertNotEqual(before, canonical_json(canonical_form(graph, injected)))

    def test_the_golden_file_covers_the_matrix(self):
        expected = (
            len(standard_loop_workflow_names())
            * len(STEP_COUNTS)
            * len(GUIDANCE_SCALES)
        )
        self.assertEqual(len(GOLDEN), expected)


class TestControlFlowMatchesGolden(unittest.TestCase):
    """The heart of the refactor: the served graph is what it always was."""

    def test_every_example_and_configuration(self):
        examples = standard_loop_workflow_names()
        self.assertGreaterEqual(len(examples), 6, "example discovery looks wrong")

        for example in examples:
            for steps in STEP_COUNTS:
                for guidance in GUIDANCE_SCALES:
                    with self.subTest(example=example, steps=steps, guidance=guidance):
                        expected = golden_form(example, steps, guidance)
                        got, graph, injected = expanded_form(example, steps, guidance)
                        self.assertEqual(
                            expected,
                            got,
                            "\n" + describe_mismatch(expected, got, graph, injected),
                        )

    def test_through_the_registration_json_boundary(self):
        """The path the server actually takes: to_json, from_dict, expand.

        The only check that catches a field missing from the region IR's
        ``to_dict``, which the in-process comparison would pass.
        """
        for example in REPRESENTATIVE:
            for steps in STEP_COUNTS:
                for guidance in GUIDANCE_SCALES:
                    with self.subTest(example=example, steps=steps, guidance=guidance):
                        expected = golden_form(example, steps, guidance)
                        got, graph, injected = expanded_form(
                            example, steps, guidance, via_json=True
                        )
                        self.assertEqual(
                            expected,
                            got,
                            "\n" + describe_mismatch(expected, got, graph, injected),
                        )

    def test_authoring_produces_a_region_not_a_macro_node(self):
        workflow = build_workflow(REPRESENTATIVE[0])
        self.assertEqual(len(workflow.regions), 1)
        self.assertFalse(hasattr(workflow, "denoise_nodes"))

    def test_invariants_hold_for_every_example(self):
        """Expansion raises on a malformed graph, so reaching here is the check."""
        for example in standard_loop_workflow_names():
            with self.subTest(example=example):
                expand(example, 2, 7.5)


class TestZeroStepsIsFixed(unittest.TestCase):
    """``num_inference_steps=0`` used to leave the VAE reading a node that was
    never emitted: the legacy patch step ran only on the last iteration, so with
    no iterations it never ran. The loop now aliases its results to the initial
    carry, so the latents pass straight through.
    """

    def test_expansion_succeeds_and_has_no_dangling_edges(self):
        for example in REPRESENTATIVE:
            for guidance in GUIDANCE_SCALES:
                with self.subTest(example=example, guidance=guidance):
                    # The invariant check raises on a dangling edge, so a clean
                    # expansion is the assertion.
                    graph, _ = expand(example, 0, guidance)
                    self.assertGreater(len(graph.workflow_nodes), 0)

    def test_the_output_resolves_to_a_real_node(self):
        graph, _ = expand(REPRESENTATIVE[0], 0, 7.5)
        produced = {
            io.name
            for node in graph.workflow_nodes
            for io in node.get_outputs().values()
        }
        for name in graph.outputs:
            self.assertIn(name, produced)


class TestExecutorView(unittest.TestCase):
    """Self-consistency of what the coordinator derives from the graph.

    This used to be a comparison against the legacy unroller. With that gone it
    checks the properties the executor actually depends on.
    """

    def test_every_node_has_a_depth(self):
        for example in REPRESENTATIVE:
            with self.subTest(example=example):
                graph, injected = expand(example, 3, 7.5)
                view = executor_view(graph, injected)
                self.assertEqual(sum(view["depth"].values()), len(graph.workflow_nodes))

    def test_reference_counts_equal_the_consuming_edge_count(self):
        """A count that is too low frees a tensor a consumer still needs."""
        for example in REPRESENTATIVE:
            for steps in (1, 3):
                with self.subTest(example=example, steps=steps):
                    graph, _ = expand(example, steps, 7.5)
                    expected = {}
                    for node in graph.workflow_nodes:
                        for io in node.get_inputs().values():
                            if io is None or io.source_type == SourceType.INPUT:
                                continue
                            expected[io.name] = expected.get(io.name, 0) + 1
                    self.assertEqual(
                        dependency.build_tensor_reference_count(graph), expected
                    )

    def test_the_only_sink_is_the_output(self):
        """Nothing is computed and thrown away.

        A second sink means some node's result reaches no consumer and is not an
        output either. That used to happen: the CFG workflows encoded the negative
        prompt above the branch, so an idle T5-XXL pass sat in every non-CFG graph.
        Enclosing the whole loop in the cond is what fixed it, and this is what
        stops it coming back.
        """
        for example in standard_loop_workflow_names():
            for guidance in GUIDANCE_SCALES:
                with self.subTest(example=example, guidance=guidance):
                    graph, injected = expand(example, 2, guidance)
                    sinks = executor_view(graph, injected)["sinks"]
                    self.assertEqual(
                        sum(sinks.values()),
                        1,
                        "expected the decode to be the only sink",
                    )


class TestHandWrittenExampleMatchesHelper(unittest.TestCase):
    """``flux_schnell.register_txt2img_control_flow_workflow`` writes the loop out
    by hand with ``for_range``/``cond``. It must come out the same as the
    ``denoise_loop`` version -- otherwise the example teaches the API wrongly,
    which is worse than having no example.
    """

    HAND_WRITTEN = "flux_schnell.register_txt2img_control_flow_workflow"
    VIA_DENOISE_LOOP = "flux_schnell.register_txt2img_workflow"

    def test_matches_the_denoise_loop_version(self):
        for steps in STEP_COUNTS:
            for guidance in GUIDANCE_SCALES:
                with self.subTest(steps=steps, guidance=guidance):
                    expected, _, _ = expanded_form(
                        self.VIA_DENOISE_LOOP, steps, guidance
                    )
                    got, graph, injected = expanded_form(
                        self.HAND_WRITTEN, steps, guidance
                    )
                    self.assertEqual(
                        expected,
                        got,
                        "\n" + describe_mismatch(expected, got, graph, injected),
                    )

    def test_survives_the_json_boundary(self):
        expected, _, _ = expanded_form(self.VIA_DENOISE_LOOP, 3, 7.5)
        got, _, _ = expanded_form(self.HAND_WRITTEN, 3, 7.5, via_json=True)
        self.assertEqual(expected, got)

    def test_it_authors_a_branch_containing_a_loop_per_side(self):
        """One cond at the top, a loop in each branch.

        Not a cond inside one loop: the branch has to enclose the whole loop, so
        that the negative prompt's encoders live inside it and are absent from the
        graph when guidance is off.
        """
        workflow = build_workflow(self.HAND_WRITTEN)
        self.assertEqual(len(workflow.regions), 1)
        top = workflow.regions[0]
        self.assertEqual(top.region_kind, "cond")
        for body in (top.then_body, top.else_body):
            nested = list(body.iter_regions())
            self.assertEqual(
                [region.region_kind for region in nested],
                ["loop"],
                "expected exactly one loop per branch",
            )


if __name__ == "__main__":
    unittest.main()
