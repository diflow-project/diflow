"""CPU-only tests for automatic workflow benchmark planning."""

import unittest

from benchmark_ops.profiler import (
    ProfileSettings,
    _split_occurrences,
    build_inputs,
    plan_case,
    solve_occurrences,
)
from benchmark_ops.shapes import Shape, ShapeSweep
from benchmark_ops.workflow_cases import WorkflowProfileCase, load_workflow_factory

SINGLE_SHAPE_SWEEP = ShapeSweep(batch_sizes=(1, 2), resolutions=((512, 512),))
FLUX_FACTORY = "workflow_hub/flux_schnell/register_txt2img_workflow.py:create_workflow"


def _flux_case(cfg_guidance_scale: float = 1.0) -> WorkflowProfileCase:
    return WorkflowProfileCase(
        name="flux_schnell",
        suite="test",
        factory=FLUX_FACTORY,
        factory_kwargs={"model_path": "/tmp/model"},
        inputs={
            "prompt": "hello",
            "negative_prompt": "bad",
            "cfg_guidance_scale": cfg_guidance_scale,
            "seed": 0,
            "guidance_scale": 0.0,
        },
    )


def _occurrences_by_op(case: WorkflowProfileCase):
    solved = solve_occurrences(
        workflow=case.build_workflow(),
        case=case,
        shape=Shape(batch_size=1, height=512, width=512),
        profile_steps=2,
    )
    return {
        (signature[0], signature[1]): counts for signature, counts in solved.items()
    }


class TestBuildInputs(unittest.TestCase):
    def test_shape_inputs_are_overridden_without_mutating_the_case(self):
        case = WorkflowProfileCase(
            name="case",
            suite="test",
            factory="unused.py:create_workflow",
            inputs={"prompt": "hello", "height": 64, "num_inference_steps": 50},
        )
        inputs = build_inputs(case, Shape(batch_size=1, height=1024, width=768), 2)

        self.assertEqual(inputs["height"], 1024)
        self.assertEqual(inputs["width"], 768)
        self.assertEqual(inputs["num_inference_steps"], 2)
        self.assertEqual(case.inputs["height"], 64)


class TestSplitOccurrences(unittest.TestCase):
    def test_occurrences_are_split_proportionally(self):
        self.assertEqual(
            _split_occurrences({"constant": 4, "per_step": 2}, 1, 2),
            {"constant": 2, "per_step": 1},
        )


class TestSolveOccurrences(unittest.TestCase):
    def test_loop_body_scales_per_step(self):
        occurrences = _occurrences_by_op(_flux_case())

        self.assertEqual(
            occurrences[("Flux1Schnell", "default")],
            {"constant": 0, "per_step": 1},
        )
        self.assertEqual(
            occurrences[("FluxSchnellFlowMatchEulerDiscreteScheduler", "step")],
            {"constant": 0, "per_step": 1},
        )
        self.assertEqual(
            occurrences[("CLIP_Flux", "default")],
            {"constant": 1, "per_step": 0},
        )

    def test_classifier_free_guidance_doubles_conditioned_ops(self):
        occurrences = _occurrences_by_op(_flux_case(cfg_guidance_scale=2.0))

        self.assertEqual(
            occurrences[("CLIP_Flux", "default")],
            {"constant": 2, "per_step": 0},
        )
        self.assertEqual(
            occurrences[("Flux1Schnell", "default")],
            {"constant": 0, "per_step": 2},
        )
        self.assertEqual(
            occurrences[
                (
                    "FluxSchnellFlowMatchEulerDiscreteScheduler",
                    "step_classifier_free_guidance",
                )
            ],
            {"constant": 0, "per_step": 1},
        )


class TestPlanCase(unittest.TestCase):
    def test_non_batchable_ops_are_only_planned_at_batch_one(self):
        planned = plan_case(
            _flux_case(),
            ProfileSettings(profile_steps=2),
            SINGLE_SHAPE_SWEEP,
        )
        batch_sizes_by_op = {}
        for entry in planned:
            batch_sizes_by_op.setdefault(entry["op_id"], set()).add(
                entry["shape"].batch_size
            )

        self.assertEqual(
            batch_sizes_by_op["FluxSchnellFlowMatchEulerDiscreteScheduler"], {1}
        )
        self.assertEqual(batch_sizes_by_op["IndexedTensor"], {1})
        self.assertEqual(batch_sizes_by_op["Flux1Schnell"], {1, 2})

    def test_profile_steps_do_not_change_measurement_count(self):
        case = _flux_case()
        few = plan_case(case, ProfileSettings(profile_steps=2), SINGLE_SHAPE_SWEEP)
        many = plan_case(case, ProfileSettings(profile_steps=8), SINGLE_SHAPE_SWEEP)
        self.assertEqual(len(few), len(many))


class TestWorkflowFactoryLoading(unittest.TestCase):
    def test_invalid_factory_references_are_reported(self):
        with self.assertRaises(FileNotFoundError):
            load_workflow_factory("workflow_hub/missing.py:create_workflow")
        with self.assertRaises(AttributeError):
            load_workflow_factory(
                "workflow_hub/flux_schnell/register_txt2img_workflow.py:missing"
            )
        with self.assertRaises(ValueError):
            load_workflow_factory(
                "workflow_hub/flux_schnell/register_txt2img_workflow.py"
            )


if __name__ == "__main__":
    unittest.main()
