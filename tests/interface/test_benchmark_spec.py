import unittest

from diflow.interface import BenchmarkSpec, Workflow
from diflow.interface.workflow_expand import expand_workflow


class TestBenchmarkSpec(unittest.TestCase):
    def test_defaults_match_the_automatic_sweep(self):
        spec = BenchmarkSpec(inputs={"prompt": "hello"})

        self.assertEqual(spec.batch_sizes, (1, 2, 4, 8))
        self.assertEqual(
            spec.resolutions,
            ((256, 256), (512, 512), (1024, 1024)),
        )
        self.assertEqual(spec.warmup, 2)
        self.assertEqual(spec.repeats, 5)

    def test_empty_non_shape_inputs_are_supported(self):
        self.assertEqual(BenchmarkSpec(inputs={}).inputs, {})

    def test_inputs_are_copied(self):
        inputs = {"prompt": "hello"}
        spec = BenchmarkSpec(inputs=inputs)
        inputs["prompt"] = "changed"

        self.assertEqual(spec.inputs["prompt"], "hello")

    def test_invalid_values_are_rejected(self):
        with self.assertRaises(ValueError):
            BenchmarkSpec(inputs={"prompt": "x"}, batch_sizes=(0,))
        with self.assertRaises(ValueError):
            BenchmarkSpec(inputs={"prompt": "x"}, resolutions=((0, 256),))

    def test_expand_preserves_benchmark_metadata(self):
        spec = BenchmarkSpec(inputs={"prompt": "hello"})
        workflow = Workflow("empty", benchmark=spec)

        expanded = expand_workflow(workflow, {})

        self.assertIs(expanded.benchmark, spec)


if __name__ == "__main__":
    unittest.main()
