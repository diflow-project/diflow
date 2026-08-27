"""Operators must not silently substitute randomly-initialised weights.

They used to: a ``model_path`` that did not exist fell through to a dummy config,
the pipeline ran end to end, and the image came out as noise. Every tensor had a
plausible shape and magnitude, so it read as a bug in the denoising logic. It cost
a full debugging pass to trace back to a wrong path in a test harness.
"""

import logging
import unittest

from diflow.operators.base import has_pretrained_weights
from diflow.operators.utils import get_op

# One per family that loads weights from a path.
OPS_THAT_LOAD_WEIGHTS = [
    "Flux1Schnell",
    "Flux1Dev",
    "CLIP_Flux",
    "T5_Flux",
    "Flux1VAE",
    "FluxSchnellFlowMatchEulerDiscreteScheduler",
    "FluxFlowMatchEulerDiscreteScheduler",
    "ZImage",
    "Qwen3_ZImage",
    "ZImageVAE",
    "ZImageFlowMatchEulerDiscreteScheduler",
    "Flux2Klein",
    "Qwen3_Flux2Klein",
    "Flux2VAE",
    "Flux2FlowMatchEulerDiscreteScheduler",
]


class TestHasPretrainedWeights(unittest.TestCase):
    def test_missing_path_raises_and_names_it(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            has_pretrained_weights("/nope/not/here", "SomeOp")
        message = str(ctx.exception)
        self.assertIn("/nope/not/here", message)
        self.assertIn("SomeOp", message)
        self.assertIn("randomly initialised", message)

    def test_none_is_allowed_but_warns(self):
        """The memory-profiling entry points pass None on purpose."""
        with self.assertLogs("diflow.operators.base", logging.WARNING) as logs:
            self.assertFalse(has_pretrained_weights(None, "SomeOp"))
        self.assertIn("UNINITIALISED", "\n".join(logs.output))

    def test_existing_path_is_accepted(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(has_pretrained_weights(tmp, "SomeOp"))


class TestOperatorsRefuseAMissingPath(unittest.TestCase):
    """Wired up at every call site, not just available."""

    def test_initialize_raises_for_each_op(self):
        for op_id in OPS_THAT_LOAD_WEIGHTS:
            with self.subTest(op=op_id):
                op = get_op(op_id, "/dummy/model/path")
                with self.assertRaises(FileNotFoundError):
                    op.initialize("/dummy/model/path", "cpu")

    def test_operators_without_dummy_configs_require_model_path(self):
        for op_id in (
            "CLIP_Flux",
            "T5_Flux",
            "ZImage",
            "Qwen3_ZImage",
            "ZImageVAE",
            "ZImageFlowMatchEulerDiscreteScheduler",
            "Flux2Klein",
            "Qwen3_Flux2Klein",
            "Flux2VAE",
            "Flux2FlowMatchEulerDiscreteScheduler",
        ):
            with self.subTest(op=op_id):
                op = get_op(op_id)
                with self.assertRaisesRegex(ValueError, "model_path is required"):
                    op.initialize(None, "cpu")


if __name__ == "__main__":
    unittest.main()
