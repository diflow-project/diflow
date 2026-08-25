"""Tests for the worker's input deserialization.

This exists because a change to ``NodeIO`` broke every served request while the
whole interface test suite stayed green: nothing under ``tests/interface`` reaches
the worker. ``_deserialize_inputs`` touches no worker state, so it can be called
unbound.
"""

import base64
import io
import unittest

import torch
from PIL import Image

from diflow.backend.worker import DistributedWorker
from diflow.interface.node_io import NodeIO, SourceType
from diflow.interface.workflow_node import WorkflowNode
from diflow.operators.custom.indexed_tensor import IndexedTensor


def deserialize(inputs, node):
    # No worker state is touched, so an unbound call avoids standing up ZMQ,
    # NVSHMEM and a GPU.
    return DistributedWorker._deserialize_inputs(None, inputs, node)


class TestDeserializeInputs(unittest.TestCase):
    def setUp(self):
        self.node = WorkflowNode(
            op=IndexedTensor(),
            inputs={
                "tensor": NodeIO(
                    name="Producer_x:out",
                    data_type=torch.Tensor,
                    source_type=SourceType.NODE,
                    source_node="Producer_x",
                ),
                "index": NodeIO(
                    name="seed", data_type=int, source_type=SourceType.INPUT
                ),
            },
        )

    def test_non_image_inputs_pass_through(self):
        """Exercises the ``if input_info and ...`` presence check.

        A ``NodeIO`` that raised on truth-testing made this blow up for every
        request, with the failure surfacing as a dead worker thread.
        """
        out = deserialize({"tensor": None, "index": 3}, self.node)
        self.assertEqual(out, {"tensor": None, "index": 3})

    def test_input_absent_from_the_node_is_still_passed_through(self):
        """input_info is None here, so the presence check must short-circuit."""
        out = deserialize({"unexpected": 7}, self.node)
        self.assertEqual(out, {"unexpected": 7})

    def test_base64_image_input_is_decoded(self):
        node = WorkflowNode(
            op=IndexedTensor(),
            inputs={
                "tensor": NodeIO(
                    name="init_image",
                    data_type=Image.Image,
                    source_type=SourceType.INPUT,
                ),
                "index": NodeIO(
                    name="seed", data_type=int, source_type=SourceType.INPUT
                ),
            },
        )
        buffer = io.BytesIO()
        Image.new("RGB", (4, 4), (10, 20, 30)).save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode()

        out = deserialize({"tensor": encoded, "index": 0}, node)
        self.assertIsInstance(out["tensor"], Image.Image)
        self.assertEqual(out["tensor"].size, (4, 4))

    def test_undecodable_image_raises_with_the_input_name(self):
        node = WorkflowNode(
            op=IndexedTensor(),
            inputs={
                "tensor": NodeIO(
                    name="init_image",
                    data_type=Image.Image,
                    source_type=SourceType.INPUT,
                ),
            },
        )
        with self.assertRaises(ValueError) as ctx:
            deserialize({"tensor": "not-base64!!"}, node)
        self.assertIn("tensor", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
