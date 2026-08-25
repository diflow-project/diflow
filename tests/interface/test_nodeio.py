import unittest
from typing import List

from PIL.Image import Image
from torch import Tensor

from diflow.interface.node_io import NodeIO, SourceType


class TestNodeIO(unittest.TestCase):
    """Test cases for the NodeIO class."""

    def test_basic_types(self):
        """Test NodeIO with basic Python types."""
        test_cases = [
            (int, "builtins.int"),
            (float, "builtins.float"),
            (str, "builtins.str"),
            (bool, "builtins.bool"),
        ]

        for data_type, expected_data_type in test_cases:
            with self.subTest(data_type=data_type):
                node_io = NodeIO(name="test", data_type=data_type)
                io_dict = node_io.to_dict()

                self.assertEqual(io_dict["name"], "test")
                self.assertEqual(io_dict["data_type"], expected_data_type)
                self.assertIsNone(io_dict.get("source_type"))
                self.assertIsNone(io_dict.get("source_node"))

                # Test roundtrip
                reconstructed = NodeIO.from_dict(io_dict)
                self.assertEqual(reconstructed.name, "test")
                self.assertEqual(reconstructed.data_type, data_type)
                self.assertIsNone(reconstructed.source_type)
                self.assertIsNone(reconstructed.source_node)

    def test_special_types(self):
        """Test NodeIO with special types like Tensor and Image."""
        test_cases = [
            (Tensor, "torch.Tensor"),
            (Image, "PIL.Image.Image"),
        ]

        for data_type, expected_data_type in test_cases:
            with self.subTest(data_type=data_type):
                node_io = NodeIO(name="test", data_type=data_type)
                io_dict = node_io.to_dict()

                self.assertEqual(io_dict["name"], "test")
                self.assertEqual(io_dict["data_type"], expected_data_type)
                self.assertIsNone(io_dict.get("source_type"))
                self.assertIsNone(io_dict.get("source_node"))

                reconstructed = NodeIO.from_dict(io_dict)
                self.assertEqual(reconstructed.name, "test")
                self.assertEqual(reconstructed.data_type, data_type)
                self.assertIsNone(reconstructed.source_type)
                self.assertIsNone(reconstructed.source_node)

    def test_generic_types(self):
        """Test NodeIO with generic types like List."""
        test_cases = [
            (List[int], "typing.List[builtins.int]"),
            (List[float], "typing.List[builtins.float]"),
            (List[Tensor], "typing.List[torch.Tensor]"),
            (List[Image], "typing.List[PIL.Image.Image]"),
            (List[List[Tensor]], "typing.List[typing.List[torch.Tensor]]"),
            (List[List[Image]], "typing.List[typing.List[PIL.Image.Image]]"),
        ]
        for data_type, expected_data_type in test_cases:
            with self.subTest(data_type=data_type):
                node_io = NodeIO(name="test", data_type=data_type)
                io_dict = node_io.to_dict()

                self.assertEqual(io_dict["name"], "test")
                self.assertEqual(io_dict["data_type"], expected_data_type)
                self.assertIsNone(io_dict.get("source_type"))
                self.assertIsNone(io_dict.get("source_node"))

                reconstructed = NodeIO.from_dict(io_dict)
                self.assertEqual(reconstructed.name, "test")
                self.assertEqual(reconstructed.data_type, data_type)
                self.assertIsNone(reconstructed.source_type)
                self.assertIsNone(reconstructed.source_node)

    def test_source_type_and_node(self):
        """Test NodeIO with source type and node information."""
        node_io = NodeIO(
            name="test",
            data_type=int,
            source_type=SourceType.NODE,
            source_node="source_node",
        )
        io_dict = node_io.to_dict()

        self.assertEqual(io_dict["name"], "test")
        self.assertEqual(io_dict["data_type"], "builtins.int")
        self.assertEqual(io_dict["source_type"], SourceType.NODE.value)
        self.assertEqual(io_dict["source_node"], "source_node")

        reconstructed = NodeIO.from_dict(io_dict)
        self.assertEqual(reconstructed.name, "test")
        self.assertEqual(reconstructed.data_type, int)
        self.assertEqual(reconstructed.source_type, SourceType.NODE)
        self.assertEqual(reconstructed.source_node, "source_node")

    def test_size(self):
        """Test NodeIO with size information."""
        node_io = NodeIO(
            name="test",
            data_type=Tensor,
            size=[10, 10],
        )
        io_dict = node_io.to_dict()

        self.assertEqual(io_dict["name"], "test")
        self.assertEqual(io_dict["data_type"], "torch.Tensor")
        self.assertEqual(io_dict["size"], [10, 10])

        reconstructed = NodeIO.from_dict(io_dict)
        self.assertEqual(reconstructed.name, "test")
        self.assertEqual(reconstructed.data_type, Tensor)
        self.assertEqual(reconstructed.size, [10, 10])

    def test_size_list(self):
        """Test NodeIO with size information."""
        node_io = NodeIO(
            name="test",
            data_type=List[Tensor],
            size=[[10, 10], [100, 100]],
        )
        io_dict = node_io.to_dict()

        self.assertEqual(io_dict["name"], "test")
        self.assertEqual(io_dict["data_type"], "typing.List[torch.Tensor]")
        self.assertEqual(io_dict["size"], [[10, 10], [100, 100]])

        reconstructed = NodeIO.from_dict(io_dict)
        self.assertEqual(reconstructed.name, "test")
        self.assertEqual(reconstructed.data_type, List[Tensor])
        self.assertEqual(reconstructed.size, [[10, 10], [100, 100]])


if __name__ == "__main__":
    unittest.main()
