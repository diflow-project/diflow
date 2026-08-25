import unittest

import torch

from diflow.operators.custom.latents_generator import LatentsGenerator


class TestLatentsGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test case - initialize model once for all tests"""
        cls.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize LatentsGenerator
        cls.latents_generator = LatentsGenerator()

    def test_execution_without_input_latents(self):
        """Test model execution without providing input latents"""
        result = self.latents_generator.execute(
            model_components={},  # No model components needed
            device=self.device,
            batch_size=1,
            num_channels_latents=4,
            height=512,
            width=512,
            dtype=(
                torch.float16 if str(self.device).startswith("cuda") else torch.float32
            ),
            seed=42,
        )

        # Check output format
        self.assertIn("latents", result)
        latents = result["latents"]

        # Check if output is a tensor
        self.assertIsInstance(latents, torch.Tensor)

        # Check tensor properties
        self.assertEqual(latents.device.type, self.device)
        self.assertEqual(
            latents.shape, (1, 4, 64, 64)
        )  # 512/8 = 64 due to vae_scale_factor
        if str(self.device).startswith("cuda"):
            self.assertEqual(latents.dtype, torch.float16)
        else:
            self.assertEqual(latents.dtype, torch.float32)

    def test_execution_with_input_latents(self):
        """Test model execution with provided input latents"""
        input_latents = torch.randn(1, 4, 64, 64)

        result = self.latents_generator.execute(
            model_components={},
            device=self.device,
            latents=input_latents,
            dtype=(
                torch.float16 if str(self.device).startswith("cuda") else torch.float32
            ),
        )

        # Check if output matches input latents
        self.assertIn("latents", result)
        output_latents = result["latents"]

        self.assertEqual(output_latents.shape, input_latents.shape)
        self.assertEqual(output_latents.device.type, self.device)

    @classmethod
    def tearDownClass(cls):
        """Clean up resources"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
