import unittest

from benchmark_ops.shapes import (
    Shape,
    ShapeSweep,
    group_by_resolution,
    parse_batch_sizes,
    parse_resolutions,
)


class TestShapeSweep(unittest.TestCase):
    def test_expand_iterates_batch_sizes_within_each_resolution(self):
        sweep = ShapeSweep(batch_sizes=(1, 2), resolutions=((512, 512), (1024, 768)))

        self.assertEqual(
            sweep.expand(),
            [
                Shape(batch_size=1, height=512, width=512),
                Shape(batch_size=2, height=512, width=512),
                Shape(batch_size=1, height=1024, width=768),
                Shape(batch_size=2, height=1024, width=768),
            ],
        )

    def test_from_dict_falls_back_to_defaults(self):
        self.assertEqual(ShapeSweep.from_dict(None), ShapeSweep.default())
        self.assertEqual(ShapeSweep.from_dict({}), ShapeSweep.default())

    def test_from_dict_keeps_partial_overrides(self):
        sweep = ShapeSweep.from_dict({"batch_sizes": [8]})

        self.assertEqual(sweep.batch_sizes, (8,))
        self.assertEqual(sweep.resolutions, ShapeSweep.default().resolutions)

    def test_reference_shape_is_smallest_batch_at_first_resolution(self):
        sweep = ShapeSweep(
            batch_sizes=(4, 1, 2), resolutions=((1024, 1024), (512, 512))
        )

        self.assertEqual(
            sweep.reference_shape(),
            Shape(batch_size=1, height=1024, width=1024),
        )

    def test_group_by_resolution_preserves_sweep_order(self):
        sweep = ShapeSweep(batch_sizes=(1, 4), resolutions=((512, 512), (1024, 1024)))

        self.assertEqual(
            group_by_resolution(sweep.expand()),
            {(512, 512): [1, 4], (1024, 1024): [1, 4]},
        )

    def test_shape_round_trips_through_dict(self):
        shape = Shape(batch_size=2, height=768, width=512)

        self.assertEqual(Shape.from_dict(shape.to_dict()), shape)


class TestCliParsing(unittest.TestCase):
    def test_parse_batch_sizes(self):
        self.assertEqual(parse_batch_sizes("1,2, 4"), (1, 2, 4))

    def test_parse_batch_sizes_rejects_non_positive(self):
        with self.assertRaises(ValueError):
            parse_batch_sizes("0")

    def test_parse_batch_sizes_rejects_empty(self):
        with self.assertRaises(ValueError):
            parse_batch_sizes(",")

    def test_parse_resolutions_reads_height_then_width(self):
        self.assertEqual(
            parse_resolutions("512x512,1024x768"), ((512, 512), (1024, 768))
        )

    def test_parse_resolutions_requires_separator(self):
        with self.assertRaises(ValueError):
            parse_resolutions("512")


if __name__ == "__main__":
    unittest.main()
