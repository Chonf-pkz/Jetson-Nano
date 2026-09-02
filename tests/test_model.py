import unittest

import torch

from src.model import LaneTracker
from src.preprocessing_config import MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH


class CompactModelTests(unittest.TestCase):
    def test_output_shape_and_range(self):
        model = LaneTracker(pretrained=False)
        sample = torch.randn(4, 3, MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH)
        output = model(sample)
        self.assertEqual(tuple(output.shape), (4,))
        self.assertTrue(bool(torch.all(output >= -1.0)))
        self.assertTrue(bool(torch.all(output <= 1.0)))

    def test_model_is_small_enough_for_nano(self):
        _, total = LaneTracker().count_parameters()
        self.assertLess(total, 300_000)


if __name__ == '__main__':
    unittest.main()
