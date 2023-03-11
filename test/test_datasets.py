import unittest
import sys
sys.path.append("../NMR_Upscale_UW_DIRECT")
from datasets import GHzData

class TestGHzData(unittest.TestCase):

    def test_data_loading(self):
        # Test the function with default arguments
        dataset = GHzData()
        # Check that the files list is not empty
        self.assertGreater(len(dataset.tensor_low), 0, "The dataset is empty")
        # Check that the shape of tensor_low and tensor_high is the same
        self.assertEqual(dataset.tensor_low.shape, dataset.tensor_high.shape, f"The shapes of tensor_low and tensor_high are not the same. tensor_low has shape {dataset.tensor_low.shape}, while tensor_high has shape {dataset.tensor_high.shape}")

        x, y = dataset[0]
        # Check that the shape of tensor_low and tensor_high is the same
        self.assertGreater(x.sum(), 0.0, "The sample has at least one peak")

if __name__ == '__main__':
    unittest.main()
