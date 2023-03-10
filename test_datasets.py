import unittest
from your_module import GHzData

class TestGHzData(unittest.TestCase):

    def test_data_loading(self):
        # Test the function with default arguments
        dataset = GHzData()
        # Check that the files list is not empty
        self.assertGreater(len(dataset.files), 0,"The files list is empty" )
        # Check that the shape of tensor_low and tensor_high is the same
        self.assertEqual(dataset.tensor_low.shape, dataset.tensor_high.shape, " The shapes of tensor_low and tensor_high are not the same. tensor_low has shape f"{dataset.tensor_low.shape}", while tensor_high has shape f" {dataset.tensor_high.shape}"
")

if __name__ == '__main__':
    unittest.main()
