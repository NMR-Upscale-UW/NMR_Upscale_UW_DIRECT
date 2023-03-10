import argparse
import random
import torch
import time
from model_params import name2params
from models import name2model
from torch import nn
from datasets import GHzData
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import Subset, DataLoader
import os
import train 
import unittest
from train import regression_dataset_loop, test_loop, train_valid_loop, main


class TestRegressionDatasetLoop(unittest.TestCase):

    def setUp(self):
        # Set up some sample inputs and labels
        self.inputs = torch.randn((32, 10, 3))
        self.labels = torch.randn((32, 1))
        # Create a sample model, optimizer, and criterion
        self.model = torch.nn.Linear(3, 1)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = torch.nn.MSELoss()

    def test_regression_dataset_loop_train(self):
        # Test the function when is_train=True
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(self.inputs,                                                                               self.labels))
        epoch_loss, per_batch_loss = regression_dataset_loop(self.model, self.optimizer,                                                                self.criterion, loader,                                                                    is_train=True)
        # Check that the epoch_loss is a scalar
        self.assertIsInstance(epoch_loss, float, "epoch_loss should be a float, but it is                                                     of type f"{type(epoch_loss)}"")
        # Check that the per_batch_loss is a list with the same length as the number of batches
        self.assertEqual(len(per_batch_loss), len(loader)," per_batch_loss should have             length {len(loader)}, but it has length f"{len(per_batch_loss)}"")

    def test_regression_dataset_loop_eval(self):
        # Test the function when is_train=False
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(self.inputs,                                                                               self.labels))
        epoch_loss, per_batch_loss = regression_dataset_loop(self.model, self.optimizer,                                                                self.criterion, loader,                                                                    is_train=False)
        # Check that the epoch_loss is a scalar
        self.assertIsInstance(epoch_loss, float, "epoch_loss should be a float, but it is                                                     of type f"{type(epoch_loss)}"")
        # Check that the per_batch_loss is a list with the same length as the number of batches
        self.assertEqual(len(per_batch_loss), len(loader)," per_batch_loss should                   have length {len(loader)}, but it has length f"{len(per_batch_loss)}"")


if __name__ == '__main__':
    unittest.main()

    


class TestTestLoop(unittest.TestCase):

    def setUp(self):
        # Set up some sample inputs and labels
        self.inputs = torch.randn((32, 10, 3))
        self.labels = torch.randn((32, 1))
        # Create a sample model and criterion
        self.model = torch.nn.Linear(3, 1)
        self.criterion = torch.nn.MSELoss()

    def test_test_loop(self):
        # Test the function with a sample test loader
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(self.inputs, self.labels))
        test_loss, to_save = test_loop(self.model, self.criterion, test_loader)
        # Check that the test_loss is a scalar
        self.assertIsInstance(test_loss, float, "epoch_loss should be a float, but it is                                                     of type f"{type(test_loss)}"")
        # Check that the to_save is a list with the same length as the number of batches
        self.assertEqual(len(to_save), len(test_loader)," per_batch_loss should                     have length {len(loader)}, but it has length f"{len(per_batch_loss)}"")

if __name__ == '__main__':
    unittest.main()



class TestTrainValidLoop(unittest.TestCase):

    def setUp(self):
        # Set up some sample inputs and labels
        self.inputs = torch.randn((32, 10, 3))
        self.labels = torch.randn((32, 1))
        # Create a sample model, optimizer, and criterion
        self.model = torch.nn.Linear(3, 1)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = torch.nn.MSELoss()
        # Create sample train and validation loaders
        self.train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(self.inputs, self.labels))
        self.valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(self.inputs, self.labels))

    def test_train_valid_loop(self):
        # Test the function with a sample train and validation loader
        model = train_valid_loop(self.model, self.train_loader, self.valid_loader,                                          self.optimizer, self.criterion, num_epochs=5)
        # Check that the model is an instance of torch.nn.Module
        self.assertIsInstance(model, torch.nn.Module, "model should be an instance of                                     torch.nn.Module, but it is of type f"{type(model)}"")

if __name__ == '__main__':
    unittest.main()




class TestMain(unittest.TestCase):

    def test_main(self):
        # Test the function with some sample arguments
        args = argparse.Namespace(model_name='mlp', num_epochs=5, high_resolution_frequency=400, data_dir='./data/', random_key=1234, train_split=0.7, valid_split=0.15)
        main(args)
        # Check that the predictions.pt file is created
        self.assertTrue(os.path.exists("predictions.pt"), "The predictions.pt file does not                                                             exist.")

if __name__ == '__main__':
    unittest.main()
