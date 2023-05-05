import argparse
import random
import torch
import time
from torch import nn
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import Subset, DataLoader
import os
import unittest

import sys

sys.path.append("../NMR_Upscale_UW_DIRECT")
from datasets import GHzData
import train 
from train import regression_dataset_loop, train_valid_loop, main
from model_params import name2params
from models import name2model


class TestRegressionDatasetLoop(unittest.TestCase):

    def setUp(self):
        # Set up some sample inputs and labels
        self.inputs = torch.randn((32, 10, 3))
        self.labels = torch.randn((32, 10, 1))
        # Create a sample model, optimizer, and criterion
        self.model = torch.nn.Linear(3, 1)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = torch.nn.MSELoss()

    def test_regression_dataset_loop_train(self):
        # Test the function when is_train=True
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(self.inputs, self.labels))
        epoch_loss, per_batch_loss, model = regression_dataset_loop(self.model, self.optimizer, self.criterion, loader, is_train=True)
        self.assertIsInstance(model, nn.Module)
        # Check that the epoch_loss is a scalar
        self.assertIsInstance(epoch_loss, float, f"epoch_loss should be a float, but it is of type {type(epoch_loss)}")
        # Check that the per_batch_loss is a list with the same length as the number of batches
        self.assertEqual(len(per_batch_loss), len(loader), f"per_batch_loss should have length {len(loader)}, but it has length {len(per_batch_loss)}")

    def test_regression_dataset_loop_eval(self):
        # Test the function when is_train=False
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(self.inputs, self.labels))
        epoch_loss, per_batch_loss, model = regression_dataset_loop(self.model, self.optimizer, self.criterion, loader, is_train=False)
        self.assertIsInstance(model, nn.Module)
        # Check that the epoch_loss is a scalar
        self.assertIsInstance(epoch_loss, float, f"epoch_loss should be a float, but it is of type {type(epoch_loss)}")
        # Check that the per_batch_loss is a list with the same length as the number of batches
        self.assertEqual(len(per_batch_loss), len(loader), f"per_batch_loss should have length {len(loader)}, but it has length {len(per_batch_loss)}")

class TestTrainValidLoop(unittest.TestCase):

    def setUp(self):
        # Set up some sample inputs and labels
        self.inputs = torch.randn((32, 10, 3))
        self.labels = torch.randn((32, 10, 1))
        # Create a sample model, optimizer, and criterion
        self.model = torch.nn.Linear(3, 1)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = torch.nn.MSELoss()
        # Create sample train and validation loaders
        self.train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(self.inputs, self.labels))
        self.valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(self.inputs, self.labels))

    def test_train_valid_loop(self):
        # Test the function with a sample train and validation loader
        train_loss_epoch, valid_loss_epoch, best_model_ckpt = train_valid_loop(self.model, self.train_loader, self.valid_loader, self.optimizer, self.criterion, num_epochs=1)
        # Check that the model is an instance of torch.nn.Module
        self.assertTrue(len(train_loss_epoch) >= len(valid_loss_epoch))
        self.assertTrue('epoch' in best_model_ckpt)
        self.assertIsInstance(best_model_ckpt['epoch'], int)
        self.assertTrue('avg_valid_loss' in best_model_ckpt)
        self.assertIsInstance(best_model_ckpt['avg_valid_loss'], float)
        self.assertTrue('state_dict' in best_model_ckpt)


class TestMain(unittest.TestCase):

    def test_main(self):
        # Test the function with some sample arguments
        args = argparse.Namespace(
            model_name='mlp', 
            num_epochs=1, 
            high_resolution_frequency=400,
            data_dir='./data/', 
            random_key=1234, 
            train_split=0.7, 
            valid_split=0.15, 
            save_dir='./results_test/', 
            limit_for_test=1)
        main(args)
        # Check that the predictions.pt file is created
        self.assertTrue(os.path.exists("./results_test/400MHz/mlp/predictions.pt"), "The predictions.pt file does not exist.")

if __name__ == '__main__':
    unittest.main()
