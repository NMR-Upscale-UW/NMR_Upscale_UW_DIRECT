# Import the necessary libraries for logging experiments with Comet.ml
# from comet_ml import Experiment

# Import core PyTorch functionality: PyTorch base, Neural Network module, and CUDA device support
import torch
import torch.nn as nn

# Import additional Python libraries for data manipulation, file I/O, and plotting
import numpy as np
import os
import glob
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
from torch.cuda.amp import autocast, GradScaler

# Import PyTorch classes for building a neural network model
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout

# Import PyTorch optimizer classes for gradient-based optimization
from torch.optim import Adam, SGD, Adagrad, RMSprop, SparseAdam, LBFGS, Adadelta

# Import utility from scikit-learn for splitting data into train and validation sets
from sklearn.model_selection import train_test_split

# Import additional Python libraries for mathematical operations and plotting
import math
import matplotlib.pyplot
import time

# Import PyTorch Functional API for activation functions and other operations
import torch.nn.functional as F

# Import nmrsim library for NMR simulation
import nmrsim
from nmrsim import plt
from nmrsim import Multiplet

# Import random functions for generating random numbers
from random import randint, uniform

# Import itertools for generating Cartesian products of input iterables
from itertools import product

# Import Optuna library for hyperparameter optimization
import optuna

# Import PyTorch optim module for optimization algorithms
import torch.optim as optim

import csv

# Set the device to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} device")

class GHzData(Dataset):
    def __init__(self):
        # Load the list of CSV file paths.
        self.files = glob.glob(os.path.join('/home/fostooq/NMR_Upscale_UW_DIRECT/sprectra_data/400MHz/spectral_data_*.csv'))

        self.y_60 = []  # List to store 60 MHz data.
        self.y_400 = []  # List to store 400 MHz data.

        # Loop through each file in the list of files.
        for self.file in self.files:
            # Read each file into a pandas DataFrame.
            self.df = pd.read_csv(self.file)

            # Extract 60 MHz and 400 MHz intensity columns and convert them to numpy arrays.
            self.array_60 = self.df['60MHz_intensity'].to_numpy()
            self.array_400 = self.df['400MHz_intensity'].to_numpy()

            # Append the arrays to the respective lists for 60 MHz and 400 MHz data.
            self.y_60.append(self.array_60)
            self.y_400.append(self.array_400)

        # Convert the 60 MHz list to a tensor, change dtype to float, and add a dimension (n, 1, 5500).
        self.tensor_60 = torch.Tensor(self.y_60).float().unsqueeze(1).to(device)

        # Convert the 400 MHz list to a tensor, change dtype to float, and add a dimension (n, 1, 5500).
        self.tensor_400 = torch.Tensor(self.y_400).float().unsqueeze(1).to(device)

        # Store the number of samples in the dataset.
        self.num_samples = len(self.y_60)

    def __getitem__(self, index):  # Method to retrieve an item from the dataset using an index.
        return self.tensor_60[index], self.tensor_400[index]

    def __len__(self):  # Method to return the total number of samples in the dataset.
        return self.num_samples

# Establishing and loading data into notebook
dataset = GHzData()

#Splitting the data
train_X, test_X, train_y, test_y = train_test_split(dataset.tensor_60, dataset.tensor_400,
                                                    test_size=0.1)

# Splits train data into validation data
train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y,
                                                      test_size=0.1)
# Creating datasets
train_dataset = TensorDataset(train_X, train_y)
test_dataset = TensorDataset(test_X, test_y)
valid_dataset = TensorDataset(valid_X, valid_y)

class CNN(nn.Module):
    def __init__(self, num_layers, num_channels, kernel_size, drop_out):
        super().__init__()
        prev_dim = num_channels
        k = kernel_size
        layers = [nn.Conv1d(1, prev_dim, kernel_size=k, padding='same'), nn.ReLU(), nn.Dropout(p=drop_out)]

        for _ in range(1, num_layers):
            layers.append(nn.Conv1d(prev_dim, num_channels, kernel_size=k, padding='same'))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=drop_out))

        layers.append(nn.Conv1d(prev_dim, 1, kernel_size=k, padding='same'))
        self.m = nn.Sequential(*layers)

    def forward(self, x):
        return self.m(x)
def fit(model, dataloader, optimizer, criterion, accumulation_steps=4):
    model.train()
    running_loss = 0.0
    scaler = GradScaler()

    for i, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device).to(next(model.parameters()).dtype), targets.to(device)

        # Forward pass
        outputs = model(inputs)

        # Compute the loss
        loss = criterion(outputs, targets)

        # Normalize the loss
        loss = loss / accumulation_steps

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        # Update the model weights after accumulating gradients from 'accumulation_steps' mini-batches
        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * accumulation_steps

    return running_loss / len(dataloader)

def validate(model, dataloader, optimizer, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)


def objective_structure(trial):


    # Define model hyperparameters
    num_layers = trial.suggest_int('num_layers', 3, 5, step=2)
    num_channels = trial.suggest_int('num_channels', 32, 64, step=32)
    kernel_size = trial.suggest_int('kernel_size', 3, 5, step=2)
    drop_out = trial.suggest_float("drop_out", 0.0, 0.5)

    model = CNN(num_layers, num_channels, kernel_size, drop_out).to(device)
    criterion = nn.MSELoss()

    num_epoch = 30

    # Initialize best value and loss history
    best_value_inner = float('inf')
    loss_history = {'train': [], 'val': []}

    def objective_inner(trial):
    
        # Define optimizer hyperparameters
        optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'Adadelta', 'Adagrad', 'RMSprop', 'SGD'])
        lr = trial.suggest_float('lr', 1e-2, 1e-1, log=True)
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
        batch_size = trial.suggest_int('batch_size', 64, 96, step=32)
        
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
  

        # Training loop
        for epoch in range(num_epoch):
            train_epoch_loss = fit(model, train_dataloader, optimizer, criterion)
            loss_history['train'].append(train_epoch_loss)

            val_epoch_loss = validate(model, valid_dataloader, optimizer, criterion)
            loss_history['val'].append(val_epoch_loss)

            trial.report(val_epoch_loss, epoch)

            # Handle pruning
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # Save the best model
        nonlocal best_value_inner
        if trial.value < best_value_inner:
            best_value_inner = trial.value
            PATH = f"model_{trial.number}.pt"
            torch.save({
                'trial_number': trial.number,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_value_inner,
            }, PATH)

        return trial.value

    study_inner = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
    study_inner.optimize(objective_inner, n_trials=2)

    return best_value_inner

study = optuna.create_study(direction='minimize')
study.optimize(objective_structure, n_trials=2)

# Get the best trial
trial = study.best_trial

# Save the best trial parameters into a csv file
with open('best_trial.csv', 'w', newline='') as csvfile:
    fieldnames = ['Parameter', 'Value']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerow({'Parameter': 'Trial Value', 'Value': trial.value})
    for key, value in trial.params.items():
        writer.writerow({'Parameter': key, 'Value': value})

