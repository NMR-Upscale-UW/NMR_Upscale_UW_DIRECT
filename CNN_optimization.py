# Loading in libraries necessary for CNN
import torch
import torch.nn as nn
import numpy as np
import os
import glob
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD, Adagrad, RMSprop, SparseAdam, LBFGS, Adadelta
from sklearn.model_selection import train_test_split
import math
import matplotlib.pyplot
import time
import torch.nn.functional as F
import nmrsim
from nmrsim import plt
from itertools import product
import CNN_model



def objective(trial):
    # Generate the model
    model = LinearVAE(trial).to(device)
    # Generate optimizers
    # Try Adam, AdaDelta, Adagrad, RMSprop, SGD
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'Adadelta', 'Adagrad', 'RMSprop', 'SGD'])
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    batch_size_trial = trial.suggest_int('batch_size', 64, 256, step=64)
    num_epochs = trial.suggest_int('num_epochs', 5, 50, step=5)
    criterion = nn.MSELoss()
    # Load Data
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
    # Batch size change to higher batch sizes
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_trial, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size_trial, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size_trial, shuffle=True)
    # Paste training loop here
    for epoch in range(num_epochs):
        train_epoch_loss = fit(model, train_dataloader, optimizer, criterion)
        val_epoch_loss = validate(model, valid_dataloader, optimizer, criterion)
    trial.report(train_epoch_loss, epoch)
    # Handle pruning
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()
    return train_epoch_loss
