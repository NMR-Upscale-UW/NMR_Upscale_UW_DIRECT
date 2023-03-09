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


PRINT_EVERY_N = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def regression_dataset_loop(model, optimizer, criterion, loader, is_train):
    if is_train:
        model.train()
    else:
        model.eval()

    running_loss = 0
    per_batch_loss = []
    for index, (inputs, labels) in enumerate(loader):
        # Make sure model and data are on the same device
        N, L, C = inputs.shape
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        if is_train:
            optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        if is_train:
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        per_batch_loss.append(loss.item())
    epoch_loss = running_loss / len(loader)
    return epoch_loss, per_batch_loss

def test_loop(model, criterion, test_loader):
    model.eval() # Model to evaluation mode
    test_loss = 0.0
    to_save = []
    for inputs, labels in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item() 
        to_save.append((inputs, outputs, loss.item()))
    return test_loss / len(test_loader), to_save


def train_valid_loop(model, train_loader, valid_loader, optimizer, criterion, num_epochs):
    t0 = time.time()
    train_loss_epoch = []
    valid_loss_epoch = []
    for e in range(num_epochs):
        # Run train loop
        tepoch_loss, tper_batch_loss = regression_dataset_loop(
            model, optimizer, criterion, train_loader, True)
        train_loss_epoch.append(tepoch_loss)

        # Run validation loop
        with torch.no_grad():
            vepoch_loss, vper_batch_loss = regression_dataset_loop(
                model, optimizer, criterion, valid_loader, False)
        valid_loss_epoch.append(vepoch_loss)
    
        if(int(e) % PRINT_EVERY_N) == 0:
            print(f'[{round(time.time()-t0, 5)} s] Epoch {e} loss: {vepoch_loss :.4f}')
    print(f'Time Elapsed: {round(time.time()-t0, 5)} seconds')
    return model

def main(args):
    assert args.train_split + args.valid_split < 1, "Leave some space for test!"
    if args.model_name in name2params.keys():
        params = name2params[args.model_name]
        model = name2model[args.model_name](params)
    else:
        raise ValueError(f"args.model_name should be one of {name2params.keys()}. Currently defined: {args.model_name}")

    
    criterion = nn.MSELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=params.lr) # Optimization function

    # Establishing and loading data into notebook
    dataset = GHzData(args.data_dir, args.high_resolution_frequency)
    dset_size = len(dataset)
    train_size = int(dset_size*args.train_split)
    valid_size = int(dset_size*args.valid_split)
    test_size = dset_size - train_size - valid_size
    dev_size = train_size + valid_size

    #Splitting the data
    dev_indices, test_indices, _, __= train_test_split(
        np.arange(dset_size), np.arange(dset_size), train_size=dev_size, test_size=test_size)
    train_indices, valid_indices, _, __= train_test_split(
        dev_indices, dev_indices, train_size=train_size, test_size=valid_size)

    train_set = Subset(dataset, train_indices)
    valid_set = Subset(dataset, valid_indices)
    test_set = Subset(dataset, test_indices)

    train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True)
    valid_dataloader = DataLoader(valid_set, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=128, shuffle=True)

    # train model
    model.to(device)
    optimized_model = train_valid_loop(
        model, train_dataloader, valid_dataloader, optimizer, criterion, args.num_epochs
    )

    test_epoch_loss, to_save = test_loop(
        optimized_model, criterion, test_dataloader)

    print(f"TEST AVG Loss: {test_epoch_loss}")
    torch.save(to_save, "predictions.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='mlp')
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--high_resolution_frequency', type=int, default=400)
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--random_key', type=int, default=12345)
    parser.add_argument('--train_split', type=float, default=0.7)
    parser.add_argument('--valid_split', type=float, default=0.15)

    args = parser.parse_args()
    
    # for reproducibility
    random.seed(args.random_key)
    np.random.seed(args.random_key)
    torch.random.manual_seed(args.random_key)
    torch.cuda.manual_seed_all(args.random_key)
    os.environ["PL_GLOBAL_SEED"] = str(args.random_key)

    main(args)
