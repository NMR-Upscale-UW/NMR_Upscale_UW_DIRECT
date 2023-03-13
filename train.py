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
    model = model.to(device)
    if is_train:
        model.train()
    else:
        model.eval()

    running_loss = 0
    per_batch_loss = []
    for index, (inputs, labels) in enumerate(loader):
        print(f"{index+1}/{len(loader)}", end='\r')
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
    return epoch_loss, per_batch_loss, model

def test_loop(model, criterion, test_loader):
    model = model.to(device)
    model.eval() # Model to evaluation mode
    test_loss = 0.0
    to_save = []
    for index, (inputs, labels) in enumerate(test_loader):
        print(f"{index+1}/{len(test_loader)}", end='\r')
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item() 
        to_save.append((inputs, outputs, labels, loss.item()))
    return test_loss / len(test_loader), to_save


def train_valid_loop(model, train_loader, valid_loader, optimizer, criterion, num_epochs):
    t0 = time.time()
    train_loss_epoch = []
    valid_loss_epoch = []
    best_valid_loss = float("inf")
    best_state_dict = None
    best_epoch = 0
    for e in range(num_epochs):
        # Run train loop
        print(f"Train Epoch {e}")
        tepoch_loss, tper_batch_loss, model = regression_dataset_loop(
            model, optimizer, criterion, train_loader, True)
        train_loss_epoch.append(tepoch_loss)

        # Run validation loop
        print(f"Validation Epoch {e}")
        with torch.no_grad():
            vepoch_loss, vper_batch_loss, model = regression_dataset_loop(
                model, optimizer, criterion, valid_loader, False)
        if vepoch_loss < best_valid_loss: #save the model with the best validation loss
            # torch.save(
            #     {'epoch': e, 'avg_valid_loss':vepoch_loss, 'state_dict': model.state_dict()}, os.path.join(save_dir, 'best_model.pt'))
            best_state_dict = model.state_dict()
            best_epoch = e
            best_valid_loss = vepoch_loss
        valid_loss_epoch.append(vepoch_loss)
    
        if(int(e) % PRINT_EVERY_N) == 0:
            print(f'[{round(time.time()-t0, 5)} s] Epoch {e} loss: {vepoch_loss :.4f}')
    print(f'Time Elapsed: {round(time.time()-t0, 5)} seconds')
    return train_loss_epoch, valid_loss_epoch, {
        'epoch': best_epoch,
        'avg_valid_loss': best_valid_loss,
        'state_dict': best_state_dict,
    }

def main(args):
    assert args.train_split + args.valid_split < 1, "Leave some space for test!"

    args.save_dir = os.path.join(args.save_dir, f"{args.high_resolution_frequency}MHz", args.model_name)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.model_name in name2params.keys():
        params = name2params[args.model_name]
        model = name2model[args.model_name](params)
    else:
        raise ValueError(f"args.model_name should be one of {name2params.keys()}. Currently defined: {args.model_name}")

    
    criterion = nn.MSELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=params.lr) # Optimization function

    # Establishing and loading data into notebook
    print("Loading Dataset...")
    dataset = GHzData(args.data_dir, args.high_resolution_frequency)
    if args.limit_for_test == 1:
        dset_size = 50
    else:
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
    test_dataloader = DataLoader(test_set, batch_size=1, shuffle=True)

    # train model
    model.to(device)
    train_loss_epoch, valid_loss_epoch, best_model_ckpt = train_valid_loop(
        model, train_dataloader, valid_dataloader, optimizer, criterion, args.num_epochs
    )

    best_model_ckpt['hyperparams'] = dict(name2params[args.model_name])

    print(f"Loading model with best validation performance")
    print(f"Avg Validation Loss: {best_model_ckpt['avg_valid_loss']} at epoch {best_model_ckpt['epoch']}")
    model.load_state_dict(best_model_ckpt['state_dict'])
    test_epoch_loss, to_save = test_loop(model, criterion, test_dataloader)

    print(f"Avg Test Loss: {test_epoch_loss}")
    torch.save(
        {'train_loss': train_loss_epoch, 'valid_loss': valid_loss_epoch},
        os.path.join(args.save_dir, "loss_curves.pt")
    )
    torch.save(best_model_ckpt, os.path.join(args.save_dir, "best_model.pt"))
    torch.save(to_save, os.path.join(args.save_dir, "predictions.pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='mlp', help="The name of the model you implemented in models.py")
    parser.add_argument('--num_epochs', type=int, default=30, help="How many epochs to train/validate the model")
    parser.add_argument('--high_resolution_frequency', type=int, default=400, help="What resolution to upscale the 60MHz measurement to")
    parser.add_argument('--data_dir', type=str, default='./data/', help="Where the dataset is stored")
    parser.add_argument('--random_key', type=int, default=12345, help="The random seed/key to use for reproducibility")
    parser.add_argument('--train_split', type=float, default=0.7, help="The fraction of the entire dataset to use for the train set")
    parser.add_argument('--valid_split', type=float, default=0.15, help="The fraction of the entire dataset to use for the validation set")
    parser.add_argument('--save_dir', type=str, default='./results/', help="Where the results should be stored")
    parser.add_argument('--limit_for_test', type=int, default=0, help="If 1 it will limit the number of samples in the dataset for faster testing")

    args = parser.parse_args()
    
    # for reproducibility
    random.seed(args.random_key)
    np.random.seed(args.random_key)
    torch.random.manual_seed(args.random_key)
    torch.cuda.manual_seed_all(args.random_key)
    os.environ["PL_GLOBAL_SEED"] = str(args.random_key)

    main(args)
