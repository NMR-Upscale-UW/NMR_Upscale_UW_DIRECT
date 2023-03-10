'''
This module utilizies Optuna to help training the data and get the best
hyperparameter for the model. 
The module is divided into the following sections:
1) Setup: import packages, device setting
2) Data Preprocessing: load data, spectral data tensor creations
3) Model Fitting
4) Validation: value loss computation
5) Final loss
'''
# Setup
# Loading in libraries necessary for CNN
import torch
import torch.nn as nn
import numpy as np
import os
import glob
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import math
import matplotlib.pyplot
import time
import torch.nn.functional as F
import nmrsim
from nmrsim import plt
from itertools import product
import statistics
from tqdm import tqdm
import optuna 

# Device Setting
# whether to run on GPU or CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} device")

#Checking how many files are in repository for training, testing, and validation
files = glob.glob('./Spectral_Data/spectral_data/400MHz/*.csv')
print('Total number of files: ', len(files))

# Data Preprocessing
class GHzData(Dataset):
    '''
    Load datasets and convert them into proper format(tensor)
    :param Dataset: Dataset for processing
    '''
    def __init__(self):
        '''
        extract desired spectras from the dataset
        Example: 60MHz data vs 400MHz data
        '''
        # Data loading starting with list of csv strings
        self.files = glob.glob(os.path.join('./Spectral_Data/spectral_data/400MHz', 
                                                       'spectral_data_*.csv'))

        self.y_60 = [] # Establishes a list for 60 MHz data
        self.y_400 = [] # Establishes a list for 400 MHz data

        for self.file in self.files: # For loop for each file in files
            self.df = pd.read_csv(self.file) # Reads each into a pandas dataframe
            self.array_60 = self.df['60MHz_intensity'].to_numpy() # Takes 60MHz intensity to np
            self.array_400 = self.df['400MHz_intensity'].to_numpy() # Takes 400MHz intensity to np
            self.y_60.append(self.array_60) # Appends all arrays to 60MHz list
            self.y_400.append(self.array_400) # Appends all arrays to 400MHz list
            
        # Creates a 60 MHz tensor from list, converts to float, unsqueezes to have shape (n, 1, 5500)
        self.tensor_60 = torch.Tensor(self.y_60).float().unsqueeze(1).to(device)        

        # Creates a 400 MHz tensor from list, converts to float, unsqueezes to have shape (n, 1, 5500)
        self.tensor_400 = torch.Tensor(self.y_400).float().unsqueeze(1).to(device)
        
        # Track the length of number of samples in frame
        self.num_samples = len(self.y_60)

    def __getitem__(self, index):  
        '''
        establishes an index for the tensors
        '''
        return self.tensor_60[index], self.tensor_400[index]
    
    def __len__(self):
        '''
        Returns variable number of samples
        '''
        return self.num_samples
    
# Model Fitting
def fit(model, dataloader, optimizer, criterion):
    '''
    With the model we had fitted and trained, compute the loss
    :param model: the generated VAE model
    :param dataloader: loaded data
    :param optimizer: optimzer we generated
    :param criterion: mean squared error (squared L2 norm) between each element in the input x and target y
    :return train_loss: the training loss
    '''
    model.train()
    running_loss = 0.0
    for i, data in enumerate(dataloader):
        data, _ = data
        data = data.to(device)
        optimizer.zero_grad()
        reconstruction, mu, logvar = model(data)
        bce_loss = criterion(reconstruction, data)
        loss = final_loss(bce_loss, mu, logvar)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_loss = running_loss/len(dataloader.dataset)
    return train_loss


# Model Validation
def validate(model, dataloader, criterion):
    '''
    This function evalutes the generatedf model and computes its validation loss

    param: model: the generated VAE model
    param: dataloader: loaded data
    param
    : criterion: criterion that measeaures the 
        mean squared error (squared L2 norm) between each element in the input x and target y
    return: val_loss: the validation loss
    '''
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            data, _ = data
            data = data.to(device)
            reconstruction, mu, logvar = model(data)
            bce_loss = criterion(reconstruction, data)
            loss = final_loss(bce_loss, mu, logvar)
            running_loss += loss.item()
        
    val_loss = running_loss/len(dataloader.dataset)
    return val_loss

def final_loss(bce_loss, mu, logvar):
    """
    This function will add the reconstruction loss (BCELoss) and the 
    KL-Divergence.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param bce_loss: recontruction loss
    :param mu: the mean from the latent vector
    :param logvar: log variance from the latent vector
    :return BCE+KDE: the total suk of BCELoss and KL_Divergence
    """
    BCE = bce_loss 
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


features = 5500
encode = 512
class LinearVAE(nn.Module):
    '''
    #Define a simple linear VAE for Trial Optimization
    '''
    def __init__(self,trial):
        '''
        Encoding data
        '''
        super(LinearVAE, self).__init__()
 
        # encoder
        self.enc1 = nn.Linear(in_features=5500, out_features=encode)
        self.enc2 = nn.Linear(in_features=encode, out_features=features*2)
 
        # decoder 
        self.dec1 = nn.Linear(in_features=features, out_features=encode)
        self.dec2 = nn.Linear(in_features=encode, out_features=5500)
    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample
 
    def forward(self, x):
        '''
        Reparameterization of the feature values and decoding
        '''
        x = x.view(-1, 5500)
        x = x.unsqueeze(1)
        # encoding
        x = F.relu(self.enc1(x))
        x = self.enc2(x).view(-1, 2, features)

        # get `mu` and `log_var`
        mu = x[:, 0, :] # the first feature values as mean
        log_var = x[:, 1, :] # the other feature values as variance
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
 
        # decoding
        x = F.relu(self.dec1(z))
        #print(x.shape)
        reconstruction = torch.sigmoid(self.dec2(x))
        reconstruction = reconstruction.unsqueeze(1)
        #print(reconstruction.shape)
        return reconstruction, mu, log_var


def objective(trial):
    '''
    Generate, train, and optimize the linear VAE model
    '''
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

# Using optuna to find the optimal parameter

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10)

trial = study.best_trial

print('Training Loss: {}'.format(trial.value))
print("Best hyperparameters: {}".format(trial.params))