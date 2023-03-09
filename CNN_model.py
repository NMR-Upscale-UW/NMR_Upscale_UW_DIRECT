import torch
import torch.nn as nn
import numpy as np
import os
import glob
import pandas as pd
import time
import nmrsim
from nmrsim import plt
from itertools import product
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# whether to run on GPU or CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} device")

# Checking how many files are in repository for training, testing, and validation
files = glob.glob('/home/fostooq/NMR_Upscale_UW_DIRECT/spectral_data/400MHz/spectral_data_*.csv')
print('Total number of files: ', len(files))


class GHzData(Dataset):
    """
    Dataset class for loading GHz data.
    """

    def __init__(self):
        """
        Initializes the dataset by loading the data.
        """
        # Data loading starting with list of csv strings
        self.files = glob.glob(os.path.join('/home/fostooq/NMR_Upscale_UW_DIRECT/spectral_data/400MHz/spectral_data_*.csv'))
        self.y_60 = []  # Establishes a list for 60 MHz data
        self.y_400 = []  # Establishes a list for 400 MHz data
        for self.file in self.files:  # For loop for each file in files
            self.df = pd.read_csv(self.file)  # Reads each into a pandas dataframe
            self.array_60 = self.df['60MHz_intensity'].to_numpy()  # Takes 60MHz intensity to np
            self.array_400 = self.df['400MHz_intensity'].to_numpy()  # Takes 400MHz intensity to np
            self.y_60.append(self.array_60)  # Appends all arrays to 60MHz list
            self.y_400.append(self.array_400)  # Appends all arrays to 400MHz list

        # Creates a 60 MHz tensor from list, converts to float, unsqueezes to have shape (n, 1, 5500)
        self.tensor_60 = torch.Tensor(self.y_60).float().unsqueeze(1).to(device)

        # Creates a 400 MHz tensor from list, converts to float, unsqueezes to have shape (n, 1, 5500)
        self.tensor_400 = torch.Tensor(self.y_400).float().unsqueeze(1).to(device)

        # Track the length of number of samples in frame
        self.num_samples = len(self.y_60)

    def __getitem__(self, index):  # establishes an index for the tensors
        """
        Returns the data at the given index.
        
        Args:
            index (int): index of the data to be returned.

        Returns:
            tensor_60 (torch.Tensor): tensor of 60MHz data at the given index.
            tensor_400 (torch.Tensor): tensor of 400MHz data at the given index.
        """
        return self.tensor_60[index], self.tensor_400[index]

    def __len__(self):  # Returns variable number of samples
        """
        Returns the number of samples in the dataset.
        
        Returns:
            num_samples (int): number of samples in the dataset.
        """
        return self.num_samples
    
class NeuralNetwork(nn.Module):
    """
    Neural Network class for GHz data.
    """

    def __init__(self):
        """
        Initializes the neural network.
        """
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, padding='same')
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding='same')
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding='same')
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding='same')
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Conv1d(in_channels=128, out_channels=1, kernel_size=3, padding='same')

    def forward(self, x):
        """
        Defines the forward pass of the neural network.

        Args:
            x (torch.Tensor): tensor of data to pass through the network.

        Returns:
            x (torch.Tensor): tensor after passing through the network.
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.relu4(x)
        x = self.fc5(x)
        return x

    
model = NeuralNetwork().to(device) # Assigns model to variable model, sends to gpu
    
def train_test_val_split(tensor_60, tensor_400):
    """
    Splits the GHz data into training, testing, and validation sets.

    Args:
        tensor_60 (torch.Tensor): tensor of 60 MHz data.
        tensor_400 (torch.Tensor): tensor of 400 MHz data.

    Returns:
        train_X (torch.Tensor): tensor of training data for 60 MHz.
        train_y (torch.Tensor): tensor of training data for 400 MHz.
        test_X (torch.Tensor): tensor of testing data for 60 MHz.
        test_y (torch.Tensor): tensor of testing data for 400 MHz.
        valid_X (torch.Tensor): tensor of validation data for 60 MHz.
        valid_y (torch.Tensor): tensor of validation data for 400 MHz.
    """
    # Establishing and loading data into notebook
    dataset = GHzData()

    # Splitting the data
    train_X, test_X, train_y, test_y = train_test_split(dataset.tensor_60, dataset.tensor_400,
                                                        test_size=0.1)

    # Splits train data into validation data
    train_X, valid_X, train_y, valid_y = train_test_split(train_X, train_y,
                                                          test_size=0.1)
    return train_X, train_y, test_X, test_y, valid_X, valid_y


def data_loader(train_X, train_y, test_X, test_y, valid_X, valid_y):
    """
    Loads the GHz data into PyTorch data loaders.

    Args:
        train_X (torch.Tensor): tensor of training data for 60 MHz.
        train_y (torch.Tensor): tensor of training data for 400 MHz.
        test_X (torch.Tensor): tensor of testing data for 60 MHz.
        test_y (torch.Tensor): tensor of testing data for 400 MHz.
        valid_X (torch.Tensor): tensor of validation data for 60 MHz.
        valid_y (torch.Tensor): tensor of validation data for 400 MHz.

    Returns:
        train_dataloader (torch.utils.data.DataLoader): data loader for training set.
        valid_dataloader (torch.utils.data.DataLoader): data loader for validation set.
        test_dataloader (torch.utils.data.DataLoader): data loader for testing set.
    """
    # Creating datasets
    train_dataset = TensorDataset(train_X, train_y)
    test_dataset = TensorDataset(test_X, test_y)
    valid_dataset = TensorDataset(valid_X, valid_y)

    # Batch size change to higher batch sizes
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True)

    return train_dataloader, valid_dataloader, test_dataloader


def loss_function():
    """
    Returns the Mean Squared Error loss function for the model.
    """
    criterion = nn.MSELoss() 
    return criterion


def optimizer():
    """
    Returns the RMSprop optimizer function for the model.
    """
    optimizer = RMSprop(model.parameters(), lr=0.001)
    return optimizer


def model_train_evaluation(num_epochs, train_dataloader, valid_dataloader):
    """
    Trains and evaluates the performance of the model on the given data.

    Parameters:
        num_epochs (int): Number of epochs to train the model.
        train_dataloader (DataLoader): DataLoader object containing training data.
        valid_dataloader (DataLoader): DataLoader object containing validation data.

    Returns:
        epoch_loss (float): Loss of the model after training.
    """
    num_epochs = 30
    time_ = time.time()
    train_loss_epoch = []
    valid_loss_epoch = []
    for e in range(num_epochs):
        running_loss = 0.0
        for index, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(1)
        epoch_loss = running_loss / len(train_dataloader.dataset)
        train_loss_epoch.append(epoch_loss)
        model.eval()
        valid_loss = 0.0
        valid_correct = 0
        valid_total = 0
        loss_list_test = []
        for inputs, labels in valid_dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            valid_loss += loss.item() * inputs.size(1)
            _, labels = torch.min(labels, 1)
            _, predicts = torch.min(outputs.data, 1)
            predicts = predicts.unsqueeze(1)
            valid_total += labels.size(0)
            valid_correct += (predicts == labels).float().mean()
        epoch_loss = valid_loss / len(valid_dataloader.dataset)
        valid_loss_epoch.append(epoch_loss)
        if (int(e) % 10) == 0:
            print(f'Epoch {e} loss: {epoch_loss:.4f}')
    print(f'Time Elapsed: {round(time.time()-time_, 5)} seconds')
    return epoch_loss


def model_test():
    """
    Evaluates the performance of the trained model on the test dataset.

    Returns:
    accuracy (float): The percentage accuracy of the model on the test dataset.
    test_loss (float): The mean loss of the model on the test dataset.
    """
    model.eval() # Model to evaluation mode

    test_loss = 0.0
    test_correct = 0
    test_total = 0
    loss_list_test = []

    # Loop for testing
    for inputs, labels in test_dataloader:
        #Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * inputs.size(1)
        loss_list_test.append(loss)

        _, labels = torch.min(labels, 1)
        _, predicts = torch.min(outputs.data, 1)
        predicts = predicts.unsqueeze(1)
        test_total += labels.size(0)
        test_correct += (predicts == labels).float().mean()

    accuracy = (test_correct / test_total)*100
    test_loss /= len(test_dataloader.dataset)
    print(f' Mean Loss of Function: {test_loss}, Accuracy: {accuracy}')
    print(labels.shape, outputs.shape)
    
    return (accuracy, test_loss)

def model_prediction(inputs):
    """
    Generates predictions for new input using the trained model.

    Args:
    inputs (Tensor): The input tensor for which predictions are to be generated.

    Returns:
    predictions_numpy (pandas DataFrame): A DataFrame containing the predicted output values for the input tensor.
    """
    model.eval()
    with torch.no_grad():
        for inputs, _ in test_dataloader:
            predictions = model(inputs)

    predictions_numpy = predictions.cpu().numpy().reshape(10,-1)
    predictions_numpy = pd.DataFrame(predictions_numpy)

    return predictions_numpy