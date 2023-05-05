import glob
import pandas as pd
import torch
import os
from torch.utils.data import Dataset
from preprocessing.utils import folder2tensors

class GHzData(Dataset):
    def __init__(self, data_dir='./data/', high_res=400):
        super().__init__()
        save_file_path = os.path.join(data_dir, f'{high_res}MHz.pt')

        # Check if you have already preprocessed the dataset folder 
        if os.path.exists(save_file_path):
            dset = torch.load(save_file_path)
            self.tensor_low = dset['low']
            self.tensor_high = dset['high']
        else:
            # preprocess the dataset folder if you haven't
            self.tensor_low, self.tensor_high = folder2tensors(data_dir, high_res)
            
    def __getitem__(self, index): # establishes an index for the tensors
        return self.tensor_low[index], self.tensor_high[index]
    
    def __len__(self): # Returns variable number of samples
        return self.tensor_low.shape[0]
    