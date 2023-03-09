import glob
import pandas as pd
import torch
import os
from torch.utils.data import Dataset

class GHzData(Dataset):
    def __init__(self, data_dir='./data/', high_res=400):
        super().__init__()
        # Data loading starting with list of csv strings
        files = glob.glob(os.path.join(data_dir, f'{high_res}MHz', 'spectral_data_*.csv'))

        y_low = [] # Establishes a list for 60 MHz data
        y_high = [] # Establishes a list for 400 MHz data

        for file in files: # For loop for each file in files
            df = pd.read_csv(file) # Reads each into a pandas dataframe
            array_low = df['60MHz_intensity'].to_numpy() # Takes 60MHz intensity to np
            array_high = df[f'{high_res}MHz_intensity'].to_numpy() # Takes 400MHz intensity to np
            y_low.append(array_low) # Appends all arrays to 60MHz list
            y_high.append(array_high) # Appends all arrays to 400MHz list
            
        # Creates a 60 MHz tensor from list, converts to float, unsqueezes to have shape (n, 1, 5500)
        self.tensor_low = torch.Tensor(y_low).float().unsqueeze(1)

        # Creates a high resolution MHz tensor from list, converts to float, unsqueezes to have shape (n, 1, 5500)
        self.tensor_high = torch.Tensor(y_high).float().unsqueeze(1)
        
    def __getitem__(self, index): # establishes an index for the tensors
        return self.tensor_low[index], self.tensor_high[index]
    
    def __len__(self): # Returns variable number of samples
        return self.tensor_low.shape[0]
    