import torch
from torch import nn


class MLP(nn.Module): # MLP = Multi Layer Perceptron
    def __init__(self, cfg):
        super().__init__()
        prev_dim  = cfg.hidden_dims[0]
        # First make the first transformation that will be passed to the hidden dims
        layers = [nn.Linear(cfg.input_dim, prev_dim), nn.ReLU(), nn.Dropout(p=cfg.p_drop)]
        
        # Make the hidden layers
        for next_dim in cfg.hidden_dims[1:]:
            layers.append(nn.Linear(prev_dim, next_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=cfg.p_drop))
            prev_dim = next_dim

        # Make the last transformation
        layers.append(nn.Linear(prev_dim, cfg.output_dim))

        # define full model
        self.m = nn.Sequential(*layers)
            
    def forward(self, x):  
        return self.m(x)

class CNN(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        prev_dim  = cfg.hidden_dims[0]
        k = cfg.kernel_size
        # First make the first transformation that will be passed to the hidden dims
        layers = [nn.Conv1d(cfg.input_dim, prev_dim, kernel_size=k, padding=1), nn.ReLU(), nn.Dropout(p=cfg.p_drop)]
        
        # Make the hidden layers
        for next_dim in cfg.hidden_dims[1:]:
            layers.append(nn.Conv1d(prev_dim, next_dim, kernel_size=k, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=cfg.p_drop))
            prev_dim = next_dim

        # Make the last transformation
        layers.append(nn.Conv1d(prev_dim, cfg.output_dim, kernel_size=k, padding=1))

        # define full model
        self.m = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.m(x)

name2model = {
    'mlp': MLP,
    'cnn': CNN,
}