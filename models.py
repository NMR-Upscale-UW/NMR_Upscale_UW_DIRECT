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
        layers = [nn.Conv1d(cfg.input_dim, prev_dim, kernel_size=k, padding=cfg.padding), nn.ReLU(), nn.Dropout(p=cfg.p_drop)]
        
        # Make the hidden layers
        for next_dim in cfg.hidden_dims[1:]:
            layers.append(nn.Conv1d(prev_dim, next_dim, kernel_size=k, padding=cfg.padding))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=cfg.p_drop))
            prev_dim = next_dim

        # Make the last transformation
        layers.append(nn.Conv1d(prev_dim, cfg.output_dim, kernel_size=k, padding=cfg.padding))

        # define full model
        self.m = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.m(x)

class ConvVAE(nn.Module):
    def __init__(self, cfg):
        super(ConvVAE, self).__init__()

        self.encoder = CNN(cfg['encoder'])
        self.decoder = CNN(cfg['decoder'])
        
    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample.unsqueeze(1)
 
    def forward(self, x):
        # encoding
        x = self.encoder(x)

        # get `mu` and `log_var`
        mu = x[:, 0, :] 
        log_var = x[:, 1, :]
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
 
        # decoding
        reconstruction = torch.sigmoid(self.decoder(z))
        # reconstruction = reconstruction.unsqueeze(1)
        return reconstruction, mu, log_var

name2model = {
    'mlp': MLP,
    'cnn': CNN,
    'conv_vae': ConvVAE,
}