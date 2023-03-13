import torch
from torch import nn
import math


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

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x.permute(1,0,2)
        x = x + self.pe[:x.size(0)]
        x = x.permute(1,0,2)

        return self.dropout(x)

class TransformerModel(nn.Module):

    def __init__(self, cfg):
        
        super().__init__()

        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(cfg.d_model, cfg.p_drop)
        encoder_layers = nn.TransformerEncoderLayer(
            cfg.d_model, cfg.nhead, cfg.d_hid, cfg.p_drop)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, cfg.nlayers)
        self.encoder = nn.Linear(1, cfg.d_model)
        self.d_model = cfg.d_model
        self.decoder = nn.Linear(cfg.d_model, 1)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, batch_input) -> torch.Tensor:
        B, C, L = batch_input.shape
        src = self.encoder(batch_input.permute(0, 2, 1))
        # src is [B, L, D]
        src = self.pos_encoder(src)        
        # Pytorch Transformer Encoder layer takes in the dimensions [length, batch, dimension/channle]
        src = src.permute(1, 0, 2) 
        output = self.transformer_encoder(src)
        output = output.permute(1, 0, 2)
        output = torch.nn.functional.relu(output)
        output = self.decoder(output)
        output = output.permute(0, 2, 1) # [B, L, D] -> [B, D, L], and D = 1

        return output



name2model = {
    'mlp': MLP,
    'cnn': CNN,
    'conv_vae': ConvVAE,
    'transformer': TransformerModel,
}