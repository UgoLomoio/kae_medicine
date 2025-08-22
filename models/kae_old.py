"""
This module contains the implementation of a Kolmogorov-Arnold based AutoEncoder (KAE) model using PyTorch.
"""

from models.kan import KAN
import torch
from torch import nn


class KAN_Encode(torch.nn.Module):

    def __init__(self, input_dim, layers_hidden, latent_dim=512, base_activation=torch.nn.SiLU):
        super(KAN_Encode, self).__init__()

        self.kan = KAN([input_dim, *layers_hidden], base_activation=base_activation)
        self.act = nn.ReLU()
        self.dense = nn.Linear(layers_hidden[-1], latent_dim)

    def forward(self, x: torch.Tensor):
        if x.ndim == 3:
            x = x.squeeze(1)
        x = self.kan(x)
        x = self.act(x)
        x = self.dense(x)
        return x


class KAN_Decode(torch.nn.Module):
    
    def __init__(self, latent_dim, layers_hidden, output_dim, base_activation=torch.nn.SiLU):
        super(KAN_Decode, self).__init__()

        self.dense = nn.Linear(latent_dim, layers_hidden[0])
        self.act = base_activation()
        self.kan = KAN([layers_hidden[0], *layers_hidden[1:], output_dim], base_activation=base_activation)

    def forward(self, x: torch.Tensor):
        if x.ndim == 3:
            x = x.squeeze(1)
        x = self.dense(x)
        x = self.act(x)
        x = self.kan(x)
        return x

class KAE(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        layers_hidden,
        latent_dim = 64,
        base_activation = torch.nn.SiLU
        ):
        super(KAE, self).__init__()

        self.encoder = KAN_Encode(input_dim, layers_hidden, latent_dim, base_activation=base_activation)
        self.decoder = KAN_Decode(latent_dim, layers_hidden[::-1], input_dim, base_activation=base_activation)

        #self.encoder.apply(self._init_weights)
        #self.decoder.apply(self._init_weights)

    def forward(self, x: torch.Tensor):
        if x.ndim == 3:
            x = x.squeeze(1)
        x = self.encode(x)
        x = self.decode(x)
        return x

    def encode(self, x: torch.Tensor):
        x = self.encoder(x)
        return x
    
    def decode(self, x: torch.Tensor):
        x = self.decoder(x)
        return x

    def _init_weights(self, module):
        # Skip activation functions
        if isinstance(module, (nn.SiLU, nn.ReLU, nn.GELU, nn.Tanh, nn.Sigmoid)):
            return
        
        # Only apply kaiming_normal_ to modules with weight tensors
        if hasattr(module, 'weight') and isinstance(module.weight, torch.Tensor):
            nn.init.kaiming_normal_(module.weight, nonlinearity='silu')
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)


    
