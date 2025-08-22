"""
This module contains the implementation of a Kolmogorov-Arnold based AutoEncoder (KAE) model using PyTorch.
"""

from models.kan import KAN
import torch
from torch import nn

class KAN_Encode(torch.nn.Module):

    def __init__(self, input_dim, layers_hidden, base_activation=torch.nn.SiLU, dropout = 0.0, bn = False):
        super(KAN_Encode, self).__init__()

        self.kan = KAN([input_dim, *layers_hidden[:-1]], base_activation=base_activation)
        self.act = base_activation()
        self.dense = nn.Linear(layers_hidden[-2], layers_hidden[-1])
        print(f"KAN layer input dimension: {input_dim}, output dimension: {layers_hidden[:-1]}")
        print(f"Dense layer input dimension: {layers_hidden[-2]}, output dimension: {layers_hidden[-1]}")
        
        if bn:
            self.bn = nn.BatchNorm1d(layers_hidden[-1])
            print(f"BatchNorm layer input dimension: {layers_hidden[-1]}")
        else:
            self.bn = None
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x: torch.Tensor):
        if x.ndim == 3:
            x = x.squeeze(1)
        x = self.kan(x)
        x = self.dense(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.act(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x

class Linear_Decode(torch.nn.Module):
    
    def __init__(self, layers_hidden, output_dim, act=torch.nn.SiLU):

        super(Linear_Decode, self).__init__()

        self.act = act

        blocks = []
        for i in range(len(layers_hidden) - 1):
            blocks.append(nn.Linear(layers_hidden[i], layers_hidden[i + 1]))
            #if i < len(layers_hidden) - 1:  # Don't apply activation on the last layer    
            blocks.append(self.act())
            print(f"Dense layer input dimension: {layers_hidden[i]}, output dimension: {layers_hidden[i + 1]}")
        blocks.append(nn.Linear(layers_hidden[-1], output_dim))
        print(f"Dense layer input dimension: {layers_hidden[-1]}, output dimension: {output_dim}")  
        self.decode = nn.Sequential(*blocks)
    
    def forward(self, x: torch.Tensor):
        if x.ndim == 3:
            x = x.squeeze(1)
        x = self.decode(x)
        return x
    
class KAN_Decode(torch.nn.Module):
    
    def __init__(self, layers_hidden, output_dim, base_activation=torch.nn.SiLU):

        super(KAN_Decode, self).__init__()

        self.dense = nn.Linear(layers_hidden[0], layers_hidden[1])
        print(f"Dense layer input dimension: {layers_hidden[0]}, output dimension: {layers_hidden[1]}")
        self.act = base_activation
        self.kan = KAN([*layers_hidden[1:-1], output_dim], base_activation=base_activation)
        print(f"KAN layer input dimension: {layers_hidden}, output dimension: {output_dim}")
        self.output_dense = nn.Linear(output_dim, output_dim)

    def forward(self, x: torch.Tensor):
        if x.ndim == 3:
            x = x.squeeze(1)
        x = self.kan(x)
        #x = self.output_dense(x)
        #x = self.act(x)
        return x

class KAE(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        layers_hidden,
        base_activation = torch.nn.SiLU,
        dropout = 0.0,
        bn = False
        ):
        super(KAE, self).__init__()

        self.encoder = KAN_Encode(input_dim, layers_hidden, base_activation=base_activation, dropout=dropout, bn=bn)
        #self.decoder = KAN_Decode(layers_hidden[::-1], input_dim, base_activation=base_activation)
        self.decoder = Linear_Decode([layers_hidden[::-1][0]], input_dim, act=base_activation)

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


    
