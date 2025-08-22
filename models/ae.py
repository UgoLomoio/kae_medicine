"""
This module contains the implementation of the AutoEncoder class based on MLP layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearBlock(nn.Module):
    """
    LinearBlock class for MLP layers.
    """

    def __init__(self, input_dim, output_dim, dropout = 0.0, bn = False, act=nn.ReLU):
        """
        Initialize the LinearBlock.

        Args:
            input_dim (int): Dimension of the input data.
            output_dim (int): Dimension of the output data.
            dropout (float): Dropout rate (default is 0.0, no dropout).
            act (callable): Activation function.
        """
        super(LinearBlock, self).__init__()
        self.dense = nn.Linear(input_dim, output_dim)
        if act is not None:
            self.act = act()
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        if bn:
            self.bn = nn.BatchNorm1d(output_dim)
        else:
            self.bn = None

    def forward(self, x):
        x = self.dense(x)
        if self.bn is not None:
            if x.ndim == 3:
                x = x.squeeze(1)
            x = self.bn(x)
            if x.ndim == 2:
                x = x.unsqueeze(1)
        x = self.act(x) if hasattr(self, 'act') else x
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        return x

class AutoEncoder(nn.Module):
    """
    AutoEncoder class based on MLP layers.
    """

    def __init__(self, input_dim, hidden_dims, dropout=0.0, bn = True, act=nn.ReLU):
        """
        Initialize the AutoEncoder.

        Args:
            input_dim (int): Dimension of the input data.
            hidden_dims (list of int): Dimensions of the hidden layers.
            dropout (float): Dropout rate (default is 0.0, no dropout).
            act (callable): Activation function.
        """
        super(AutoEncoder, self).__init__()

        hidden_dims = hidden_dims.copy()
        self.dropout = dropout
        self.act = act
        self.bn = bn
        # Encoder
        blocks = []
        for i in range(len(hidden_dims)):
            input_dim_b = input_dim if i == 0 else hidden_dims[i-1]
            # Last layer has no dropout
            if i == len(hidden_dims) - 1:
               dropout = 0.0
            else:
                dropout = self.dropout
            #dropout = 0.0 
            
            #print(f"Layer {i}: input_dim={input_dim_b}, output_dim={hidden_dims[i]}")
            block = LinearBlock(input_dim_b, hidden_dims[i], dropout=dropout, act=self.act, bn=bn)
            blocks.append(block)
        self.encoder = nn.Sequential(*blocks)
        
        # Decoder
        blocks = []
        hidden_dims.reverse()
        input_dim_b = hidden_dims[0]  # Set input_dim for the first layer of the decoder
        for i in range(1, len(hidden_dims)+1):
            if i == len(hidden_dims):
                # Last layer of the decoder should match the input dimension
                output_dim = input_dim
                dropout = 0.0  # No dropout in the last layer
                act = None
            else:
                dropout = self.dropout
                act = self.act
                output_dim = hidden_dims[i]
                
            #print(f"Layer {i}: input_dim={input_dim_b}, output_dim={output_dim}")
            block = LinearBlock(input_dim_b, output_dim, dropout=dropout, bn = False, act=act)
            input_dim_b = output_dim  # Update input_dim for the next layer
            blocks.append(block)
        self.decoder = nn.Sequential(*blocks)
        
    def forward(self, x):
        """
        Forward pass through the AutoEncoder.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Reconstructed data.
        """
        #print(f"Input shape: {x.shape}")
        encoded = self.encoder(x)
        #print(f"Encoded shape: {encoded.shape}")
        decoded = self.decoder(encoded)
        #print(f"Decoded shape: {decoded.shape}")
        return decoded
    
    def loss_function(self, x, reconstructed_x):
        """
        Compute the loss function for the AutoEncoder.
        Args:
            x (torch.Tensor): Original data.
            reconstructed_x (torch.Tensor): Reconstructed data.
        Returns:
            torch.Tensor: Computed loss.
        """
        return F.mse_loss(reconstructed_x, x)
    
    def encode(self, x):
        """
        Encode the input data.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Encoded data.
        """
        return self.encoder(x)

    def decode(self, x):
        """
        Decode the input data.

        Args:
            x (torch.Tensor): Encoded data.

        Returns:
            torch.Tensor: Decoded data.
        """
        return self.decoder(x)