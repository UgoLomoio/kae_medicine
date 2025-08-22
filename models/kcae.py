"""
This module contains the implementation of a Kolmogorov-Arnold based Convolutional Autoencoder (KCAE) model using PyTorch.
"""

from models.kcn import KCN, KCNTranspose
import torch
from torch import nn

class PixelShuffle1D(torch.nn.Module):
    """
    1D pixel shuffler. https://arxiv.org/pdf/1609.05158.pdf
    Upscales sample length, downscales channel length
    "short" is input, "long" is output
    """
    def __init__(self, upscale_factor):
        super(PixelShuffle1D, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        batch_size = x.shape[0]
        short_channel_len = x.shape[1]
        short_width = x.shape[2]

        long_channel_len = short_channel_len // self.upscale_factor
        long_width = self.upscale_factor * short_width

        x = x.contiguous().view([batch_size, self.upscale_factor, long_channel_len, short_width])
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, long_channel_len, long_width)

        return x

class KCAE(nn.Module):
    def __init__(
        self,
        input_dim,
        layers_hidden, 
        kernel_sizes=3,
        strides=1,
        paddings=1,
        base_activation=nn.SiLU(),
        pixel_shuffle=True,
        dropout=0.0,
    ):
        
        super(KCAE, self).__init__()

        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * len(layers_hidden)
        if isinstance(strides, int):
            strides = [strides] * len(layers_hidden)
        if isinstance(paddings, int):
            paddings = [paddings] * len(layers_hidden)

        self.encoder = KCN([input_dim, *layers_hidden], kernel_sizes, strides, paddings, base_activation=base_activation)
        self.pixel_shuffle = pixel_shuffle
        if not self.pixel_shuffle:
            self.decoder = KCNTranspose([*layers_hidden[::-1][:-1], layers_hidden[::-1][-1]], kernel_sizes[::-1][:-1], strides[::-1][:-1], paddings[::-1][:-1], base_activation=base_activation)
            self.after_decoder = nn.Sequential(
                nn.ConvTranspose1d(layers_hidden[::-1][-1], input_dim, kernel_size=kernel_sizes[::-1][-1], stride=strides[::-1][-1], padding=paddings[::-1][-1]),
            )
        else:
            self.decoder = nn.Sequential(
                #KCN([layers_hidden[-1], 4], [3, 3], [1, 1], [1, 1], base_activation=base_activation),
                nn.Conv1d(layers_hidden[-1], 4, kernel_size=3, stride=1, padding=1),
                #nn.BatchNorm1d(4),
                #base_activation(),
                #nn.Dropout(dropout),
                PixelShuffle1D(upscale_factor=4)
            )

        self.final_block = nn.Sequential(
            *[
                nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=1),  # Final layer to match the input dimension
            ]
        )
        
    
    def forward(self, x: torch.Tensor):

        x = self.encode(x)
        x = self.decode(x)
        return x
    
    def encode(self, x: torch.Tensor):
        x = self.encoder(x)
        return x
    
    def decode(self, x: torch.Tensor):
        x = self.decoder(x)
        if not self.pixel_shuffle:
            x = self.after_decoder(x)
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