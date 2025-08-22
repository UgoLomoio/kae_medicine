"""
This module contains the implementation of a Convolutional Autoencoder (CAE) and a Denoising Convolutional Autoencoder (DCAE) using PyTorch.
"""

import torch.nn as nn
import torch
import torch.nn.functional as F

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=0.0, kernel_size=3, stride=1, padding=1, bn = True, act = nn.ReLU):
        
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        if bn:
            self.bn = nn.BatchNorm1d(out_channels)
        else: 
            self.bn = None
        if dropout > 0.0:
            self.do = nn.Dropout(dropout)
        else:
            self.do = None
        if act:
            self.act = act()
        else:
            self.act = None

    def forward(self, x):

        x = self.conv(x)

        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        if self.do is not None:
            x = self.do(x)

        return x

class ConvTransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0, kernel_size=3, stride=1, padding=1, bn = True, act = nn.ReLU):

        super(ConvTransposeBlock, self).__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding)
        if bn:
            self.bn = nn.BatchNorm1d(out_channels)
        else: 
            self.bn = None
        if dropout > 0.0:
            self.do = nn.Dropout(dropout)
        else:
            self.do = None
        if act:
            self.act = act()
        else:
            self.act = None
        self.pixel_shuffle = PixelShuffle1D(stride) 

    def forward(self, x):

        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        if self.do is not None:
            x = self.do(x)
        return x

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

class PixelUnshuffle1D(torch.nn.Module):
    """
    Inverse of 1D pixel shuffler
    Upscales channel length, downscales sample length
    "long" is input, "short" is output
    """
    def __init__(self, downscale_factor):
        super(PixelUnshuffle1D, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, x):
        batch_size = x.shape[0]
        long_channel_len = x.shape[1]
        long_width = x.shape[2]

        short_channel_len = long_channel_len * self.downscale_factor
        short_width = long_width // self.downscale_factor

        x = x.contiguous().view([batch_size, long_channel_len, short_width, self.downscale_factor])
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view([batch_size, short_channel_len, short_width])
        return x
    

class CAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, kernel_sizes, strides, paddings, dropout = 0.0, latent_dim = None, act = nn.ReLU, pixel_shuffle = True):
        
        super(CAE, self).__init__()

        hidden_dims = hidden_dims.copy()
        kernel_sizes = kernel_sizes.copy()
        strides = strides.copy()
        paddings = paddings.copy()
        self.dropout = dropout
        self.act = act
        self.last_enc_ch = hidden_dims[-1]
        self.latent_dim = latent_dim 
        self.width = None 

        blocks = []
        for i in range(len(hidden_dims)):
            kernel_size = kernel_sizes[i]
            stride = strides[i]
            padding = paddings[i]
            input_dim_b = input_dim if i == 0 else hidden_dims[i-1]
            act = self.act 
            #Last layer has no activation function and dropout
            if i <= len(hidden_dims) - 1:
                dropout = self.dropout #0.0
                blocks.append(ConvBlock(input_dim_b, hidden_dims[i], kernel_size=kernel_size, stride=stride, padding=padding, bn = True, dropout=dropout, act=act))
            else:
                dropout = self.dropout
                blocks.append(ConvBlock(hidden_dims[i-1], hidden_dims[i], kernel_size=kernel_size, stride=stride, padding=padding, bn = False, dropout=dropout, act=act)) 

        self.encoder = nn.Sequential(*blocks)
        self.pixel_shuffle = pixel_shuffle

        if not pixel_shuffle:   
            blocks = []
            hidden_dims.reverse()
            kernel_sizes.reverse()
            strides.reverse()
            paddings.reverse()
            input_dim_b = hidden_dims[0]
            for i in range(1, len(hidden_dims)+1):
                kernel_size = kernel_sizes[i-1]
                stride = strides[i-1]
                padding = paddings[i-1]
                if i == len(hidden_dims):
                    # Last layer of the decoder should match the input dimension
                    output_dim = input_dim
                    dropout = 0.0 #self.dropout # No dropout in the last layer
                    bn = False
                    act = None
                else:
                    output_dim = hidden_dims[i]
                    dropout = 0.0#self.dropout
                    bn = False
                    act = self.act
                #print(f"Layer {i}: input_dim={input_dim_b}, output_dim={output_dim}")
                blocks.append(ConvTransposeBlock(input_dim_b, output_dim, kernel_size=kernel_size, stride=stride, padding=padding, dropout=dropout, bn = bn, act= act))
                input_dim_b = output_dim
        else:
            blocks = []
            blocks.append(nn.Conv1d(hidden_dims[-1], 16, kernel_size=3, stride=1, padding=1))  # Prepare for pixel shuffle
            blocks.append(nn.BatchNorm1d(16))
            blocks.append(act())
            #if dropout > 0.0:
            #    blocks.append(nn.Dropout(dropout))
            # Pixel shuffle layer to upscale the feature maps
            blocks.append(PixelShuffle1D(upscale_factor=16))

        self.decoder = nn.Sequential(*blocks)

        self.final_block = nn.Sequential(
                *[
                    nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=1),  # Final layer to match the input dimension
                ]
        )

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x) 
        return x
    
    def encode(self, x):
        """
        Encode the input tensor into the latent space.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Encoded tensor.
        """
        x = self.encoder(x)
        if not self.pixel_shuffle:
            x = x.view(x.size(0), -1)  # Flatten the tensor
            self.width = x.shape[-1]
        return x
    
    def decode(self, x):
        """
        Decode the latent tensor back to the original space.
        Args:
            x (torch.Tensor): Latent tensor.
        Returns:
            torch.Tensor: Decoded tensor.
        """
        if self.pixel_shuffle:
            x = self.decoder(x)
            x = x.view(x.size(0), 1, -1)  # Reshape to match the expected output
        else:
            width = int((x.size(1) / self.last_enc_ch))
            x = x.view(x.size(0), self.last_enc_ch, width)
            x = self.decoder(x)
        return x