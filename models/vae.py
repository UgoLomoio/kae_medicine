"""
This module contains the implementation of the Variational AutoEncoder class with convolutional/MLP block.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cae import ConvBlock, ConvTransposeBlock
from models.ae import LinearBlock

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout = 0.0, conv=False, width = None, beta = 1.0, kernel_sizes=None, strides=None, paddings=None, batch_size = 32, window_size = 4096, act=nn.ReLU):

        super(VAE, self).__init__()
        if conv:
            block = ConvBlock
            block_transpose = ConvTransposeBlock
        else:
            block = LinearBlock

        hidden_dims = hidden_dims.copy()
        self.conv = conv
        self.batch_size = batch_size
        self.window_size = window_size
        self.dropout = dropout
        self.act = act
        self.beta = beta
        
        if conv:
            kernel_sizes = kernel_sizes.copy()
            strides = strides.copy()
            paddings = paddings.copy()
            self.last_ch_enc = hidden_dims[-1]
        else: 
            self.last_ch_enc = 1

        blocks = []
        for i, h_dim in enumerate(hidden_dims):
            input_dim_b = input_dim if i == 0 else hidden_dims[i-1]
            act = self.act
            if conv:
                if i == len(hidden_dims) - 1:
                    bn = True  # No batch normalization in the last layer
                else:
                    bn = True
            dropout = 0.0 # No dropout in the encoder
            #print(f"Layer {i}: input_dim={input_dim_b}, output_dim={h_dim}")
            if conv:
                blocks.append(block(input_dim_b, h_dim, dropout=dropout, kernel_size=kernel_sizes[i], stride=strides[i], padding=paddings[i], bn=bn, act=act))
            else:
                blocks.append(block(input_dim_b, h_dim, dropout=dropout, act=act))
        self.encoder = nn.Sequential(*blocks)

        if self.conv:
            if width is None:
                ks_sum = sum(kernel_sizes)
                strides_sum = sum(strides)
                paddings_sum = sum(paddings)
                self.width = self.last_ch_enc * (window_size - ks_sum + 1) // strides_sum + 2 * paddings_sum
            else:
                self.width = width
            self.latent_dim = self.width
        else:
            self.latent_dim = hidden_dims[-1]
        
        if not self.conv:
            self.fc_mu = nn.Linear(self.latent_dim, self.latent_dim)
            self.fc_logvar = nn.Linear(self.latent_dim, self.latent_dim)
        else:
            self.fc_mu = nn.Linear(self.width, self.width)
            self.fc_logvar = nn.Linear(self.width, self.width)
            
        blocks = []
        hidden_dims.reverse()
        if conv:
            strides.reverse()
            kernel_sizes.reverse()
            paddings.reverse()
        input_dim_b = hidden_dims[0]
        for i in range(1, len(hidden_dims)+1):
            if i == len(hidden_dims):
                output_dim = input_dim
                dropout = 0.0  # No dropout in the last layer
                act = None
                bn = False
            else:
                dropout = self.dropout
                act = self.act
                bn = False  
                output_dim = hidden_dims[i]
            #print(f"Layer {i}: input_dim={input_dim_b}, output_dim={output_dim}")
            if conv:
                blocks.append(block_transpose(input_dim_b, output_dim, dropout=dropout, kernel_size=kernel_sizes[i-1], bn = bn, stride=strides[i-1], padding=paddings[i-1], act=act))
            else:
                blocks.append(block(input_dim_b, output_dim, dropout=dropout, act=act))
            input_dim_b = output_dim
        self.decoder = nn.Sequential(*blocks)   

    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [B x C x W]
        :return: (Tensor) Tuple of mu and log_var
        """
        result = self.encoder(input)
        if self.conv:
            #print(f"Encoder output shape: {result.shape}")
            result = result.view(result.shape[0], 1, -1)
            if result.shape[-1] != self.width:
                raise ValueError(f"Expected width {self.width}, but got {result.shape[-1]}. Check your input size and model configuration.")
         
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        #print(f"Latent vector shape: {result.shape}")
        mu = self.fc_mu(result)
        log_var = self.fc_logvar(result)
        return mu, log_var


    def decode(self, z):
        """
        Decodes the latent vector z back to the original data space.
        :param z: (Tensor) Latent vector [B x 1 x latent_dim]
        :return: (Tensor) Decoded output
        """
        if self.conv:
            z = z.view(self.batch_size, self.last_ch_enc, -1)
        # Pass through decoder network
        decoded = self.decoder(z)
        return decoded


    def forward(self, x):
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        self.batch_size = x.shape[0]
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
    
    def loss_function(self, criterion, *args, **kwargs):
        
        x, x_recon, mu, logvar = args
        kld_weight = kwargs["M_N"]
        
        rec_loss = criterion(x_recon, x)
        if criterion.reduction == "sum":
            rec_loss = rec_loss / x.shape[0]

        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kld_loss = self.beta*kld_weight*kld_loss
        loss = rec_loss + kld_loss
        return {
            'loss': loss,
            'recon_loss': rec_loss.detach(),
            'kld_loss': -kld_loss.detach()
        }
    
    def sample(self,
               num_samples,
               current_device):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples to generate
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        self.batch_size = num_samples
        if self.conv:
            if self.latent_dim is not None:
                # Sample from the latent space
                z = torch.randn(num_samples, 1, self.latent_dim)
            else:
                # Sample from the latent space
                z = torch.randn(num_samples, 1, self.width)
        else:
            z = torch.randn(num_samples, self.last_ch_enc, self.latent_dim)
        z = z.to(current_device)
        samples = self.decode(z)
        return samples
        