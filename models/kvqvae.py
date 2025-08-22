"""
Kolmogorov Arnold Vector Quantized Variational Autoencoder (K-VQVAE) for signal reconstuction and generation.
"""

from models.kan import KAN
from models.kcn import KCN
from models.vqvae import VectorQuantizer1d
import torch
from torch import nn
from torch.nn import functional as F

class KVQVAE(nn.Module):

    def __init__(
        self,
        input_dim,
        layers_hidden,
        strides=1,
        paddings=0,
        kernel_sizes=3,
        emb_dim = 512,
        window_size=4096,
        beta = 0.25,
        conv=False,
        base_activation=nn.SiLU,
    ):
        super(KVQVAE, self).__init__()

        self.conv = conv
        self.beta = beta

        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * len(layers_hidden)
        if isinstance(strides, int):
            strides = [strides] * len(layers_hidden)
        if isinstance(paddings, int):
            paddings = [paddings] * len(layers_hidden)

        self.layers_hidden = layers_hidden

        if conv:
            self.encoder = KCN([1, *layers_hidden[:]], kernel_sizes=kernel_sizes, strides=strides, paddings=paddings, base_activation=base_activation)
            self.decoder = KCN([*layers_hidden[::-1], 1], kernel_sizes=kernel_sizes[::-1], strides=strides[::-1], paddings=paddings[::-1], base_activation=base_activation)
        else:
            self.encoder = KAN([input_dim, *layers_hidden[:]], base_activation=base_activation)
            self.decoder = KAN([*layers_hidden[::-1], input_dim], base_activation=base_activation)
        
        if self.conv:
            self.latent_dim = emb_dim
            self.last_enc_ch = 1
        else:
            self.latent_dim = layers_hidden[-1]
            self.last_enc_ch = layers_hidden[-1]

        self.vq_layer = VectorQuantizer1d(layers_hidden[-1], emb_dim, self.beta)
        
    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        return result

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D x  W]
        :return: (Tensor) [B x C x W]
        """

        result = self.decoder(z)
        return result

    def forward(self, input):
        if input.ndim == 3:
            input = input.squeeze(1)
        encoding = self.encode(input)
        if encoding.ndim == 2:
            encoding = encoding.unsqueeze(1)
        quantized_inputs, vq_loss = self.vq_layer(encoding)
        return [self.decode(quantized_inputs), input, vq_loss]

    def loss_function(self,
                      *args,
                      **kwargs):
        """
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        vq_loss = args[2]

        recons_loss = F.mse_loss(recons, input)

        loss = recons_loss + vq_loss
        return {'loss': loss,
                'recon_loss': recons_loss,
                'vq_loss':vq_loss}

    def sample(self,
               num_samples,
               current_device):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        self.batch_size = num_samples
        if self.conv:
            z = torch.randn(num_samples, 1, self.latent_dim)
        else:
            z = torch.randn(num_samples, self.last_enc_ch, self.latent_dim)
        z = z.to(current_device)
        samples = self.decode(z)
        return samples


