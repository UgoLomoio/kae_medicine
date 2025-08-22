"""
Kolmogorov Arnold Variational Autoencoder (KVAE) for signal reconstuction and generation.
"""

from models.kan import KAN
from models.kcn import KCN, KCNTranspose
import torch
from torch import nn
from torch.nn import functional as F

class KVAE(nn.Module):

    def __init__(
        self,
        input_dim,
        layers_hidden,
        strides=1,
        paddings=0,
        kernel_sizes=3,
        width = None,
        window_size=4096,
        beta = 1.0,
        conv=False,
        base_activation=nn.SiLU
    ):
        super(KVAE, self).__init__()

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
            self.encoder = KCN([1, *layers_hidden], kernel_sizes=kernel_sizes, strides=strides, paddings=paddings, base_activation=base_activation)
            self.decoder = KCNTranspose([*layers_hidden[::-1][:-1], layers_hidden[::-1][-1]], kernel_sizes=kernel_sizes[::-1][:-1], strides=strides[::-1][:-1], paddings=paddings[::-1][:-1], base_activation=base_activation)
        else:
            self.encoder = KAN([input_dim, *layers_hidden], base_activation=base_activation)
            self.decoder = KAN([*layers_hidden[::-1][:-1], layers_hidden[::-1][-1]], base_activation=base_activation)
        
        if self.conv:
            if width is not None:
                self.width = width
            else:
                self.width = layers_hidden[-1]
            self.last_enc_ch = 1
        else:
            self.width = layers_hidden[-1]
            self.last_enc_ch = layers_hidden[-1]

        if self.width is not None:
            self.fc_mu = KAN([self.width,  self.width], base_activation=base_activation)
            self.fc_sigma = KAN([self.width,  self.width], base_activation=base_activation)

        if self.conv:
            self.after_decoder = nn.Sequential(
                nn.ConvTranspose1d(layers_hidden[::-1][-1], 1, kernel_size=kernel_sizes[::-1][-1], stride=strides[:-1][-1], padding=paddings[::-1][-1]),
            )
        else:
            self.after_decoder = nn.Linear(layers_hidden[::-1][-1], input_dim)
        
        self.batch_size = None  # Placeholder for batch size, to be set during training or inference

    def encode(self, input):
        
        posterior_dist = self.encoder(input)
        # Split the result into miu and sigma components
        # of the latent Gaussian distribution
        if self.conv:
            posterior_dist = posterior_dist.view(posterior_dist.shape[0], 1, -1)
            #posterior_dist = posterior_dist.squeeze(1)
            if posterior_dist.shape[-1] != self.width:
                raise ValueError(f"Expected width {self.width}, but got {posterior_dist.shape[-1]}. Check your input size and model configuration.")
        miu = self.fc_mu(posterior_dist)
        log_sigma = self.fc_sigma(posterior_dist)

        return [miu, log_sigma]

    def decode(self, z):
        priori_dist = self.decoder(z)
        priori_dist = self.after_decoder(priori_dist)
        return priori_dist

    def reparameterize(self, miu, log_sigma):
        std = torch.exp(0.5 * log_sigma)
        eps = torch.randn_like(std)
        return eps * std + miu

    def forward(self, x, **kwargs):

        if self.batch_size is None:
            self.batch_size = x.shape[0]
            
        if not self.conv:
            if x.ndim == 3:
                x = x.squeeze(1)
        miu, log_sigma = self.encode(x)
        z = self.reparameterize(miu, log_sigma)
        if self.conv:
            z = z.view(self.batch_size, self.last_enc_ch, -1)
        return  [self.decode(z), miu, log_sigma]

    def loss_function(self, criterion,
                    *args, **kwargs):
        
        input, ouput, miu, log_sigma = args
        kld_weight = kwargs["M_N"]
        
        recons_loss = criterion(input, ouput)
        if criterion.reduction == "sum":
            recons_loss = recons_loss / input.shape[0]
            
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_sigma - miu ** 2 - log_sigma.exp(), dim = 1), dim = 0)
        loss = recons_loss + self.beta*kld_weight*kld_loss
        return {'loss': loss, 'recon_loss':recons_loss.detach(), 'kld_loss':-kld_loss.detach()}

    def sample(self,
            num_samples:int,
            current_device: int, **kwargs):
        """
            Samples from the latent space and return the corresponding
            image space map.
            :param num_samples: (Int) Number of samples
            :param current_device: (Int) Device to run the model
            :return: (Tensor)
        """

        
        z = torch.randn(num_samples,
                    self.layers_hidden[-1])
        z = z.to(current_device)
        samples = self.decode(z)
        return samples

    def _init_weights(self, module):
        # Skip activation functions
        if isinstance(module, (nn.SiLU, nn.ReLU, nn.GELU, nn.Tanh, nn.Sigmoid)):
            return
        
        # Only apply kaiming_normal_ to modules with weight tensors
        if hasattr(module, 'weight') and isinstance(module.weight, torch.Tensor):
            nn.init.kaiming_normal_(module.weight, nonlinearity='silu')
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)


