"""
This module contains the implementation of the VQ- Variational AutoEncoder class with convolutional/MLP block.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cae import ConvBlock, ConvTransposeBlock
from models.ae import LinearBlock

class VectorQuantizer1d(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 beta: float = 0.25):
        super(VectorQuantizer1d, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents):
        latents = latents.permute(0, 2, 1).contiguous()  # [B x D x W] -> [B x W x D]
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # [BW x D]

        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BW x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BW, 1]

        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BW x K]

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BW, D]
        quantized_latents = quantized_latents.view(latents_shape)  # [B x W x D]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()

        return quantized_latents.permute(0, 2, 1).contiguous(), vq_loss  # [B x D x W]
    
class VQVAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout = 0.0, conv=False, emb_dim = 128, beta = 0.25, kernel_sizes=None, strides=None, paddings=None, batch_size = 32, window_size = 4096, act=nn.ReLU):

        super(VQVAE, self).__init__()
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
            #No dropout in the last layer
            if i == len(hidden_dims) - 1:
                dropout = 0.0
                if conv:
                    bn = False
            else:
                if conv:
                    bn = True
                dropout = self.dropout
            #print(f"Layer {i}: input_dim={input_dim_b}, output_dim={h_dim}")
            if conv:
                blocks.append(block(input_dim_b, h_dim, dropout=dropout, kernel_size=kernel_sizes[i], stride=strides[i], padding=paddings[i], bn=bn, act=act))
            else:
                blocks.append(block(input_dim_b, h_dim, dropout=dropout, act=act))
        
        self.encoder = nn.Sequential(*blocks)

        self.vq_layer = VectorQuantizer1d(h_dim,
                                        emb_dim,
                                        self.beta
                                        )
                    
        
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
            else:
                dropout = self.dropout
                act = self.act
                output_dim = hidden_dims[i]
            #print(f"Layer {i}: input_dim={input_dim_b}, output_dim={output_dim}")
            if conv:
                blocks.append(block_transpose(input_dim_b, output_dim, dropout=dropout, kernel_size=kernel_sizes[i-1], stride=strides[i-1], padding=paddings[i-1], act=act))
            else:
                blocks.append(block(input_dim_b, output_dim, dropout=dropout, act=act))
            input_dim_b = output_dim
        self.decoder = nn.Sequential(*blocks)   

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        return result

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D x W]
        :return: (Tensor) [B x C x W]
        """
        result = self.decoder(z)
        return result

    def forward(self, input):
        encoding = self.encode(input)
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
    
    def interpolate_sample(self, input1, input2, num_steps=10):
        """
        Generate samples by interpolating between two inputs in latent space
        """
        self.eval()
        with torch.no_grad():
            # Encode both inputs
            z1 = self.encode(input1)
            z2 = self.encode(input2)
            
            # Quantize
            q1, _ = self.vq_layer(z1)
            q2, _ = self.vq_layer(z2)
            
            # Interpolate in quantized space
            alphas = torch.linspace(0, 1, num_steps).to(input1.device)
            interpolated_samples = []
            
            for alpha in alphas:
                interpolated_z = (1 - alpha) * q1 + alpha * q2
                # Re-quantize the interpolated vector
                quantized_interp, _ = self.vq_layer(interpolated_z)
                sample = self.decode(quantized_interp)
                interpolated_samples.append(sample)
                
            return torch.stack(interpolated_samples)

    def _sample_with_prior(self, num_samples, current_device, prior_model):
        """
        Sample from 1D VQ-VAE using a trained prior model
        Input format: [B, 1, W] where W is the sequence length
        
        :param num_samples: Number of samples to generate
        :param current_device: Device to run the model on
        :param prior_model: Trained autoregressive model for 1D sequences
        :return: Generated samples of shape [B, 1, W]
        """
        self.eval()
        with torch.no_grad():
            # Calculate the sequence length of the latent space after encoding
            latent_sequence_length = self._get_latent_sequence_length()
            
            # Sample discrete latent codes using the prior model
            # Shape: [batch_size, latent_sequence_length]
            samples = prior_model.sample(
                num_samples,
                current_device
            )
            
            # Flatten indices for embedding lookup
            flat_samples = samples.view(-1)  # [batch_size * sequence_length]
            
            # Get quantized vectors from codebook
            quantized_latents = self.vq_layer.embedding(flat_samples)  # [B*W, embedding_dim]
            
            # Reshape to [batch_size, sequence_length, embedding_dim]
            quantized_latents = quantized_latents.view(
                num_samples, self.window_size, self.vq_layer.D
            )
            
            # Convert to format expected by decoder: [B, embedding_dim, W]
            # This matches the output format of your VectorQuantizer1d
            quantized_latents = quantized_latents.permute(0, 2, 1).contiguous()
            
            # Decode the quantized latents to generate samples
            samples = self.decode(quantized_latents)  # Output: [B, 1, W]
            
        return samples

    def _get_latent_sequence_length(self):
        """
        Calculate the latent sequence length after encoding [B, 1, W] input
        """
        # For 1D convolutions, calculate downsampling from your encoder
        input_length = self.window_size  # W dimension
        
        # Calculate based on your conv layers
        # You need to track the actual conv parameters from your encoder
        # This is a simplified calculation - adjust based on your architecture
        
        # Example: if you have conv layers that downsample by factor of 8 total
        downsampling_factor = 32  # Adjust based on your actual encoder
        latent_length = input_length // downsampling_factor
        
        return latent_length

    def sample(self, num_samples, current_device, method='prior', **kwargs):
        """
        Samples from the VQ-VAE model
        
        :param method: 'prior' (requires prior_model), 'interpolate' (requires input1, input2)
        """
        if method == 'prior':
            prior_model = kwargs.get('prior_model')
            if prior_model is None:
                raise ValueError("Prior model required for prior sampling")
            return self._sample_with_prior(num_samples, current_device, prior_model)
        
        elif method == 'interpolate':
            input1 = kwargs.get('input1')
            input2 = kwargs.get('input2')
            if input1 is None or input2 is None:
                raise ValueError("Two inputs required for interpolation sampling")
            return self.interpolate_sample(input1, input2, num_samples)
        
        else:
            raise ValueError("Method must be 'prior' or 'interpolate'")