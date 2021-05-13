"""Module containing the main VAE class."""

import torch
import torch.nn as nn
import pdb

from vae.utils.initialization import weights_init
from .decoders import DecoderBurgess, DecoderRezendeViola, IntegrationDecoderCNCVAE
from .encoders import EncoderBurgess, IntegrationEncoderCNCVAE


class VAE(nn.Module):
    def __init__(self, img_size, latent_dim, encoding_type=None, dense_size=128):
        """
        Class which defines model and forward pass.

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).
        """
        super(VAE, self).__init__()

        if list(img_size[1:]) not in [[32, 32], [64, 64]]:
            raise RuntimeError(
                f"{img_size} sized images not supported. Only (None, 32, 32)"
                + "and (None, 64, 64) supported. Build your own architecture "
                + "or reshape images!")

        self.latent_dim = latent_dim
        self.img_size = img_size
        self.num_pixels = self.img_size[1] * self.img_size[2]

        if encoding_type == 'IntegrativeCNCVAE':
            self.encoder = IntegrationEncoderCNCVAE(
                data_size=img_size,
                dense_units=dense_size,
                latent_dim=latent_dim)
            self.decoder = IntegrationEncoderCNCVAE(
                data_size=img_size,
                dense_units=dense_size,
                latent_dim=latent_dim)
        elif encoding_type == 'TamingVAEs':
            self.encoder = EncoderBurgess(img_size, self.latent_dim)
            self.decoder = DecoderRezendeViola(img_size, self.latent_dim)
        else:
            self.encoder = EncoderBurgess(img_size, self.latent_dim)
            self.decoder = DecoderBurgess(img_size, self.latent_dim)

        self.reset_parameters()

    def reparameterize(self, mean, logvar):
        """
        Samples from a normal distribution using the reparameterization trick.

        Parameters
        ----------
        mean : torch.Tensor
            Mean of the normal distribution. Shape (batch_size, latent_dim)

        logvar : torch.Tensor
            Diagonal log variance of the normal distribution.
            Shape : (batch_size, latent_dim)
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + std * eps
        else:
            return mean

    def forward(self, x):
        """
        Forward pass of model.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        
        latent_dist = self.encoder(x)
        latent_sample = self.reparameterize(*latent_dist)
        reconstruct = self.decoder(latent_sample)

        return reconstruct, latent_dist, latent_sample

    def reset_parameters(self):
        self.apply(weights_init)

    def sample_latent(self, x):
        """
        Returns a sample from the latent distribution.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        latent_dist = self.encoder(x)
        latent_sample = self.reparameterize(*latent_dist)

        return latent_sample
