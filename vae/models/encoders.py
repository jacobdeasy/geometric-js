"""Module containing the encoders."""

import numpy as np
import torch
import torch.nn as nn
import pdb


class EncoderBurgess(nn.Module):
    def __init__(self, img_size,
                 latent_dim=10):
        r"""Encoder of the model proposed in [1].

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).

        latent_dim : int
            Dimensionality of latent output.

        Model Architecture (transposed for decoder)
        ------------
        - 4 convolutional layers (each with 32 channels), (4 x 4 kernel),
        (stride of 2)
        - 2 fully connected layers (each of 256 units)
        - Latent distribution:
            - 1 fully connected layer of 20 units (log variance and mean for
            10 Gaussians)

        References:
            [1] Burgess, Christopher P., et al. "Understanding disentangling in
            $\beta$-VAE." arXiv preprint arXiv:1804.03599 (2018).
        """
        super(EncoderBurgess, self).__init__()

        # Layer parameters
        hid_channels = 32
        kernel_size = 4
        hidden_dim = 256
        self.latent_dim = latent_dim
        self.img_size = img_size
        # Shape required to start transpose convs
        self.reshape = (hid_channels, kernel_size, kernel_size)
        n_chan = self.img_size[0]

        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)
        self.conv1 = nn.Conv2d(
            n_chan, hid_channels, kernel_size, **cnn_kwargs)
        self.conv2 = nn.Conv2d(
            hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.conv3 = nn.Conv2d(
            hid_channels, hid_channels, kernel_size, **cnn_kwargs)

        # If input image is 64x64 do fourth convolution
        if self.img_size[1] == self.img_size[2] == 64:
            self.conv_64 = nn.Conv2d(
                hid_channels, hid_channels, kernel_size, **cnn_kwargs)

        # Fully connected layers
        self.lin1 = nn.Linear(np.product(self.reshape), hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)

        # Fully connected layers for mean and variance
        self.mu_logvar_gen = nn.Linear(hidden_dim, self.latent_dim * 2)

    def forward(self, x):
        batch_size = x.size(0)
        
        # Convolutional layers with ReLu activations
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        if self.img_size[1] == self.img_size[2] == 64:
            x = torch.relu(self.conv_64(x))

        # Fully connected layers with ReLu activations
        x = x.view((batch_size, -1))
        x = torch.relu(self.lin1(x))
        x = torch.relu(self.lin2(x))

        # Fully connected layer for log variance and mean
        # Log std-dev in paper (bear in mind)
        mu_logvar = self.mu_logvar_gen(x)
        mu, logvar = mu_logvar.view(-1, self.latent_dim, 2).unbind(-1)

        return mu, logvar


class IntegrationEncoderCNCVAE(nn.Module):
    def __init__(self, data_size, dense_units=128, latent_dim=16):
        r"""Encoder of the concatanation VAE [1].

        Parameters
        ----------
        data_size : int
            Dimensionality of the input data
        
        dense_units : int
            Number of units for the dense layer

        latent_dim : int
            Dimensionality of latent output.

        Model Architecture (transposed for decoder)
        ------------
        - 1 fully connected layer with units defined by dense_units
        - Latent distribution:
            - 1 fully connected layer of latent_dim units (log variance and mean for
            10 Gaussians)

        References:
            [1] Simidjievski, Nikola et al. “Variational Autoencoders for Cancer
                Data Integration: Design Principles and Computational Practice.” 
                Frontiers in genetics vol. 10 1205. 11 Dec. 2019,
                doi:10.3389/fgene.2019.01205
        """
        super(IntegrationEncoderCNCVAE, self).__init__()

        self.data_size = data_size
        self.dense_units = dense_units
        self.latent_dim = latent_dim

        # define encoding layers
        self.encode = nn.Linear(self.data_size, self.dense_units)
        self.embed = nn.Linear(self.dense_units, self.latent_dim * 2)


    def forward(self, x):
        x = self.encode(x)
        z = self.embed(x)
        mu, logvar = z.view(-1, self.latent_dim, 2).unbind(-1)

        return mu, logvar
