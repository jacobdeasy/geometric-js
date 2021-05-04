"""Module containing the decoders."""

import numpy as np
import torch
import torch.nn as nn
import pdb


class DecoderBurgess(nn.Module):
    def __init__(self, img_size,
                 latent_dim=10):
        r"""Decoder of the model proposed in [1].

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
        super(DecoderBurgess, self).__init__()

        # Layer parameters
        hid_channels = 32
        kernel_size = 4
        hidden_dim = 256
        self.img_size = img_size
        # Shape required to start transpose convs
        self.reshape = (hid_channels, kernel_size, kernel_size)
        n_chan = self.img_size[0]
        self.img_size = img_size

        # Fully connected layers
        self.lin1 = nn.Linear(latent_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, np.product(self.reshape))

        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)
        # If input image is 64x64 do fourth convolution
        if self.img_size[1] == self.img_size[2] == 64:
            self.convT_64 = nn.ConvTranspose2d(
                hid_channels, hid_channels, kernel_size, **cnn_kwargs)

        self.convT1 = nn.ConvTranspose2d(
            hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.convT2 = nn.ConvTranspose2d(
            hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.convT3 = nn.ConvTranspose2d(
            hid_channels, n_chan, kernel_size, **cnn_kwargs)

    def forward(self, z):
        batch_size = z.size(0)

        # Fully connected layers with ReLu activations
        x = torch.relu(self.lin1(z))
        x = torch.relu(self.lin2(x))
        x = torch.relu(self.lin3(x))
        x = x.view(batch_size, *self.reshape)

        # Convolutional layers with ReLu activations
        if self.img_size[1] == self.img_size[2] == 64:
            x = torch.relu(self.convT_64(x))
        x = torch.relu(self.convT1(x))
        x = torch.relu(self.convT2(x))
        # Sigmoid activation for final conv layer
        x = torch.sigmoid(self.convT3(x))

        return x


class DecoderRezendeViola(nn.Module):
    def __init__(self, img_size,
                 latent_dim=10):
        r"""Decoder of the model used in [1].

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
            [1] Danilo Jimenez Rezende and Fabio Viola. Taming vaes, 2018.
        """
        
        super(DecoderRezendeViola, self).__init__()

        # Layer parameters
        hid_channels = 32
        kernel_size = 4
        hidden_dim = 256
        self.img_size = img_size
        # Shape required to start transpose convs
        self.reshape = (hid_channels, kernel_size, kernel_size)
        n_chan = self.img_size[0]
        self.img_size = img_size

        # Fully connected layers
        self.lin1 = nn.Linear(latent_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, np.product(self.reshape))

        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)
        # If input image is 64x64 do fourth convolution
        if self.img_size[1] == self.img_size[2] == 64:
            self.convT_64 = nn.ConvTranspose2d(
                hid_channels, hid_channels, kernel_size, **cnn_kwargs)

        self.convT1 = nn.ConvTranspose2d(
            hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.convT2 = nn.ConvTranspose2d(
            hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.convT3 = nn.ConvTranspose2d(
            hid_channels, 2 * n_chan, kernel_size, **cnn_kwargs)
        
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
            # std = torch.zeros_like(mean) + 0.25
            eps = torch.randn_like(std)
            return mean + std * eps
        else:
            return mean

    def forward(self, z):
        
        batch_size = z.size(0)

        # Fully connected layers with ReLu activations
        x = torch.relu(self.lin1(z))
        x = torch.relu(self.lin2(x))
        x = torch.relu(self.lin3(x))
        x = x.view(batch_size, *self.reshape)

        # Convolutional layers with ReLu activations
        if self.img_size[1] == self.img_size[2] == 64:
            x = torch.relu(self.convT_64(x))
        x = torch.relu(self.convT1(x))
        x = torch.relu(self.convT2(x))
        # Sigmoid activation for final conv layer
        
        x = torch.sigmoid(self.convT3(x))
        out = self.reparameterize(x[:,0,:,:].view(-1, self.img_size[0],self.img_size[1], self.img_size[2]), x[:,1,:,:].view(-1, self.img_size[0],self.img_size[1], self.img_size[2]))

        return out


class IntegrationDecoderCNCVAE(nn.Module):
    def __init__(self, data_size, latent_dim=16, dense_units=128):
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
        super(IntegrationDecoderCNCVAE, self).__init__()
        self.data_size = data_size
        self.dense_units = dense_units
        self.latent_dim = latent_dim

        # define decoding layers
        self.de_embed = nn.Linear(self.latent_dim, self.dense_units)
        self.decode = nn.Linear(self.dense_units, self.self.latent_dim)


    def forward(self, z):
        hidden = self.de_embed(z)
        x = self.decode(hidden)

        return x