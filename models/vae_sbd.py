from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
import math
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


class VAE_SBD(nn.Module):
    def __init__(self, input_size, n_feat, n_pool, channels_start, exp_factor=2, weights=None, train_new=True):
        """

        :param input_size: shape of image in the form of a list of length 3 [h, w, ch] where h == w
        :param n_feat: number of features in embedding space
        :param n_pool: number of pooling layers in cnn network
        :param channels_start: number of chanels in first conv block
        :param exp_factor: factor by which number of channels in conv blocks increases (defaulted to 2)
        :param weights: parameter specifying path to saved weights (defaulted to None)
        :param train_new: boolean specifying whether a new model should be trained (defaulted to True)
        """
        super(VAE_SBD, self).__init__()

        # check input dims
        try:
            assert (len(input_size) == 3)
            assert (input_size[1] == input_size[2])
            assert (isinstance(n_pool, int))
            assert (isinstance(channels_start, int))
            assert (isinstance(exp_factor, (int, float)) and exp_factor > 0)
        except AssertionError:
            print("VAE parameter asserions not satisfied\n"
                  "Make sure input size is a list specifying image shapes where 'image width' = 'image height'\n"
                  "Make sure n_pool layers, and channels_start are integers\n"
                  "Make sure exp_factor is a positive number")

        # set up parameters for generator script
        cur_ch = input_size[0]
        next_ch = channels_start
        cur_dim = input_size[1]

        # set up necessary accumulator lists
        enc_layers = []

        # build encoder/decoder based on input parameters
        for i in range(n_pool):
            # add necessary layers to encoder
            enc_layers.append(torch.nn.Conv2d(cur_ch, next_ch, 3, padding=1))
            enc_layers.append(torch.nn.ReLU())
            enc_layers.append(torch.nn.Conv2d(next_ch, next_ch, 3, padding=1))
            enc_layers.append(torch.nn.ReLU())
            enc_layers.append(torch.nn.MaxPool2d(2, stride=2, return_indices=False))

            # update generator variables
            cur_ch = next_ch
            next_ch = math.ceil(next_ch * exp_factor)
            cur_dim = cur_dim // 2

        dec_layers = nn.Sequential(
            nn.Conv2d(in_channels=n_feat+2, out_channels=64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=input_size[0], kernel_size=3, padding=1),
        )

        self.fc = nn.Sequential(nn.Linear(in_features=cur_dim * cur_dim * cur_ch, out_features=1024),
                                nn.ReLU(),
                                nn.Linear(in_features=1024, out_features=256))

        self.mu = nn.Linear(256, n_feat)
        self.sigma = nn.Linear(256, n_feat)

        x = torch.linspace(-1, 1, input_size[1])
        y = torch.linspace(-1, 1, input_size[1])
        x_grid, y_grid = torch.meshgrid(x, y)
        self.register_buffer('x_grid', x_grid.view((1, 1) + x_grid.shape))
        self.register_buffer('y_grid', y_grid.view((1, 1) + y_grid.shape))

        # create encoder/decoder
        self.encoder = EncoderV1(nn.ModuleList(enc_layers))
        self.decoder = DecoderV1(dec_layers)
        if weights is not None and not train_new:
            self.load_state_dict(torch.load(weights))

        # save necessary variables
        self.shape = input_size
        self.weights = weights
        self.state_size = (cur_ch, cur_dim, cur_dim)

    def get_state_size(self):
        return self.state_size

    def forward(self, x):

        enc = self.encoder(x)
        enc = enc.view(enc.size(0), -1)

        fc = self.fc(enc)

        mu = self.mu(fc)
        logvar = self.sigma(fc)

        z = self.reparameterize(mu, logvar)

        batch_size = z.size(0)
        z = z.view(z.shape + (1, 1))
        broadcast = z.expand(-1, -1, self.shape[1], self.shape[1])
        x = torch.cat((self.x_grid.expand(batch_size, -1, -1, -1),
                       self.y_grid.expand(batch_size, -1, -1, -1),
                       broadcast), dim=1)

        dec = self.decoder(x)
        return enc, dec, mu, logvar

    def train_batch(self, x):

        enc, dec, mu, logvar = self.forward(x)

        loss = self.loss_function(x, dec, mu, logvar, beta=1)

        return enc, dec, loss

    def loss_function(self, img, dec, mu, logvar, beta):

        mse_crit = torch.nn.MSELoss()

        mse_loss = mse_crit(img, dec)
        lk_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return mse_loss + lk_loss * beta

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def save_model(self, path = None):
        if path is not None:
            torch.save(self.state_dict(), path)
        elif self.weights is not None:
            torch.save(self.state_dict(), self.weights)


class EncoderV1(nn.Module):
    def __init__(self, layers):
        super(EncoderV1, self).__init__()

        self.layers = layers

    def forward(self, input):

        # get encoding
        enc = input
        for i, e_layer in enumerate(self.layers):
            enc = e_layer(enc)

        return enc


class DecoderV1(nn.Module):
    def __init__(self, layers):
        super(DecoderV1, self).__init__()

        self.layers = layers

    def forward(self, input):

        dec = input
        for i, d_layer in enumerate(self.layers):
            dec = d_layer(dec)

        return dec