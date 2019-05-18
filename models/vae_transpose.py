import torch
import torch.nn as nn


class VAE_Transpose(nn.Module):
    def __init__(self, input_size, n_feat, n_pool, channels_start, exp_factor=2, dropout = 0.2, weights=None, train_new=True):
        """

        :param input_size: shape of image in the form of a list of length 3 [h, w, ch] where h == w
        :param n_feat: number of features in embedding space
        :param n_pool: number of pooling layers in cnn network
        :param channels_start: number of chanels in first conv block
        :param exp_factor: factor by which number of channels in conv blocks increases (defaulted to 2)
        :param weights: parameter specifying path to saved weights (defaulted to None)
        :param train_new: boolean specifying whether a new model should be trained (defaulted to True)
        """

        super(VAE_Transpose, self).__init__()

        encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
        )

        cur_dim = 3
        cur_ch = 256

        enc_layers_fc = nn.Sequential(
            nn.Linear(cur_ch * cur_dim * cur_dim, cur_ch * cur_dim),
            nn.Linear(cur_ch * cur_dim, cur_ch),
            nn.Linear(cur_ch, n_feat * 2)
        )
        dec_layers_fc = nn.Sequential(
            nn.Linear(n_feat, cur_ch),
            nn.Linear(cur_ch, cur_ch*cur_dim),
            nn.Linear(cur_ch*cur_dim, cur_ch*cur_dim*cur_dim),
        )

        decoder = nn.Sequential(
            nn.ConvTranspose2d(cur_ch, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=5, stride=1),
            # nn.Sigmoid(),
        )

        self.encoder = Encoder(encoder, enc_layers_fc)
        self.decoder = Decoder(decoder, dec_layers_fc, cur_dim, cur_ch)

        if weights is not None and not train_new:
            self.load_state_dict(torch.load(weights))

        # save necessary variables
        self.shape = input_size
        self.weights = weights

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):

        enc = self.encoder(x)

        mu, logvar = torch.chunk(enc, 2, dim=1)

        z = self.reparameterize(mu, logvar)

        dec = self.decoder(z)
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

    def save_model(self, path=None):
        if path is not None:
            torch.save(self.state_dict(), path)
        elif self.weights is not None:
            torch.save(self.state_dict(), self.weights)


class Encoder(nn.Module):
    def __init__(self, layers, layers_fc):
        super(Encoder, self).__init__()

        self.layers = layers
        self.layers_fc = layers_fc

    def forward(self, input):

        # get encoding
        enc = input
        for i, e_layer in enumerate(self.layers):
            enc = e_layer(enc)

        enc = enc.view(enc.size(0), -1)
        for i, e_layer_fc in enumerate(self.layers_fc):
            enc = e_layer_fc(enc)

        return enc


class Decoder(nn.Module):
    def __init__(self, layers, layers_fc, dim, ch):
        super(Decoder, self).__init__()

        self.layers = layers
        self.layers_fc = layers_fc
        self.dim = dim
        self.ch = ch

    def forward(self, input):

        dec = input

        for i, d_layer_fc in enumerate(self.layers_fc):
            dec = d_layer_fc(dec)

        dec = dec.view(dec.size(0), self.ch, self.dim, self.dim)

        for i, d_layer in enumerate(self.layers):
            dec = d_layer(dec)

        return dec