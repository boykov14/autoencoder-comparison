import torch
import torch.nn as nn

class Autoencoder(nn.Module):
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

        super(Autoencoder, self).__init__()

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
        dec_layers = []

        # build encoder/decoder based on input parameters
        for i in range(n_pool):
            # add necessary layers to encoder
            enc_layers.append(torch.nn.Conv2d(cur_ch, next_ch, 3, padding=1))
            enc_layers.append(torch.nn.ReLU())
            enc_layers.append(torch.nn.Conv2d(next_ch, next_ch, 3, padding=1))
            enc_layers.append(torch.nn.ReLU())
            enc_layers.append(torch.nn.MaxPool2d(2, stride=2, return_indices=True))

            # add necessary layers to decoder
            dec_layers.insert(0, torch.nn.ConvTranspose2d(next_ch, cur_ch, 3, padding=1))
            dec_layers.insert(0, torch.nn.ConvTranspose2d(next_ch, next_ch, 3, padding=1))
            dec_layers.insert(0, torch.nn.ConvTranspose2d(next_ch, next_ch, 3, padding=1))
            dec_layers.insert(0, torch.nn.ConvTranspose2d(next_ch, next_ch, 3, padding=1))
            dec_layers.insert(0, torch.nn.MaxUnpool2d(2, stride=2))

            # update generator variables
            cur_ch = next_ch
            next_ch = next_ch * exp_factor
            cur_dim = cur_dim // 2

        enc_layers_fc = [
            nn.Linear(cur_dim * cur_dim * cur_ch, n_feat)
        ]

        dec_layers_fc = [
            nn.Linear(n_feat, cur_dim * cur_dim * cur_ch)
        ]

        # create encoder
        self.encoder = Encoder(nn.ModuleList(enc_layers), nn.ModuleList(enc_layers_fc))
        self.decoder = Decoder(nn.ModuleList(dec_layers), nn.ModuleList(dec_layers_fc), cur_dim, cur_ch)

        if weights is not None and not train_new:
            self.load_state_dict(torch.load(weights))

        # save necessary variables
        self.shape = input_size
        self.weights = weights

    def forward(self, x):

        enc, indices = self.encoder(x, return_indices=True)
        dec = self.decoder(enc, indices)
        return enc, dec

    def train_batch(self, x):

        criterion = torch.nn.MSELoss()

        enc, dec = self.forward(x)

        loss = criterion(dec, x)

        return enc, dec, loss

    def save_model(self, path = None):
        if path is not None:
            torch.save(self.state_dict(), path)
        elif self.weights is not None:
            torch.save(self.state_dict(), self.weights)

class Encoder(nn.Module):
    def __init__(self, layers, layers_fc):
        super(Encoder, self).__init__()

        self.layers = layers
        self.layers_fc = layers_fc

    def forward(self, input, return_indices):

        indices = []

        # get encoding
        enc = input
        for i, e_layer in enumerate(self.layers):
            if type(e_layer) is torch.nn.MaxPool2d:
                enc, ind = e_layer(enc)
                if return_indices:
                    indices.append(ind)
            else:
                enc = e_layer(enc)

        enc = enc.view(enc.size(0), -1)
        for i, e_layer_fc in enumerate(self.layers_fc):
            enc = e_layer_fc(enc)

        if return_indices:
            return enc, indices
        else:
            return enc


class Decoder(nn.Module):
    def __init__(self, layers, layers_fc, dim, ch):
        super(Decoder, self).__init__()

        self.layers = layers
        self.layers_fc = layers_fc
        self.dim = dim
        self.ch = ch

    def forward(self, input, indices):

        ind_idx = len(indices) - 1
        dec = input

        for i, d_layer_fc in enumerate(self.layers_fc):
            dec = d_layer_fc(dec)

        dec = dec.view(dec.size(0), self.ch, self.dim, self.dim)

        for i, d_layer in enumerate(self.layers):
            if type(d_layer) is torch.nn.MaxUnpool2d:
                dec = d_layer(dec, indices[ind_idx])
                ind_idx -= 1
            else:
                dec = d_layer(dec)

        return dec