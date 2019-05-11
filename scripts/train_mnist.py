from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from models.simple_autoencoder import Autoencoder
from models.autoencoder_transpose import Autoencoder_Transpose
from models.vae_unpool import VAE_Unpool
from models.vae_transpose import VAE_Transpose

import numpy as np
import matplotlib.pyplot as plt

def train_vae_sbd():

    visualize = False

    epochs = 100
    batch_size = 100
    img_size = [1,28,28]
    train_new=True
    device='cpu'
    lr =0.0001

    model = Autoencoder_Transpose(
        input_size=img_size,
        n_feat=128,
        n_pool=2,
        channels_start=12,
        exp_factor=2,
        weights="..//Weights//Autoencoder_Transpose_test1.pt",
        train_new=train_new
    )

    model = model.to(device)

    # set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # define transforms for train/test
    transforms_train = transforms.Compose([
           transforms.ToTensor(),
           transforms.Normalize((0.1307,), (0.3081,))
       ])

    transforms_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    # create data loaders
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True, transform=transforms_train),
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms_test),
        batch_size=batch_size, shuffle=True)

    for epoch in range(1, epochs):
        train_model(model, optimizer, train_loader, device, visualize)
        # test_model(model, optimizer, test_loader, device)


def train_model(model, optimizer, train_loader, device, visualize):
    model.train()

    # set up plotting
    if visualize:
        fig = plt.figure(1, figsize=(8, 4))  # (figsize=(10,5))
        fig.subplots_adjust(left=0.05, right=0.95)
        ax2 = fig.add_subplot(221)
        ax1 = fig.add_subplot(222)

    for idx, (img, lab) in enumerate(train_loader):
        img, lab = img.to(device), lab.to(device)
        optimizer.zero_grad()

        enc, dec, loss = model.train_batch(img)

        # loss = loss_function(img, dec, mu, logvar, beta=1.0)

        loss.backward()
        optimizer.step()

        # display one image
        if visualize:
            pred_img = np.asarray(dec[0].data.to('cpu')).transpose((1, 2, 0))[:,:,0]
            sample_img_ = np.asarray(img[0].data.to('cpu')).transpose((1, 2, 0))[:,:,0]
            ax1.imshow(pred_img, animated=True)
            ax2.imshow(sample_img_, animated=True)
            plt.show(block=False)
            plt.pause(1 / 20)

        if idx % 10 == 0:
            print('[{}/{}]\tLoss: {:.6f}'.format(
                idx,
                len(train_loader),
                loss.item()))

        if idx % 100 == 0 and idx > 0:
            print("Saving Model Progress")
            model.save_model()

        model.save_model()

def test_model(model, test_loader, device):
    model.eval()
    loss_total = 0
    div_factor = 0

    for idx, (img, lab) in enumerate(test_loader):
        img, lab = img.to(device), lab.to(device)

        enc, dec, mu, logvar = model(img)

        loss = loss_function(img, dec, mu, logvar, beta=1.5)
        loss_total += loss
        div_factor += img.size(0)

        if idx % 10 == 0:
            print('Testing: [{}/{}]\tLoss: {:.6f}'.format(
                idx,
                len(test_loader),
                loss.item()))

    print("Total loss: {:.6f}".format(
        loss_total/div_factor
    ))
    model.train()

def loss_function(img, dec, mu, logvar, beta):

    mse_crit = torch.nn.MSELoss()

    mse_loss = mse_crit(img, dec)
    lk_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return mse_loss + lk_loss * beta

if __name__ == "__main__":
    train_vae_sbd()