#!/usr/bin/python
# Author: Suzanna Sia

import torch

import torch.nn as nn

class DCGeneratorModel(nn.Module):
    def __init__(self, model_configs):
        # strided 2D convolution transpose (blows up nxn into mxm) were m>n.
        # each paired with 2d batch norm 
        # relu activation
        super(DCGeneratorModel, self).__init__()

        self.nz = model_configs.latent_size
        self.ngf = model_configs.depth_feature_maps_generator
        self.nc = model_configs.n_channels

        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.nz, self.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(self.ngf * 8, self.ngf *4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.ngf, self.nc, 4, 2, 1, bias=False),

            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


    def save_image(self, fixed_noise):
        with torch.no_grad():
            fake = self.main(fixed_noise).detach().cpu()
        return fake
