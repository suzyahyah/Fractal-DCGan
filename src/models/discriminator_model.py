#!/usr/bin/python
# Author: Suzanna Sia

import torch.nn as nn

class DCDiscriminatorModel(nn.Module):
    def __init__(self, model_configs):
        super(DCDiscriminatorModel, self).__init__()

        self.nc = model_configs.n_channels
        self.ndf = model_configs.depth_feature_maps_discriminator
        
        self.main = nn.Sequential(
            nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.ndf*2, self.ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.ndf*4, self.ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.ndf *8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
