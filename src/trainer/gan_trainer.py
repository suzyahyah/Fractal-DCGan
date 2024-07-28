#!/usr/bin/python
# Author: Suzanna Sia

import torch
import torch.nn as nn 
import torch.optim as optim
import torchvision.utils as vutils
from tqdm import tqdm

import numpy as np
rd = lambda x: np.around(x, 2)

class GANTrainer():
    def __init__(self, training_configs, netG, netD, dataloader, nz, device):

        self.lr = training_configs.lr
        self.nz = nz
        self.device = device

        self.beta1 = training_configs.adam_optimizer.beta1
        self.num_epochs = training_configs.num_epochs
        self.criterion = nn.BCELoss()
        self.netG = netG
        self.netD = netD
        self.dataloader = dataloader

        self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.lr, betas=(self.beta1, 0.999))


        self.fixed_noise = torch.randn(64, self.nz, 1, 1).to(device)


    # ganhacks, conduct different mini-batches for real and fake images. 
    #
    def train(self):

        img_list = []

        for epoch in tqdm(range(self.num_epochs)):
            for i, batch in enumerate(self.dataloader):

                fake_images = self._generate_fake_images(batch)
                lossD = self._train_discriminator(batch, fake_images)
                lossG = self._train_generator(batch, fake_images)

            if epoch % 10 == 0:
                print(f"Epoch: {epoch}, LossD: {rd(lossD)}, lossG: {rd(lossG)}")
                fake_image_check = self.netG.save_image(self.fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake_image_check, padding=2, normalize=True))
                torch.save(fake_image_check, f"data/fake_images/img_{epoch}.pt")


    def _train_discriminator(self, batch, fake_images):

        self.netD.zero_grad()

        lossD_real = self._train_discriminator_on_real_images(batch)
        lossD_fake = self._train_discriminator_on_fake_images(fake_images)

        #lossD = lossD_real + lossD_err
        self.optimizerD.step()
        return lossD_real.detach().item() + lossD_fake.detach().item()


    def _train_discriminator_on_real_images(self, batch):

        real_ = batch[0].to(self.device)
        bsize = batch[0].size(0)
        labels = torch.ones(bsize, dtype=torch.float, device=self.device)
        # prediction score on 1
        output_d = self.netD(real_).squeeze()[:, 1]
        lossD_real = self.criterion(output_d, labels)
        lossD_real.backward()
        #D_x = output.mean().item()
        return lossD_real

    def _train_discriminator_on_fake_images(self, fake_images):
        bsize = len(fake_images)
        labels = torch.zeros(bsize, dtype=torch.float, device=self.device)

        output = self.netD(fake_images.detach()).view(-1)

        lossD_fake = self.criterion(output, labels)
        lossD_fake.backward()
        #D_G_z1 = output.mean().item()
        return lossD_fake


    #def _train_generator(self):
    def _generate_fake_images(self, batch):
        bsize = batch[0].size(0)
        noise = torch.randn((bsize, self.nz, 1, 1), device=self.device)
        gen_fake_images = self.netG(noise)
        return gen_fake_images

    def _train_generator(self, batch, fake_images):
        self.netG.zero_grad()
        bsize = batch[0].size(0)
        labels = torch.ones(bsize, dtype=torch.float, device=self.device)
        output = self.netD(fake_images).view(-1)
        lossG = self.criterion(output, labels)
        lossG.backward()

        self.optimizerG.step()
        return lossG.detach().item()


