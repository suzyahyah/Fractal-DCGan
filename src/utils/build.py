#!/usr/bin/python
# Author: Suzanna Sia

from datasets.image_dataset import ImageDataset
from models.generator_model import DCGeneratorModel
from models.discriminator_model import DCDiscriminatorModel 
from models.utils import weights_init
from trainer.gan_trainer import GANTrainer

from torch.utils.data import DataLoader

def build_dataset(image_config):
    return ImageDataset(image_config).dataset

def build_dataloader(dataloader_configs, dataset):
    # we'll just use the default dataloader here
    dataloader = DataLoader(dataset,
                            batch_size=dataloader_configs.batch_size,
                            shuffle=True,
                            num_workers=dataloader_configs.workers)
    return dataloader

def build_model_generator(model_config, device):
    netG = DCGeneratorModel(model_config)
    netG.apply(weights_init).to(device)
    return netG


def build_model_discriminator(model_config, device):
    netD = DCDiscriminatorModel(model_config)
    netD.apply(weights_init).to(device)
    return netD

def build_trainer(training_configs, generator, discriminator, dataloader, nz, device):
    trainer = GANTrainer(training_configs, generator, discriminator, dataloader, nz, device)
    return trainer
