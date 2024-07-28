#!/usr/bin/python
# Author: Suzanna Sia
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from utils import build
from omegaconf import OmegaConf

seed = 0
random.seed(seed)
torch.manual_seed(seed)
#torch.use_deterministic_algorithms(True)
# this code references the DCGan Pytorch Tutorial but heavily refactored.

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = build.build_dataset(args.data_configs)
    dataloader = build.build_dataloader(args.dataloader_configs, dataset)

    generator = build.build_model_generator(args.model_configs, device)
    discriminator = build.build_model_discriminator(args.model_configs, device)

    
    trainer = build.build_trainer(args.training_configs, 
                                  generator, 
                                  discriminator, 
                                  dataloader, 
                                  args.model_configs.latent_size,
                                  device)

    trainer.train()
    generator.generate()


if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model_config', default='configs/model.yaml')
    argparser.add_argument('--training_config', default='configs/training.yaml')
    argparser.add_argument('--data_config', default='configs/data.yaml')
    args = argparser.parse_args()

    args_dict = {}
    args = vars(args)
    for key in args:
        args_dict.update(OmegaConf.load(args[key]))
    args = argparse.Namespace(**args_dict)

    main(args)
