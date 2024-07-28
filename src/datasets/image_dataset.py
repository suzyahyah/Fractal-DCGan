#!/usr/bin/python
# Author: Suzanna Sia
import torchvision.datasets as dset
import torchvision.transforms as transforms
from omegaconf import OmegaConf

class ImageDataset():
    def __init__(self, image_config):
        self.data_dir = image_config.data_dir
        self.image_size = image_config.image_size
        self.dataset = self._make_dset()
        

    def _make_dset(self):
        dataset = dset.ImageFolder(root=self.data_dir, 
                                    transform=self.transform())

        return dataset


    def transform(self):
        return transforms.Compose([transforms.Resize(self.image_size), transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



