# import libraries

import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import os

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import pytorch_lightning as pl
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader,random_split
from pytorch_lightning import Trainer, seed_everything

class MyImageModule(pl.LightningDataModule):

    def __init__(self, batch_size):
        super().__init__()
        self.trains_dims = None
        self.batch_size = batch_size
        self.train_dir = './images/training/'
        self.test_dir = './images/test/'

    # def prepare_data(self):

    # self.train_data = datasets.ImageFolder(train_dir, transform=transform)
    # self.val_data = datasets.ImageFolder(val_dir, transform=transform)
    # elf.test_data = datasets.ImageFolder(test_dir, transform=transform)

    def setup(self, step):
        transform = transforms.Compose([
            # you can add other transformations in this list
            # transforms.Grayscale(num_output_channels=1),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        train_data = datasets.ImageFolder(self.train_dir, transform=transform)
        train_data, val_data = random_split(train_data, [170, 21])

        self.train_data = train_data
        self.val_data = val_data
        # self.test_data = datasets.ImageFolder(test_dir, transform=transform)

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size)
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(self.val_data, batch_size=self.batch_size)
        return val_loader

        # def test_dataloader(self):
        # test_loader = torch.utils.data.DataLoader(self.test_data, batch_size = self.batch_size)
        # return test_loader