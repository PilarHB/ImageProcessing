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
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import Trainer, seed_everything


class MyImageModule(pl.LightningDataModule):

    def __init__(self, batch_size, dataset_size=None):
        super().__init__()
        self.trains_dims = None
        self.batch_size = batch_size
        self.data_dir = './images/'
        self.dataset_size = dataset_size

    # def prepare_data(self):

    # self.train_data = datasets.ImageFolder(train_dir, transform=transform)
    # self.val_data = datasets.ImageFolder(val_dir, transform=transform)
    # elf.test_data = datasets.ImageFolder(test_dir, transform=transform)

    def setup(self, stage=None):
        self.transform = transforms.Compose([
            # you can add other transformations in this list
            # transforms.Grayscale(num_output_channels=1),
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.augmentation = transforms.Compose([
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Build Dataset
        dataset = datasets.ImageFolder(self.data_dir)

        dataset_size = len(dataset) if self.dataset_size is None else min(len(dataset), self.dataset_size)
        train_size = int(0.7 * dataset_size)
        val_size = int(0.5 * (dataset_size - train_size))
        test_size = int(dataset_size - train_size - val_size)
        # test_size = len(dataset) - train_size - val_size
        # train_data = datasets.ImageFolder(self.train_dir, transform=transform)
        self.train_data, self.val_data, self.test_data = random_split(dataset,
                                                                      [train_size, val_size, test_size],
                                                                      generator=torch.Generator().manual_seed(42))
        print("Len Train Data", len(self.train_data))
        print("Len Val Data", len(self.val_data))
        print("Len Test Data", len(self.test_data))
        # Data Augmentation for Training
        self.train_data.dataset.transform = self.augmentation
        # Transform Data
        self.val_data.dataset.transform = self.transform
        self.test_data.dataset.transform = self.transform

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size)
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(self.val_data, batch_size=self.batch_size)
        return val_loader

    def test_dataloader(self):
        test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=self.batch_size)
        return test_loader

    # TODO: Método para acceder a las clases
    def find_classes(self):
        classes = [d.name for d in os.scandir(self.data_dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes

