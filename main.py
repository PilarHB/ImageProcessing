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
from torch.utils.data import Dataset, DataLoader,random_split


from sklearn.metrics import confusion_matrix
#from plotcm import plot_confusion_matrix
import pdb

torch.set_printoptions(linewidth=120)

class CNN(pl.LightningModule):

    # defines the network
    def __init__(self):
        super(CNN, self).__init__()
        # PyTorch uses NCHW
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 28, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(28, 10, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2))
        self.dropout1 = torch.nn.Dropout(0.25)
        self.fc1 = torch.nn.Linear(1960, 18)
        self.dropout2 = torch.nn.Dropout(0.08)
        self.fc2 = torch.nn.Linear(18, 10)
        # we are also defing some variable for counting purposes
        self.valTotal = 0
        self.valCorrect = 0
        self.trainTotal = 0
        self.trainCorrect = 0

    # mandatory
    def forward(self, t):
        # evaluating the batch data as it moves forward in the netowrk
        print("Beginning")
        print(t.shape)
        t = self.layer1(t)
        print("After layer1")
        print(t.shape)
        t = self.layer2(t)
        print("After layer2")
        print(t.shape)
        t = self.dropout1(t)
        # t = torch.relu(self.fc1(t.view(t.size(0), -1)))
        print(t.shape)
        t = t.view(t.size(0), -1)
        print("Before fc1")
        print(t.shape)
        t = self.fc1(t)
        t = torch.relu(t)
        t = F.leaky_relu(self.dropout2(t))

        return F.softmax(self.fc2(t))

        # return t
        # return torch.relu(self.l1(x.view(x.size(0), -1)))

    # trainning loop
    def training_step(self, batch, batch_idx):
        x, y = batch
        # x = x.view(x.size(0),-1)
        y_hat = self(x)
        loss = F.nll_loss(y_hat, y)

        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    # If you need to do something with all the outputs of each training_step
    # def training_epoch_end(self, training_step_outputs):
    # for pred in training_step_outputs:

    # define optimizers
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

    # validation loop
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss)
        pred = ...
        return {'loss': loss, 'pred': pred}


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
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize([64, 64], interpolation=2),
            transforms.ToTensor()
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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(len(os.listdir('./images/training/')))
    print(len(os.listdir('./images/training/fail')))
    print(len(os.listdir('./images/training/success')))

    image_module = MyImageModule(batch_size=4)
    model = CNN()
    trainer = pl.Trainer(default_root_dir='./checkpoints', gpus=1)
    trainer.fit(model, image_module)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
