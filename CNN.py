import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from collections import OrderedDict
import os

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
import torch.nn as nn
from torch.nn import Module
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import pytorch_lightning as pl
import torchvision.models as models
from typing import Optional
from torch.utils.data import Dataset, DataLoader,random_split
from pytorch_lightning.metrics.functional import accuracy


from sklearn.metrics import confusion_matrix
#from plotcm import plot_confusion_matrix
import pdb


#  --- Utility functions ---

class CNN(pl.LightningModule):

    # defines the network
    def __init__(self,
                 input_shape: list = [3, 256, 256],
                 backbone: str = 'vgg16',
                 train_bn: bool = True,
                 milestones: tuple = (5, 10),
                 batch_size: int = 8,
                 lr: float = 1e-2,
                 lr_scheduler_gamma: float = 1e-1,
                 num_workers: int = 6):
        super(CNN, self).__init__()
        # parameters
        self.dim = input_shape
        self.backbone = backbone
        self.train_bn = train_bn
        self.milestones = milestones
        self.batch_size = batch_size
        self.lr = lr
        self.lr_scheduler_gamma = lr_scheduler_gamma
        self.num_workers = num_workers

        self.__build_model()

    def __build_model(self):
        """Define model layers & loss."""

        # 1. Load pre-trained network: choose the model for the pretrained network
        model_func = getattr(models, self.backbone)
        # backbone = model_func(pretrained=True)
        self.feature_extractor = model_func(pretrained=True)

        # _layers = list(self.feature_extractor.children())[:-1]
        # print(_layers)
        # self.feature_extractor = torch.nn.Sequential(*_layers)

        # freeze(module=self.feature_extractor, train_bn=self.train_bn)

        # 2. Classifier:
        # If.eval() is used, then the layers are frozen.
        self.feature_extractor.eval()

        # 3. Loss:
        # self.loss_func = F.cross_entropy

        # PyTorch uses NCHW
        # classes are two: success or failure
        num_target_classes = 2
        n_sizes = self._get_conv_output(self.dim)
        # self.feature_extractor.classifier[6] = nn.Linear(in_features=self.feature_extractor.classifier[6].in_features, out_features=2)
        self.fc = nn.Linear(n_sizes, num_target_classes)

    # mandatory
    def forward(self, t):

        """Forward pass. Returns logits."""

        # 1. Feature extraction:
        t = self.feature_extractor(t)
        t = t.squeeze(-1).squeeze(-1)

        # 2. Classifier (returns logits):
        t = self.fc(t)

        return t

    def set_parameter_requires_grad(model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

        # returns the size of the output tensor going into the Linear layer from the conv block.
    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))

        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    # returns the feature tensor from the conv block
    def _forward_features(self, x):
        x = self.feature_extractor(x)
        return x

    # loss function
    def loss(self, logits, labels):
        return self.loss_func(input=logits, target=labels)

    # trainning loop
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        # 2. Compute loss & accuracy:
        train_loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        # num_correct = torch.eq(y_bin.view(-1), y_true.view(-1)).sum()
        self.log('train_loss', train_loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True)

        # 3. Outputs:
        # tqdm_dict = {'train_loss': train_loss}
        # output = OrderedDict({'loss': train_loss,
        #                       # 'num_correct': num_correct,
        #                       'log': tqdm_dict,
        #                       'progress_bar': tqdm_dict})
        # return output
        return train_loss

    # If you need to do something with all the outputs of each training_step

    # define optimizers
    def configure_optimizers(self):
        # return torch.optim.Adam(self.parameters(), lr=0.02)
        return torch.optim.SGD(self.feature_extractor.parameters(), lr=0.001, momentum=0.9)

    # validation loop
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        val_loss = F.nll_loss(logits, y)

        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return val_loss


    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        test_loss = F.nll_loss(logits, y)
        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.log('test_loss', test_loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        return test_loss
