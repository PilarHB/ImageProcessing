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
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import pytorch_lightning as pl
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader,random_split


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
        #_layers = list(backbone.children())[:-1]
        #print(_layers)
        # self.feature_extractor = torch.nn.Sequential(*_layers)

        # freeze(module=self.feature_extractor, train_bn=self.train_bn)

        # 2. Classifier:
         #self.feature_extractor = models.vgg16(pretrained=True)
        # self.feature_extractor = torch.nn.Sequential(*_layers)
        self.feature_extractor.eval()

        # 3. Loss:
        self.loss_func = F.binary_cross_entropy_with_logits

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

        imgs, labels = batch
        # imgs = imgs.view(imgs.size(0), -1)
        preds = self(imgs)

        # 2. Compute loss & accuracy:
        # Calculate Loss
        train_loss = F.cross_entropy(preds, labels)
        # Calculate Correct
        _, preds = torch.max(preds, 1)
        num_correct = torch.sum(preds == labels).float() / preds.size(0)

        # logs = {'train_loss': train_loss, 'train_correct': num_correct}
        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        # self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # return {'loss': train_loss, 'log': logs, 'progress_bar': logs}

        # 3. Outputs:
        tqdm_dict = {'train_loss': train_loss}
        output = OrderedDict({'loss': train_loss,
                              'num_correct': num_correct,
                              'log': tqdm_dict,
                              'progress_bar': tqdm_dict})

        return output

    # If you need to do something with all the outputs of each training_step
    def training_epoch_end(self, outputs):
        """Compute and log training loss and accuracy at the epoch level."""

        train_loss_mean = torch.stack([output['loss']
                                       for output in outputs]).mean()
        train_acc_mean = torch.stack([output['num_correct']
                                      for output in outputs]).sum().float()
        train_acc_mean /= (len(outputs) * self.batch_size)
        return {'log': {'train_loss': train_loss_mean,
                        'train_acc': train_acc_mean,
                        'step': self.current_epoch}}


    # define optimizers
    def configure_optimizers(self):
        # return torch.optim.Adam(self.parameters(), lr=0.02)

        return torch.optim.SGD(self.feature_extractor.parameters(), lr=0.001, momentum=0.9)

    # validation loop
    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        # imgs = imgs.view(imgs.size(0),-1)
        preds = self(imgs)

        # Calculate Loss
        val_loss = F.cross_entropy(preds, labels)

        # Calculate Correct
        _, preds = torch.max(preds, 1)
        num_correct = torch.sum(preds == labels).float() / preds.size(0)

        # logs = {'val_loss': val_loss, 'val_correct': num_correct}
        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {'val_loss': val_loss,'num_correct': num_correct}
        # return {'loss': val_loss, 'log': logs, 'progress_bar': logs}

    def validation_epoch_end(self, outputs):
        """Compute and log validation loss and accuracy at the epoch level."""

        val_loss_mean = torch.stack([output['val_loss']
                                     for output in outputs]).mean()
        val_acc_mean = torch.stack([output['num_correct']
                                    for output in outputs]).sum().float()
        val_acc_mean /= (len(outputs) * self.batch_size)
        return {'log': {'val_loss': val_loss_mean,
                        'val_acc': val_acc_mean,
                        'step': self.current_epoch}}

    # # Aggegate Validation Result
    # def validation_end(self, outputs):
    #     avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     avg_correct = torch.stack([x['val_correct'] for x in outputs]).mean()
    #     logs = {'avg_val_loss': avg_loss, 'avg_val_correct': avg_correct}
    #     torch.cuda.empty_cache()
    #     return {'avg_val_loss': avg_loss, 'log': logs}