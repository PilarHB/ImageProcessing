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


from sklearn.metrics import confusion_matrix
#from plotcm import plot_confusion_matrix
import pdb

#  --- Utility functions ---



class CNN(pl.LightningModule):

    # defines the network
    def __init__(self):
        super(CNN, self).__init__()
        # classes are two: success or failure
        # PyTorch uses NCHW
        # classes are two: success or failure
        num_target_classes = 2
        # choose the model for the pretrained network
        # self.feature_extractor = models.resnet18(pretrained=True)
        # num_ftrs = self.feature_extractor.fc.in_features
        # model_ft.fc = nn.Linear(num_ftrs, 2)
        self.feature_extractor = models.vgg16(pretrained=True)
        self.feature_extractor.eval()

        # use the pretrained model to classify success-fail (2 image classes)
        # print(self.feature_extractor.classifier[6].out_features)
        # self.feature_extractor.classifier[6] = nn.Linear(in_features=self.feature_extractor.classifier[6].in_features, out_features=2)
        self.classifier = nn.Linear(self.feature_extractor.classifier[6].out_features, num_target_classes)
        # self.classifier = nn.Linear(num_ftrs, num_target_classes)

        # # we are also defining some variable for counting purposes
        # self.valTotal = 0
        # self.valCorrect = 0
        # self.trainTotal = 0
        # self.trainCorrect = 0

    # mandatory
    def forward(self, t):

        representations = self.feature_extractor(t)
        t = self.classifier(representations)
        return t


    # trainning loop
    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        # x = x.view(x.size(0),-1)
        preds = self(imgs)
        # Calculate Loss
        loss = F.cross_entropy(preds, labels)
        # Calculate Correct
        _, preds = torch.max(preds, 1)
        correct = torch.sum(preds == labels).float() / preds.size(0)

        logs = {'train_loss': loss, 'train_correct': correct}
        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        # self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'log': logs, 'progress_bar': logs}

    # If you need to do something with all the outputs of each training_step
    # def training_epoch_end(self, training_step_outputs):
    # for pred in training_step_outputs:

    # define optimizers
    def configure_optimizers(self):
        # return torch.optim.Adam(self.parameters(), lr=0.02)
        return torch.optim.SGD(self.feature_extractor.parameters(), lr=0.001, momentum=0.9)

    # validation loop
    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        # x = x.view(x.size(0),-1)
        preds = self(imgs)
        # Calculate Loss
        # loss = F.nll_loss(preds, labels)
        loss = F.cross_entropy(preds, labels)
        # Calculate Correct
        _, preds = torch.max(preds, 1)
        correct = torch.sum(preds == labels).float() / preds.size(0)

        logs = {'train_loss': loss, 'train_correct': correct}
        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'log': logs, 'progress_bar': logs}

    # Aggegate Validation Result
    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_correct = torch.stack([x['val_correct'] for x in outputs]).mean()
        logs = {'avg_val_loss': avg_loss, 'avg_val_correct': avg_correct}
        torch.cuda.empty_cache()
        return {'avg_val_loss': avg_loss, 'log': logs}