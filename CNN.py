import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from collections import OrderedDict, Iterable
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

from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning.metrics.functional import accuracy

from sklearn.metrics import confusion_matrix
# from plotcm import plot_confusion_matrix
import pdb


class CNN(pl.LightningModule):

    # defines the network
    def __init__(self,
                 input_shape: list = [3, 256, 256],
                 backbone: str = 'vgg16',
                 train_bn: bool = True,
                 milestones: tuple = (5, 10),
                 batch_size: int = 8,
                 learning_rate: float = 1e-2,
                 lr_scheduler_gamma: float = 1e-1,
                 num_workers: int = 6):
        super(CNN, self).__init__()
        # parameters
        self.dim = input_shape
        self.backbone = backbone
        self.train_bn = train_bn
        self.milestones = milestones
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lr_scheduler_gamma = lr_scheduler_gamma
        self.num_workers = num_workers

        # reference dummy image for logging graph
        # self.reference_image=torch.rand((1,1,28,28))

        self.__build_model()

    def __build_model(self):
        """Define model layers & loss."""

        # 1. Load pre-trained network: choose the model for the pretrained network
        model_func = getattr(models, self.backbone)
        # backbone = model_func(pretrained=True)
        self.feature_extractor = model_func(pretrained=True)

        _layers = list(self.feature_extractor.children())[:-1]
        # print(_layers)
        # self.feature_extractor = torch.nn.Sequential(*_layers)

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
        features = t.squeeze(-1).squeeze(-1)
        # 2. Classifier (returns logits):
        t = self.fc(features)
        # We want the probability to sum 1
        t = F.log_softmax(t, dim=1)
        return features, t

    def set_parameter_requires_grad(model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

        ## for name, child in res_mod.named_children():
        ##    if name in ['layer3', 'layer4']:
        ##        print(name + 'has been unfrozen.')
        ##        for param in child.parameters():
        ##            param.requires_grad = True
        ##    else:
        ##        for param in child.parameters():
        ##            param.requires_grad = False

        # also need to update optimization function
        # only optimize those that require grad

        ## optimizer_conv = torch.optim.SGD(filter(lambda x: x.requires_grad, res_mod.parameters()), lr=0.001, momentum=0.9)

    # returns the size of the output tensor going into the Linear layer from the conv block.
    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))

        output_feat = self._forward_features(input)
        print("output_feat")
        # print(output_feat)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        print("n_size")
        print(n_size)
        return n_size

    def get_size(self):
        n_sizes = self._get_conv_output(self.dim)
        return n_sizes

    # returns the feature tensor from the conv block
    def _forward_features(self, x):
        x = self.feature_extractor(x)
        return x

    # loss function
    def loss(self, logits, labels):
        return self.loss_func(input=logits, target=labels)

    # trainning loop
    def training_step(self, batch, batch_idx):
        # x = images , y = batch, logits = labels
        x, y = batch
        logits = self(x)

        # 2. Compute loss & accuracy:
        train_loss = F.cross_entropy(logits[1], y)
        # train_loss = self.loss(logits[1], y)
        # print(train_loss)
        preds = torch.argmax(logits[1], dim=1)
        num_correct = torch.eq(preds.view(-1), y.view(-1)).sum()
        acc = accuracy(preds, y)

        # Logging
        self.log('train_loss', train_loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True)
        self.log('num_correct', num_correct, on_step=True, on_epoch=True, logger=True)

        # 3. Outputs:
        # 3. Outputs:
        tqdm_dict = {'train_loss': train_loss}
        output = OrderedDict({'loss': train_loss,
                              'num_correct': num_correct,
                              'log': tqdm_dict,
                              'progress_bar': tqdm_dict})
        return output


    def training_epoch_end(self, outputs):
        """Compute and log training loss and accuracy at the epoch level."""

        # Logging activations

        train_loss_mean = torch.stack([output['loss']
                                       for output in outputs]).mean()
        train_acc_mean = torch.stack([output['num_correct']
                                      for output in outputs]).sum().float()
        train_acc_mean /= (len(outputs) * self.batch_size)

        # Logging scalars
        self.logger.experiment.add_scalar('Loss/Train',
                                          train_loss_mean,
                                          self.current_epoch)

        self.logger.experiment.add_scalar('Accuracy/Train',
                                          train_acc_mean,
                                          self.current_epoch)
        # Logging histograms
        self.custom_histogram_adder()

        tensorboard_logs = {'train_loss': train_loss_mean, "train_accuracy": train_acc_mean}
        return {'train_loss': train_loss_mean, 'log': tensorboard_logs}

    # If you need to do something with all the outputs of each training_step

    # validation loop
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        # 2. Compute loss & accuracy:
        # val_loss = self.loss(logits, y)
        val_loss = F.cross_entropy(logits[1], y)
        preds = torch.argmax(logits[1], dim=1)
        # num_correct = torch.sum(preds == y).float() / preds.size(0)
        num_correct = torch.eq(preds.view(-1), y.view(-1)).sum()
        acc = accuracy(preds, y)

        self.log('val_loss', val_loss, on_step=True, on_epoch=True, logger=True)
        self.log('val_acc', acc, on_step=True, on_epoch=True, logger=True)
        self.log('num_correct', num_correct, on_step=True, on_epoch=True, logger=True)

        return {'val_loss': val_loss,
                'val_acc': acc,
                'num_correct': num_correct}

    def validation_epoch_end(self, outputs):
        """Compute and log validation loss and accuracy at the epoch level."""

        val_loss_mean = torch.stack([output['val_loss']
                                     for output in outputs]).mean()
        val_acc_mean = torch.stack([output['num_correct']
                                    for output in outputs]).sum().float()
        val_acc_mean /= (len(outputs) * self.batch_size)

        # Adding logs to TensorBoard
        self.logger.experiment.add_scalar("Loss/Val", val_loss_mean, self.current_epoch)
        self.logger.experiment.add_scalar("Accuracy/Val", val_acc_mean, self.current_epoch)

        tensorboard_logs = {'val_loss': val_loss_mean, "val_accuracy": val_acc_mean}
        return {'val_loss': val_loss_mean, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        # 2. Compute loss & accuracy:
        # test_loss = self.loss(logits, y)
        test_loss = F.cross_entropy(logits[1], y)
        preds = torch.argmax(logits[1], dim=1)
        # num_correct = torch.sum(preds == y).float() / preds.size(0)
        num_correct = torch.eq(preds.view(-1), y.view(-1)).sum()

        acc = accuracy(preds, y)
        self.log('test_loss', test_loss, on_step=True, on_epoch=True, logger=True)
        self.log('test_acc', acc, on_step=True, on_epoch=True, logger=True)
        self.log('num_correct', num_correct, on_step=True, on_epoch=True, logger=True)

        logs = {'test_loss': test_loss}
        return {'test_loss': test_loss,
                'num_correct': num_correct,
                'log': logs,
                'progress_bar': logs}

    def test_epoch_end(self, outputs):
        # OPTIONAL
        # The code that runs as a validation epoch finished
        # Used for metric evaluation
        test_loss_mean = torch.stack([output['test_loss']
                                     for output in outputs]).mean()
        test_acc_mean = torch.stack([output['num_correct']
                                    for output in outputs]).sum().float()
        test_acc_mean /= (len(outputs) * self.batch_size)

        # Logging Data to TensorBoard
        self.logger.experiment.add_scalar("Loss/Test", test_loss_mean, self.current_epoch)
        self.logger.experiment.add_scalar("Accuracy/Test", test_acc_mean, self.current_epoch)

        tensorboard_logs = {'test_loss': test_loss_mean, "test_accuracy": test_acc_mean}
        return {'test_loss': test_loss_mean, 'log': tensorboard_logs}

    # define optimizers
    def configure_optimizers(self):

        # optimizer2 = torch.optim.Adam(self.feature_extractor.parameters(), lr=self.learning_rate)
        optimizer1 = torch.optim.SGD(self.parameters(), lr=0.002, momentum=0.9)
        # Decay LR by a factor of 0.1 every 7 epochs
        scheduler1 = lr_scheduler.StepLR(optimizer1, step_size=7, gamma=0.1)
        # return torch.optim.SGD(self.feature_extractor.parameters(), lr=self.learning_rate, momentum=0.9)
        return (
            # {'optimizer': optimizer1, 'lr_scheduler': scheduler1, 'monitor': 'metric_to_track'}
            {'optimizer': optimizer1, 'lr_scheduler': scheduler1}
            # {'optimizer': optimizer2, 'lr_scheduler': scheduler2},
        )

    def custom_histogram_adder(self):
        # A custom defined function that adds Histogram to TensorBoard
        # Iterating over all parameters and logging them
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)






