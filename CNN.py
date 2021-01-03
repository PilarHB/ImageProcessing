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
from pytorch_lightning.metrics.functional import accuracy, auroc, precision, recall, confusion_matrix, f1, fbeta
import pytorch_lightning.metrics


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

        # build the model
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

    def set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

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

        # 2. Compute loss & metrics:
        return self._calculate_step_metrics(logits, y)


    def training_epoch_end(self, outputs):
        """Compute and log training loss and accuracy at the epoch level."""
        self._calculate_epoch_metrics(outputs, name='Train')

    # validation loop
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        # 2. Compute loss & metrics:

        outputs = self._calculate_step_metrics(logits, y)
        self.log("val_loss", outputs["loss"])
        return outputs


    def validation_epoch_end(self, outputs):
        """Compute and log validation loss and accuracy at the epoch level."""
        self._calculate_epoch_metrics(outputs, name='Val')

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        # 2. Compute loss & metrics:
        return self._calculate_step_metrics(logits, y)

    def test_epoch_end(self, outputs):

        self._calculate_epoch_metrics(outputs, name='Test')

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

    # TODO: Refactor internal metrics
    def _calculate_step_metrics(self, logits, y):
        # prepare the metrics
        loss = F.cross_entropy(logits[1], y)
        # train_loss = self.loss(logits[1], y)
        preds = torch.argmax(logits[1], dim=1)
        num_correct = torch.eq(preds.view(-1), y.view(-1)).sum()
        acc = accuracy(preds, y)
        f1_score = f1(preds, y, num_classes=2, average='weighted')
        fb05_score = fbeta(preds, y, num_classes=2, average='weighted', beta=0.5)
        fb2_score = fbeta(preds, y, num_classes=2, average='weighted', beta=2)
        cm = confusion_matrix(preds, y, num_classes=2)
        prec = precision(preds, y, num_classes=2, class_reduction='weighted')
        rec = recall(preds, y, num_classes=2, class_reduction='weighted')
        # au_roc = auroc(preds, y, pos_label=1)

        return {'loss': loss,
                'acc': acc,
                'f1_score': f1_score,
                'f05_score': fb05_score,
                'f2_score': fb2_score,
                'precision': prec,
                'recall': rec,
                # 'auroc': au_roc,
                'confusion_matrix': cm,
                'num_correct': num_correct}

    def _calculate_epoch_metrics(self, outputs, name):

        # Logging activations
        loss_mean = torch.stack([output['loss']
                                 for output in outputs]).mean()
        acc_mean = torch.stack([output['num_correct']
                                for output in outputs]).sum().float()
        acc_mean /= (len(outputs) * self.batch_size)

        f1_score = torch.stack([output['f1_score']
                                for output in outputs]).mean()

        f05_score = torch.stack([output['f05_score']
                                 for output in outputs]).mean()
        f2_score = torch.stack([output['f2_score']
                                for output in outputs]).mean()
        precision = torch.stack([output['precision']
                                 for output in outputs]).mean()
        recall = torch.stack([output['recall']
                              for output in outputs]).mean()
        # Logging scalars
        self.logger.experiment.add_scalar(f'Loss/{name}',
                                          loss_mean,
                                          self.current_epoch)

        self.logger.experiment.add_scalar(f'Accuracy/{name}',
                                          acc_mean,
                                          self.current_epoch)

        self.logger.experiment.add_scalar(f'F1_Score/{name}',
                                          f1_score,
                                          self.current_epoch)
        self.logger.experiment.add_scalar(f'F05_Score/{name}',
                                          f05_score,
                                          self.current_epoch)
        self.logger.experiment.add_scalar(f'F2_Score/{name}',
                                          f2_score,
                                          self.current_epoch)
        self.logger.experiment.add_scalar(f'Precision/{name}',
                                          precision,
                                          self.current_epoch)
        self.logger.experiment.add_scalar(f'Recall/{name}',
                                          recall,
                                          self.current_epoch)
