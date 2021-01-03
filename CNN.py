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
from pytorch_lightning.metrics.functional import accuracy, auroc
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

        # prepare the metrics
        self.accuracy = pl.metrics.Accuracy()
        self.f1 = pl.metrics.F1(average='weighted')
        self.fb05 = pl.metrics.FBeta(num_classes=2, average='weighted', beta=0.5)
        self.fb2 = pl.metrics.FBeta(num_classes=2, average='weighted', beta=2)

        self.cm = pl.metrics.ConfusionMatrix(num_classes=2)
        self.prec = pl.metrics.Precision(num_classes=2, average='micro')
        self.recall = pl.metrics.Recall(num_classes=2, average='micro')
        # self.auroc = pl.metrics.ROC()

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

        # 2. Compute loss & accuracy:
        train_loss = F.cross_entropy(logits[1], y)
        # train_loss = self.loss(logits[1], y)
        # print(train_loss)
        preds = torch.argmax(logits[1], dim=1)
        num_correct = torch.eq(preds.view(-1), y.view(-1)).sum()
        acc = self.accuracy(preds, y)
        f1_score = self.f1(preds, y)
        f05_score = self.fb05(preds, y)
        f2_score = self.fb2(preds, y)
        precision = self.prec(preds, y)
        recall = self.recall(preds, y)

        # Logging
        # self.log('train_loss', train_loss, on_step=True, on_epoch=True, logger=True)
        # self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True)
        # self.log('train_num_correct', num_correct, on_step=True, on_epoch=True, logger=True)
        # self.log('train_f1_score', f1_score, on_step=True, on_epoch=True, logger=True)

        return {'loss': train_loss,
                'acc': acc,
                'f1_score': f1_score,
                'f05_score': f05_score,
                'f2_score': f2_score,
                'precision': precision,
                'recall': recall,
                'num_correct': num_correct}


    def training_epoch_end(self, outputs):
        """Compute and log training loss and accuracy at the epoch level."""

        # Logging activations
        train_loss_mean = torch.stack([output['loss']
                                       for output in outputs]).mean()
        train_acc_mean = torch.stack([output['num_correct']
                                      for output in outputs]).sum().float()
        train_acc_mean /= (len(outputs) * self.batch_size)

        train_f1_score = torch.stack([output['f1_score']
                                     for output in outputs]).mean()

        train_f05_score = torch.stack([output['f05_score']
                                      for output in outputs]).mean()
        train_f2_score = torch.stack([output['f2_score']
                                     for output in outputs]).mean()
        train_precision = torch.stack([output['precision']
                                      for output in outputs]).mean()
        train_recall = torch.stack([output['recall']
                                      for output in outputs]).mean()



        # Logging scalars
        self.logger.experiment.add_scalar('Loss/Train',
                                          train_loss_mean,
                                          self.current_epoch)

        self.logger.experiment.add_scalar('Accuracy/Train',
                                          train_acc_mean,
                                          self.current_epoch)

        self.logger.experiment.add_scalar('F1_Score/Train',
                                          train_f1_score,
                                          self.current_epoch)
        self.logger.experiment.add_scalar('F05_Score/Train',
                                          train_f05_score,
                                          self.current_epoch)
        self.logger.experiment.add_scalar('F2_Score/Train',
                                          train_f2_score,
                                          self.current_epoch)
        self.logger.experiment.add_scalar('Precision/Train',
                                          train_precision,
                                          self.current_epoch)
        self.logger.experiment.add_scalar('Recall/Train',
                                          train_recall,
                                          self.current_epoch)
        # Logging histograms
        self.custom_histogram_adder()


    # If you need to do something with all the outputs of each training_step

    # validation loop
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        # 2. Compute loss & accuracy:
        # val_loss = self.loss(logits, y)
        val_loss = F.cross_entropy(logits[1], y)
        preds = torch.argmax(logits[1], dim=1)
        num_correct = torch.eq(preds.view(-1), y.view(-1)).sum()
        acc = accuracy(preds, y)
        f1_score = self.f1(preds, y)
        f05_score = self.fb05(preds, y)
        f2_score = self.fb2(preds, y)
        precision = self.prec(preds, y)
        recall = self.recall(preds, y)

        self.log('val_loss', val_loss)
        # .log('val_acc', acc, on_step=True, on_epoch=True, logger=True)
        # self.log('val_num_correct', num_correct, on_step=True, on_epoch=True, logger=True)

        # logs = {'val_loss': val_loss}
        return {'loss': val_loss,
                'acc': acc,
                'f1_score': f1_score,
                'f05_score': f05_score,
                'f2_score': f2_score,
                'precision': precision,
                'recall': recall,
                'num_correct': num_correct}
                # 'log': logs}

    def validation_epoch_end(self, outputs):
        """Compute and log validation loss and accuracy at the epoch level."""

        val_loss_mean = torch.stack([output['loss']
                                     for output in outputs]).mean()
        val_acc_mean = torch.stack([output['num_correct']
                                    for output in outputs]).sum().float()
        val_acc_mean /= (len(outputs) * self.batch_size)

        val_f1_score = torch.stack([output['f1_score']
                                    for output in outputs]).mean()
        val_f05_score = torch.stack([output['f05_score']
                                      for output in outputs]).mean()
        val_f2_score = torch.stack([output['f2_score']
                                     for output in outputs]).mean()
        val_precision = torch.stack([output['precision']
                                      for output in outputs]).mean()
        val_recall = torch.stack([output['recall']
                                      for output in outputs]).mean()

        # Adding logs to TensorBoard
        self.logger.experiment.add_scalar("Loss/Val",
                                          val_loss_mean,
                                          self.current_epoch)
        self.logger.experiment.add_scalar("Accuracy/Val",
                                          val_acc_mean,
                                          self.current_epoch)
        self.logger.experiment.add_scalar('F1_Score/Val',
                                          val_f1_score,
                                          self.current_epoch)
        self.logger.experiment.add_scalar('F05_Score/Val',
                                          val_f05_score,
                                          self.current_epoch)
        self.logger.experiment.add_scalar('F2_Score/Val',
                                          val_f2_score,
                                          self.current_epoch)
        self.logger.experiment.add_scalar('Precision/Val',
                                          val_precision,
                                          self.current_epoch)
        self.logger.experiment.add_scalar('Recall/Val',
                                          val_recall,
                                          self.current_epoch)

        # tensorboard_logs = {'val_loss': val_loss_mean,
        #                     "val_accuracy": val_acc_mean,
        #                     "val_f1_score": val_f1_score}
        # return {'val_loss': val_loss_mean, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        # 2. Compute loss & accuracy:
        # test_loss = self.loss(logits, y)
        test_loss = F.cross_entropy(logits[1], y)
        preds = torch.argmax(logits[1], dim=1)
        num_correct = torch.eq(preds.view(-1), y.view(-1)).sum()
        f1_score = self.f1(preds, y)
        acc = accuracy(preds, y)
        f05_score = self.fb05(preds, y)
        f2_score = self.fb2(preds, y)
        precision = self.prec(preds, y)
        recall = self.recall(preds, y)

        # self.log('test_loss', test_loss, on_step=True, on_epoch=True, logger=True)
        # self.log('test_acc', acc, on_step=True, on_epoch=True, logger=True)
        # self.log('num_correct', num_correct, on_step=True, on_epoch=True, logger=True)

        logs = {'test_loss': test_loss}
        return {'loss': test_loss,
                'num_correct': num_correct,
                'f1_score': f1_score,
                'f05_score': f05_score,
                'f2_score': f2_score,
                'precision': precision,
                'recall': recall,
                'log': logs,
                'progress_bar': logs}

    def test_epoch_end(self, outputs):
        # OPTIONAL
        # The code that runs as a validation epoch finished
        # Used for metric evaluation
        test_loss_mean = torch.stack([output['loss']
                                      for output in outputs]).mean()
        test_acc_mean = torch.stack([output['num_correct']
                                     for output in outputs]).sum().float()
        test_acc_mean /= (len(outputs) * self.batch_size)
        test_f1_score = torch.stack([output['f1_score']
                                     for output in outputs]).mean()
        test_f05_score = torch.stack([output['f05_score']
                                      for output in outputs]).mean()
        test_f2_score = torch.stack([output['f2_score']
                                     for output in outputs]).mean()
        test_precision = torch.stack([output['precision']
                                      for output in outputs]).mean()
        test_recall = torch.stack([output['recall']
                                      for output in outputs]).mean()

        # Logging Data to TensorBoard
        self.logger.experiment.add_scalar("Loss/Test",
                                          test_loss_mean,
                                          self.current_epoch)
        self.logger.experiment.add_scalar("Accuracy/Test",
                                          test_acc_mean,
                                          self.current_epoch)
        self.logger.experiment.add_scalar('F1_Score/Test',
                                          test_f1_score,
                                          self.current_epoch)
        self.logger.experiment.add_scalar('F05_Score/Test',
                                          test_f05_score,
                                          self.current_epoch)
        self.logger.experiment.add_scalar('F2_Score/Test',
                                          test_f2_score,
                                          self.current_epoch)
        self.logger.experiment.add_scalar('Precision/Test',
                                          test_precision,
                                          self.current_epoch)
        self.logger.experiment.add_scalar('Recall/Test',
                                          test_recall,
                                          self.current_epoch)

        # tensorboard_logs = {'test_loss': test_loss_mean,
        #                     "test_accuracy": test_acc_mean,
        #                     "test_f1_score": test_f1_score}
        # return {'test_loss': test_loss_mean, 'log': tensorboard_logs}

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
    # def _calculate_step_metrics(self):
    #
    #     return
    #
    # def _calculate_epoch_metrics(self):
    #
    #     return
