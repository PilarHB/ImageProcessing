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


from sklearn.metrics import confusion_matrix
#from plotcm import plot_confusion_matrix
import pdb

torch.set_printoptions(linewidth=120)

class CNN(pl.LightningModule):

    # defines the network
    def __init__(self):
        super(CNN, self).__init__()
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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("Cuda:", torch.cuda.is_available())
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(dev)
    # print("Cuda:", torch.cuda.get_device_name(0))
    if dev.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')
    print(len(os.listdir('./images/training/')))
    print(len(os.listdir('./images/training/fail')))
    print(len(os.listdir('./images/training/success')))

    # Config  ################################################
    # criterion = nn.CrossEntropyLoss()
    batch_size = 8
    # img_size = 224
    # epoch = 2

    # Callbacks  ################################################
    # Save Model
    # checkpoint_callback = ModelCheckpoint(filepath='./checkpoints', monitor='val_loss',
    #                                        save_best_only=True, mode='min', save_weights_only=True)
    # EarlyStopping
    # earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.0, patience=2)

    # Load images  ################################################
    image_module = MyImageModule(batch_size=batch_size)

    # Set a seed  ################################################
    seed_everything(42)

    # Load model  ################################################
    model = CNN()

    # Trainer  ################################################
    trainer = pl.Trainer(default_root_dir='./checkpoints',gpus=1,deterministic=True)
    # trainer = pl.Trainer(default_root_dir='./checkpoints')
    trainer.fit(model, image_module)

    #Predict
    # model = CNN.load_from_checkpoint(PATH)
    # model.freeze()

    # x = some_images_from_cifar10()
    # predictions = model(x)
