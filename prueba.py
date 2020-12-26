# import libraries
import itertools
import os
import io
from collections import Iterable

import torch
import numpy as np
import re
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torchvision
from pytorch_lightning import seed_everything, metrics
from pytorch_lightning.metrics import classification
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.metrics.functional import precision_recall_curve, auc
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from CNN import CNN
from MyImageModule import MyImageModule

torch.set_printoptions(linewidth=120)
MODEL_CKPT_PATH = 'model/'
MODEL_CKPT = 'model/model-{epoch:02d}-{val_loss:.2f}'

# --- FUNCTIONS ----
from CNN import CNN
from MyImageModule import MyImageModule
from ImageModel import ImageModel


def plot_ROC_curve(preds, targets, pos_label):
    # Compute ROC curve and ROC area for each class
    y_true = targets.detach().numpy()
    y_pred = preds.detach().numpy()

    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Curve ROC Label:', pos_label)
    plt.legend(loc="lower right")
    plt.show()

@torch.no_grad()
def evaluate(self, model, loader):
    y_true = []
    y_pred = []
    for imgs, labels in loader:
        logits = model(imgs)
        y_true.extend(labels)
        y_pred.extend(logits.detach().numpy())
    return np.array(y_true), np.array(y_pred)


# make a class prediction for one row of data
@torch.no_grad()
def predict(model, loader):
    # convert row to data
    y_pred = []
    y_true = []
    for img, labels in loader:
        logits = model(img)
        # print(logits[1])
        y_pred.extend(logits[1].detach().numpy())
        y_true.extend(labels)
    return np.array(y_true), np.array(y_pred)

@torch.no_grad()
def get_all_preds(model, loader):
    all_preds = torch.tensor([])
    all_targets = torch.tensor([])
    for batch in loader:
        images, labels = batch
        preds = model(images)
        all_preds = torch.cat((all_preds, preds[1]), dim=0)
        all_targets = torch.cat((all_targets, labels), dim=0)
    return all_preds, all_targets


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.savefig("confusion_matrix.png", format='png')

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

@torch.no_grad()
def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


def find_classes(dir):
    classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    classes.sort()
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes


def add_pr_curve_tensorboard(writer, class_index, test_probs, test_preds, classes, global_step=0):
    '''
    Takes in a "class_index" from 0 to 9 and plots the corresponding
    precision-recall curve
    '''
    tensorboard_preds = test_preds == class_index
    tensorboard_probs = test_probs[:, class_index]

    writer.add_pr_curve(classes[class_index],
                        tensorboard_preds,
                        tensorboard_probs,
                        global_step=global_step)
    writer.close()

@torch.no_grad()
def get_probabilities(model, testloader):
    class_probs = []
    class_preds = []
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            output = model(images)
            class_probs_batch = [F.softmax(el, dim=0) for el in output[1]]
            _, class_preds_batch = torch.max(output[1], 1)

            class_probs.append(class_probs_batch)
            class_preds.append(class_preds_batch)

    test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
    test_preds = torch.cat(class_preds)

    return test_probs, test_preds


def show_activations(model):
    # _layers = list(model.children())[:-1]
    _layers = list(model.children())
    # print(_layers)
    for layer in _layers:
        print("layer", layer)
        if isinstance(layer, Iterable):
            for i in layer:
                print("sublayer", i)
        # else:
        #     print(layer)


# --MAIN ------------------------------------------------------------------------------------------------------

# instantiate class to handle model
image_model = ImageModel()
# checkpoint_callback, early_stop_callback = image_model.config_callbacks()

image_module = MyImageModule(batch_size=32)
image_module.setup()

# image_model.call_trainer()

##################################################################################################################
# CHECKING METRICS AND FUNCTIONS
##################################################################################################################

# print("Len test dataset: ", len(image_module.test_data))
# test_set = image_module.test_data
# print("test set:", test_set)
# print("Len targets: ", len(image_module.test_data.targets))

inference_model = image_model.inference_model()
print("Inference model:", inference_model)
# print("Test Dataloader:", image_module.test_dataloader())
y_true, y_pred = predict(inference_model, image_module.test_dataloader())
# ("y_true", y_true)
# print("y_pred", y_true)

test_preds, test_targets = get_all_preds(inference_model, image_module.test_dataloader())
# print("Test preds", test_preds)
# print("Test_targets", test_targets)

# Plot metrics - Precision-Recall Curve
precision, recall, _ = precision_recall_curve(torch.tensor(y_pred), torch.tensor(y_true))
image_model.plot_precision_recall_curve(recall, precision)
# image_model.plot_roc_curve(y_true, y_pred)

# With tensors
preds_correct1 = get_num_correct(test_preds, test_targets)
print("--With tensors--")
print('total correct:', preds_correct1)
print('accuracy:', preds_correct1 / len(image_module.test_data))
# # Without tensors
preds_correct2 = get_num_correct(torch.Tensor(y_pred), torch.Tensor(y_true))
print("--Without tensors--")
print('total correct:', preds_correct2)
print('accuracy:', preds_correct2 / len(image_module.test_data))

# Confusion Matrix
cm = confusion_matrix(test_targets, test_preds.argmax(dim=1))
class_names = find_classes('./images/')
print(class_names)
plot_confusion_matrix(cm, class_names, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues)

# test_probs, test_preds = get_probabilities(inference_model, image_module.test_dataloader())
# print("test_preds", test_preds.shape)
# print("test_probs", test_probs.shape)
# plot all the pr curves
# for i in range(len(class_names)):
#     add_pr_curve_tensorboard(image_model.writer, i, test_probs, class_names, test_preds)

# Samples required by the custom ImagePredictionLogger callback to log image predictions.
# val_samples = next(iter(image_module.val_dataloader()))
# val_imgs, val_labels = val_samples[0], val_samples[1]
# print(val_imgs.shape)
# print(val_labels.shape)
# grid = torchvision.utils.make_grid(val_samples[0], nrow=8, padding=2)
# write to tensorboard
# image_model.writer.add_image('prueba_hola', grid)
# image_model.writer.add_graph(inference_model, val_imgs)
# image_model.writer.close()

# show_activations(inference_model)
