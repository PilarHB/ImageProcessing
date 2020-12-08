# import libraries

import os
import torch
import numpy as np
import re
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning import seed_everything, metrics
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.metrics.functional import precision_recall_curve, auc
from sklearn.preprocessing import label_binarize

from CNN import CNN
from MyImageModule import MyImageModule

torch.set_printoptions(linewidth=120)


# --- FUNCTIONS ----
def evaluate(model, loader):
    y_true = []
    y_pred = []
    for imgs, labels in loader:
        logits = model(imgs)

        y_true.extend(labels)
        y_pred.extend(logits.detach().numpy())

    return np.array(y_true), np.array(y_pred)


def plot_precision_recall_curve(recall, precision):
    fig, ax = plt.subplots()
    ax.step(recall, precision, color='r', alpha=0.99, where='post')
    ax.fill_between(recall, precision, alpha=0.2, color='b', step='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    # plt.legend(loc="lower right")
    plt.title('Precision Recall Curve')
    fig.savefig("./stat_images/precision_recall_curve.png", format='png')


def plot_roc_curve(y_true, y_pred):
    # Compute ROC curve and ROC area for each class
    y_test = y_true
    y_score = y_pred

    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score, pos_label=2)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    fig.savefig("./stat_images/ROC_curve.png", format='png')


def load_best_model(MODEL_CKPT_PATH):
    # Load best model  ################################################
    model_ckpts = os.listdir(MODEL_CKPT_PATH)
    losses = []
    for model_ckpt in model_ckpts:
        # print(model_ckpt)
        loss = re.findall("\d+\.\d+", model_ckpt)
        # print(loss)
        if not loss:
            losses = losses
        else:
            losses.append(float(loss[0]))

    losses = np.array(losses)
    best_model_index = np.argsort(losses)[0]
    best_model = model_ckpts[best_model_index]
    print(best_model)
    return best_model


def config_parameter():
    # criterion = nn.CrossEntropyLoss()
    batch_size = 8
    # img_size = 224
    # Number of epochs to train for
    num_epochs = 15
    # Flag for feature extracting. When False, we finetune the whole model,when True we only update the reshaped layer params
    # feature_extract = True
    checkpoint_callback = ModelCheckpoint(filepath=MODEL_CKPT,
                                          monitor='val_loss',
                                          save_top_k=3,
                                          mode='min',
                                          save_weights_only=True)
    # EarlyStopping  ################################################
    # Monitor a validation metric and stop training when it stops improving.
    early_stop_callback = EarlyStopping(monitor='val_loss',
                                        min_delta=0.0,
                                        patience=2,
                                        verbose=False,
                                        mode='min')
    return batch_size, num_epochs, checkpoint_callback, early_stop_callback


# --- MAIN ----
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

    # Config  ################################################
    # criterion = nn.CrossEntropyLoss()
    batch_size = 8
    # img_size = 224
    # Number of epochs to train for
    num_epochs = 15
    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    feature_extract = True

    # Callbacks  ################################################
    # Save the model after every epoch by monitoring a quantity.
    MODEL_CKPT_PATH = 'model/'
    MODEL_CKPT = 'model/model-{epoch:02d}-{val_loss:.2f}'
    # Other options: save_top_k=3
    checkpoint_callback = ModelCheckpoint(filepath=MODEL_CKPT,
                                          monitor='val_loss',
                                          save_top_k=3,
                                          mode='min',
                                          save_weights_only=True)
    # EarlyStopping  ################################################
    # Monitor a validation metric and stop training when it stops improving.
    early_stop_callback = EarlyStopping(monitor='val_loss',
                                        min_delta=0.0,
                                        patience=2,
                                        verbose=False,
                                        mode='min')

    # Load images  ################################################
    image_module = MyImageModule(batch_size=batch_size)
    image_module.setup()
    # Samples required by the custom ImagePredictionLogger callback to log image predictions.
    # val_samples = next(iter(image_module.val_dataloader()))
    # val_imgs, val_labels = val_samples[0], val_samples[1]
    # print(val_imgs.shape, val_labels.shape)

    # Set a seed  ################################################
    seed_everything(42)

    # Load model  ################################################
    model = CNN()

    # Trainer  ################################################
    trainer = pl.Trainer(max_epochs=num_epochs,
                         default_root_dir='./checkpoints',
                         gpus=1,
                         deterministic=True,
                         callbacks=[early_stop_callback],
                         checkpoint_callback=checkpoint_callback)

    trainer.fit(model, image_module)

    # Test  ################################################
    trainer.test()
    y_true, y_pred = evaluate(model, image_module.test_dataloader())

    # Load best model  ################################################
    # best_model = load_best_model(MODEL_CKPT_PATH)
    # print("Best model:", best_model)

    # Evaluate model  ################################################
    # inference_model = CNN.load_from_checkpoint(MODEL_CKPT_PATH + best_model)
    # y_true, y_pred = evaluate(inference_model, image_module.test_dataloader())

    # Generate binary correctness labels across classes
    binary_ground_truth = label_binarize(y_true,
                                         classes=np.arange(0, 1).tolist())
    # print("binary_ground_truth", binary_ground_truth)

    # precision_micro, recall_micro, _ = precision_recall_curve(binary_ground_truth.ravel(), y_pred.ravel())
    precision, recall, thresholds = precision_recall_curve(torch.tensor(y_pred), torch.tensor(y_true))

    # print("Precision:", precision)
    # print("Recall:", recall)

    # Plot metrics - Precision-Recall Curve
    plot_precision_recall_curve(recall, precision)
    # Plot metrics - ROC Curve
    # plot_roc_curve(y_true, y_pred)
