# import libraries

import os
import torch
import numpy as np
import re
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import label_binarize

from CNN import CNN
from MyImageModule import MyImageModule

torch.set_printoptions(linewidth=120)

# --- FUNCTIONS ----
def evaluate(model, loader):
    y_true = []
    y_pred = []
    for imgs, labels in loader:
        logits = inference_model(imgs)

        y_true.extend(labels)
        y_pred.extend(logits.detach().numpy())

    return np.array(y_true), np.array(y_pred)


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


    """Train the model.
    Args:
        args: Model hyper-parameters
    Note:
        For the sake of the example, the images dataset will be downloaded
        to a temporary directory.
    """

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
    val_samples = next(iter(image_module.val_dataloader()))
    val_imgs, val_labels = val_samples[0], val_samples[1]
    print(val_imgs.shape, val_labels.shape)

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
                         checkpoint_callback =checkpoint_callback)

    trainer.fit(model, image_module)

    # Test  ################################################
    trainer.test()

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

    inference_model = CNN.load_from_checkpoint(MODEL_CKPT_PATH + best_model)
    y_true, y_pred = evaluate(inference_model, image_module.test_dataloader())
    print("Y true:", y_true)
    print("Y pred:", y_pred)

    # generate binary correctness labels across classes
    binary_ground_truth = label_binarize(y_true,
                                         classes=np.arange(0, 1).tolist())

    # compute a PR curve with sklearn like you normally would
    precision_micro, recall_micro, _ = precision_recall_curve(binary_ground_truth.ravel(),
                                                              y_pred.ravel())
    print("Precision micro:", precision_micro)
    print("Recall micro:", recall_micro)



