# import libraries

import os
import torch
import numpy as np
import re
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torchvision
import torchvision.transforms.functional as F
from pytorch_lightning import seed_everything, metrics
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.metrics.functional import precision_recall_curve, auc
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize
from PIL import Image
from torchvision import transforms
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.tensorboard import SummaryWriter


from CNN import CNN
from MyImageModule import MyImageModule

torch.set_printoptions(linewidth=120)


class ImageModel():
    def __init__(self,
                 batch_size=8,
                 num_epochs=15,
                 img_size=256,
                 feature_extract=True):
        super(ImageModel, self).__init__()
        # Parameters
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.img_size = img_size
        # Flag for feature extracting. When False, we finetune the whole model,when True we only update the reshaped
        # layer params
        self.feature_extract = feature_extract
        # criterion = nn.CrossEntropyLoss()
        # Save the model after every epoch by monitoring a quantity.
        current_path = os.path.dirname(os.path.realpath(__file__))
        self.MODEL_CKPT_PATH = os.path.join(current_path, 'model/')
        self.MODEL_CKPT = os.path.join(self.MODEL_CKPT_PATH, 'model-{epoch:02d}-{val_loss:.2f}')
        # Set a seed  ################################################
        seed_everything(42)
        # Load model  ################################################
        self.model = CNN()
        self.image_module = MyImageModule(batch_size=self.batch_size)
        # self.image_module = MyImageModule(batch_size=self.batch_size)
        self.activation = {}
        self.writer = SummaryWriter('tb_logs')

    def config_callbacks(self):
        # Checkpoint  ################################################
        # Saves the models so it is possible to access afterwards
        checkpoint_callback = ModelCheckpoint(dirpath= self.MODEL_CKPT_PATH,
                                              filename=self.MODEL_CKPT,
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
        return checkpoint_callback, early_stop_callback

    def call_trainer(self):
        # Load images  ################################################
        self.image_module.setup()

        # Samples required by the custom ImagePredictionLogger callback to log image predictions.
        val_samples = next(iter(self.image_module.val_dataloader()))
        # val_imgs, val_labels = val_samples[0], val_samples[1]
        # print(val_imgs.shape)
        # print(val_labels.shape)
        grid = torchvision.utils.make_grid(val_samples[0], nrow=8, padding=2)
        # write to tensorboard
        self.writer.add_image('prueba', grid)
        self.writer.close()

        # Load callbacks ########################################
        checkpoint_callback, early_stop_callback = self.config_callbacks()

        # Logger ################################################
        # Tensorboard Logger used
        logger = TensorBoardLogger('tb_logs', name='my_model')

        # Trainer  ################################################
        trainer = pl.Trainer(max_epochs=self.num_epochs,
                             gpus=1,
                             logger=logger,
                             deterministic=True,
                             callbacks=[early_stop_callback, checkpoint_callback])

        trainer.fit(model=self.model, datamodule=self.image_module)
        # Test  ################################################
        trainer.test()

    def evaluate(self, model, loader):
        y_true = []
        y_pred = []
        for imgs, labels in loader:
            logits = model(imgs)
            y_true.extend(labels)
            y_pred.extend(logits.detach().numpy())
        return np.array(y_true), np.array(y_pred)

    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()
        return hook

    # Returns the size of features tensor
    def get_size_features(self, model):

        feature_size = model.get_size()
        return feature_size

    @torch.no_grad()
    def evaluate_image(self, image, model):
        image_tensor = self.image_preprocessing(image)
        model.feature_extractor.classifier[6].register_forward_hook(self.get_activation('classifier[6]'))
        output = model(image_tensor)
        # print("Features", self.activation['classifier[6]'])
        # print("Output", output[0])
        features = output[0]
        # features_size = output[0].shape
        return features.detach().numpy()

    def image_preprocessing(self, image):
        transform = transforms.Compose([
            # you can add other transformations in this list
            # transforms.Grayscale(num_output_channels=1),
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform(image).unsqueeze(0)
        print(image_tensor.shape)
        return image_tensor

    @torch.no_grad()
    def inference_model(self):
        best_model = self.load_best_model()
        inference_model = self.model.load_from_checkpoint(self.MODEL_CKPT_PATH + best_model)
        return inference_model

    # TODO: Revisar este método, creo que no es necesario
    def evaluate_model(self):
        inference_model = self.inference_model()
        print("Inference model:", inference_model)
        print("Test Dataloader:", self.image_module.test_dataloader())
        y_true, y_pred = image_model.evaluate(inference_model, self.image_module.test_dataloader())
        return y_true, y_pred

    def load_best_model(self):
        # Load best model  ################################################
        model_ckpts = os.listdir(self.MODEL_CKPT_PATH)
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
        return best_model

    def load_model(self, name):
        # model_ckpts = os.listdir(self.MODEL_CKPT_PATH)
        model = self.model.load_from_checkpoint(self.MODEL_CKPT_PATH + name)
        print(model)
        return model

    # TODO: método para mostrar todas las métricas ()



# --- MAIN ----
if __name__ == '__main__':
    print("Cuda:", torch.cuda.is_available())
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(dev)
    # print("Cuda:", torch.cuda.get_device_name(0))
    if dev.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        # print('Memory Usage:')
        # print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        # print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')

    # Config  ################################################
    image_model = ImageModel()
    checkpoint_callback, early_stop_callback = image_model.config_callbacks()

    # Train model  ################################################
    # image_model.call_trainner()
    # y_true, y_pred = evaluate(model, image_module.test_dataloader())

    # Load best model  ################################################
    best_model = image_model.load_best_model()
    print("Best model:", best_model)
    inference_model = image_model.inference_model()
    print(inference_model)

    #  Evaluate output  ################################################
    # image = Image.open("./images/fail/img1605601451.8657722.png")
    # image_tensor = image_model.image_preprocessing(image)
    # image_model.evaluate_image(image, inference_model)

    # feature_size = image_model.get_size_features(inference_model)
    # print(feature_size)

    # Evaluate model  ################################################
    # inference_model = image_model.inference_model()
    # y_true, y_pred = image_model.evaluate_model()

    # Generate binary correctness labels across classes
    # binary_ground_truth = label_binarize(y_true,
    #                                     classes=np.arange(0, 1).tolist())
    # print("binary_ground_truth", binary_ground_truth)

    # precision_micro, recall_micro, _ = precision_recall_curve(binary_ground_truth.ravel(), y_pred.ravel())
    # precision, recall, _ = precision_recall_curve(torch.tensor(y_pred), torch.tensor(y_true))

    # Plot metrics - Precision-Recall Curve
    # image_model.plot_precision_recall_curve(recall, precision)
    # Plot metrics - ROC Curve
    # plot_roc_curve(y_true, y_pred)
