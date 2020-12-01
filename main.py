# import libraries

import os
import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from CNN import CNN
from MyImageModule import MyImageModule

torch.set_printoptions(linewidth=120)


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
