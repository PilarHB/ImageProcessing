# Image Processing
Image Preprocessing and Image Processing for Image Recognition.

ImageProcessing is a repository for Image Recognition. The objective of the code is to train a model capable 
of classifying the input images as 'success' or 'fail'. The output of the model will be a tensor of features and the
result of classification. 

## Definition of the problem

## Elements in the code
This codes uses Pytorch Lightning, Python 3.7 and Tensorboard. 

- **CNN.py**: Convolutional Neural Network using a Transfer Learning pretrained model. (LightningModule)
- **ImageModel.py**: Python class used for train, test and load the models. 
- **MyImageModule.py**: Class for loading our image dataset (LightningDataModule)
- **ModelMetrics.py**: Python class to load and evaluate the trained models. The metrics are saved in a directory called 'models_metrics'. 


## Model

## Metrics

In ModelMetrics.py calling `get_test_metrics(display=True)` the following metrics and the plots are available
1. Confusion Matrix
2. ROC Curve
3. Precision-Recall Curve
4. Stats score: TP, FP, TN and FN
5. F1, F2 and F0.5 Score


## Setup

- To train the model
    1. Select the type of Transfer Learning Model
    2. Select the size of dataset, if `None`, dataset_size is the whole dataset
    `image_model = ImageModel(model_name='resnet50', dataset_size=2692)`
    
- To test the model
    1. Initialize ImageModel() `image_model = ImageModel(model_name='resnet50', dataset_size=2692)`
    2. Initialize ImageModule() 
    - `image_module = MyImageModule(dataset_size=2692, batch_size=8)`
    - `image_module.setup()`
    3. Load model to test: `load_model()` or `load_best_model()`
    - `inference_model = image_model.load_model(name_model)`
    4. Predict results `test_preds, test_targets = get_all_preds(inference_model, image_module.test_dataloader())`
    5. Get the metrics 
    - `metrics = ModelMetrics(test_preds, test_targets, name_model, parent_model='resnet50')`
    - `metrics.get_test_metrics(display=True)`
    


- Tensorboard code can be access at http://localhost:6006/ introducing in command line : `tensorboard --logdir=tb_logs`





 


 