# PyTorch Vision Models

## Description 
This repository contains code for various copmuter vision models in PyTorch.

## Folder Structure
```
vision_model/      
|- __init__.py       
|- train.py         # script to train model 
|- detect.py        # script to run model inference
|- configs          # Configuration files for model training
|- model            # Directory of model architecture
|- utils            # Directory containing util/data functions 
```

## Model Training
```
python train.py --config train_config_path
```

## Model Inference
```
python detect.py --img img_path
```