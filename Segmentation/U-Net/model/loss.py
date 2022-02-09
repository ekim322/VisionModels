import torch.nn as nn

# TODO: Implement dice loss

def UNET_loss_fn(preds, targets):
    return 

def cross_entropy_loss_fn(preds, targets):
    loss_fn = nn.CrossEntropyLoss()
    return loss_fn(preds, targets)