import torch
import numpy as np

def check_config_key(cfg, key):
    """
    Return True if valid value exists for config[key]
    """
    if key in cfg:
        if cfg[key] is not None:
            return True
    return False

def np_to_tensor(image, is_mask=False):
    """
    Process numpy array to be input to model
    """
    if is_mask:
        img = torch.as_tensor(image).long()
    else:
        if len(image.shape)==2:
            image = np.expand_dims(image, axis=-1)
        img = image / np.max(image)
        img = np.transpose(image, (2,0,1))
        img = torch.as_tensor(img).float()
    return img