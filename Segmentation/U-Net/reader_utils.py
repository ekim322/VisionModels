import numpy as np
import cv2
import torch

import matplotlib.pyplot as plt

from model.unet import UNet

model_path = 'saved_weight/cityscapes3_val.pt'
class_keys = {'bg': 0, 'road': 1, 'car': 2}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_VehicleNet(model_path = model_path):
    model = UNet(n_channels=3, n_classes=len(class_keys))
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    return model

def np_to_tensor(image, is_mask=False):
    """
    Process numpy array to be input to model
    """
    if is_mask:
        img = torch.as_tensor(image).long()
    else:
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis = -1)
        img = image / np.max(image)
        img = np.transpose(image, (2, 0, 1))
        img = torch.as_tensor(img).float()
    return img.unsqueeze(0).to(device)

def tensor_to_np(image):
    img = image.cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    return img

def detect_img(image, model):
    """
    Returns segmentation
    """
    img = image.copy()

#     if img.shape[-1] == 3:
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    img = cv2.resize(img, (512, 512))
    input_tnsr = np_to_tensor(img)

    model_output = model(input_tnsr)[0]
    mask = torch.argmax(model_output, dim = 0).cpu().numpy()
    mask = cv2.resize(mask.astype('uint8'), (image.shape[1], image.shape[0]))

#     ticks = filter_class(mask, 1)
#     ticks_clean = clean_mask(ticks)

    cars = filter_class(mask, 2)

#     needle = filter_class(mask, 3)
#     needle_clean = clean_mask(needle)

    return cars

def filter_class(image, key):
    """
    Returns segmentation mask of class_key
    """
    img = image.copy()
    img[img!=key] = 0
    img[img==key] = 1
    return img

def display_eq(image, M, B):
    img = image.copy()
    for x in range(img.shape[1]):
        y = round(M*x + B)
        if img.shape[-1] != 3:
            img[y, x] = 1
        else:
            img[y, x, :] = (255, 0, 0)
    plt.imshow(img)

def clean_mask(image):
    """
    Morph operation to reduce noise
    """
    found = False
    k_size = 7
    while not found:
        kernel = np.ones((k_size, k_size), np.uint8)
        filtered = cv2.morphologyEx(image.astype('uint8'), cv2.MORPH_OPEN,
                                    kernel)
        if np.sum(filtered) < np.sum(image) / 2:
            k_size -= 2
        else:
            found = True
    return filtered

def find_needle_equation(needle):
    """
    Returns best fitting line
    """
    found = False
    k_size = 7
    while not found:
        kernel = np.ones((k_size, k_size), np.uint8)
        eroded = cv2.erode(needle, kernel) # Erode to reduce noise
        if np.sum(eroded) < np.sum(needle) / 2:
            k_size -= 2
        else:
            found = True
    y, x = np.array(np.where(eroded == 1)) # [0]=rows, [1]=cols
    M, B = np.polyfit(x, y, 1) # Find best fitting line

    return M, B
