import sys
import copy
import time
import torchvision 
import torch 
import numpy as np

def accuracy(y, Y):
    Y = Y.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    y = np.argmax(y, axis=-1)
    acc = np.mean((y == Y).astype(int))
    return acc

