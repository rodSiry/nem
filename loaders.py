import torchvision
import os
from tqdm import tqdm
import torch 
import torch.nn.functional as F
import numpy as np
import random
import torch.distributions as distr
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import time
import math
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from utils import save_pickle, load_pickle

def iterate(d, bs=64, shuffle=True):
    loader = DataLoader(d, batch_size=bs, shuffle=shuffle, num_workers=16)
    while True:
        for s in enumerate(loader):
            yield s

def provide_datasets(dataset, seq_len, data_path=''):
    if dataset == 'mnist':
        t = transforms.Compose([transforms.Resize(32), transforms.Grayscale(1), transforms.ToTensor()])
        data = torchvision.datasets.MNIST(data_path, transform=t, download=True)
        data = iterate(data, bs=seq_len)
        data2 = torchvision.datasets.MNIST(data_path, transform=t, train=False, download=True)
        data2 = iterate(data2, bs=seq_len)

    elif dataset == 'svhn':
        t = transforms.Compose([transforms.Resize(32), transforms.Grayscale(1), transforms.ToTensor()])
        data = torchvision.datasets.SVHN(data_path, transform=t, download=True, split='train')
        data = iterate(data, bs=seq_len)
        data2 = torchvision.datasets.SVHN(data_path, transform=t, download=True, split='test')
        data2 = iterate(data2, bs=seq_len)

    elif dataset == 'cifar100':
        t = transforms.Compose([transforms.Resize(32), transforms.Grayscale(1), transforms.ToTensor()])
        data = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=t)
        data = iterate(data, bs=seq_len)
        data2 = torchvision.datasets.CIFAR100(root=data_path, train=False, download=True, transform=t)
        data2 = iterate(data2, bs=seq_len)

    elif dataset == 'cifar10':
        t = transforms.Compose([transforms.Resize(32), transforms.Grayscale(1), transforms.ToTensor()])
        data = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=t)
        data = iterate(data, bs=seq_len)
        data2 = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=t)
        data2 = iterate(data2, bs=seq_len)

    return data, data2


