import torchvision.transforms as transforms
from torchvision.datasets.folder import *
import torchvision
from torch.utils.data import Dataset, DataLoader 
from tqdm import tqdm
import numpy as np
import os 
import random
import torch
import pickle


#call these functions once to create dump datasets of small images

def bake_miniImageNet(root='/data/chercheurs/siry191/data'):
    path = os.path.join(root, 'tiny-imagenet-200')


    t = transforms.Compose([transforms.Resize(16), transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])

    #bake train
    dico_train = {}
    train_path = os.path.join(path, 'train')
    for j, c in tqdm(enumerate(os.listdir(train_path))):
        train_class_path = os.path.join(train_path, c, 'images')
        dico_train[j] = []
        for f in os.listdir(train_class_path):
            filepath = os.path.join(train_path, c, 'images', f)
            image = default_loader(filepath)
            image = t(image).view(-1)
            dico_train[j].append((image, j))

    dico = {}
    dico['train'] = dico_train
    dico['test'] = dico_train
    F = open(os.path.join(root, 'tiny_imagenet.pk'), 'wb')
    pickle.dump(dico, F)
    F.close()

def bake_cifar100(root='/home/rodrigue/data/', train=True):
    t = transforms.Compose([transforms.Resize(16), transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
    dset_train = torchvision.datasets.CIFAR100(root=root, train=True,  transform=t, download=True)
    dset_val = torchvision.datasets.CIFAR100(root=root, train=False,  transform=t, download=True)

    dico = {}
    dico['train'] = {}
    for i in tqdm(range(len(dset_train))):
        xi, yi = dset_train.__getitem__(i)
        try:
            dico['train'][yi].append((xi.view(-1), yi))
        except:
            dico['train'][yi] = [(xi.view(-1), yi)]

    dico['test'] = {}
    for i in tqdm(range(len(dset_val))):
        xi, yi = dset_val.__getitem__(i)
        try:
            dico['test'][yi].append((xi.view(-1), yi))
        except:
            dico['test'][yi] = [(xi.view(-1), yi)]

    #F = open(os.path.join(root, 'cifar100_cache.pk'), 'wb')
    F = open(os.path.join(root, 'micro_cifar100.pk'), 'wb')
    pickle.dump(dico, F)
    F.close()

def bake_mnist(root='/data/chercheurs/siry191/data', train=True):
    t = transforms.Compose([transforms.Resize(16), transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
    dset_train = torchvision.datasets.MNIST(root=root, train=True, transform=t, download=True)
    dset_val = torchvision.datasets.MNIST(root=root, train=False, transform=t, download=True)

    dico = {}
    dico['train'] = {}
    for i in tqdm(range(len(dset_train))):
        xi, yi = dset_train.__getitem__(i)
        try:
            dico['train'][yi].append((xi.view(-1), yi))
        except:
            dico['train'][yi] = [(xi.view(-1), yi)]

    dico['test'] = {}
    for i in tqdm(range(len(dset_val))):
        xi, yi = dset_val.__getitem__(i)
        try:
            dico['test'][yi].append((xi.view(-1), yi))
        except:
            dico['test'][yi] = [(xi.view(-1), yi)]

    F = open(os.path.join(root, 'mnist_cache.pk'), 'wb')
    pickle.dump(dico, F)
    F.close()

def bake_svhn(root='/data/chercheurs/siry191/data', train=True):
    t = transforms.Compose([transforms.Resize(16), transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
    dset_train = torchvision.datasets.SVHN(root=root, split='train',  transform=t, download=True)
    dset_val = torchvision.datasets.SVHN(root=root, split='test',  transform=t, download=True)

    dico = {}
    dico['train'] = {}
    for i in tqdm(range(len(dset_train))):
        xi, yi = dset_train.__getitem__(i)
        try:
            dico['train'][yi].append((xi.view(-1), yi))
        except:
            dico['train'][yi] = [(xi.view(-1), yi)]

    dico['test'] = {}
    for i in tqdm(range(len(dset_val))):
        xi, yi = dset_val.__getitem__(i)
        try:
            dico['test'][yi].append((xi.view(-1), yi))
        except:
            dico['test'][yi] = [(xi.view(-1), yi)]

    F = open(os.path.join(root, 'svhn_cache.pk'), 'wb')
    pickle.dump(dico, F)
    F.close()

bake_cifar100()
