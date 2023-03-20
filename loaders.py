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
import copy

def sort_by_class(s):
    sy, indices = torch.sort(s[1][1])
    return s[0], (s[1][0][indices], sy)

def iterate(d, bs=64, shuffle=True, correlation='random'):
    loader = DataLoader(d, batch_size=bs, shuffle=shuffle, num_workers=16)
    while True:
        for s in enumerate(loader):
            if correlation=='random':
                choice = random.choice([True, False])
                if choice:
                    yield sort_by_class(s)
                else:
                    yield s
            elif correlation=='iid':
                yield s
            elif correlation=='ci':
                yield sort_by_class(s)

#pre-bake and pickle the database that will be used by the sequence generator
def bake_database(data_path='/home/rodrigue/data', target_filename='database.pk', resolution=16, bs=1):
    t = transforms.Compose([transforms.Resize(resolution), transforms.Grayscale(1), transforms.ToTensor()])

    main_dict = {}

    datasets = []
    mnist_train = torchvision.datasets.MNIST(data_path, transform=t, download=True, train=True)
    mnist_test  = torchvision.datasets.MNIST(data_path, transform=t, download=True, train=False)
    datasets.append(('mnist', mnist_train, mnist_test))

    svhn_train = torchvision.datasets.SVHN(data_path, transform=t, download=True, split='train')
    svhn_test  = torchvision.datasets.SVHN(data_path, transform=t, download=True, split='test')
    datasets.append(('svhn', svhn_train, svhn_test))

    cifar10_train = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=t)
    cifar10_test = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=t)
    datasets.append(('cifar10', cifar10_train, cifar10_test))

    cifar100_train = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=t)
    cifar100_test = torchvision.datasets.CIFAR100(root=data_path, train=False, download=True, transform=t)
    datasets.append(('cifar100', cifar100_train, cifar100_test))
    
    for dataset_name, train_set, test_set in datasets:
        train_loader = DataLoader(train_set, batch_size=bs, shuffle=False, num_workers=1)
        test_loader = DataLoader(test_set, batch_size=bs, shuffle=False, num_workers=1)
        
        train_dict = {}
        c = 0
        for x, y in train_loader:
            xc = copy.deepcopy(x)
            try:
                train_dict[y.item()].append(xc)
            except:
                train_dict[y.item()] = [xc]
        for y in train_dict:
            train_dict[y] = torch.cat(train_dict[y], 0)

        test_dict = {}
        c = 0
        for x, y in test_loader:
            xc = copy.deepcopy(x)
            try:
                test_dict[y.item()].append(xc)
            except:
                test_dict[y.item()] = [xc]
        for y in train_dict:
            test_dict[y] = torch.cat(test_dict[y], 0)
        
        main_dict[dataset_name] = {'train':train_dict, 'test':test_dict}
    
    save_pickle(main_dict, os.path.join(data_path, target_filename))

#continual sequence generator
class SequenceGenerator():

    def __init__(self, file_path='/home/rodrigue/data/database.pk'):
        self.dict = load_pickle(file_path)

    def gen_sequence(self, seq_len=1000, dataset_list=['mnist', 'cifar10', 'svhn'], correlation='ci', fold='test', shuffle_class=True, shuffle_dset=False):
        
        seq_chunks_x = []
        seq_chunks_y = []

        dset_len = seq_len // len(dataset_list)

        for i, dset in enumerate(dataset_list):
            dset_chunks_x = []
            dset_chunks_y = []

            cur_dset_len = dset_len
            if i == len(dataset_list)-1:
                cur_dset_len = dset_len + seq_len % dset_len

            class_len = cur_dset_len // len(self.dict[dset][fold].keys())
            for j, y in enumerate(self.dict[dset][fold].keys()):
                chunk_len = class_len
                if j == len(self.dict[dset][fold].keys())-1:
                    chunk_len = class_len + cur_dset_len % class_len

                permutation = torch.randperm(self.dict[dset][fold][y].shape[0])
                chunk = self.dict[dset][fold][y][permutation][:chunk_len]
                dset_chunks_x.append(chunk)
                dset_chunks_y.append((y * torch.ones(chunk_len)).long())
            
            seq_chunks_x += dset_chunks_x
            seq_chunks_y += dset_chunks_y

        seq_x = torch.cat(seq_chunks_x, 0)
        seq_y = torch.cat(seq_chunks_y, 0)

        if correlation == 'iid':
            permutation = torch.randperm(seq_x.shape[0])
            seq_x, seq_y = seq_x[permutation], seq_y[permutation]

        return seq_x, seq_y
