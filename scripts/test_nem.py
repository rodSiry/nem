import sys 
sys.path.insert(0, '../')
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
import random
import torch.optim as optim
from models.nem import NEMModel
from loaders.sequence_generator import SequenceGenerator
from utils import save_pickle
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import numpy as np

#test of the update rule on randomized sequences

def test_meta(n_samples=20, d_h=16, d_a=1, n_classes=10, n_times=10, root='/data/chercheurs/siry191/data', snap_path='/data/chercheurs/siry191/data', exp_name='fast_a', force_shuffle=False):

    #source = cached_dataset(root=root, filename='cifar100_cache.pk')
    #source = cached_dataset(root=root, filename='micro_cifar100.pk')
    #source = cached_dataset(root=root, filename='cifar100.pk')
    #source = cached_dataset(root=root, filename='omniglot.pk')
    source = cached_dataset(root=root, filename='svhn_cache.pk')
    #source = cached_dataset(root=root, filename='svhn_cache.pk')
    #source = cached_dataset(root=root, filename='tiny_imagenet.pk')
    #xt, yt, xv, yv = source.get_non_stationary_sequence(n_samples=n_samples, n_classes=n_classes)
    #perfos = torch.zeros(xt.shape[0])

    clsf = CustomPropModel(d_h=d_h, d_a=d_a)
    save = torch.load(os.path.join(snap_path,name+'.pt'))
    clsf.load_state_dict(save['model'])

    mean_test = 0
    pbar = tqdm(range(n_times))
    for m in pbar:
        arch = Arch(n_out=200, n_in=256, ly_range=[3, 3], h_range=[n_h, n_h], n_neur=d_h)
        clsf.eval()
        clsf.cuda()

        #training sequence
        xt, yt, xv, yv = source.get_non_stationary_sequence(max_seq=n_samples*n_classes, n_samples=n_samples, n_classes=n_classes)
        xt, yt, xv, yv = xt.cuda(), yt.cuda(), xv.cuda(), yv.cuda()

        #NEM training
        state = arch.get_init().cuda()
        for xi, yi in zip(xt, yt):
            xi, yi = xi.unsqueeze(0), yi.unsqueeze(0)
            state = clsf.update(arch, state, xi, yi, n_steps=n_steps).detach()

        #evaluate on train sequence
        acc_train = 0
        for j, (xi, yi) in enumerate(zip(xt, yt)):
            xi, yi = xi.unsqueeze(0), yi.unsqueeze(0)
            loss, acc_ = clsf.inference(arch, state, xi, yi, n_steps=n_steps)
            acc_train += acc_
            #perfos[j] += acc_
        acc_train = acc_train / xt.shape[0]

        #evaluate on test set
        acc_test = 0
        for xi, yi in zip(xv, yv):
            xi, yi = xi.unsqueeze(0), yi.unsqueeze(0)
            loss, acc_ = clsf.inference(arch, state, xi, yi, n_steps=n_steps)
            acc_test += acc_
        acc_test = acc_test / xv.shape[0]

        pbar.set_description('train: '+str(acc_train)+ ' test: '+ str(acc_test))
        mean_test += acc_test

    mean_test = mean_test / n_times
    print(mean_test)


