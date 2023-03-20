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
from utils import save_pickle, load_pickle, make_gif, accuracy
from loaders import SequenceGenerator

#perform standard MLP training with Adam
def train_sgd_baseline(data_path='/home/rodrigue/data', n_repeats=3, seq_len=1000, datasets=['cifar10'], mode='full_iid', make_filters_gif=False, n_max_iters=1000, lr=1e-4):
    
    n_iters = seq_len
    if mode == 'full_iid':
        n_iters = 1000

    data = SequenceGenerator()

    frames = []
    mean_acc = 0

    for n in range(n_repeats):

        correlation = mode

        if mode == 'full_iid':
            correlation = 'iid'

        Xtr, Ytr = data.gen_sequence(seq_len, datasets, correlation=correlation, fold='train')
        D = Xtr.shape[-1]
        Xtr, Ytr = Xtr.view(Xtr.shape[0], -1).cuda(), Ytr.cuda()
        Xts, Yts = data.gen_sequence(seq_len, datasets, correlation=correlation, fold='test')
        Xts, Yts = Xts.view(Xtr.shape[0], -1).cuda(), Yts.cuda()

        net = nn.Sequential(
            nn.Linear(Xtr.shape[-1], 128),
            nn.LayerNorm([128]),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm([128]),
            nn.LeakyReLU(),
            nn.Linear(128, torch.max(Ytr).item()+1)
        )

        net.cuda()
        opt = optim.Adam(net.parameters(), lr=lr)

        cur_acc = 0
        for i in range(n_iters-8):
            net.train()

            if mode == 'ci':
                y = net(Xtr[i].unsqueeze(0))
                loss = F.cross_entropy(y, Ytr[i].unsqueeze(0))
            else:
                y = net(Xtr)
                loss = F.cross_entropy(y, Ytr)

            opt.zero_grad()
            loss.backward()
            opt.step()

            net.eval()

            yt = net(Xts)
            acc = accuracy(yt, Yts).mean()
                
            #we retain best test acc (cheat that favors this baseline)
            if acc > cur_acc:
                cur_acc = acc

            if make_filters_gif and n==n_repeats-1:
                if i%10==0:
                    bf_img = net[0].weight.view(-1, 1, D, D)
                    bf_img = torchvision.utils.make_grid(bf_img, normalize=True, nrow=8)
                    samples_img = torchvision.utils.make_grid(Xtr[i:i+8].view(-1, 1, D, D))
                    if mode=='full_iid':
                        samples_img = 0*samples_img
                    img = torch.cat([samples_img, bf_img], -2)
                    img = img.permute(1, 2, 0)
                    frames.append(img.cpu().numpy())
                 
        mean_acc += cur_acc

    if make_filters_gif:
        make_gif('results/sgd'+mode+'.gif', frames)

    return mean_acc / n_repeats

datasets = ['cifar10', 'mnist', 'svhn']
lr = 1e-2
acc = train_sgd_baseline(n_repeats=1, lr=lr, mode='ci', datasets=datasets, seq_len=10000, n_max_iters=5000, make_filters_gif=True)
print(acc)
"""
for _ in range(20):
    acc = train_sgd_baseline(n_repeats=1, lr=lr, mode='full_iid', datasets=datasets, seq_len=10000, n_max_iters=5000)
    print('lr:', lr, 'acc:', acc)
    lr = 0.5*lr
"""
