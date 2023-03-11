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
from models import MetaNNProp, Population

def accuracy(y, Y):
    Y = Y.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    y = np.argmax(y, axis=-1)
    acc = (y == Y).astype(int)
    return acc
exp_name = 'nem_fast_lr'
data_path = '/home/rodrigue/data'
n_pop = 1000
seq_len = 2000
n_classes = 10
input_dim = 1024
n_iters = 5
dataset = 'mnist'

logs = {}
logs['mean'] = []
logs['best'] = []


with torch.no_grad():

    pop = Population(n_pop=n_pop, n_base_in=input_dim, n_base_out=n_classes)
    
    pop.init_meta_param()
    pop = load_pickle('results/snapshot/nem_ret/1000.pk')
    #pop.n_base_hidden = 512


    total_acc = 0
    total_loss = 0

    for n in range(n_iters+1):
        if n == 1:

            img = Xtr.view(-1, 1, 32, 32)
            torchvision.utils.save_image(img, 'results/seq.png')

        #sample sequence
        pop.init_base_param()

        _, (Xtr, Ytr) = next(data)
        Xtr, Ytr = Xtr.cuda(), Ytr.long().cuda()
        Xtr = Xtr.view(seq_len, -1)

        _, (Xts, Yts) = next(data2)
        Xts, Yts = Xts.cuda(), Yts.long().cuda()
        Xts = Xts.view(seq_len, -1)

        for i in range(seq_len):
            x_train, y_train = Xtr[i].unsqueeze(0).expand(n_pop, -1), Ytr[i].unsqueeze(0).expand(n_pop)
            pop.update(x_train, y_train)

            if n == n_iters and i%100==0:
                bf = pop.get_best_filters(total_acc)
                img = torchvision.utils.make_grid(bf, normalize=True)
                torchvision.utils.save_image(img, 'results/filters_'+str(i)+'.png')

        for i in range(seq_len):
            x_test, y_test = Xts[i].unsqueeze(0).expand(n_pop, -1), Yts[i].unsqueeze(0).expand(n_pop)
            #x_test, y_test = Xtr[i].unsqueeze(0).expand(n_pop, -1), Ytr[i].unsqueeze(0).expand(n_pop)
            losses, acc = pop.inference(x_test, y_test)
            total_acc += acc
            total_loss += losses

        print(total_acc.mean()/n_iters/seq_len, max(total_acc)/n_iters/seq_len)
        logs['mean'].append(total_acc.mean())
        logs['best'].append(np.max(total_acc))
