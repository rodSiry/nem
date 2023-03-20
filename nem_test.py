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
from utils import save_pickle, load_pickle, make_gif, convolve
from models import Population
from loaders import SequenceGenerator
import matplotlib.pyplot as plt



def accuracy(y, Y):
    Y = Y.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    y = np.argmax(y, axis=-1)
    acc = (y == Y).astype(int)
    return acc


def test_nem_model(data_path='/home/rodrigue/data', datasets = ['cifar10', 'mnist', 'svhn'], seq_len=10000, n_pop=1000, n_classes=100, input_dim=256, n_iters=1, D=16):

    logs = {}
    logs['mean'] = []
    logs['best'] = []

    frames = []

    data = SequenceGenerator()

    with torch.no_grad():

        pop = Population(n_pop=n_pop, n_base_in=input_dim, n_base_out=n_classes)
        pop.init_meta_param()
        pop = load_pickle('results/snapshot/nem_small/10000.pk')
        pop.n_base_out = 10
        #pop.n_base_hidden = 512


        total_acc = 0
        total_loss = 0

        for n in range(n_iters+1):
            hidden = []
            if n == 1:

                img = Xtr.view(-1, 1, D, D)
                torchvision.utils.save_image(img, 'results/seq.png')

            #sample sequence
            pop.init_base_param()

            Xtr, Ytr = data.gen_sequence(seq_len, datasets, correlation='ci', fold='train')
            Xtr, Ytr = Xtr.view(Xtr.shape[0], -1).cuda(), Ytr.cuda()
            Xts, Yts = data.gen_sequence(seq_len, datasets, correlation='ci', fold='test')
            Xts, Yts = Xts.view(Xtr.shape[0], -1).cuda(), Yts.cuda()
            


            for i in range(Xtr.shape[0]-8):
                x_train, y_train = Xtr[i].unsqueeze(0).expand(n_pop, -1), Ytr[i].unsqueeze(0).expand(n_pop)
                pop.update(x_train, y_train)

                if n == n_iters and i%10==0:
                    #bf_img = net[0].weight.view(-1, 1, 32, 32)
                    bf_img = pop.get_best_filters(total_acc).cpu().detach()
                    bf_img = torchvision.utils.make_grid(bf_img.view(-1, 1, D, D), normalize=True, nrow=8)
                    samples_img = torchvision.utils.make_grid(Xtr[i:i+8].view(-1, 1, D, D).cpu())
                    img = torch.cat([samples_img, bf_img], -2)
                    img = img.permute(1, 2, 0)
                    frames.append(img.detach().cpu().numpy())
         
                if n == n_iters:
                    hidden.append(pop.get_best_hidden(total_acc).cpu().detach())

            for i in range(Xts.shape[0]):
                x_test, y_test = Xts[i].unsqueeze(0).expand(n_pop, -1), Yts[i].unsqueeze(0).expand(n_pop)
                #x_test, y_test = Xtr[i].unsqueeze(0).expand(n_pop, -1), Ytr[i].unsqueeze(0).expand(n_pop)
                losses, acc = pop.inference(x_test, y_test)
                total_acc += acc
                total_loss += losses

            print(total_acc.mean()/n_iters/Xts.shape[0], max(total_acc)/n_iters/Xts.shape[0])
            logs['mean'].append(total_acc.mean())
            logs['best'].append(np.max(total_acc))
            if n == n_iters:
                hidden = torch.stack(hidden, 0)
                plot_x = hidden[:, 0, :].cpu().detach().numpy()
                for k in range(plot_x.shape[-1]):
                    plt.plot(convolve(plot_x[:, k]))
                plt.show()

        make_gif('results/nem_'+'_'.join(datasets)+'_.gif', frames)
