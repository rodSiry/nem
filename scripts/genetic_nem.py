#the NEM model, but re-implemented for genetic (or ES) optimization

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

def accuracy(y, Y):
    Y = Y.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    y = np.argmax(y, axis=-1)
    acc = (y == Y).astype(int)
    return acc

def iterate(d, bs=64, shuffle=True):
    loader = DataLoader(d, batch_size=bs, shuffle=shuffle, num_workers=16)
    while True:
        for s in enumerate(loader):
            yield s

 
#class to instantiate each tiny function of the update rule

class MetaNNPop():
    def __init__(self, n_pop, n_in, n_hidden, n_out):
        self.w1 = torch.randn(n_pop, n_in, n_hidden).cuda() / math.sqrt(n_in)
        self.w2 = torch.randn(n_pop, n_hidden, n_out).cuda() / math.sqrt(n_hidden)
        
        self.b1 = torch.zeros(n_pop, 1, n_hidden).cuda()

    def apply(self, x):
        y = torch.bmm(x, self.w1)
        y = y + self.b1
        y = torch.relu(y)
        y = torch.bmm(y, self.w2)
        return y
    
    def evolve(self, indices, thr=100):
        n_pop = self.w1.shape[0]//2
        self.w1 = self.w1[indices]
        self.w2 = self.w2[indices]
        self.b1 = self.b1[indices]

        self.w1 = torch.cat([self.w1[:n_pop], self.w1[:n_pop]], 0)
        self.w2 = torch.cat([self.w2[:n_pop], self.w2[:n_pop]], 0)
        self.b1 = torch.cat([self.b1[:n_pop], self.b1[:n_pop]], 0)

        self.w1[n_pop:] = self.w1[n_pop:] + 1e-2 * torch.randn(self.w1[n_pop:].shape).cuda()
        self.w2[n_pop:] = self.w2[n_pop:] + 1e-2 * torch.randn(self.w2[n_pop:].shape).cuda()
        self.b1[n_pop:] = self.b1[n_pop:] + 1e-2 * torch.randn(self.b1[n_pop:].shape).cuda()
        
#class to instantiate, evaluate and evolve a population of candidate update rules

class Population():
    def __init__(self, n_pop=100, n_base_layers=3, n_base_in=512, n_base_out=10, n_base_hidden=128, n_meta_state=5, n_meta_act=5, eps=1e-4, cuda=True):
        self.n_pop = n_pop

        self.n_base_layers = n_base_layers
        self.n_base_in = n_base_in
        self.n_base_out = n_base_out
        self.n_base_hidden = n_base_hidden

        self.n_meta_state = n_meta_state
        self.n_meta_act   = n_meta_act

        self.cuda = cuda
        self.eps = eps

    def init_base_param(self):
        self.base_param = {}

        h = []
        w = []
        b = []

        for i in range(self.n_base_layers):
            if i == 0:
                din = self.n_base_in
                dout = self.n_base_hidden

            elif i == self.n_base_layers - 1:
                din = self.n_base_hidden
                dout = self.n_base_out

            else:
                din = self.n_base_hidden
                dout = self.n_base_hidden

            w.append(torch.randn(self.n_pop, din, dout).cuda())
            h.append(torch.zeros(self.n_pop, dout, self.n_meta_state).cuda())

        self.base_param = {'w':w, 'h':h}

    def init_meta_param(self, replace=True):

        inner = MetaNNPop(self.n_pop, 2*self.n_meta_state + self.n_meta_act, 10, self.n_meta_state)
        to_prev = MetaNNPop(self.n_pop, self.n_meta_state, 10, self.n_meta_state)
        to_next = MetaNNPop(self.n_pop, self.n_meta_state, 10, self.n_meta_state)
        to_act    = MetaNNPop(self.n_pop, self.n_meta_state + self.n_meta_act, 10, self.n_meta_act)
        expand = MetaNNPop(self.n_pop, 1, 10, self.n_meta_act)
        shrink = MetaNNPop(self.n_pop, self.n_meta_act, 10, 1)
        self.meta = {'inner':inner, 'to_prev':to_prev, 'to_next':to_next, 'to_act':to_act, 'expand':expand, 'shrink':shrink}

    def evolve(self, criterion, threshold=5):
        #indices = torch.argsort(criterion)#torch.from_numpy(criterion))
        indices = torch.argsort(torch.from_numpy(criterion))
        indices = torch.flip(indices, (0, ))
        for k in self.meta:
            self.meta[k].evolve(indices)

    def normalize(self, x):
        mean = x.mean(1, keepdim=True).expand(x.shape)
        std  = x.std(1, keepdim=True).expand(x.shape)
        return (x - mean) / (std + 1e-10)

    def normalize_last(self, x):
        mean = x.mean(-1).unsqueeze(-1)
        std  = x.std(-1).unsqueeze(-1)
        return (x - mean) / (std + 1e-10)
        
    def inference(self, x, Y):

        #forward pass

        y = x.unsqueeze(-1)
        y = self.meta['expand'].apply(y)

        forward_msgs = [y]

        for wi, hi in zip(self.base_param['w'], self.base_param['h']):
            y = torch.bmm(y.permute(0, 2, 1), wi)
            y = y.permute(0, 2, 1)
            y = self.normalize(y)
            forward_msgs.append(y)
            y = torch.cat([y, hi], -1)
            y = self.meta['to_act'].apply(y)
    
        y = self.meta['shrink'].apply(y)
        y = y.squeeze(-1)

        losses = F.cross_entropy(y, Y, reduction='none')
        acc = accuracy(y, Y)
        
        return losses, acc

    def update(self, x, Y):

        #forward pass

        y = x.unsqueeze(-1)
        y = self.meta['expand'].apply(y)

        forward_msgs = [y]

        for wi, hi in zip(self.base_param['w'], self.base_param['h']):
            y = torch.bmm(y.permute(0, 2, 1), wi)
            y = y.permute(0, 2, 1)
            y = self.normalize(y)
            forward_msgs.append(y)
            y = torch.cat([y, hi], -1)
            y = self.meta['to_act'].apply(y)
    
        #backward pass & state update

        cur_backward_msg = F.one_hot(Y, self.n_base_out).float()
        cur_backward_msg = cur_backward_msg.unsqueeze(-1).expand(-1, -1, self.n_meta_state)

        backward_msgs = [cur_backward_msg]

        for i in range(self.n_base_layers):
            j = self.n_base_layers - i - 1
            cur_forward_msg = forward_msgs[j + 1]
            update_input = torch.cat([self.base_param['h'][j], cur_forward_msg, cur_backward_msg], -1)
            self.base_param['h'][j] = torch.clip(self.meta['inner'].apply(update_input), -1, 1)
            cur_backward_msg = torch.bmm(self.base_param['w'][j], cur_backward_msg)
            cur_backward_msg = self.normalize(cur_backward_msg)
            backward_msgs.append(cur_backward_msg)

        #base weight matrix update

        for i, wi in enumerate(self.base_param['w']):
            cur_forward_msg  = forward_msgs[i]
            cur_backward_msg = backward_msgs[self.n_base_layers - i - 1]
            fwd = self.meta['to_prev'].apply(cur_forward_msg)
            bwd = self.meta['to_next'].apply(cur_backward_msg)
            dwi = torch.bmm(fwd, bwd.permute(0, 2, 1))
            self.base_param['w'][i] = wi + self.eps * dwi

#

def train_nem(data_path = '/home/rodrigue/data', n_pop = 1000, n_rep = 10, seq_len = 100, n_classes = 100, input_dim = 1024, n_iters = 1000000000000):

#t = torchvision.transforms.ToTensor()
#data = torchvision.datasets.MNIST('/home/rodrigue/data', transform=t, download=True)

t = transforms.Compose([transforms.Resize(32), transforms.Grayscale(1), transforms.ToTensor()])
data = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=t)
data = iterate(data, bs=seq_len)


with torch.no_grad():
    pop = Population(n_pop=n_pop, n_base_in=input_dim, n_base_out=n_classes)
    pop.init_meta_param()

    for _ in range(n_iters):
        total_acc = 0
        total_loss = 0
        for k in range(n_rep):
            #sample sequence
            pop.init_base_param()
            _, (x, y) = next(data)
            x, y = x.cuda(), y.long().cuda()
            x = x.view(seq_len, -1)

            for i in range(seq_len):
                x_train, y_train = x[i].unsqueeze(0).expand(n_pop, -1), y[i].unsqueeze(0).expand(n_pop)
                j = random.randint(0, i)
                x_test, y_test = x[j].unsqueeze(0).expand(n_pop, -1), y[j].unsqueeze(0).expand(n_pop)
                pop.update(x_train, y_train)
                losses, acc = pop.inference(x_test, y_test)
                total_acc += acc
                total_loss += losses
        pop.evolve(total_acc)
        #pop.sort_meta_param(total_loss)
        print(total_acc.mean() / n_rep/seq_len)
        #print(total_acc.mean() / n_rep / seq_len)
