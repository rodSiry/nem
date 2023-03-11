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

def accuracy(y, Y):
    Y = Y.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    y = np.argmax(y, axis=-1)
    acc = (y == Y).astype(int)
    return acc

def gaussian_mutation(x):
    n = torch.randn(x.shape).cuda()
    return x + 0.01 * n

def nonlocal_sparse(x):
    n = (torch.rand(x.shape).cuda() - 0.5) * 2
    d = torch.bernoulli(torch.ones(x.shape) * p).cuda()
    x = (1 - d) * x + d * n 
    x = torch.clip(x, -1, 1)
    return x

class MLPPopulation():
    def __init__(self, n_pop, n_in, n_hidden, n_out, mutation_operator=gaussian_mutation):
        self.w1 = torch.randn(n_pop, n_in, n_hidden).cuda() / math.sqrt(n_in)
        self.w2 = torch.randn(n_pop, n_hidden, n_out).cuda() / math.sqrt(n_hidden)
        
        self.b1 = torch.zeros(n_pop, 1, n_hidden).cuda()
        self.mutation_operator = mutation_operator

    def apply(self, x):
        y = torch.bmm(x, self.w1)
        y = y + self.b1
        y = torch.relu(y)
        y = torch.bmm(y, self.w2)
        return y
    
    def mutation_operator(self, x):
        return self.mutation_operator(x)
   
    def evolve(self, indices, thr=100):
        n_pop = self.w1.shape[0]//2
        self.w1 = self.w1[indices]
        self.w2 = self.w2[indices]
        self.b1 = self.b1[indices]

        self.w1 = torch.cat([self.w1[:n_pop], self.w1[:n_pop]], 0)
        self.w2 = torch.cat([self.w2[:n_pop], self.w2[:n_pop]], 0)
        self.b1 = torch.cat([self.b1[:n_pop], self.b1[:n_pop]], 0)

        self.w1[n_pop:] = self.mutation_operator(self.w1[n_pop:])
        self.w2[n_pop:] = self.mutation_operator(self.w2[n_pop:])
        self.b1[n_pop:] = self.mutation_operator(self.b1[n_pop:])
        
class Population():
    def __init__(self, n_pop=100, n_base_layers=3, n_base_in=512, n_base_out=10, n_base_hidden=128, n_meta_state=5, n_meta_act=5, eps=1, cuda=True, mutation_operator=gaussian_mutation):
        self.n_pop = n_pop

        self.n_base_layers = n_base_layers
        self.n_base_in = n_base_in
        self.n_base_out = n_base_out
        self.n_base_hidden = n_base_hidden

        self.n_meta_state = n_meta_state
        self.n_meta_act   = n_meta_act

        self.cuda = cuda
        self.eps = eps
        self.mutation_operator = mutation_operator

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

        inner = MLPPopulation(self.n_pop, 2*self.n_meta_state + self.n_meta_act, 10, self.n_meta_state, mutation_operator=self.mutation_operator)
        to_prev = MLPPopulation(self.n_pop, self.n_meta_state, 10, self.n_meta_state, mutation_operator=self.mutation_operator)
        to_next = MLPPopulation(self.n_pop, self.n_meta_state, 10, self.n_meta_state, mutation_operator=self.mutation_operator)
        to_act    = MLPPopulation(self.n_pop, self.n_meta_state + self.n_meta_act, 10, self.n_meta_act, mutation_operator=self.mutation_operator)
        expand = MLPPopulation(self.n_pop, 1, 10, self.n_meta_act, mutation_operator=self.mutation_operator)
        shrink = MLPPopulation(self.n_pop, self.n_meta_act, 10, 1, mutation_operator=self.mutation_operator)
        self.meta = {'inner':inner, 'to_prev':to_prev, 'to_next':to_next, 'to_act':to_act, 'expand':expand, 'shrink':shrink}

    def evolve(self, criterion, threshold=5):
        indices = torch.argsort(torch.from_numpy(criterion))
        indices = torch.flip(indices, (0, ))
        for k in self.meta:
            self.meta[k].evolve(indices)

    def normalize(self, x):
        mean = x.mean(1, keepdim=True).expand(x.shape)
        std  = x.std(1, keepdim=True).expand(x.shape)
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
            fwd = fwd / torch.sqrt(fwd.pow(2).sum(-1).unsqueeze(-1) + 1e-10)
            bwd = self.meta['to_next'].apply(cur_backward_msg)
            bwd = bwd / torch.sqrt(bwd.pow(2).sum(-1).unsqueeze(-1) + 1e-10)
            dwi = torch.bmm(fwd, bwd.permute(0, 2, 1))
            self.base_param['w'][i] = torch.clip(wi + self.eps * dwi, -3, 3)


