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

class LearnerPopulation(nn.Module):
    def __init__(self, n_pop=10, n_base_in=512, n_base_out=100, n_base_hidden=256, n_base_layers=2, n_h=16, n_a=16):
        super().__init__()

        self.n_base_layers = n_base_layers
        self.n_pop = n_pop
        self.n_base_in  = n_in
        self.n_base_out = n_out
        self.n_h   = n_h
        self.n_a   = n_a

    def init_base_params(self):

        self.base_weights = []
        self.base_inner_states = []

        for i in range(self.n_base_layers):
            if i == 0:
                self.base_weights.append(torch.randn(self.n_pop, self.n_base_hidden, self.n_base_in).cuda())
                self.base_inner_states.append(torch.randn(self.n_pop, self.n_base_hidden, self.n_h).cuda())
            elif i == self.n_base_layers - 1:
                self.base_weights.append(torch.randn(self.n_pop, self.n_base_out, self.n_base_hidden).cuda())
                self.base_inner_states.append(torch.randn(self.n_pop, self.n_base_out, self.n_h).cuda())
            else:
                self.base_weights.append(torch.randn(self.n_pop, self.n_base_hidden, self.n_base_hidden).cuda())
                self.base_inner_states.append(torch.randn(self.n_pop, self.n_base_hidden, self.n_h).cuda())

    def init_meta_params(self):

        self.i2v = torch.randn(self.n_pop, self.n_a, 1).cuda()
        self.v2i = torch.randn(self.n_pop, 1, self.n_a).cuda()

        self.ha2h_w = torch.randn(self.n_pop, self.n_h, self.n_a + self.n_h).cuda()
        self.ha2h_b = torch.randn(self.n_pop, self.n_h).cuda()

        self.ha2a_w = torch.randn(self.n_pop, self.n_a, self.n_a + self.n_h).cuda()
        self.ha2a_b = torch.randn(self.n_pop, self.n_a).cuda()

    def forward(self, x, y):
