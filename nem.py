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
from models import Population
from loaders import provide_datasets

def train_nem(dataset_name='cifar10', exp_name='nem_ret', data_path='/home/rodrigue/data', n_pop=1000, n_rep=1, seq_len=[1000, 5000], n_classes=10, input_dim=1024, n_iters=1000000000000):
    
    curriculum_index=0
    train_data, test_data = provide_datasets(dataset_name, seq_len[curriculum_index], data_path=data_path)

    n=0

    os.makedirs(os.path.join('results/snapshot', exp_name), exist_ok=True)

    logs = {}
    logs['mean'] = []
    logs['best'] = []
    
    pop = Population(n_pop=n_pop, n_base_in=input_dim, n_base_out=n_classes)
    pop.init_meta_param()
    
    while True:
        
        with torch.no_grad():

            total_acc = 0
            total_loss = 0

            for k in range(n_rep):
                pop.init_base_param()

                #sample train sequence
                _, (Xtr, Ytr) = next(train_data)
                Xtr, Ytr = Xtr.cuda(), Ytr.long().cuda()
                Xtr = Xtr.view(seq_len[curriculum_index], -1)

                #sample test data
                _, (Xts, Yts) = next(test_data)
                Xts, Yts = Xts.cuda(), Yts.long().cuda()
                Xts = Xts.view(seq_len[curriculum_index], -1)

                for i in range(seq_len[curriculum_index]):
                    x_train, y_train = Xtr[i].unsqueeze(0).expand(n_pop, -1), Ytr[i].unsqueeze(0).expand(n_pop)
                    j = random.randint(0, seq_len[curriculum_index]-1)
                    x_test, y_test = Xts[j].unsqueeze(0).expand(n_pop, -1), Yts[j].unsqueeze(0).expand(n_pop)
                    pop.update(x_train, y_train)
                    losses, acc = pop.inference(x_test, y_test)
                    total_acc += acc
                    total_loss += losses

            pop.evolve(total_acc)
            total_acc = total_acc / n_rep / seq_len[curriculum_index]
            print(total_acc.mean(), np.max(total_acc))
            logs['mean'].append(total_acc.mean())
            logs['best'].append(np.max(total_acc))

            if n%100 == 0:
                save_pickle(pop, os.path.join('results/snapshot', exp_name, str(seq_len[curriculum_index])+'.pk'))
                save_pickle(logs, os.path.join('results/snapshot', exp_name, str(seq_len[curriculum_index])+'_logs.pk'))

            if np.max(total_acc) >= 0.9:
                curriculum_index += 1
                if curriculum_index > len(seq_len):
                    print("Training finished.")
                    break
                print("Level-up : ", seq_len[curriculum_index])
                train_data, test_data = provide_datasets(dataset_name, seq_len[curriculum_index])

            n += 1

train_nem()
