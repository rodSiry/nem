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

#loads a dump and provides randomized input sequences with desired characteristics

class SequenceGenerator():
    def __init__(self, dump_path='/data/chercheurs/siry191/data/', filename='cifar100_cache.pk'):
        filepath = os.path.join(dump_path, filename)
        F = open(filepath, 'rb')
        self.dico = pickle.load(F)
        F.close()

    def get_non_stationary_sequence(self, max_seq=2, n_samples=100, n_classes=100, train=False):

        xt = []
        yt = []
        xv = []
        yv = []

        #shuffle labels
        label_permutation = np.random.permutation(100)

        n_tot = n_samples * n_classes
        if train:
            n_classes = min(random.randint(2, 100), n_classes)
        else:
            n_classes = n_classes
        n_samples = max(n_tot // n_classes, 1)
        classes = random.sample(range(0, len(self.dico['train'].keys())), n_classes)

        #fetch samples
        for i, c in enumerate(classes):
            train_samples_c = random.sample(self.dico['train'][c], min(n_samples, len(self.dico['train'][c])))
            test_samples_c  = random.sample(self.dico['test'][c], len(self.dico['test'][c]))

            xt += [x[0] for x in train_samples_c]
            xv += [x[0] for x in test_samples_c]
            yt += [c for x in train_samples_c]
            yv += [c for x in test_samples_c]

        xt = torch.stack(xt, 0)
        yt = torch.Tensor(yt).long()
        xv = torch.stack(xv, 0)
        yv = torch.Tensor(yv).long()

        do_shuffle = bool(random.getrandbits(1))
        if do_shuffle and train:
            perm = torch.randperm(xt.shape[0])
            xt = xt[perm]
            yt = yt[perm]
        return xt[:max_seq, :], yt[:max_seq], xv[:, :], yv
