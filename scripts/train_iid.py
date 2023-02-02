import sys 
sys.path.insert(0, '../')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import math
import random
import torch.optim as optim
import torchvision.transforms as transforms
from models.metrics import accuracy
from loaders.sequence_generator import SequenceGenerator
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

def iterate(d, bs=64, shuffle=True, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    loader = DataLoader(d,  batch_size=bs, shuffle=shuffle, num_workers=8)
    while True:
        for s in enumerate(loader):
            yield s
def corrupt(x):
    r = x
    rd1 = random.randint(0, x.shape[-1]-1)
    r[:, :rd1] = 0
    return r

def corrupt2d(x):
    r = x
    rd1 = random.randint(0, x.shape[-1]-1)
    r[:, :rd1] = 0
    return r

#build sparse layer with log(N)N elements
class SparseLinear(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()

        n_elem = n_out * math.floor(math.log(n_in))
        indices_1 = torch.randint(0, n_in-1, (n_elem, ))
        indices_2 = torch.Tensor(range(n_out)).unsqueeze(-1).expand(-1, math.floor(math.log(n_in))).reshape(-1)
        values = torch.randn(indices_2.shape) / math.sqrt(math.floor(math.log(n_in)))
        matrix = torch.sparse_coo_tensor(indices=torch.stack([indices_1, indices_2], 0), values=values, size=(n_in, n_out))
        self.w = nn.Parameter(matrix)
        self.b = nn.Parameter(torch.zeros(1, n_out))

    def forward(self, x):
        y = torch.sparse.mm(self.w.t(), x.t()).t()
        y = y + self.b
        return y
        
"""
class BlockSparseLinear(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_in
    def forward(self, x):
        y = torch.sparse.mm(self.w.t(), x.t()).t()
        y = y + self.b
        return y
"""

class ResidualFF(nn.Module):
    def __init__(self, n_input, n_out, spectral=False, sampling='down'):
        super().__init__()
        self.lin = nn.Linear(n_input, n_out)
        #self.bn = nn.BatchNorm1d(n_out)

    def forward(self, x):
        y = torch.relu(self.lin(x))
        return y + x



class ResidualBlock(nn.Module):
    def __init__(self, n_input, n_out, spectral=False, sampling='down'):
        super(ResidualBlock, self).__init__()
        k = 3
        p = k // 2
        self.short = nn.Conv2d(n_input, n_out, 1)
        self.conv1 = nn.Conv2d(n_input, n_out, k, padding=p)
        self.conv2 = nn.Conv2d(n_out, n_out, k, padding=p)
        self.bn1 = nn.BatchNorm2d(n_out)
        self.bn2 = nn.BatchNorm2d(n_out)
        self.sampling = sampling
        self.last = nn.Conv2d(n_out, n_out, 2, 2)

    def forward(self, x):
        y = F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))))
        y = self.short(x) + y
        if self.sampling=='down':
            y = self.last(y)
            return y



def train_iid(n_it=10, n_samples=10, n_samples_buffer=10, n_classes=100, n_val=100, n_times=50, conv=True, data_path='/data/chercheurs/siry191/data/', snap_path='/data/chercheurs/siry191/data/'):

    source = SequenceGenerator(dump_path=data_path, filename='micro_cifar100.pk')
    t = transforms.Compose([transforms.RandomAffine(90, translate=(0.1, 0.1), scale=(0.8, 1)), transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2), transforms.Resize(16), transforms.Grayscale(), transforms.ToTensor()])

    rracc_train = 0
    rracc_test  = 0
    pbar = tqdm(range(n_times))
    bs  = 64
    t_test = transforms.Compose([transforms.Resize(16), transforms.Grayscale(), transforms.ToTensor()])
    dset = iterate(torchvision.datasets.CIFAR100(root=data_path, train=True, transform=t_test), bs=64)
    test_dset = iterate(torchvision.datasets.CIFAR100(root=data_path, train=False, transform=t_test), bs=64)

    for t in pbar:

        if not conv:
            n_h = 4096
            model = nn.Sequential(
                    nn.Linear(256, n_h),
                    ResidualFF(n_h, n_h),
                    ResidualFF(n_h, n_h),
                    ResidualFF(n_h, n_h),
                    nn.Linear(n_h, 100))

        else:
            n_h = 128
            model = nn.Sequential(
                    ResidualBlock(1, n_h),
                    ResidualBlock(n_h, n_h*2),
                    ResidualBlock(n_h*2, n_h*4),
                    nn.Flatten(),
                    nn.Linear(16*n_h, 100),)

        model.cuda()
        opt = optim.Adam(model.parameters(), lr=1e-4)

        x, y, xv, yv = source.get_non_stationary_sequence(max_seq=n_samples*n_classes, n_samples=n_samples, n_classes=n_classes)
        x, y, xv, yv = x.cuda(), y.cuda(), xv.cuda(), yv.cuda()
        
        if conv:
            x = x.view(-1,  1, 16, 16)
            xv = xv.view(-1,  1, 16, 16)

        dico = {}

        for xi, yi in zip(x, y):
            try:
                dico[yi.item()].append(xi)
            except:
                dico[yi.item()] = [xi]
        
        xb, yb = [], []
        for k in dico.keys():
            xb = xb + dico[k][:n_samples_buffer]
            yb = yb + [k] * len(dico[k][:n_samples_buffer])

        xb = torch.stack(xb, 0).cuda()
        yb = torch.Tensor(yb).long().cuda()

        racc = 0
        model.train()
        for i in range(n_val*n_it):
            #rand = torch.randperm(xb.shape[0])
            _, (xc, yc) = next(dset)
            xc, yc = xc.cuda(), yc.cuda()

            if not conv:
                xc = xc.view(xc.shape[0], -1)

            yc_ = model(xc)
            loss = F.cross_entropy(yc_, yc)

            opt.zero_grad()
            loss.backward()
            opt.step()

            racc += accuracy(yc_, yc)

            if i%n_val == 0:
                tacc = racc / n_val
                racc = 0
                print(tacc)

        xv,yv = xv.cpu(), yv.cpu()

        model.eval().cpu()
        yv_ = model(xv)
        acc = accuracy(yv_, yv)
        print('train_acc', tacc, 'final_acc:', acc)




def train_nearest(n_it=10, n_samples=10, n_samples_buffer=10, n_classes=100, n_val=100, n_times=50, conv=True, root='/data/chercheurs/siry191/data/', snap_path='/data/chercheurs/siry191/data/'):

    #source = cached_dataset(root=root, filename='tiny_imagenet.pk')
    #source = cached_dataset(root=root, filename='cifar100_cache.pk')
    source = cached_dataset(root=root, filename='micro_cifar100.pk')
    #source = cached_dataset(root=root, filename='mnist_cache.pk')
    #source = cached_dataset(root=root, filename='svhn_cache.pk')
    rracc_train = 0
    rracc_test  = 0

    pbar = tqdm(range(n_times))
    for t in pbar:

        x, y, xv, yv = source.get_non_stationary_sequence(max_seq=n_samples*n_classes, n_samples=n_samples, n_classes=n_classes)

        dico = {}

        for xi, yi in zip(x, y):
            try:
                dico[yi.item()].append(xi)
            except:
                dico[yi.item()] = [xi]
        
        xb, yb = [], []
        for k in dico.keys():
            xb = xb + dico[k][:n_samples_buffer]
            yb = yb + [k] * len(dico[k][:n_samples_buffer])

        xb = torch.stack(xb, 0)
        yb = torch.Tensor(yb).long()

        racc = 0

        proto_x = xb
        proto_y = yb

        test_x = xv
        test_y = yv

        racc = 0
        for k in range(test_x.shape[0]):
            x1 = proto_x.unsqueeze(1)
            x2 = test_x[k].unsqueeze(0).unsqueeze(0)
            d = (x1 - x2).pow(2).sum(-1).squeeze(-1)
            amx = torch.argmax(-d)
            y_ = proto_y[amx]
            racc += float(y_ == test_y[k])
        print(racc / test_x.shape[0])



train_iid(n_it=100, n_samples_buffer=500,  n_samples=500, n_classes=100, n_val=100, n_times=1, conv=True, data_path='/data/chercheurs/siry191/data/', snap_path='/data/chercheurs/siry191/data/')
#unsup_iid(n_samples=1, n_classes=500, n_val=100, n_times=100, conv=False, root='/data/chercheurs/siry191/data/', snap_path='/data/chercheurs/siry191/data/')

#train_iid(n_it=9000, n_samples_buffer=200, n_samples=200, n_classes=50, n_val=100, n_times=10, conv=False, root='/data/chercheurs/siry191/data/', snap_path='/data/chercheurs/siry191/data/')
#train_iid(n_it=9000, n_samples_buffer=12, n_samples=200, n_classes=50, n_val=100, n_times=10, conv=False, root='/data/chercheurs/siry191/data/', snap_path='/data/chercheurs/siry191/data/')

#train_iid(n_it=9000, n_samples_buffer=500, n_samples=500, n_classes=10, n_val=100, n_times=10, conv=False, root='/data/chercheurs/siry191/data/', snap_path='/data/chercheurs/siry191/data/')
#train_nearest(n_it=10000, n_samples_buffer=200,  n_samples=500, n_classes=50, n_val=100, n_times=10, conv=False, root='/data/chercheurs/siry191/data/', snap_path='/data/chercheurs/siry191/data/')
#train_iid(n_it=6000, n_samples_buffer=500, n_samples=500, n_classes=10, n_val=100, n_times=10, conv=False, root='/data/chercheurs/siry191/data/', snap_path='/data/chercheurs/siry191/data/')
