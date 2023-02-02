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

#curriculum meta-learning for the NEM model

def train_curriculum_NEM(data_path='/home/rodrigue/data', snapshot_path='/home/rodrigue/data/', exp_name='curriculum', starting_snapshot=None):

    seq_generator = SequenceGenerator(dump_path=data_path, filename='micro_cifar100.pk')

    logs = []

    #sequence length is gradually increased during meta-learning,
    # each increase is triggered when the model reaches 0.9 memorization accuracy
    #levels = [512, 256, 128, 64, 32]
    levels = [2, 10, 50, 100, 500, 1000, 2000, 5000, 10000, 20000]
    #levels = [2000, 5000, 10000, 20000]
    level = 0

    model = NEMModel(d_h=256, d_a=256)
    ma_model = NEMModel(d_h=256, d_a=256)

    if not starting_snapshot == None:
        state = torch.load(os.path.join(snapshot_path, starting_snapshot + '.pt'))
        model.load_state_dict(state['model'])

    model.cuda()
    ma_model.cuda()

    opt = optim.Adam(model.parameters(), lr=1e-4)

    if not starting_snapshot == None:
        opt.load_state_dict(state['opt'])

    max_gen_acc = 0
    #ma_model.load_state_dict(model.state_dict())

    while True:
        seq_len = levels[level]
        means = []
        means_len = 20
        n_samples = math.ceil(seq_len / 100)
        if n_samples == 0:
            n_samples = 1
            n_classes = seq_len
        else:
            n_classes = 100

        trunc_len = min(seq_len, 512)
        n_hidden = 512
        max_patience = 5
        patience = 0
        batch = 1
        n_val = math.ceil(100 / seq_len)

        logs.append({'trunc_len':trunc_len, 'n_hidden':n_hidden, 'n_classes':n_classes, 'n_samples':n_samples, 'seq_len':seq_len, 'train':[], 'val':[],'acc':[]})

        #iters per level
        running_train_loss = 0
        running_val_loss = 0

        i = 0
        k = 0

        while True:
            x, y, x_val, y_val = seq_generator.get_non_stationary_sequence(max_seq=seq_len, n_samples=n_samples, n_classes=n_classes, train=True)
            trunc_loss = 0
            seq_train_loss = 0
            seq_val_loss = 0

            model.train()
            model.init_base_model(256, 100, n_hidden, 3)

            for j, (xc, yc) in enumerate(zip(x, y)):
                xc, yc = xc.cuda(), yc.cuda()
                model.update(xc, yc)

                #r = random.randint(0, j)
                r = random.randint(0, x.shape[0]-1)
                xv = x[r].cuda()
                yv = y[r].cuda()

                train_loss, acc = model.inference(xv, yv)
                trunc_loss += train_loss
                seq_train_loss += train_loss.item()

                r = random.randint(0, x_val.shape[0]-1)
                xv = x_val[r].cuda()
                yv = y_val[r].cuda()

                val_loss, acc = model.inference(xv, yv)
                trunc_loss += val_loss
                seq_val_loss += val_loss.item()


                if (j%trunc_len==trunc_len-1) or j==x.shape[0]-1:
                    trunc_loss.backward()
                    trunc_loss = 0
                    model.detach_base_model()
                    k += 1
                    if (k%batch == 0):
                        nn.utils.clip_grad_norm_(model.parameters(), 1)
                        opt.step()
                        opt.zero_grad()
                        #ma_model.moving_average_update(model)
                        #model.load_state_dict(ma_model.state_dict())



            seq_train_loss = seq_train_loss / x.shape[0]
            seq_val_loss = seq_val_loss / x.shape[0]
            running_train_loss += seq_train_loss 
            running_val_loss += seq_val_loss 

            if i % n_val == n_val - 1:

                #testing round

                model.eval()
                model.init_base_model(256, 100, n_hidden, 3)

                x, y, xv, yv = seq_generator.get_non_stationary_sequence(max_seq=seq_len, n_samples=n_samples, n_classes=n_classes)
                for j, (xc, yc) in enumerate(zip(x, y)):
                    xc, yc = xc.cuda(), yc.cuda()
                    model.update(xc, yc)
                    model.detach_base_model()

                seq_train_acc = 0
                for j, (xc, yc) in enumerate(zip(x, y)):
                    xc, yc = xc.cuda(), yc.cuda()
                    loss, acc = model.inference(xc, yc)
                    seq_train_acc += acc
                seq_train_acc = seq_train_acc / x.shape[0]

                seq_val_acc = 0
                for j, (xc, yc) in enumerate(zip(xv, yv)):
                    xc, yc = xc.cuda(), yc.cuda()
                    loss, acc = model.inference(xc, yc)
                    seq_val_acc += acc
                seq_val_acc = seq_val_acc / xv.shape[0]



                if seq_val_acc > max_gen_acc:
                    max_gen_acc = seq_val_acc
                    torch.save({'model': model.state_dict(), 'opt': opt.state_dict()}, os.path.join(snapshot_path, exp_name+'_'+str(seq_len)+'.pt'))

                running_train_loss = running_train_loss / n_val
                running_val_loss = running_val_loss / n_val
                
                print(exp_name, 'SEQ-'+str(seq_len), 'meta-train-train:', running_train_loss, 'meta-train-test:', running_val_loss, 'meta-test-acc:', seq_train_acc, seq_val_acc)
                logs[-1]['train'].append(running_train_loss)
                logs[-1]['val'].append(running_val_loss)
                logs[-1]['acc'].append(seq_train_acc)
                save_pickle(logs, 'results/csv/'+exp_name+'.pk')

                running_train_loss = 0
                running_val_loss = 0

                means.append(seq_train_acc)
                if len(means)>means_len:
                    means.pop(0)
                avg = sum(means) / means_len

                if avg>0.9:
                    level += 1
                    level = min(level, len(levels)-1)
                    max_gen_acc = 0
                    print('level up -----------------------------------------------')
                    break
            i += 1

train_curriculum_NEM(exp_name='curriculum_norm')
