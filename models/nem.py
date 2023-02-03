import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets.folder import *
from models.metrics import accuracy
import math
import random
import time
from torchvision import transforms as transforms

class Normalize(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        mu = x.mean(0, keepdim=True)
        sg = x.var(0, keepdim=True)
        y = (x - mu) / (torch.sqrt(sg + 1e-6))
        return y

#initialization of the base (fast) parameters dict
def create_base_model(n_layers=2, n_in=1024, n_out=100, n_hidden=128, d_hidden_state=256, cuda=True):
    dico = {}

    for i in range(n_layers):

        if i == 0:
            d_in  = n_in
            d_out = n_hidden

        elif i == n_layers-1:
            d_in  = n_hidden
            d_out = n_out

        else:
            d_in  = n_hidden
            d_out = n_hidden

        dico['l'+str(i)] = torch.randn(d_out, d_in) / math.sqrt(d_in)
        dico['b'+str(i)] = torch.zeros(d_out)
        dico['h'+str(i)] = torch.zeros(d_out, d_hidden_state)

    if cuda:
        for k in dico.keys():
            dico[k] = dico[k].cuda()

    dico['n_layers'] = n_layers
    dico['n_out'] = n_out
    dico['n_in']  = n_in

    return dico


#NEM update rule module
class NEMModel(nn.Module):
    def __init__(self, d_h=256, d_a=256):
        super().__init__()

        self.gamma = 0.001

        self.inner = nn.GRUCell(d_h + d_a, d_h)

        self.to_vec = nn.Sequential(
            nn.Linear(1, d_a),
            )

        self.to_first_msg = nn.Sequential(
            nn.Linear(1, d_h),
            )

        self.from_vec = nn.Sequential(
            nn.Linear(d_a, 1),
            )

        self.to_act = nn.Sequential(
            nn.LayerNorm([d_a + d_h]),
            nn.Linear(d_a+d_h, d_a), 
            nn.LeakyReLU(),
            )

        self.to_update_next = nn.Sequential(
            nn.Linear(d_h, d_h),
            )

        self.to_update_prev = nn.Sequential(
            nn.Linear(d_h, d_h),
            )

        self.d_hidden_state = d_h
        self.d_activation   = d_a

    def moving_average_update(self, cur_value, tau=0.999):
        for p, cp in zip(self.parameters(), cur_value.parameters()):
            p.data = tau * p.data + (1 - tau) * cp.data

    def init_base_model(self, base_n_in, base_n_out, base_n_hidden, base_n_layers):
        self.state = create_base_model(n_in=base_n_in, n_out=base_n_out, n_hidden=base_n_hidden, n_layers=base_n_layers, d_hidden_state=self.d_hidden_state, cuda=True)

    def detach_base_model(self):
        for k in self.state.keys():
            try:
                self.state[k] = self.state[k].detach()
            except:
                pass

    def inference(self, x, Y, dropout=False):

        x = x.unsqueeze(-1)
        y = self.to_vec(x)

        if dropout:
            drop_layer = random.randint(0, self.state['n_layers'] - 1)

        for i in range(self.state['n_layers']):

            w = self.state['l'+str(i)]
            b = self.state['b'+str(i)]
            h = self.state['h'+str(i)]

            if dropout:
                if i == drop_layer:
                    b = torch.bernoulli(torch.ones(y.shape[0], 1).cuda() * 0.9)
                    y = y * b

            y = torch.matmul(w, y) #/ math.sqrt(y.shape[-1])
            y = self.to_act(torch.cat([y, h], -1)) + y

        y = self.from_vec(y)
        loss = F.cross_entropy(y.squeeze(-1).unsqueeze(0), Y.unsqueeze(-1))
        acc = accuracy(y.squeeze(-1).unsqueeze(0), Y.unsqueeze(-1))
        return loss, acc

    def update(self, x, Y, dropout=False):

        #forward pass
        x = x.unsqueeze(-1)
        y = self.to_vec(x)

        if dropout:
            drop_layer = random.randint(0, self.state['n_layers'] - 1)

        m_prev = []
        for i in range(self.state['n_layers']):

            w = self.state['l'+str(i)]
            b = self.state['b'+str(i)]
            h = self.state['h'+str(i)]

            if dropout:
                if i == drop_layer:
                    b = torch.bernoulli(torch.ones(y.shape[0], 1).cuda() * 0.9)
                    y = y * b

            y = torch.matmul(w, y) #/ math.sqrt(y.shape[-1])
            m_prev.append(y)
            y = self.to_act(torch.cat([y, h], -1)) + y

        #backward pass
        m_next = torch.zeros(self.state['n_out'], self.d_hidden_state).cuda()
        m_next[Y] = 1

        for i in range(self.state['n_layers']):

            j = self.state['n_layers'] - i - 1
            update_feed = torch.cat([m_next, m_prev[j]], -1)
            self.state['h'+str(j)] = self.inner(update_feed, self.state['h'+str(j)])
            m_next = torch.matmul(self.state['l'+str(j)].t(), self.state['h'+str(j)]) #/ math.sqrt(m_next.shape[-1])

        #weight update
        s_prev = self.to_first_msg(x).squeeze(0)
        for i in range(self.state['n_layers']):
            s_next = self.state['h'+str(i)]
            p_next = self.to_update_next(s_next)
            p_prev = self.to_update_prev(s_prev)
            p_next = p_next / torch.sqrt(p_next.pow(2).sum(-1).unsqueeze(-1) + 1e-6)
            p_prev = p_prev / torch.sqrt(p_prev.pow(2).sum(-1).unsqueeze(-1) + 1e-6)
            dw = torch.matmul(p_next, p_prev.t())
            self.state['l'+str(i)] = self.state['l'+str(i)] + self.gamma * dw
            #self.state['l'+str(i)] = self.state['l'+str(i)] / torch.sqrt(self.state['l'+str(i)].pow(2).sum(-1).unsqueeze(-1) + 1e-6)
            s_prev = s_next


"""
x = torch.randn(256).cuda()
y = torch.ones(1).long().cuda()
model = NEMModel()
model.cuda()
model.init_base_model(256, 100, 1024, 3)
model.update(x, y)
"""
