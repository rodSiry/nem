import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import math
import random
import numpy as np

def accuracy(y, Y):
    Y = Y.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    y = np.argmax(y, axis=-1)
    acc = (y == Y).astype(int)
    acc = torch.Tensor(acc).cuda()
    return acc

#tensor container for any mlp architecture, may be instantiated either as parameterdict or tensordict
def create_mlp_structure(batch, n_layers=2, n_in=1024, n_out=100, n_hidden=128, n_h=5, cuda=True, parameter=False, init_mode='network'):
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

        if init_mode=='zero':
            dico['l'+str(i)] = torch.zeros(batch, d_out, d_in) 
            dico['b'+str(i)] = torch.zeros(batch, d_out)

        elif init_mode=='noise':
            pos_l = torch.randn(batch//2, d_out, d_in) 
            pos_b = torch.randn(batch//2, d_out)
            dico['l'+str(i)] = torch.cat([pos_l, -pos_l], 0) 
            dico['b'+str(i)] = torch.cat([pos_b, -pos_b], 0) 


        elif init_mode=='network':
            dico['l'+str(i)] = torch.randn(batch, d_out, d_in) / math.sqrt(d_in)
            dico['b'+str(i)] = torch.zeros(batch, d_out)

    if cuda:
        for k in dico.keys():
            dico[k] = dico[k].cuda()

    if parameter:
        for k in dico.keys():
            dico[k] = nn.Parameter(dico[k])
        return nn.ParameterDict(dico)

    else:
        return dico


#handles parralel inner and outer updates of particles of the meta-model, and PES grad accumulation
class PESModel(nn.Module):

    def __init__(self, d_h=5, sigma=0.01, gamma=0.1):
        super().__init__()

        self.base_n_layers = 2
        self.base_n_hidden = 32
        self.base_n_in = 256

        self.meta_n_layers = 2
        self.meta_n_hidden = 5

        self.theta = create_mlp_structure(batch=1, n_layers=2, n_in=2, n_out=1, n_hidden=self.meta_n_hidden, parameter=True, init_mode='network')
        self.sigma = sigma
        self.gamma = gamma

    def init_inner_episode(self, n_particles=100):
        self.cumnoise = create_mlp_structure(batch=n_particles, n_layers=2, n_in=2, n_out=1, n_hidden=self.meta_n_hidden, init_mode='zero')
        self.cumgrad  = create_mlp_structure(batch=1, n_layers=2, n_in=2, n_out=1, n_hidden=self.meta_n_hidden, init_mode='zero')
        self.state    = create_mlp_structure(batch=n_particles, n_layers=2, n_in=self.base_n_in, n_out=100, n_hidden=self.base_n_hidden, init_mode='network')

    def inference_and_cumgrad(self, x, Y):

        n_particles = x.shape[0]

        #forward pass in base network
        with torch.no_grad():
            y = x.unsqueeze(-1)
            for i in range(self.base_n_layers):
                y = torch.bmm(self.state['l'+str(i)], y)
                y = y + self.state['b'+str(i)].unsqueeze(-1)

            y = y.squeeze(-1)
            loss = F.cross_entropy(y, Y, reduction='none')
            #loss = -accuracy(y, Y)

        #accumulate and place meta-grad
        for k in self.theta.keys():
            expand_size = [n_particles] + [1] * (len(self.cumgrad[k].shape) - 1)
            loss_expand = loss.view(expand_size)
            self.cumgrad[k] = self.cumgrad[k] + torch.sum(self.cumnoise[k] * loss_expand, 0).unsqueeze(0)
            self.cumgrad[k] = 1 / (self.sigma * self.sigma * n_particles) * self.cumgrad[k]
            self.theta[k].grad = self.cumgrad[k]

        return loss.mean().item()
 
    def update(self, x, Y):
        n_particles = x.shape[0]

        #forward pass in base network
        diff_list = []
        diff_list_keys = []

        y = x.unsqueeze(-1)

        for i in range(self.base_n_layers):
            self.state['l'+str(i)].requires_grad_()
            self.state['b'+str(i)].requires_grad_()

            diff_list.append(self.state['l'+str(i)])
            diff_list.append(self.state['b'+str(i)])
            diff_list_keys.append('l'+str(i))
            diff_list_keys.append('b'+str(i))

            y = torch.bmm(self.state['l'+str(i)], y)
            y = y + self.state['b'+str(i)].unsqueeze(-1)

        #grad computation in base network
        y = y.squeeze(-1)
        loss = F.cross_entropy(y, Y)
        grad = autograd.grad(loss, diff_list)
        diff_dict = dict(zip(diff_list_keys, grad))

        #particles update
        noise = create_mlp_structure(batch=n_particles, n_layers=2, n_in=2, n_out=1, n_hidden=self.meta_n_hidden, init_mode='noise')

        theta = {}
        for k in self.theta.keys():
            self.cumnoise[k] = self.cumnoise[k] + self.sigma * noise[k]
            theta[k] = torch.cat([self.theta[k]]*n_particles, 0) + self.sigma * noise[k]

        with torch.no_grad():
            new_state = {}
            for k in self.state.keys():
                base_tsr = self.state[k]
                base_tsr_grad = diff_dict[k]
                update_feed = torch.cat([base_tsr.unsqueeze(-1), base_tsr_grad.unsqueeze(-1)], -1).view(-1, 2)
                dw = update_feed.unsqueeze(-1)

                #update rule application
                for l in range(self.meta_n_layers):

                    shape_w = theta['l' + str(l)].shape
                    theta_w_expand = theta['l'+ str(l)].unsqueeze(1).expand(shape_w[0], int(update_feed.shape[0] / n_particles), shape_w[1], shape_w[2]).reshape(-1, shape_w[1], shape_w[2])

                    shape_b = theta['b' + str(l)].shape
                    theta_b_expand = theta['b'+ str(l)].unsqueeze(1).expand(shape_w[0], int(update_feed.shape[0] / n_particles), shape_w[1]).reshape(-1, shape_w[1])

                    dw = torch.bmm(theta_w_expand, dw)
                    dw = dw + theta_b_expand.unsqueeze(-1)
                    if l<self.meta_n_layers-1:
                        dw = F.leaky_relu(dw)

                dw = dw.view(base_tsr.shape)
                new_state[k] = base_tsr + torch.tanh(dw) * self.gamma

            self.state = new_state

"""
net = PESModel()
net.cuda()
x = torch.randn(10, 1024).cuda()
y = torch.ones(10).long().cuda()
net.init_inner_episode(n_particles=10)
state = net.update(x, y)
loss  = net.inference_and_cumgrad(x, y)
print(loss)
"""
