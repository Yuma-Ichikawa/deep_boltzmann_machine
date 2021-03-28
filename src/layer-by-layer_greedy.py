# Import modules
import numpy as np
import pandas as pd
import scipy 
import time
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader
from torchvision.utils import make_grid
from torch import nn, optim

class pre_train_RBM(nn.Module):
        """
        a layer-by-layer greedy learning algorithm with RBM
        Params
        b : bias of visible unit
        c : bias of hidden unit
        w : weight
        k : Sampling Times 
        """
        def __init__(self, n_vis=784, n_hid=128, k=15, epoch=25, learning_rate=0.001, batch_size=100, initial_std=0.01, seed=0, verbose=0, device='cpu'):
            super(pre_train_RBM, self).__init__()
            self.n_hid=n_hid
            self.n_vis=n_vis
            self.device = device
            self.w = torch.empty((n_hid, n_vis), device=device).normal_(mean=0, std=initial_std)
            self.k = k
            self.epoch = epoch
            self.batch_size=batch_size
            self.learning_rate = learning_rate
            self.verbose=verbose
        
        def visible_to_hidden(self, v):
            """
            sampling hidden units from visible units for layers
            """
            p = torch.sigmoid(torch.mm(v, self.w.t()))
            return p.bernoulli()

        def two_visible_to_hidden(self, v):
            """
            sampling hidden units from visible units for 1st layer
            """
            p = torch.sigmoid(2*(torch.mm(v, self.w.t())))
            return p.bernoulli()        

        def visible_to_hidden_prob(self, v):
            """
            compute P(h=1 | v) for l layers
            """
            return torch.sigmoid(torch.mm(v, self.w.t()))

        def two_visible_to_hidden_prob(self, v):
            """
            compute P(h=1 | v) for L layer
            """
            return torch.sigmoid(2*(torch.mm(v, self.w.t())))

        def hidden_to_visible(self, h):
            """
            sampling visible units from hidden units for l layers
            """
            p = torch.sigmoid(torch.mm(h, self.w))
            return p.bernoulli()

        def two_hidden_to_visible(self, h):
            """
            sampling visible units from two hidden units for l layers
            """
            p = torch.sigmoid(2*(torch.mm(h, self.w)))
            return p.bernoulli()

        def free_energy(self, v):
            """
            Caluculating Free Energy
            """
            w_x_h = torch.mm(v, self.w.t())
            h_term = torch.sum(F.softplus(w_x_h), dim=1)
            return torch.mean(-h_term)
 
        def loss(self, v):
            """
            Caluculating Pseudo-likelihood 
            """
            flip = torch.randint(0, v.size()[1], (1,))
            v_fliped = v.clone()
            v_fliped[:, flip] = 1 - v_fliped[:, flip]
            free_energy = self.free_energy(v)
            free_energy_fliped = self.free_energy(v_fliped)
            return  v.size()[1]*F.softplus(free_energy_fliped - free_energy)

        def batch_fit_visible(self, v_pos):
            """
            1step pretrain visible units and 1st hidden layer
            """        
            ph_pos = self.two_visible_to_hidden_prob(v_pos)
            # Caluculate negative term
            v_neg = self.hidden_to_visible(self.h_samples)
            ph_neg = self.two_visible_to_hidden_prob(v_neg)

            lr = (self.learning_rate) / v_pos.size()[0]
            # Update W
            update = torch.matmul(ph_pos.t(), v_pos) - torch.matmul(ph_neg.t(), v_neg)
            self.w += lr * update
            # memory PCD method
            self.h_samples = ph_neg.bernoulli()

        def batch_fit_last_hidden(self, v_pos):
            """
            1step pretrain L-1 hidden layer and L hidden layer
            """        
            ph_pos = self.visible_to_hidden_prob(v_pos)
            # negative part
            v_neg = self.two_hidden_to_visible(self.h_samples)
            ph_neg = self.visible_to_hidden_prob(v_neg)

            lr = (self.learning_rate) / v_pos.size()[0]
            # Update W
            update = torch.matmul(ph_pos.t(), v_pos) - torch.matmul(ph_neg.t(), v_neg)
            self.w += lr * update
            # memory PCD method
            self.h_samples = ph_neg.bernoulli()
        
        def batch_fit(self, v_pos):
            """
            1step pretrain the other layers
            """        
            ph_pos = self.visible_to_hidden_prob(v_pos)
            # negative part
            v_neg = self.hidden_to_visible(self.h_samples)
            ph_neg = self.visible_to_hidden_prob(v_neg)

            lr = (self.learning_rate) / v_pos.size()[0]
            # Update W
            update = torch.matmul(ph_pos.t(), v_pos) - torch.matmul(ph_neg.t(), v_neg)
            self.w += lr * update
            # memory PCD method
            self.h_samples = ph_neg.bernoulli()
        
        def fit_1st_layer(self, train_loader):
            """
            Pretrain visible units and 1st hidden layer
            """
            self.losses = []
            # Initialize hidden units
            self.h_samples = torch.zeros(self.batch_size, self.n_hid).to(self.device)
            start_pretrain = time.time()
            for epoch in range(self.epoch):
                running_loss = 0.0
                verbose = self.verbose
                start = time.time()
                for id, (data, target) in enumerate(train_loader):
                    data = data.to(self.device)
                    target = target.to(self.device)
                    self.batch_fit_visible(data)
                    running_loss += self.loss(data)
                self.losses.append(running_loss/self.n_vis)
                if verbose:
                    end = time.time()
                    print(f'Epoch {epoch+1} pseudo-likelihood = {(running_loss/self.n_vis):.4f}, {end-start:.2f}s')
            end_pretrain = time.time()
            print(f'pretrain time : {start_pretrain - end_pretrain:.2f} s')
            
        def fit(self, train_loader):
            """
            Pretrain L-1 hidden layer and L hidden layer
            """
            self.losses = []
            # Initialize hidden units
            self.h_samples = torch.zeros(self.batch_size, self.n_hid).to(self.device)
            start_pretrain = time.time()
            for epoch in range(self.epoch):
                running_loss = 0.0
                verbose = self.verbose
                start = time.time()
                for id, (data, target) in enumerate(train_loader):
                    data = data.to(self.device)
                    target = target.to(self.device)
                    self.batch_fit(data)
                    running_loss += self.loss(data)
                self.losses.append(running_loss/self.n_vis)
                if verbose:
                    end = time.time()
                    print(f'Epoch {epoch+1} pseudo-likelihood = {(running_loss/self.n_vis):.4f}, {end-start:.2f}s')
            end_pretrain = time.time()
            print(f'pretrain time : {start_pretrain - end_pretrain:.2f} s')

        def fit_last_layer(self, train_loader):
            """
            Pretrain the other layers
            """
            self.losses = []
            # Initialize hidden units
            self.h_samples = torch.zeros(self.batch_size, self.n_hid).to(self.device)
            start_pretrain = time.time()
            for epoch in range(self.epoch):
                running_loss = 0.0
                verbose = self.verbose
                start = time.time()
                for id, (data, target) in enumerate(train_loader):
                    data = data.to(self.device)
                    target = target.to(self.device)
                    self.batch_fit_last_hidden(data)
                    running_loss += self.loss(data)
                self.losses.append(running_loss/self.n_vis)
                if verbose:
                    end = time.time()
                    print(f'Epoch {epoch+1} pseudo-likelihood = {(running_loss/self.n_vis):.4f}, {end-start:.2f}s')
            end_pretrain = time.time()
            print(f'pretrain time : {start_pretrain - end_pretrain:.2f} s')