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


class DBM_3layer:
    def __init__(self, n_vis=784, n_hid1=128, n_hid2=64, epoch=10, 
                mean_iter = 100, gibbs_iter=5, learning_rate=0.01, 
                batch_size=100, initial_std=0.01, pretrain=True, device='cpu'):
        """
        Parameters
        ---------------
        learning_rate : 学習率(learing rate)
        n_vis = 784 : 可視ユニット数(number of visible units)
        n_hid1 : １層目の隠れユニット数(number of 1-layer hidden units)
        n_hid2 : ２層目の隠れユニット数(number of 2-layer hidden units)
        epoch : エポック数(epch)
        mean_iter : 反復方程式のiteration数(Iteration of meanfield equation)
        gibbs_iter : vをサンプルときのギブスサンプリンングのモンテカルロステップ(Iteration of gibbs sampling)
        """
        self.n_vis = n_vis
        self.n_hid1 = n_hid1
        self.n_hid2 = n_hid2
        self.device = device
        self.w1 = torch.empty((n_hid1, n_vis), device=self.device).normal_(mean=0, std=initial_std)
        self.w2 = torch.empty((n_hid2, n_hid1), device=self.device).normal_(mean=0, std=initial_std)
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.mean_iter = mean_iter
        self.gibbs_iter = gibbs_iter
        self.batch_size = batch_size
        self.pretrain = pretrain

    def rbm_learning_layers(self,train_loader, v, n_hid):
        """Learning a single RBM
        Returns
        -----------
        out : 特徴点(Feature points)
        w : 学習後の重み(Pre-learned weight)
        """
        # Instance pre_train_rbm
        v = v.to(self.device)
        rbm =  pre_train_RBM(n_vis=v.size()[1], n_hid=n_hid, learning_rate=0.01, verbose=False, device=self.device)
        # Learning RBM 
        rbm.fit(train_loader)
        w = rbm.w
        feature = rbm.visible_to_hidden(v)
        return feature, w

    def rbm_learning_1st_layer(self, train_loader, v, n_hid):
        """Learning a single RBM
        Returns
        -----------
        out : 特徴点(Feature points)
        w : 学習後の重み(Pre-learned weight)
        """
        v = v.to(self.device)
        # Instance pre_train_rbm
        rbm =  pre_train_RBM(n_vis=v.size()[1], n_hid=n_hid, learning_rate=0.01, verbose=False, device=self.device)
        # Learning RBM 
        rbm.fit_1st_layer(train_loader)
        w = rbm.w
        feature = rbm.visible_to_hidden(v)
        return feature, w

    def rbm_learning_last_layer(self,train_loader, v, n_hid):
        """Learning a single RBM
        Returns
        -----------
        out : 特徴点(Feature points)
        w : 学習後の重み(Pre-learned weight)
        """
        v = v.to(self.device)
        # Instance pre_train_rbm
        rbm =  pre_train_RBM(n_vis=v.size()[1], n_hid=n_hid, learning_rate=0.01, verbose=False, device=self.device)
        # Learning RBM 
        rbm.fit_last_layer(train_loader)
        w = rbm.w
        feature = rbm.visible_to_hidden(v)
        return feature, w

    def making_loader(self, feature):
        """making Pytorch Datasets
        Returns
        ------------
        out : Pytorch dataloader
        """
        ds_train = TensorDataset(feature, y_train)
        train_loader = DataLoader(ds_train, batch_size=100, shuffle=True)
        return train_loader

    def pre_train(self, train_loader, v):
        """Greedy layerwise pretrainig
        Parameter
        --------------
        train_loader : Dataloaderをセットする(Set Dataloader)
        v : Dataloaderにする前のデータを入れる(Set samples before making Dataloader)
        Return
        --------------
        w1, w2 : Pre-trained weights
        """
        v = v.to(self.device)
        feature1, w1= self.rbm_learning_1st_layer(train_loader, v, n_hid=self.n_hid1)
        feature1 = feature1.to(self.device)
        feature1_loader = self.making_loader(feature1)
        feature2, w2 = self.rbm_learning_last_layer(feature1_loader, feature1, n_hid=self.n_hid2)
        feature2 = feature2.to(self.device)
        return w1, w2 
    
    def sample_v(self, v, gibbs=100):
        """Sampling visible units
        Return
        ----------
        v : visible units
        """
        v = v.to(self.device)
        # 初期値は0とする
        self.v = v.to(self.device)
        ph1 = torch.zeros(v.size()[0], self.n_hid1).to(self.device)
        ph2 = torch.zeros(v.size()[0], self.n_hid2).to(self.device)

        for i in range(gibbs):
            ph1 = torch.sigmoid(torch.mm(self.v, self.w1.t()) + torch.mm(self.h2, self.w2))
            self.h1 = ph1.bernoulli()
            ph2 = torch.sigmoid(torch.mm(self.h1, self.w2.t()))
            self.h2 = ph2.bernoulli()
            ph1 = torch.sigmoid(torch.mm(self.v, self.w1.t()) + torch.mm(self.h2, self.w2))
            self.h1 = ph1.bernoulli()
            pv = torch.sigmoid(torch.mm(self.h1, self.w1))
            self.v = pv.bernoulli()
        return v, self.v
    
    def sample_all_units(self, v, gibbs=100):
        """Sampling All units
        Parameters
        --------------
        v : sample data
        gibbs : モンテカルロステップ(MCS)
        Returns
        -----------
        v_gibb : 可視ユニットのサンプリング(sampling visible units)
        h1_gibb : １層目の隠れユニットのサンプリング(sampling 1st-layer hidden units)
        h2_gibb : 2層目の隠れユニットのサンプリング(sampling 2nd-layer hideen units)
        """
        self.v = v.to(self.device)
        ph1 = torch.zeros(v.size()[0], self.n_hid1).to(self.device)
        ph2 = torch.zeros(v.size()[0], self.n_hid2).to(self.device)

        for i in range(gibbs):
            ph1 = torch.sigmoid(torch.mm(self.v, self.w1.t()) + torch.mm(self.h2, self.w2))
            self.h1 = ph1.bernoulli()
            ph2 = torch.sigmoid(torch.mm(self.h1, self.w2.t()))
            self.h2 = ph2.bernoulli()
            ph1 = torch.sigmoid(torch.mm(self.v, self.w1.t()) + torch.mm(self.h2, self.w2))
            self.h1 = ph1.bernoulli()
            pv = torch.sigmoid(torch.mm(self.h1, self.w1))
            self.v = pv.bernoulli()
        v_gibb = self.v
        h1_gibb = self.h1
        h2_gibb = self.h2
        return v_gibb, h1_gibb, h2_gibb


    def mean_field_eq(self, v):
        """mean field equation 
        Parameters
        --------------
        v : sample data
        Returns
        ---------
        grad_pos1 : 一層目の隠れユニットのPositive Part(Positive part of 1st-layer hidden units)
        grad_pos2 : 二層目の隠れユニットのPositive Part(Positive part of 2nd-layer hidden units)
        """
        v = v.to(self.device)
        self.m1 = torch.zeros(v.size()[0], self.n_hid1).to(self.device)
        self.m2 = torch.zeros(v.size()[0], self.n_hid2).to(self.device)
        # Iteration of mean field equation
        for i in range(self.mean_iter):
            self.m1 = torch.sigmoid(torch.mm(v, self.w1.t()) + torch.mm(self.m2, self.w2))
            self.m2 = torch.sigmoid(torch.mm(self.m1, self.w2.t()))
        grad_pos1 = torch.mm(self.m1.t(), v) / v.shape[0]
        grad_pos2 = torch.mm(self.m2.t(), self.m1) / v.shape[0]
        return grad_pos1, grad_pos2

    def gibbs_sampling(self, v, train_gibbs_num=1):
        """
        Caluculating Negative Part
        Parameters
        --------------
        train_gibbs_num : 訓練の際のモンテカルロステップ数(MCS for training)
        v : sample data
        Returns 
        -------------
        grad_neg1 : 一層目の隠れユニットのNegative Part(Negative part of 1st-layer hidden units)
        grad_neg2 : 二層目の隠れユニットのNegative Part(Negative part of 2nd-layer hidden units)
        """
        v = v.to(self.device)
        gibb_v = torch.zeros(v.size()[0], v.size()[1]).to(self.device)
        gibb_ph1 = torch.zeros(v.size()[0], self.n_hid1).to(self.device)
        gibb_ph2 = torch.zeros(v.size()[0], self.n_hid2).to(self.device)

        self.v = v
        for i in range(train_gibbs_num):
            gibb_ph1 = torch.sigmoid(torch.mm(v, self.w1.t()) + torch.mm(self.h2, self.w2))
            self.h1 = gibb_ph1.bernoulli()
            gibb_ph2 = torch.sigmoid(torch.mm(self.h1, self.w2.t()))
            self.h2 = gibb_ph2.bernoulli()
            gibb_ph1 = torch.sigmoid(torch.mm(v, self.w1.t()) + torch.mm(self.h2, self.w2))
            self.h1 = gibb_ph1.bernoulli()
            gibb_pv = torch.sigmoid(torch.mm(self.h1, self.w1))
            self.v = gibb_pv.bernoulli()
            gibb_v = self.v
        
        # caluculate negative part
        grad_neg1 = torch.mm(gibb_ph1.t(), gibb_v) / (train_gibbs_num*v.shape[0])
        grad_neg2 = torch.mm(gibb_ph2.t(), gibb_ph1) / (train_gibbs_num*v.shape[0])
        return grad_neg1, grad_neg2

    def fit(self, v, train_loader, train_gibbs_num=50):
        """Learning DBM
        Prameters
        ------------
        v : sample data
        train_loader : 訓練データのDataloader(Dataloader for Training)
        """
        v = v.to(self.device)
        self.h2 = torch.zeros((v.size()[0], self.n_hid2), device=device)
        self.h1 = torch.zeros((v.size()[0], self.n_hid1), device=device)
        self.v = torch.zeros((v.size()[0], self.n_vis), device=device)
        # Pretrain
        if self.pretrain:
            w1, w2 = self.pre_train(train_loader, v)
            self.w1, self.w2 = w1, w2
        # Caluculate First update which iterate the gibbs sampling untile steady state 
        start_first = time.time()
        grad_pos1, grad_pos2 = self.mean_field_eq(v)
        grad_neg1, grad_neg2 = self.gibbs_sampling(v, train_gibbs_num = train_gibbs_num)
        self.w1 += self.learning_rate * (grad_pos1 - grad_pos1)
        self.w2 += self.learning_rate * (grad_pos2 - grad_neg2)
        end_first = time.time()
        print(f'【EPOCH】 : 1, train time : {start_first - end_first:.2f} s')
        # epoch number
        epoch_number = 2
        for i in range(self.epoch-1):
            start = time.time()
            grad_pos1, grad_pos2 = self.mean_field_eq(v)
            grad_neg1, grad_neg2 = self.gibbs_sampling(self.v)
            self.w1 += self.learning_rate * (grad_pos1 - grad_pos1)
            self.w2 += self.learning_rate * (grad_pos2 - grad_neg2)
            end = time.time()
            print(f'【EPOCH】 : {epoch_number}, train time : {start - end:.2f} s')
            epoch_number += 1