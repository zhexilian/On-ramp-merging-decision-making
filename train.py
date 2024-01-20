# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 19:51:19 2023

@author: 15834
"""

import numpy as np
from model import DIRL, QLoss, prob_calculation, plot_prob, plot_loss
from model import Net
import torch
from torch import nn, optim
import random
import torch.nn.functional as F
from sklearn import preprocessing
import argparse
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    # data preprocessed
    dataset = np.load("dataset.npy")
    min_max_scaler = preprocessing.MinMaxScaler()
    state_dataset = min_max_scaler.fit_transform(dataset[:,:13])
    action_dataset = min_max_scaler.fit_transform(dataset[:,13].reshape(-1, 1))

    # model train
    model = DIRL()
    criterion = QLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    action = [0,0.25,0.5,0.75,1]
    Input = torch.FloatTensor(state_dataset)
    label = torch.LongTensor(action_dataset)
    batchsize = args.batch_size
    epoch = args.epochs
    model.to(device)
    criterion.to(device)
    Loss, Prob = [], []
    for epoch in range(1,epoch+1):
        model.train()
        samples = [random.randint(0, len(label)-1) for _ in range(batchsize)]
        Input_sample = Input[[samples]]
        label_samples = label[[samples]]
        Input_sample, label_samples = Input_sample.to(device), label_samples.to(device)
        action0 = action[0] * torch.ones((batchsize, 1)).to(device)
        action1 = action[1] * torch.ones((batchsize, 1)).to(device)
        action2 = action[2] * torch.ones((batchsize, 1)).to(device)
        action3 = action[3] * torch.ones((batchsize, 1)).to(device)
        action4 = action[4] * torch.ones((batchsize, 1)).to(device)
        out0 = model(Input_sample,action0)
        out1 = model(Input_sample,action1)
        out2 = model(Input_sample,action2)
        out3 = model(Input_sample,action3)
        out4 = model(Input_sample,action4)
        loss = criterion(out0,out1,out2,out3,out4,label_samples)
        prob = prob_calculation(out0, out1, out2, out3, out4, label_samples)
        if epoch%10 == 0:
            print("第%d轮平均损失:%f"%(epoch,loss/batchsize))
            print("第%d轮专家动作概率对数:%f"%(epoch,prob))
        optimizer.zero_grad()
        # 初始化
        loss.backward()
        optimizer.step()
        Loss.append(loss.to('cpu').item())
        Prob.append(prob)
        if epoch%1000 == 0:
            plot_prob(epoch,Prob)
            plot_loss(epoch,Loss)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--epochs', type=int, help='training epochs (default: 10000)', default=10000)
    parser.add_argument('--batch_size', type=int, help='batch size (default:256)', default=256)
    parser.add_argument('--learning_rate', type=float, help='learning rate (default:0.0005)', default=0.0005)
    args = parser.parse_args()
    
    main(args)
