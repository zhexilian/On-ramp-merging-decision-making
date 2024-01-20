# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 20:39:32 2023

@author: 15834
"""

import torch 
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DIRL(nn.Module):
    def __init__(self):
        super(DIRL, self).__init__()
        self.fc1 = nn.Linear(13,20)
        self.fc2 = nn.Linear(20,32)
        self.fc21 = nn.Linear(32,64)
        self.fc22 = nn.Linear(64,32)
        self.fc3 = nn.Linear(32,20)
        self.fc4 = nn.Linear(20,3)
        self.fc5 = nn.Linear(4,10)
        self.fc6 = nn.Linear(10,32)
        self.fc7 = nn.Linear(32,10)
        self.out = nn.Linear(10, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.25)
    def forward(self, s,a):
        s = F.relu(self.fc1(s))
        s = F.relu(self.fc2(s))
        s = F.relu(self.fc21(s))
        s = F.relu(self.fc22(s))
        s = F.relu(self.fc3(s))
        s = F.relu(self.fc4(s))
        Q = F.relu(self.fc5(torch.cat((s, a), dim=1)))
        Q = F.relu(self.fc6(Q))
        Q = F.relu(self.dropout(self.fc7(Q)))
        Q = self.out(Q)
        return Q


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.hidden2 = torch.nn.Linear(n_hidden,32)
        self.hidden3 = torch.nn.Linear(32,n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)
        
    def forward(self,x):
        x = F.relu(self.hidden(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = self.out(x)
        return x

class QLoss(nn.Module):
    def __init__(self):
        super(QLoss, self).__init__()
    def forward(self,out0,out1,out2,out3,out4,label): 
        exp_out0 = torch.exp(out0)
        exp_out1 = torch.exp(out1)
        exp_out2 = torch.exp(out2)
        exp_out3 = torch.exp(out3)
        exp_out4 = torch.exp(out4)
        #求和取对数
        y1 = torch.log(exp_out0 + exp_out1 + exp_out2 + exp_out3 + exp_out4) 
        
        y2 = 0 * torch.ones((len(label),1))
        for i in range(len(label)):
            if label[i] == 0:
                y2[i] = out0[i]
            elif label[i] == 0.25:
                y2[i] = out1[i]
            elif label[i] == 0.5:
                y2[i] = out2[i]
            elif label[i] == 0.75:
                y2[i] = out3[i]
            else:
                y2[i] = out4[i]
        
        return torch.sum(y1) - torch.sum(y2)
        

def prob_calculation(out0,out1,out2,out3,out4,label):
    exp_out0 = torch.exp(out0)
    exp_out1 = torch.exp(out1)
    exp_out2 = torch.exp(out2)
    exp_out3 = torch.exp(out3)
    exp_out4 = torch.exp(out4)
    y2 = 0 * torch.ones((len(label),1)).to(device)
    for i in range(len(label)):
        if label[i] == 0:
            y2[i] = out0[i]
        elif label[i] == 0.25:
            y2[i] = out1[i]
        elif label[i] == 0.5:
            y2[i] = out2[i]
        elif label[i] == 0.75:
            y2[i] = out3[i]
        else:
            y2[i] = out4[i]
    
    prob = torch.log(torch.exp(y2)/(exp_out0 + exp_out1 + exp_out2 + exp_out3 + exp_out4))
    return torch.mean(prob).item()
    
def plot_prob(epoch,Prob):
    fig = plt.figure(figsize=(32,20))
    x = np.linspace(1,epoch,epoch)
    plt.plot(x,Prob[:epoch+1],"r-",lw=1)
    plt.yticks(fontsize=36)
    plt.xticks(fontsize=36)
    plt.xlabel("epoch",fontsize=36)
    plt.ylabel("log(P)",fontsize=36)
    plt.grid()
    plt.show()
    plt.close()    
    
def plot_loss(epoch,Loss):
    fig = plt.figure(figsize=(32,20))
    x = np.linspace(1,epoch,epoch)
    plt.plot(x,Loss[:epoch+1],"b--",lw=1)
    plt.yticks(fontsize=36)
    plt.xticks(fontsize=36)
    plt.xlabel("epoch",fontsize=36)
    plt.ylabel("Loss",fontsize=36)
    plt.grid()    
    plt.show()
    plt.close()