import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F


class CNNnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cov_30size = nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size = [30, 30], stride = 10),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(16, 32, [5, 5], 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2)
        )
        self.cov_20size = nn.Sequential(
            torch.nn.Conv2d(3, 16, [20, 20], 5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(16, 32, [5, 5], 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(32, 32, [3, 3], 2),
            torch.nn.ReLU()
        )
        self.cov_10size = nn.Sequential(
            torch.nn.Conv2d(3, 16, [10, 10], 5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(16, 32, [5, 5], 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(32, 32, [3, 3], 2),
            torch.nn.ReLU()
        )
        self.cov_5size = nn.Sequential(
            torch.nn.Conv2d(3, 16, [5, 5], 5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(16, 32, [5, 5], 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(32, 32, [3, 3], 2),
            torch.nn.ReLU()
        )
        self.dense = nn.Sequential(
            torch.nn.Linear(3200, 5),
            torch.nn.ReLU()
        )
        self.weight_W = nn.Parameter(torch.Tensor(40, 80))
        self.weight_proj = nn.Parameter(torch.Tensor(80, 80))
    
    def attention_net(self, x):
        u = torch.tanh(torch.matmul(x, self.weight_W))
        att = torch.matmul(u, self.weight_proj)
        att_score = F.softmax(att, dim=0)
        return att_score
        
    def forward(self, x):
        cov1_out = self.cov_30size(x)
        cov2_out = self.cov_20size(x)
        cov3_out = self.cov_10size(x)
        cov4_out = self.cov_5size(x)
        cov1_out = torch.reshape(cov1_out, (800,))
        cov2_out = torch.reshape(cov2_out, (800,))
        cov3_out = torch.reshape(cov3_out, (800,))
        cov4_out = torch.reshape(cov4_out, (800,))
        cov_out = torch.cat((cov1_out, cov2_out, cov3_out, cov4_out), 0)
        cov_out = torch.reshape(cov_out, (80, 40))
        att_score = self.attention_net(cov_out)
        att_out = torch.matmul(att_score, cov_out)
        att_out = torch.reshape(att_out, (3200, ))
        res = self.dense(att_out)
        res = torch.reshape(res, (1,5))
        return res

class CNN_net_30(nn.Module):
    def __init__(self):
        super().__init__()
        self.cov_30size = nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=[30, 30], stride=10),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(16, 32, [5, 5], 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2)
        )
        self.dense = nn.Sequential(
            torch.nn.Linear(800, 120),
            torch.nn.ReLU()
        )

    def forward(self, x):
        cov_out = self.cov_30size(x)
        cov_out = torch.reshape(cov_out, (800,))
        res = self.dense(cov_out)
        res = torch.reshape(res, (1, 120))
        return res;