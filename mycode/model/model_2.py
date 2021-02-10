import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F


class CNNnet(torch.nn.Moudle):
    def __init__(self):
        super().__init__()
        self.cov_30size = torch.Sequential(
            torch.nn.Conv2d(1, 16, [30, 30], 10),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(16, 32, [5, 5], 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
        )
        self.cov_20size = torch.Sequential(
            torch.nn.Conv2d(1, 16, [20, 20], 5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(16, 32, [5, 5], 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(32, 32, [3, 3], 2),
            torch.nn.ReLU()
        )
        self.cov_10size = torch.Sequential(
            torch.nn.Conv2d(1, 16, [10, 10], 5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(16, 32, [5, 5], 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(32, 32, [3, 3], 2),
            torch.nn.ReLU()
        )
        self.cov_5size = torch.Sequential(
            torch.nn.Conv2d(1, 16, [5, 5], 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(16, 32, [5, 5], 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(32, 32, [5, 5], 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(32, 32, [5, 5], 2),
            torch.nn.ReLU(),
        )
        self.dense = torch.Sequential(
            torch.nn.Linear(32*4*4, 120),
            torch.nn.ReLU(),
            torch.nn.Linear(120, 5),
            torch.nn.ReLU()
        )
        self.weight_W = nn.Parameter(torch.Tensor(4, 120))
        self.weight_proj = nn.Parameter(torch.Tensor(120, 1))
    
    def attention_net(self, x):
        u = torch.tanh(torch.matmul(x, self.weight_W))
        att = torch.matmul(u, self.weight_proj)
        att_score = F.softmax(att, dim=1)
        return att_score
        
    def forward(self, x):
        cov1_out = self.cov_30size(x)
        cov2_out = self.cov_20size(x)
        cov3_out = self.cov_10size(x)
        cov4_out = self.cov_5size(x)
        cov_out = torch.cat((cov1_out, cov2_out, cov3_out, cov4_out), 1)
        att_score = self.attention_net(cov_out)
        res = torch.matmul(cov_out, att_score)
        return res

