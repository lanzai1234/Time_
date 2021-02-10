import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

class feature_merge(nn.Module):
    def __init__(self):
        super(feature_merge, self).__init__()
        self.encoder = nn.Sequential(
            torch.nn.Conv2d(6,5,[3, 3], 1,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(5, 3, [3, 3], 1, padding=1),
            torch.nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            torch.nn.ConvTranspose2d(3, 5, [3, 3], 1, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(5, 6, [3, 3], 1, padding=1),
            torch.nn.ReLU(),
        )

    def forward(self,x):
        output = self.encoder(x)
        res = self.decoder(output)
        return  output,res

def merge(train_x, mode='channel'):
    train = []
    merge_res = []
    if mode == 'channel':
        for index in range(0,len(train_x),2):
            right = train_x[index]
            left = train_x[index]
            out = np.concatenate((right, left), axis=2)
            train.append(out)
        train = np.array(train)

    model = feature_merge()
    for item in train:
        item = Variable(item)
        out = model(item)
        merge_res.append(out.numpy())

    return  np.array([merge_res])

