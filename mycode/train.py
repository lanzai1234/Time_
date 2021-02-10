import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from preprocess import merge
from torch.autograd import Variable
from model import CNNnet, CNN_net_30
from Dataloader import walkFile
from tqdm import tqdm
from model import Gat
import matplotlib.pyplot as plt
from model import Lstm_, LSTM_autoencoder

F = (np.load("./RobotDataset/20mins_fault_data.npy"))


def Linear_train():
    model = CNNnet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_func = nn.CrossEntropyLoss()

    train_x, train_y = walkFile("G:\牛梦毫_zy1906134_医疗影像计算大作业\mycode\dataset\\train")
    train_x = merge(train_x)
    for epoches in range(10):
        print('epoch{}'.format(epoches + 1))
        # train_loss = 0.
        # train_acc = 0.
        train_loader = zip(train_x, train_y)
        for batch_x, batch_y in train_loader:
            batch_x = torch.tensor(batch_x, dtype=torch.float32)
            batch_x = torch.reshape(batch_x, (1, 3, 512, 512))
            batch_y = torch.tensor(batch_y).long()
            batch_x, batch_y = Variable(batch_x), Variable(batch_y)
            output = model(batch_x)
            loss = loss_func(output, batch_y)
            # train_loss = loss.item()
            # pred = torch.max(output, 1)[1]
            # train_correct = (pred == batch_y).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print('%.30f'%train_loss)
            print(output.retain_grad())

"""
test_loader = zip(train_x, train_y)
for batch_x, batch_y in test_loader:
    batch_x = torch.tensor(batch_x, dtype=torch.float32)
    batch_x = torch.reshape(batch_x, (1, 3, 512, 512))
    batch_y = torch.tensor(batch_y).long()
    batch_x, batch_y = Variable(batch_x), Variable(batch_y)
    output = model(batch_x)
    print(output.detach().numpy())
"""


def standerize(a):
    max_v, max_index = torch.max(a, 0)
    # print(max_v)
    min_v, min_index = torch.min(a, 0)
    a = (a - min_v)/(max_v - min_v)
    return a


def Reverse_Standrize(a, b):
    max_v, max_index = torch.max(a, 0)
    min_v, min_index = torch.min(a, 0)
    b = b * (max_v - min_v) + min_v
    return b


def Gat_Train():
    in_feature = 50
    out_feature = 50
    node_num = 18
    dropout = 0.5
    alpha = 0.2
    X = (standerize(torch.tensor(F).to(torch.float32))).t()

    model = Gat(in_feature, out_feature, out_feature, dropout, alpha, node_num)
    
    Learning_Rate = 0.05
    optimizer = torch.optim.Adam(model.parameters(), lr=Learning_Rate)

    loss_fn = nn.MSELoss(reduction="mean")
    WinDow_size = in_feature
    stride = 5
    for ep in range(500):
        for start in range(0, X.shape[1] - WinDow_size, stride):
            end = start + WinDow_size
            batch_X = X[:, start:end]
            batch_Y = X[:, end:end + 1]
            # batch_Y = standerize(batch_Y)
            Adj, Y_hat = model(batch_X)
            loss = loss_fn(Y_hat, batch_Y)

            print(ep, loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        

def LSTM_train(input_size=18, hidden_size=18, num_layers=1, batch_first=False, bias=True, dropout=0.1, bidirectional=False):
    data = torch.tensor(F).to(torch.float32)
    #std_data = standerize(data)
    model = Lstm_(18, 1)
    X_train = data[0:18 * 18 * 300].view(-1, 1, 18, input_size)
    Y_train = data.view(-1, 1, 1, hidden_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    loss_fn = nn.MSELoss()

    for ep in range(500):
        for index in range(X_train.size()[0]):
            X = X_train[index]
            Y = Y_train[index * 18 + 1]
            Y_hat = model(X)
            loss = loss_fn(Y_hat, Y)

            print(ep, loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    X_test = data[10000: 10500].view(-1, 1, input_size)
    Y_test = data[10500: 11000]
    predict, _ = model(X_test)
    predict = Reverse_Standrize(data[10000: 10500], predict.squeeze(1))
    print(predict, Y_test)
    """
    plt.plot(range(500), Y_test[:, 0], label="real_world")
    plt.plot(range(500), predict[:, 0].detach().numpy(), label="predict")
    plt.legend()
    plt.show()
    """


def LSTM_autoencoder_train(w=1500, stride=1):
    data = torch.tensor(F[:, 0]).to(torch.float32)
    learning_Rate = 1e-2
    model = LSTM_autoencoder(w, 64)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_Rate)

    for ep in range(500):
        for start in range(data.size()[0] - w):
            end = start + w
            X = data[start:end].view(1, 1, w)
            Y_hat = model(X)

            loss = loss_fn(Y_hat, X)   
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(ep, loss.item())


if __name__ == "__main__":
    LSTM_autoencoder_train()