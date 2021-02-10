import numpy as np
import torch
import torch.nn as nn
# import torch.nn.Function as F

F_ = np.load("./RobotDataset/20mins_fault_data.npy")
N_ = np.load("./RobotDataset/20mins_normal_data.npy")

WinDow_size = 500
stride = 5

X = torch.tensor(N_).t()
# Y = X[: , torch.arange(WinDow_size, X.shape[1], stride)]


class Linear_model(nn.Module):
    def __init__(self, scale, win_size):
        super(Linear_model, self).__init__()
        self.W1 = torch.nn.Parameter(torch.randn(scale, win_size))
        self.W2 = torch.nn.Parameter(torch.randn(scale, win_size))
        self.LeakyReLU = torch.nn.LeakyReLU()
        self.scale = scale
        self.win_size = win_size

    def forward(self, x):
        mid_res = x * self.W1
        mid_res = self.LeakyReLU(mid_res)
        mid_res = mid_res * self.W2
        mid_res = self.LeakyReLU(mid_res)
        output = mid_res.sum(dim=0)
        return output


model = Linear_model(scale=X.shape[0], win_size=WinDow_size)
Learning_Rate = 0.05
loss_fn = nn.MSELoss(reduction="mean")
optimizer = torch.optim.Adam(model.parameters(), lr=Learning_Rate)

for ep in range(500):
    for start in range(0, X.shape[1] - WinDow_size, stride):
        end = start + WinDow_size
        batch_X = X[:, start:end]
        batch_Y = X[:, end:end + 1]

        Y_hat = model(batch_X)
        loss = loss_fn(Y_hat, batch_Y)

        print(ep, loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()