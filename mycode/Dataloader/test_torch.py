import torch

"""
N, D_in, H, D_out = 64, 1000, 100, 10

X = torch.randn(N, D_in)
Y = torch.randn(N, D_out)
W1 = torch.randn(D_in, H)
W2 = torch.randn(H, D_out)

Learning_rate = 1e-6

for it in range(500):
    # forward pass
    hidden = X.mm(W1)  # N * H
    hidden_relu = hidden.clamp(min = 0)  # N * H
    Y_hat = hidden_relu.mm(W2)  # N * D_in

    # loss
    loss = (Y - Y_hat).pow(2).sum().item()
    print(it, loss)

    # backward pass
    Y_hat_grad = 2.0 * (Y_hat - Y)  # N * D_in
    W2_grad = hidden_relu.t().mm(Y_hat_grad)  # H * D_in
    hidden_relu_grad = Y_hat_grad.mm(W2.t())  # N * H
    hidden_relu_grad[hidden < 0] = 0  # N * H
    W1_grad = X.t().mm(hidden_relu_grad)  # D_in * H

    W1 = W1 - Learning_rate*W1_grad
    W2 = W2 - Learning_rate*W2_grad
"""

"""
X = torch.tensor(1., requires_grad=True)
W = torch.tensor(2., requires_grad=True)
b = torch.tensor(3., requires_grad=True)

Y = W * X + b

Y.backward()
print(X.grad)
"""
"""
N, D_in, H, D_out = 64, 1000, 100, 10

X = torch.randn(N, D_in, requires_grad=True)
Y = torch.randn(N, D_out, requires_grad=True)
W1 = torch.randn(D_in, H,  requires_grad=True)
W2 = torch.randn(H, D_out,  requires_grad=True)

Learning_rate = 1e-6

for it in range(500):
    # forward pass
    Y_hat = X.mm(W1).clamp(min=0).mm(W2)  # N * H
    W1.retain_grad()
    W2.retain_grad()
    # loss
    loss = (Y - Y_hat).pow(2).sum()
    print(it, loss.item())
    # backward pass
    loss.backward()

    with torch.no_grad():
        W1 -= Learning_rate * W1.grad
        W2 -= Learning_rate * W2.grad
        W1.grad.zero_()
        W2.grad.zero_()
"""
"""
import torch.nn as nn

N, D_in, H, D_out = 64, 1000, 100, 10

X = torch.randn(N, D_in, requires_grad=True)
Y = torch.randn(N, D_out, requires_grad=True)

model = nn.Sequential(
    nn.Linear(D_in, H, bias=False),
    nn.ReLU(),
    nn.Linear(H, D_out, bias=False)
)
Loss_func = nn.MSELoss(reduction='sum')
torch.nn.init.normal_(model[0].weight)
torch.nn.init.normal_(model[2].weight)
# Learning_rate = 1e-4
# optimizer = torch.optim.Adam(model.parameters(), lr=Learning_rate)

Learning_rate = 1e-6
optimizer = torch.optim.SGD(model.parameters(), lr=Learning_rate)

for it in range(500):
    # forward pass
    Y_hat = model(X)  # N * H
    loss = Loss_func(Y_hat, Y)
    print(it, loss.item())
    
    optimizer.zero_grad()
    # backward pass
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        for param in model.parameters():
            param -= Learning_rate*param.grad
    model.zero_grad()
    
"""

import torch.nn as nn

N, D_in, H, D_out = 64, 1000, 100, 10

X = torch.randn(N, D_in, requires_grad=True)
Y = torch.randn(N, D_out, requires_grad=True)

model = nn.Sequential(
    nn.Linear(D_in, H, bias=False),
    nn.ReLU(),
    nn.Linear(H, D_out, bias=False)
)
Loss_func = nn.MSELoss(reduction='sum')
torch.nn.init.normal_(model[0].weight)
torch.nn.init.normal_(model[2].weight)
# Learning_rate = 1e-4
# optimizer = torch.optim.Adam(model.parameters(), lr=Learning_rate)

Learning_rate = 1e-6
optimizer = torch.optim.SGD(model.parameters(), lr=Learning_rate)

for it in range(500):
    # forward pass
    Y_hat = model(X)  # N * H
    loss = Loss_func(Y_hat, Y)
    print(it, loss.item())
    
    optimizer.zero_grad()
    # backward pass
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        for param in model.parameters():
            param -= Learning_rate*param.grad
    model.zero_grad()