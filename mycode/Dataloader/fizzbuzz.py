import numpy as np
import torch
import torch.nn as nn

Num_digits = 10


def fizz_buzz(i):
    if i % 15 == 0:
        return 3
    if i % 5 == 0:
        return 2
    if i % 3 == 0:
        return 1
    return 0


def fizz_buzz_encoder(i):
    return [str(i), "fizz", "buzz", "fizzbuzz"][fizz_buzz(i)]


def binary_encoder(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)][::-1])


X = torch.Tensor([binary_encoder(i, Num_digits) for i in range(101, 2 ** Num_digits)])
Y = torch.LongTensor([fizz_buzz(i) for i in range(101, 2 ** Num_digits)])

NUM_HIDDEN = 10
Learning_Rate = 0.05
batch_size = 128

model = nn.Sequential(
    nn.Linear(Num_digits, NUM_HIDDEN),
    nn.ReLU(),
    nn.Linear(NUM_HIDDEN, 4)
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=Learning_Rate)


for ep in range(500):
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        batchX = X[start:end]
        batchY = Y[start:end]

        Y_hat = model(batchX)
        loss = loss_fn(Y_hat, batchY)
        print(ep, loss.item())

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()


test_X = torch.Tensor([binary_encoder(i, Num_digits) for i in range(1, 101)])
pred_Y = model(test_X)

pred_Y =  pred_Y.max(1)[1].data.tolist()

print([fizz_buzz_encoder(i) for i in pred_Y])
