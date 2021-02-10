import numpy as np
import matplotlib.pyplot as plt

"""
a = np.array([
    [1, 3, 5, 6, 7, 8],
    [3, 2, 1, 4, 6, 2],
    [5, 1, 8, 3, 9, 3],
    [6, 4, 3, 1, 2, 6],
    [7, 6, 9, 2, 4, 5],
    [8, 2, 3, 6, 5, 9]
])

e_vals, e_vecs = np.linalg.eig(a)
index = 0

#plt.figure(12)

for e_val in e_vals:
    plot = plt.subplot(331+index)
    plot.plot(range(len(e_vecs[index])), e_vecs[index])
    index = index + 1
plt.show()
"""

N, D_in, H, D_out = 64, 1000, 100, 10

X = np.random.randn(N, D_in)
Y = np.random.randn(N, D_out)
W1 = np.random.randn(D_in, H)
W2 = np.random.randn(H, D_out)

Learning_rate = 1e-6

for it in range(500):
    # forward pass
    hidden = X.dot(W1)  # N * H
    hidden_relu = np.maximum(hidden, 0)  # N * H
    Y_hat = np.dot(hidden_relu, W2)  # N * D_in

    # loss
    loss = np.square(Y - Y_hat).mean()
    print(it, loss)

    # backward pass
    Y_hat_grad = 2.0 * (Y_hat - Y)  # N * D_in
    W2_grad = hidden_relu.T.dot(Y_hat_grad)  # H * D_in
    hidden_relu_grad = Y_hat_grad.dot(W2.T)  # N * H
    hidden_relu_grad[hidden < 0] = 0  # N * H
    W1_grad = X.T.dot(hidden_relu_grad)  # D_in * H

    W1 = W1 - Learning_rate*W1_grad
    W2 = W2 - Learning_rate*W2_grad




