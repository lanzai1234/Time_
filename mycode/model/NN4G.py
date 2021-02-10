import torch 
import torch.nn as nn

Node_Num = 5
node_feature = 10 


class NN4G(nn.Module):
    # adj 表示图的邻接矩阵，取值在0或1
    def __init__(self, adj, node_num):
        super(NN4G, self).__init__()
        self.adj = adj
        self.W = torch.randn(node_num, node_num)
    
    def forward(self, X):
        output = self.adj.mm(self.W).mm(X)
        return output


Adj = torch.Tensor([
    [0, 1, 0, 1, 1],
    [1, 0, 1, 0, 1],
    [0, 1, 0, 0, 0],
    [1, 0, 0, 0, 1],
    [1, 1, 0, 1, 0]
])
X = torch.ones(Node_Num, node_feature)
model = NN4G(adj=Adj, node_num=Node_Num)

Y_hat = model(X)
print(X, Y_hat)


