import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

"""
class GAT(nn.Moudule):
    # graph (V, E)
    # V (N, d_in)
    # E (N, N)
    def __init__(self, graph):
        super(GAT, self).__init__()
        self.V, self.E = graph
        self.N = self.V.shape[0]
        self.d_in = self.V.shape[1]

    def attention(self, H, node_index):
        # param:
        # H : (H, d_in)
        node_Tensor = torch.Tensor(H[node_index], dtype=torch.float32)
        node_neighbors = []

        for neighbor_index in range(self.N):
            if self.E[node_index][neighbor_index] == 1 and neighbor_index != node_index:
                node_neighbors.append(neighbor_index)
        
        neighbor_weights = torch.Tensor([])
        for node_neighbor in node_neighbors:
            neighbor_Tensor = torch.Tensor(H[node_neighbor], dtype=torch.float32)
            a = torch.randn(2 * self.d_in)
            weight = a.mm(torch.cat((node_Tensor, neighbor_Tensor), dim=0).t())
            neighbor_weights.add_(weight)
        
        return neighbor_weights
    
    def forward(self, x):
"""

# torch.cat() (o1, o2):tuple, dim:指定维度
# torch.mm()
# tensor.data 或 tensor.detach()可以分离出独立于计算图的tensor
# repeat_interleave(self: Tensor, repeats: _int, dim: Optional[_int]=None) 复制指定维度
# tensor.repeat()
# tensor.squeeze() 取消一个维度
# tensor.unsqueeze() 增加一个维度


class Graph_Attention_Covlayer(nn.Module):
    def __init__(self, in_Channel, out_Channel, dropout, alpha, concat=True):
        super(Graph_Attention_Covlayer, self).__init__()
        self.in_Channel = in_Channel
        self.out_Channel = out_Channel
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_Channel, out_Channel)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_Channel, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.LeakyReLU = nn.LeakyReLU(self.alpha)

    def forward(self, H, adj):
        Wh = torch.mm(H, self.W)
        Attention_Mat = self.get_Attention_Mat(Wh)
        Attention = torch.matmul(Attention_Mat, self.a).squeeze(2)
        Attention = self.LeakyReLU(Attention)

        Attention = F.softmax(Attention, dim=1)

        zero_mask = torch.zeros_like(Attention)
        Attention = torch.where(adj > 0, Attention, zero_mask)

        output = Attention.mm(Wh)

        if self.concat is True:
            output = F.elu(output)
        return output
 
    def get_Attention_Mat(self, Wh):
        N = Wh.size()[0]
        Wh_1 = Wh.repeat_interleave(N, dim=0)
        Wh_2 = Wh.repeat(N, 1)

        Attention_Mat = torch.cat((Wh_1, Wh_2), dim=1)

        return Attention_Mat.view(N, N, 2 * self.out_Channel)


class Gat(nn.Module):
    def __init__(self, n_feature, h_feature, n_class, dropout, alpha, n_nodes, K=12):
        super(Gat, self).__init__()
        self.K = K
        self.n_nodes = n_nodes
        self.Attentions = [Graph_Attention_Covlayer(n_feature, h_feature, dropout, alpha, concat=True) for _ in range(n_nodes)]
        for i, attention in enumerate(self.Attentions):
            self.add_module("attention{}".format(i), attention)
        
        self.dropout = dropout

        self.outAtt = Graph_Attention_Covlayer(h_feature * n_nodes, n_class, dropout, alpha, concat=True)

        self.V = nn.Parameter(torch.empty(n_nodes, h_feature))
        nn.init.xavier_uniform_(self.V, gain=1.414)

    def forward(self, X):
        # print(self.V.grad)
        adj = self.Adj_learn(self.K)
        mid = torch.cat([att(X, adj) for att in self.Attentions], dim=1)
        mid = F.dropout(mid, self.dropout, training=self.training)
        out = self.outAtt(mid, adj)
        out = self.V * out
        out = F.elu(out)
        return adj, out

    def Adj_learn(self, K):
        Adj_alt = self.V.mm(self.V.t())
        values, indices = torch.topk(Adj_alt.view(self.n_nodes * self.n_nodes), K, dim=0)
        zero_mask = torch.zeros_like(Adj_alt)
        one_mask = torch.ones_like(Adj_alt)
        Adj = torch.where(Adj_alt > values[-1].item(), Adj_alt, zero_mask)
        Adj = torch.where(Adj_alt > values[-1].item(), one_mask, Adj)
        return Adj

