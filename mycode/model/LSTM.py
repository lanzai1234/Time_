import torch
import torch.nn as nn


class Lstm_(nn.Module):
    def __init__(self, batch_size=18, seq_len=1):
        super(Lstm_, self).__init__()
        self.Lstm_1 = nn.LSTM(input_size=18, hidden_size=128, num_layers=1, batch_first=False, bias=True, dropout=0.1, bidirectional=False)
        self.Lstm_2 = nn.LSTM(input_size=128, hidden_size=256, num_layers=1, batch_first=False, bias=True, dropout=0.1, bidirectional=False)
        self.Linear_1 = nn.Linear(256, 64)
        self.Linear_2 = nn.Linear(64, 1)
        self.Tanh = nn.Tanh()

    def forward(self, X):
        mid_res, _ = self.Lstm_1(X)
        mid_res = self.Tanh(mid_res)
        mid_res, _ = self.Lstm_2(mid_res)
        mid_res = self.Linear_1(mid_res)
        mid_res = self.Linear_2(self.Tanh(mid_res))
        mid_res = mid_res.view(1, 1, -1)
        return mid_res


class LSTM_autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM_autoencoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.LSTM_en1 = nn.LSTM(input_size=self.input_size, hidden_size=128, num_layers=1, batch_first=False, bias=True, dropout=0.1, bidirectional=False)
        self.LSTM_en2 = nn.LSTM(input_size=128, hidden_size=self.hidden_size, num_layers=1, batch_first=False, bias=True, dropout=0.1, bidirectional=False)

        self.LSTM_de1 = nn.LSTM(input_size=self.hidden_size, hidden_size=128, num_layers=1, batch_first=False, bias=True, dropout=0.1, bidirectional=False)
        self.LSTM_de2 = nn.LSTM(input_size=128, hidden_size=self.input_size, num_layers=1, batch_first=False, bias=True, dropout=0.1, bidirectional=False)
        
        self.Tanh = nn.Tanh()

    def forward(self, X):
        __, _ = self.LSTM_en1(X)
        __, _ = self.LSTM_en2(__)
        __ = self.Tanh(__)
        out, _ = self.LSTM_de1(__)
        out, _ = self.LSTM_de2(out)
        return out
