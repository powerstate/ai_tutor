# -*- coding:utf-8 -*-
# Author: quant
# Date: 2024/6/18

"""
https://www.youtube.com/watch?v=Gl2WXLIMvKA&list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz&index=5&pp=iAQB
"""

import torch
from torch import nn

input_size = 28
sequence_length = 28
hidden_size = 256
num_layers = 2
num_classes = 10
batch_size = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size*sequence_length, num_classes)

    def forward(self, x):
        """
        x:(batch_size, seq_len, input_size)
        out:(batch_size, seq_len, hidden_size)
        """
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)   # 初始隐状态
        out, _ = self.rnn(x, h0)

        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)   # 所有的位置都输出

        # out = self.fc(out[:,-1,:])  # 取最后一个位置的输出， many to one
        return out

class SequentialRNN(nn.Module):
    def __init__(self):
        super(SequentialRNN, self).__init__()
        self.rnn = nn.Sequential(
            nn.RNN(input_size, 128, 1, batch_first=True),
            nn.RNN(128, 64, 1, batch_first=True)
        )
        self.fc = nn.Linear(64 * sequence_length, num_classes)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), 64).to(x.device)  # 初始化最后一个RNN层的隐状态
        for rnn in self.rnn:
            x, _ = rnn(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size*sequence_length, num_classes)

    def forward(self, x):
        """
        x:(batch_size, seq_len, input_size)
        out:(batch_size, seq_len, hidden_size)
        """
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)   # 初始隐状态
        out, _ = self.gru(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size*sequence_length, num_classes)

    def forward(self, x):
        """
        x:(batch_size, seq_len, input_size)
        out:(batch_size, seq_len, hidden_size)
        """
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)   # 初始隐状态
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))  # 传入隐藏状态和细胞状态
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out

models = RNN(input_size, hidden_size, num_layers, num_classes)