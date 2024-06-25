# -*- coding:utf-8 -*-
# Author: quant
# Date: 2024/6/18

"""
https://www.youtube.com/watch?v=Gl2WXLIMvKA&list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz&index=6
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

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)

    def forward(self, x):
        """
        x:(batch_size, seq_len, input_size)
        out:(batch_size, seq_len, hidden_size)
        """
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)   # 初始隐状态
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)  # 初始隐状态
        out, _ = self.rnn(x, (h0,c0))

        # out = out.reshape(out.shape[0], -1)
        # out = self.fc(out)   # 所有的位置都输出

        out = self.fc(out[:,-1,:])  # 取最后一个位置的输出， many to one
        return out
